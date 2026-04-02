"""PyAV-based video compression and decompression backend.

All ``av`` (PyAV) usage is confined to this module. Importing this file
requires PyAV and numpy to be installed.
"""

from __future__ import annotations

import threading
from fractions import Fraction
from io import BytesIO
from typing import TYPE_CHECKING, Protocol, cast

import av
import av.error
import numpy as np
from av import Packet, VideoFrame
from typing_extensions import Self

from pymcap_cli.encoding.encoder_common import (
    EncoderConfig,
    VideoEncoderError,
    build_encoder_options,
)
from pymcap_cli.encoding.encoder_common import (
    resolve_encoder as _resolve_encoder,
)
from pymcap_cli.encoding.encoder_common import (
    resolve_encoder_for_backend as _resolve_encoder_for_backend,
)
from pymcap_cli.encoding.video_protocols import DecompressedFrame

if TYPE_CHECKING:
    from av.container import InputContainer
    from av.video.codeccontext import VideoCodecContext
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# ROS message protocols (for type checking)
# ---------------------------------------------------------------------------


class CompressedImageMsg(Protocol):
    """Protocol for ROS CompressedImage messages."""

    @property
    def data(self) -> bytes | bytearray | memoryview: ...


class ImageMsg(Protocol):
    """Protocol for ROS Image messages."""

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def encoding(self) -> str: ...

    @property
    def data(self) -> bytes | bytearray | memoryview: ...


# ---------------------------------------------------------------------------
# Encoder probing
# ---------------------------------------------------------------------------


def test_encoder(encoder_name: str) -> bool:
    """Test if an encoder is available via PyAV."""
    try:
        av.CodecContext.create(encoder_name, "w")
    except (av.error.FFmpegError, ValueError):
        return False
    else:
        return True


def resolve_encoder(codec: str, *, use_hardware: bool = True) -> str:
    """Pick the best available encoder for *codec* using PyAV to probe."""
    return _resolve_encoder(codec, test_fn=test_encoder, use_hardware=use_hardware)


def resolve_encoder_for_backend(codec: str, backend: str) -> str:
    """Pick the encoder for *codec* using the specified *backend* (PyAV probe)."""
    return _resolve_encoder_for_backend(codec, backend, test_fn=test_encoder)


# ---------------------------------------------------------------------------
# Image decoding (JPEG / PNG → VideoFrame)
# ---------------------------------------------------------------------------


class _DecoderLocal(threading.local):
    """Thread-local persistent codec contexts for JPEG and PNG decoding."""

    mjpeg_ctx: VideoCodecContext | None = None
    png_ctx: VideoCodecContext | None = None


_decoder_local = _DecoderLocal()


def _get_mjpeg_ctx() -> VideoCodecContext:
    ctx = _decoder_local.mjpeg_ctx
    if ctx is None:
        ctx = cast("VideoCodecContext", av.CodecContext.create("mjpeg", "r"))
        ctx.open()
        _decoder_local.mjpeg_ctx = ctx
    return ctx


def _get_png_ctx() -> VideoCodecContext:
    ctx = _decoder_local.png_ctx
    if ctx is None:
        ctx = av.CodecContext.create("png", "r")
        ctx.open()
        _decoder_local.png_ctx = ctx
    return ctx


def _detect_image_format(data: bytes) -> str:
    if data[:2] == b"\xff\xd8":
        return "mjpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return "unknown"


def _decode_via_container(data: bytes) -> VideoFrame:
    try:
        with cast("InputContainer", av.open(BytesIO(data))) as container:
            for frame in container.decode(video=0):
                return frame
    except av.error.FFmpegError as exc:
        raise VideoEncoderError(f"Failed to decode compressed image: {exc}") from exc
    raise VideoEncoderError("Decoder produced no frames")


def decode_compressed_frame(compressed_data: bytes) -> VideoFrame:
    """Decode a compressed image (JPEG/PNG) to a VideoFrame.

    Uses persistent thread-local codec contexts for performance.
    """
    fmt = _detect_image_format(compressed_data)
    if fmt == "unknown":
        return _decode_via_container(compressed_data)

    ctx = _get_mjpeg_ctx() if fmt == "mjpeg" else _get_png_ctx()
    try:
        for frame in ctx.decode(Packet(compressed_data)):
            return frame
    except av.error.FFmpegError as exc:
        raise VideoEncoderError(f"Failed to decode compressed image: {exc}") from exc

    raise VideoEncoderError("Decoder produced no frames")


def raw_image_to_array(message: ImageMsg) -> NDArray[np.uint8]:
    """Convert a ROS Image message to an RGB numpy array."""
    if not message.data:
        raise VideoEncoderError("Image has no data")

    width = message.width
    height = message.height
    encoding = str(message.encoding).lower()
    data = bytes(message.data)

    if encoding in {"rgb", "rgb8"}:
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        return array.copy()
    if encoding in {"bgr", "bgr8"}:
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        return array[..., ::-1].copy()
    if encoding in {"mono", "mono8", "8uc1"}:
        mono_array = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        return np.repeat(mono_array[:, :, None], 3, axis=2)

    raise VideoEncoderError(f"Unsupported image encoding: {message.encoding}")


# ---------------------------------------------------------------------------
# VideoEncoder (H.264/H.265 frame encoder)
# ---------------------------------------------------------------------------


class VideoEncoder:
    """PyAV-based video encoder for converting images to compressed video."""

    _YUV420P_COMPAT = frozenset({"yuv420p", "yuvj420p"})

    def __init__(
        self,
        width: int,
        height: int,
        codec_name: str,
        quality: int = 28,
        target_fps: float = 30.0,
        gop_size: int = 30,
        *,
        preset: str | None = None,
    ) -> None:
        self.config = EncoderConfig(width=width, height=height, codec_name=codec_name)
        self._target_fps = max(target_fps, 1.0)
        self._frame_index = 0
        self._quality = quality
        self._gop_size = gop_size

        try:
            self._context: VideoCodecContext = cast(
                "VideoCodecContext", av.CodecContext.create(codec_name, "w")
            )
        except av.error.FFmpegError as exc:
            raise VideoEncoderError(f"Failed to create encoder {codec_name}: {exc}") from exc

        fps_int = max(round(self._target_fps), 1)
        self._context.width = width
        self._context.height = height
        self._context.pix_fmt = "yuv420p"
        self._context.time_base = Fraction(1, fps_int)
        self._context.framerate = Fraction(fps_int, 1)
        self._context.gop_size = gop_size
        self._context.max_b_frames = 0

        options, bit_rate = build_encoder_options(codec_name, quality, width, height, preset=preset)
        if bit_rate is not None:
            self._context.bit_rate = bit_rate
        if options:
            self._context.options = options

        try:
            self._context.open()
        except av.error.FFmpegError as exc:
            del self._context
            raise VideoEncoderError(f"Failed to open encoder {codec_name}: {exc}") from exc

    def close(self) -> None:
        """Release the native codec context."""
        if hasattr(self, "_context"):
            del self._context

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def encode(self, frame: VideoFrame) -> bytes | None:
        """Encode a single frame and return compressed video bytes, or None if buffered."""
        needs_resize = frame.width != self.config.width or frame.height != self.config.height
        needs_fmt = frame.format.name not in self._YUV420P_COMPAT
        if needs_resize or needs_fmt:
            frame = frame.reformat(
                width=self.config.width, height=self.config.height, format=self._context.pix_fmt
            )
        frame.pts = self._frame_index
        self._frame_index += 1

        try:
            packets = list(self._context.encode(frame))
        except av.error.FFmpegError as exc:
            raise VideoEncoderError(f"Encoding error: {exc}") from exc

        if not packets:
            return None
        return b"".join(bytes(packet) for packet in packets)

    def flush(self) -> bytes | None:
        """Flush remaining buffered frames from the encoder."""
        try:
            packets = list(self._context.encode(None))
        except av.error.FFmpegError:
            return None
        if not packets:
            return None
        return b"".join(bytes(packet) for packet in packets)


# ---------------------------------------------------------------------------
# PyAVVideoDecompressor (H.264/H.265 → Image)
# ---------------------------------------------------------------------------


class PyAVVideoDecompressor:
    """Decompresses H.264/H.265 video to JPEG or raw RGB using PyAV.

    Implements ``VideoDecompressorProtocol``.
    """

    def __init__(
        self,
        video_format: str = "compressed",
        jpeg_quality: int = 90,
    ) -> None:
        self._video_format = video_format
        self._jpeg_quality = jpeg_quality
        self._decoder: VideoCodecContext | None = None
        self._jpeg_encoder: VideoCodecContext | None = None

    def _ensure_decoder(self, codec: str) -> VideoCodecContext:
        if self._decoder is not None:
            return self._decoder
        codec_name = "h264" if codec == "h264" else "hevc"
        self._decoder = av.CodecContext.create(codec_name, "r")
        self._decoder.open()
        return self._decoder

    def _ensure_jpeg_encoder(self, width: int, height: int) -> VideoCodecContext:
        if self._jpeg_encoder is not None:
            return self._jpeg_encoder
        self._jpeg_encoder = cast("VideoCodecContext", av.CodecContext.create("mjpeg", "w"))
        self._jpeg_encoder.width = width
        self._jpeg_encoder.height = height
        self._jpeg_encoder.pix_fmt = "yuvj420p"
        self._jpeg_encoder.time_base = Fraction(1, 1000)
        self._jpeg_encoder.options = {
            "q:v": str(max(1, 31 - self._jpeg_quality * 31 // 100)),
        }
        self._jpeg_encoder.open()
        return self._jpeg_encoder

    def decompress(self, video_data: bytes, codec: str) -> DecompressedFrame | None:
        decoder = self._ensure_decoder(codec)
        frames = decoder.decode(Packet(video_data))
        if not frames:
            return None
        frame = frames[-1]

        if self._video_format == "compressed":
            encoder = self._ensure_jpeg_encoder(frame.width, frame.height)
            reformatted = frame.reformat(format="yuvj420p")
            reformatted.pts = 0
            packets = encoder.encode(reformatted)
            jpeg_data = b"".join(bytes(p) for p in packets)
            return DecompressedFrame(
                data=jpeg_data,
                width=frame.width,
                height=frame.height,
                is_jpeg=True,
            )

        rgb_frame = frame.reformat(format="rgb24")
        raw_data = rgb_frame.to_ndarray().tobytes()
        return DecompressedFrame(
            data=raw_data,
            width=rgb_frame.width,
            height=rgb_frame.height,
            is_jpeg=False,
        )

    def close(self) -> None:
        """Release native codec contexts."""
        self._decoder = None
        self._jpeg_encoder = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def flush(self) -> list[DecompressedFrame]:
        if self._decoder is None:
            return []
        frames = self._decoder.decode(None)
        results: list[DecompressedFrame] = []
        for frame in frames:
            if self._video_format == "compressed":
                encoder = self._ensure_jpeg_encoder(frame.width, frame.height)
                reformatted = frame.reformat(format="yuvj420p")
                reformatted.pts = 0
                packets = encoder.encode(reformatted)
                jpeg_data = b"".join(bytes(p) for p in packets)
                results.append(
                    DecompressedFrame(
                        data=jpeg_data,
                        width=frame.width,
                        height=frame.height,
                        is_jpeg=True,
                    )
                )
            else:
                rgb_frame = frame.reformat(format="rgb24")
                raw_data = rgb_frame.to_ndarray().tobytes()
                results.append(
                    DecompressedFrame(
                        data=raw_data,
                        width=rgb_frame.width,
                        height=rgb_frame.height,
                        is_jpeg=False,
                    )
                )
        return results
