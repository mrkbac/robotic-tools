"""PyAV-based video compression and decompression backend.

All ``av`` (PyAV) usage is confined to this module. Importing this file
requires PyAV and numpy to be installed.
"""

from __future__ import annotations

import threading
from fractions import Fraction
from io import BytesIO
from typing import TYPE_CHECKING, cast

import av
import av.error
from av import Packet, VideoFrame
from typing_extensions import Self

from mcap_codec_support.video.common import (
    DecompressedFrame,
    EncoderConfig,
    VideoEncoderError,
    build_encoder_options,
)
from mcap_codec_support.video.common import (
    resolve_encoder as _resolve_encoder,
)
from mcap_codec_support.video.common import (
    resolve_encoder_for_backend as _resolve_encoder_for_backend,
)

if TYPE_CHECKING:
    from av.container import InputContainer
    from av.video.codeccontext import VideoCodecContext


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
        self._context: VideoCodecContext | None = None

        try:
            self._context = cast("VideoCodecContext", av.CodecContext.create(codec_name, "w"))
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
            self._context = None
            raise VideoEncoderError(f"Failed to open encoder {codec_name}: {exc}") from exc

    def close(self) -> None:
        """Release the native codec context."""
        if self._context is not None:
            del self._context
            self._context = None

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

    def flush_packets(self) -> list[bytes]:
        """Flush remaining buffered frames as one bytes blob per packet."""
        try:
            packets = list(self._context.encode(None))
        except av.error.FFmpegError:
            return []
        return [bytes(packet) for packet in packets]


# ---------------------------------------------------------------------------
# JpegEncoder (single-frame MJPEG)
# ---------------------------------------------------------------------------


class JpegEncoder:
    """PyAV-based MJPEG encoder for converting individual frames to JPEG.

    JPEG is intra-only, so each call to ``encode`` produces exactly one
    JPEG-encoded blob with no buffering.
    """

    def __init__(self, width: int, height: int, quality: int = 90) -> None:
        if not 1 <= quality <= 100:
            raise VideoEncoderError(f"JPEG quality must be in [1, 100], got {quality}")
        self.config = EncoderConfig(width=width, height=height, codec_name="mjpeg")
        self._quality = quality
        self._context: VideoCodecContext | None = None
        try:
            self._context = cast("VideoCodecContext", av.CodecContext.create("mjpeg", "w"))
        except av.error.FFmpegError as exc:
            raise VideoEncoderError(f"Failed to create mjpeg encoder: {exc}") from exc

        self._context.width = width
        self._context.height = height
        self._context.pix_fmt = "yuvj420p"
        self._context.time_base = Fraction(1, 1000)
        # PyAV's mjpeg encoder maps q:v 1..31 (lower = better). Convert from 1..100.
        qv = max(1, min(31, 32 - quality * 31 // 100))
        self._context.options = {"q:v": str(qv)}

        try:
            self._context.open()
        except av.error.FFmpegError as exc:
            self._context = None
            raise VideoEncoderError(f"Failed to open mjpeg encoder: {exc}") from exc

        self._frame_index = 0

    def encode(self, frame: VideoFrame) -> bytes:
        """Encode a single frame to a complete JPEG blob."""
        if (
            frame.width != self.config.width
            or frame.height != self.config.height
            or frame.format.name != "yuvj420p"
        ):
            frame = frame.reformat(
                width=self.config.width, height=self.config.height, format="yuvj420p"
            )
        frame.pts = self._frame_index
        self._frame_index += 1
        try:
            packets = list(self._context.encode(frame))
        except av.error.FFmpegError as exc:
            raise VideoEncoderError(f"JPEG encoding error: {exc}") from exc
        if not packets:
            raise VideoEncoderError("MJPEG encoder produced no output")
        return b"".join(bytes(p) for p in packets)

    def close(self) -> None:
        """Release the native codec context."""
        if self._context is not None:
            del self._context
            self._context = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# PyAVVideoDecompressor (H.264/H.265/VP9/AV1 → Image)
# ---------------------------------------------------------------------------

# CompressedVideo ``format`` string -> PyAV decoder codec name.
_DECOMPRESS_CODECS = {"h264": "h264", "h265": "hevc", "hevc": "hevc", "vp9": "vp9", "av1": "av1"}


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
        codec_name = _DECOMPRESS_CODECS.get(codec.lower(), "hevc")
        self._decoder = cast("VideoCodecContext", av.CodecContext.create(codec_name, "r"))
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
