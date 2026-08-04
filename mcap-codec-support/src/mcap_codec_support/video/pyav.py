"""PyAV-based video compression and decompression backend."""

from __future__ import annotations

import threading
from collections import deque
from fractions import Fraction
from io import BytesIO
from typing import TYPE_CHECKING, Literal, cast

import av
import av.error
from av import Packet, VideoFrame
from typing_extensions import Self

from mcap_codec_support.video.common import (
    DecompressedFrame,
    EncoderConfig,
    VideoEncoderError,
    build_encoder_options,
    raw_image_to_buffer,
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

    from mcap_codec_support._protocols import RawImageMessage


# ---------------------------------------------------------------------------
# Encoder probing
# ---------------------------------------------------------------------------


def _create_codec_context(
    codec_name: str,
    direction: Literal["r", "w"],
) -> VideoCodecContext:
    return cast("VideoCodecContext", av.CodecContext.create(codec_name, direction))


def test_encoder(encoder_name: str) -> bool:
    """Test if an encoder is available via PyAV."""
    try:
        _create_codec_context(encoder_name, "w")
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
        ctx = _create_codec_context("mjpeg", "r")
        ctx.open()
        _decoder_local.mjpeg_ctx = ctx
    return ctx


def _get_png_ctx() -> VideoCodecContext:
    ctx = _decoder_local.png_ctx
    if ctx is None:
        ctx = _create_codec_context("png", "r")
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


_RAW_IMAGE_FORMATS: dict[str, str] = {
    "rgb": "rgb24",
    "rgb8": "rgb24",
    "bgr": "bgr24",
    "bgr8": "bgr24",
    "mono": "gray",
    "mono8": "gray",
    "8uc1": "gray",
}


def raw_image_to_frame(message: RawImageMessage) -> VideoFrame:
    """Convert a ROS Image message to a PyAV frame without NumPy."""
    buffer = raw_image_to_buffer(message)
    pixel_format = _RAW_IMAGE_FORMATS[buffer.encoding]
    frame = VideoFrame(buffer.width, buffer.height, pixel_format)
    plane = frame.planes[0]
    if plane.line_size < buffer.row_bytes:
        raise VideoEncoderError(
            f"PyAV plane stride {plane.line_size} is smaller than row size {buffer.row_bytes}"
        )

    if buffer.step == plane.line_size:
        plane.update(buffer.data[: plane.buffer_size])
        return frame

    packed = bytearray(plane.buffer_size)
    for row in range(buffer.height):
        source_start = row * buffer.step
        target_start = row * plane.line_size
        packed[target_start : target_start + buffer.row_bytes] = buffer.data[
            source_start : source_start + buffer.row_bytes
        ]
    plane.update(packed)
    return frame


def video_frame_to_rgb_bytes(frame: VideoFrame) -> bytes:
    """Return tightly packed RGB24 bytes for a PyAV frame."""
    rgb = frame.reformat(format="rgb24")
    plane = rgb.planes[0]
    row_bytes = rgb.width * 3
    raw = bytes(plane)
    if plane.line_size == row_bytes:
        return raw[: row_bytes * rgb.height]
    return b"".join(
        raw[row * plane.line_size : row * plane.line_size + row_bytes] for row in range(rgb.height)
    )


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
            self._context = _create_codec_context(codec_name, "w")
        except (av.error.FFmpegError, ValueError) as exc:
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

    def _require_context(self) -> VideoCodecContext:
        context = self._context
        if context is None:
            raise VideoEncoderError("Encoder is closed")
        return context

    def encode(self, frame: VideoFrame) -> bytes | None:
        """Encode a single frame and return compressed video bytes, or None if buffered."""
        context = self._require_context()
        needs_resize = frame.width != self.config.width or frame.height != self.config.height
        needs_fmt = frame.format.name not in self._YUV420P_COMPAT
        if needs_resize or needs_fmt:
            frame = frame.reformat(
                width=self.config.width, height=self.config.height, format=context.pix_fmt
            )
        frame.pts = self._frame_index
        self._frame_index += 1

        try:
            packets = list(context.encode(frame))
        except av.error.FFmpegError as exc:
            raise VideoEncoderError(f"Encoding error: {exc}") from exc

        if not packets:
            return None
        return b"".join(bytes(packet) for packet in packets)

    def flush_packets(self) -> list[bytes]:
        """Flush remaining buffered frames as one bytes blob per packet."""
        context = self._require_context()
        try:
            packets = list(context.encode(None))
        except av.error.FFmpegError:
            return []
        return [bytes(packet) for packet in packets]


# ---------------------------------------------------------------------------
# PyAVVideoDecompressor (H.264/H.265/VP9/AV1 → Image)
# ---------------------------------------------------------------------------

# CompressedVideo ``format`` string -> PyAV decoder codec name.
_DECOMPRESS_CODECS = {
    "h264": ("h264",),
    "h265": ("hevc",),
    "hevc": ("hevc",),
    "vp9": ("vp9",),
    # PyAV/FFmpeg may expose the generic ``av1`` codec as an encoder-only
    # implementation. Prefer explicit decoders that implement packet decode.
    # libdav1d buffers this packetized stream until EOF, while libaom emits one
    # frame per packet and therefore preserves read_message_decoded streaming.
    "av1": ("libaom-av1", "libdav1d", "av1"),
}


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
        self._decoder_codecs: tuple[str, ...] | None = None
        self._pending_frames: deque[VideoFrame] = deque()
        self._jpeg_encoder: VideoCodecContext | None = None
        self._jpeg_pts = 0

    def _ensure_decoder(self, codec: str) -> VideoCodecContext:
        codec_names = _DECOMPRESS_CODECS.get(codec.lower(), ("hevc",))
        if self._decoder is not None and self._decoder_codecs == codec_names:
            return self._decoder

        # A decompressor is normally bound to one stream, but the format is
        # carried in every CompressedVideo message.  If a channel switches to
        # a new codec (for example after a camera reconfiguration), flush the
        # old context before creating the new one. Otherwise delayed B-frames
        # from the old stream are silently lost and packets are fed to the
        # wrong decoder.
        if self._decoder is not None:
            self._pending_frames.extend(self._decoder.decode(None))
        self._decoder = None
        self._decoder_codecs = None
        last_error: av.error.FFmpegError | ValueError | None = None
        for codec_name in codec_names:
            try:
                decoder = _create_codec_context(codec_name, "r")
                decoder.open()
            except (av.error.FFmpegError, ValueError) as exc:
                last_error = exc
                continue
            self._decoder = decoder
            self._decoder_codecs = codec_names
            return decoder
        raise VideoEncoderError(f"No usable decoder for {codec}: {last_error}")

    def _ensure_jpeg_encoder(self, width: int, height: int) -> VideoCodecContext:
        if (
            self._jpeg_encoder is not None
            and self._jpeg_encoder.width == width
            and self._jpeg_encoder.height == height
        ):
            return self._jpeg_encoder
        self._jpeg_encoder = _create_codec_context("mjpeg", "w")
        self._jpeg_encoder.width = width
        self._jpeg_encoder.height = height
        self._jpeg_encoder.pix_fmt = "yuvj420p"
        self._jpeg_encoder.time_base = Fraction(1, 1000)
        self._jpeg_encoder.options = {
            "q:v": str(max(1, 31 - self._jpeg_quality * 31 // 100)),
        }
        self._jpeg_encoder.open()
        self._jpeg_pts = 0
        return self._jpeg_encoder

    def _frame_to_jpeg(self, frame: VideoFrame) -> DecompressedFrame:
        encoder = self._ensure_jpeg_encoder(frame.width, frame.height)
        reformatted = frame.reformat(format="yuvj420p")
        reformatted.pts = self._jpeg_pts
        self._jpeg_pts += 1
        packets = encoder.encode(reformatted)
        return DecompressedFrame(
            data=b"".join(bytes(packet) for packet in packets),
            width=frame.width,
            height=frame.height,
            is_jpeg=True,
        )

    def decompress(self, video_data: bytes, codec: str) -> DecompressedFrame | None:
        decoder = self._ensure_decoder(codec)
        self._pending_frames.extend(decoder.decode(Packet(video_data)))
        if not self._pending_frames:
            return None
        frame = self._pending_frames.popleft()

        if self._video_format == "compressed":
            return self._frame_to_jpeg(frame)

        rgb_frame = frame.reformat(format="rgb24")
        raw_data = video_frame_to_rgb_bytes(rgb_frame)
        return DecompressedFrame(
            data=raw_data,
            width=rgb_frame.width,
            height=rgb_frame.height,
            is_jpeg=False,
        )

    def close(self) -> None:
        """Release native codec contexts."""
        self._decoder = None
        self._decoder_codecs = None
        self._pending_frames.clear()
        self._jpeg_encoder = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def flush(self) -> list[DecompressedFrame]:
        if self._decoder is not None:
            self._pending_frames.extend(self._decoder.decode(None))
        results: list[DecompressedFrame] = []
        while self._pending_frames:
            frame = self._pending_frames.popleft()
            if self._video_format == "compressed":
                results.append(self._frame_to_jpeg(frame))
            else:
                rgb_frame = frame.reformat(format="rgb24")
                raw_data = video_frame_to_rgb_bytes(rgb_frame)
                results.append(
                    DecompressedFrame(
                        data=raw_data,
                        width=rgb_frame.width,
                        height=rgb_frame.height,
                        is_jpeg=False,
                    )
                )
        return results
