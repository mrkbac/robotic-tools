"""Video compression, image transcoding, and decompressor selection helpers."""

from __future__ import annotations

import contextlib
import io
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from mcap_codec_support._schemas import normalize_schema_name
from mcap_codec_support.video.common import (
    DEFAULT_FPS,
    DEFAULT_GOP_SIZE,
    SOFTWARE_CODEC_MAP,
    EncoderMode,
    VideoEncoderError,
    calculate_downscale_dimensions,
    raw_image_to_array,
    raw_image_to_pil,
)
from mcap_codec_support.video.schemas import COMPRESSED_SCHEMAS

try:
    from PIL.Image import Resampling as _Resampling

    _PIL_BILINEAR = _Resampling.BILINEAR
except ImportError:
    _PIL_BILINEAR = None  # ``raw_image_to_pil`` raises if Pillow is missing.

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from concurrent.futures import Future, ThreadPoolExecutor

    import numpy.typing as npt
    from av import VideoFrame
    from small_mcap import DecodedMessage

    from mcap_codec_support._protocols import (
        AnyVideoBackend,
        RawImageMessage,
        VideoDecompressorProtocol,
        VideoEncoderProtocol,
    )
    from mcap_codec_support.video.ffmpeg import FFmpegVideoEncoder
    from mcap_codec_support.video.pyav import VideoEncoder


# Measured aggregate NVENC throughput on this class of GPU peaks around 4-6
# concurrent sessions and then collapses (8 sessions ran *slower* than 2) —
# even when the hardware allows opening that many at all. This is the ceiling
# regardless of what ``probe_max_concurrent_encoders`` finds is openable.
MAX_USEFUL_CONCURRENT_ENCODERS = 6


def probe_max_concurrent_encoders(
    backend: AnyVideoBackend, encoder_name: str, upper_bound: int = 8
) -> int:
    """Probe how many concurrent hardware encode sessions this system can open.

    Hardware encoders (NVENC etc.) cap the number of simultaneously open
    sessions at a driver/hardware-specific limit; recordings routinely carry
    more video topics than that (a camera plus its throttled duplicate, times
    several cameras), and exceeding the limit fails encoder *creation* outright
    rather than degrading gracefully. Software encoders have no such limit, so
    this only probes for encoder names containing a known hardware marker. Tiny
    throwaway contexts keep the probe cheap; a small margin below the empirical
    result absorbs the transient contention observed near the real limit.
    """
    if not any(marker in encoder_name for marker in ("nvenc", "videotoolbox", "vaapi")):
        return MAX_USEFUL_CONCURRENT_ENCODERS
    opened: list[VideoEncoderProtocol[Any]] = []
    try:
        for _ in range(upper_bound):
            try:
                # 320x240: small enough to probe cheaply, but well above the
                # minimum resolution some hardware encoders reject (e.g. this
                # NVENC build fails avcodec_open2 below roughly 160x96).
                opened.append(backend.create_encoder(320, 240, encoder_name, 28))
            except VideoEncoderError:
                break
    finally:
        for enc in opened:
            with contextlib.suppress(Exception):
                enc.flush_packets()
    margin = 1 if len(opened) < upper_bound else 0
    return max(1, min(len(opened) - margin, MAX_USEFUL_CONCURRENT_ENCODERS))


class _PyAVCompressionBackend:
    label = "pyav"
    prefetch_supported = True

    def test_encoder(self, encoder_name: str) -> bool:
        from mcap_codec_support.video.pyav import test_encoder  # noqa: PLC0415

        return test_encoder(encoder_name)

    def resolve_encoder(self, codec: str) -> str:
        from mcap_codec_support.video.pyav import resolve_encoder  # noqa: PLC0415

        return resolve_encoder(codec)

    def decode_compressed(self, data: bytes) -> tuple[VideoFrame, int, int]:
        from mcap_codec_support.video.pyav import decode_compressed_frame  # noqa: PLC0415

        frame = decode_compressed_frame(data)
        return frame, frame.width, frame.height

    def decode_image(self, msg: DecodedMessage, schema_name: str) -> tuple[VideoFrame, int, int]:
        if schema_name in COMPRESSED_SCHEMAS:
            return self.decode_compressed(bytes(msg.decoded_message.data))

        import av  # noqa: PLC0415

        rgb_array = raw_image_to_array(msg.decoded_message)
        frame = av.VideoFrame.from_ndarray(rgb_array, format="rgb24")
        return frame, frame.width, frame.height

    def create_encoder(
        self,
        width: int,
        height: int,
        codec_name: str,
        quality: int,
        *,
        input_pix_fmt: str | None = None,
        scale: tuple[int, int] | None = None,
    ) -> VideoEncoder:
        # PyAV reformats input frames per-frame inside VideoEncoder.encode, so
        # the protocol's pix-fmt / scale knobs are FFmpeg-CLI-only.
        del input_pix_fmt, scale
        from mcap_codec_support.video.pyav import VideoEncoder  # noqa: PLC0415

        return VideoEncoder(
            width=width,
            height=height,
            codec_name=codec_name,
            quality=quality,
            target_fps=DEFAULT_FPS,
            gop_size=DEFAULT_GOP_SIZE,
        )

    def get_pix_fmt(self, topic: str) -> str | None:
        del topic
        return None


class _FfmpegCliCompressionBackend:
    label = "ffmpeg-cli"
    prefetch_supported = False

    def __init__(self) -> None:
        self._topic_pix_fmt: dict[str, str | None] = {}

    def get_pix_fmt(self, topic: str) -> str | None:
        return self._topic_pix_fmt.get(topic)

    def test_encoder(self, encoder_name: str) -> bool:
        from mcap_codec_support.video.ffmpeg import check_encoder_cli  # noqa: PLC0415

        return check_encoder_cli(encoder_name)

    def resolve_encoder(self, codec: str) -> str:
        from mcap_codec_support.video.ffmpeg import resolve_encoder  # noqa: PLC0415

        return resolve_encoder(codec)

    def decode_compressed(self, data: bytes) -> tuple[bytes, int, int]:
        from mcap_codec_support.video.ffmpeg import probe_image_dimensions  # noqa: PLC0415

        width, height = probe_image_dimensions(data)
        return data, width, height

    def decode_image(self, msg: DecodedMessage, schema_name: str) -> tuple[bytes, int, int]:
        data = bytes(msg.decoded_message.data)
        topic = msg.channel.topic

        if schema_name in COMPRESSED_SCHEMAS:
            self._topic_pix_fmt[topic] = None
            frame, width, height = self.decode_compressed(data)
            return frame, width, height

        from mcap_codec_support.video.ffmpeg import ROS_ENCODING_TO_PIX_FMT  # noqa: PLC0415

        encoding = str(msg.decoded_message.encoding).lower()
        pix_fmt = ROS_ENCODING_TO_PIX_FMT.get(encoding)
        if not pix_fmt:
            raise VideoEncoderError(f"Unsupported image encoding: {msg.decoded_message.encoding}")
        self._topic_pix_fmt[topic] = pix_fmt
        return data, msg.decoded_message.width, msg.decoded_message.height

    def create_encoder(
        self,
        width: int,
        height: int,
        codec_name: str,
        quality: int,
        *,
        input_pix_fmt: str | None = None,
        scale: tuple[int, int] | None = None,
    ) -> FFmpegVideoEncoder:
        from mcap_codec_support.video.ffmpeg import (  # noqa: PLC0415
            FFmpegVideoEncoder,
            probe_hw_mjpeg_decoder,
        )

        # JPEG path only: offload decode to a hardware MJPEG decoder when one
        # probes healthy (cached; None → CPU decode). Keeps the pipe compact
        # (JPEG, not raw frames) either way.
        decode_codec = probe_hw_mjpeg_decoder() if input_pix_fmt is None else None
        return FFmpegVideoEncoder(
            width=width,
            height=height,
            codec_name=codec_name,
            quality=quality,
            target_fps=DEFAULT_FPS,
            gop_size=DEFAULT_GOP_SIZE,
            input_pix_fmt=input_pix_fmt,
            scale=scale,
            decode_codec=decode_codec,
        )


def create_video_compression_backend(
    mode: EncoderMode, codec: str, *, do_video: bool
) -> AnyVideoBackend:
    """Select the roscompress video backend.

    ``AUTO`` prefers PyAV (in-process, no subprocess/pipe overhead) but only
    when PyAV can actually reach a hardware encoder. A pip-installed PyAV wheel
    is typically software-only, so if PyAV would fall back to a CPU encoder
    (e.g. libx264) while the system ``ffmpeg`` exposes a hardware encoder (e.g.
    NVENC), AUTO picks the ffmpeg-cli backend instead — hardware encoding
    without needing a custom PyAV build.

    ``GSTREAMER`` is **opt-in only**, never chosen by AUTO: it uses the Jetson
    hardware JPEG decoder (``nvjpegdec``), whose full-range/limited-range colour
    handling is not faithful to libjpeg on all inputs (it can crush shadows on
    full-range JFIF footage — see :mod:`mcap_codec_support.video.gstreamer`), so it
    must not be selected without the user asking for it. A timed liveness probe
    guards the explicit path against a codec stack that hangs.
    """
    if mode is EncoderMode.GSTREAMER:
        from mcap_codec_support.video.gstreamer import (  # noqa: PLC0415
            GStreamerCompressionBackend,
            probe_hw_jpeg_pipeline,
        )

        if not probe_hw_jpeg_pipeline(codec):
            raise VideoEncoderError(
                "GStreamer video pipeline did not produce output within the probe "
                "timeout — the L4T GStreamer codec stack may be unavailable or "
                "wedged. Use --video-backend ffmpeg-cli (or auto)."
            )
        return GStreamerCompressionBackend()
    if mode is EncoderMode.FFMPEG_CLI:
        return _FfmpegCliCompressionBackend()

    pyav_backend = _PyAVCompressionBackend()
    if mode is not EncoderMode.AUTO or not do_video:
        return pyav_backend

    try:
        pyav_encoder = pyav_backend.resolve_encoder(codec)
    except (ImportError, ValueError):
        # PyAV missing or no usable encoder at all — fall back to ffmpeg.
        return _FfmpegCliCompressionBackend()

    if _is_software_encoder(pyav_encoder):
        # PyAV can only do software; use ffmpeg-cli if it offers hardware.
        ffmpeg_backend = _FfmpegCliCompressionBackend()
        try:
            if not _is_software_encoder(ffmpeg_backend.resolve_encoder(codec)):
                return ffmpeg_backend
        except (ImportError, ValueError, VideoEncoderError):
            pass  # no system ffmpeg / no encoder — stay on PyAV software
    return pyav_backend


def _is_software_encoder(encoder_name: str) -> bool:
    return encoder_name in set(SOFTWARE_CODEC_MAP.values())


def prefetch_image_decodes(
    messages: Iterable[DecodedMessage],
    backend: AnyVideoBackend,
    pool: ThreadPoolExecutor,
    prefetch: int = 8,
) -> Iterator[tuple[DecodedMessage, Future[Any] | None]]:
    """Wrap message iterator to decode compressed images in background threads."""
    buffer: deque[tuple[DecodedMessage, Future[Any] | None]] = deque()

    for msg in messages:
        schema_name = normalize_schema_name(msg.schema.name) if msg.schema else ""
        if schema_name in COMPRESSED_SCHEMAS:
            data = bytes(msg.decoded_message.data)
            future: Future[Any] | None = pool.submit(backend.decode_compressed, data)
        else:
            future = None
        buffer.append((msg, future))

        if len(buffer) > prefetch:
            yield buffer.popleft()

    while buffer:
        yield buffer.popleft()


def encode_raw_image_to_jpeg(
    decoded_message: RawImageMessage, *, jpeg_quality: int, scale: int | None
) -> tuple[bytes, int, int]:
    """Encode a raw ROS Image message to JPEG using Pillow."""
    image = raw_image_to_pil(decoded_message)
    src_w, src_h = image.size
    if scale is not None:
        target_w, target_h = calculate_downscale_dimensions(src_w, src_h, scale)
    else:
        target_w, target_h = src_w, src_h

    target_w -= target_w % 2
    target_h -= target_h % 2
    if target_w < 2 or target_h < 2:
        raise VideoEncoderError(f"Source frame too small ({target_w}x{target_h}) for JPEG encoding")

    if target_w != src_w or target_h != src_h:
        image = image.resize((target_w, target_h), _PIL_BILINEAR)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=jpeg_quality)
    return buf.getvalue(), target_w, target_h


def decode_compressed_image_to_rgb_array(data: bytes) -> npt.NDArray[np.uint8]:
    """Decode JPEG/PNG compressed image bytes to an RGB (uint8) numpy array."""
    from mcap_codec_support.video.pyav import decode_compressed_frame  # noqa: PLC0415

    rgb = decode_compressed_frame(data).to_ndarray(format="rgb24")
    return np.asarray(rgb, dtype=np.uint8)


def create_video_decompressor(
    video_format: str = "compressed",
    jpeg_quality: int = 90,
    *,
    mode: EncoderMode = EncoderMode.AUTO,
) -> VideoDecompressorProtocol:
    """Create a video decompressor using the requested backend."""
    if mode == EncoderMode.PYAV:
        from mcap_codec_support.video.pyav import PyAVVideoDecompressor  # noqa: PLC0415

        return PyAVVideoDecompressor(video_format=video_format, jpeg_quality=jpeg_quality)

    if mode == EncoderMode.FFMPEG_CLI:
        from mcap_codec_support.video.ffmpeg import FFmpegVideoDecompressor  # noqa: PLC0415

        return FFmpegVideoDecompressor(video_format=video_format, jpeg_quality=jpeg_quality)

    try:
        from mcap_codec_support.video.pyav import PyAVVideoDecompressor  # noqa: PLC0415

        return PyAVVideoDecompressor(video_format=video_format, jpeg_quality=jpeg_quality)
    except ImportError:
        from mcap_codec_support.video.ffmpeg import (  # noqa: PLC0415
            FFmpegVideoDecompressor,
            find_ffmpeg,
        )

        if find_ffmpeg():
            return FFmpegVideoDecompressor(video_format=video_format, jpeg_quality=jpeg_quality)
        raise
