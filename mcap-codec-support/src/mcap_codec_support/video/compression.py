"""Video compression, image transcoding, and decompressor selection helpers."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from mcap_codec_support._schemas import normalize_schema_name
from mcap_codec_support.video.common import (
    DEFAULT_FPS,
    DEFAULT_GOP_SIZE,
    EncoderMode,
    VideoEncoderError,
    calculate_downscale_dimensions,
    raw_image_to_array,
)
from mcap_codec_support.video.schemas import COMPRESSED_SCHEMAS

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from concurrent.futures import Future, ThreadPoolExecutor

    from small_mcap import DecodedMessage

    from mcap_codec_support._protocols import VideoCompressionBackend, VideoDecompressorProtocol


class _PyAVCompressionBackend:
    label = "pyav"
    prefetch_supported = True

    def test_encoder(self, encoder_name: str) -> bool:
        from mcap_codec_support.video.pyav import test_encoder  # noqa: PLC0415

        return test_encoder(encoder_name)

    def resolve_encoder(self, codec: str) -> str:
        from mcap_codec_support.video.pyav import resolve_encoder  # noqa: PLC0415

        return resolve_encoder(codec)

    def decode_compressed(self, data: bytes) -> tuple[Any, int, int]:
        from mcap_codec_support.video.pyav import decode_compressed_frame  # noqa: PLC0415

        frame = decode_compressed_frame(data)
        return frame, frame.width, frame.height

    def decode_image(self, msg: DecodedMessage, schema_name: str) -> tuple[Any, int, int]:
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
        input_pix_fmt: str | None = None,  # noqa: ARG002
        scale: tuple[int, int] | None = None,  # noqa: ARG002
    ) -> Any:
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

    def decode_compressed(self, data: bytes) -> tuple[Any, int, int]:
        from mcap_codec_support.video.ffmpeg import probe_image_dimensions  # noqa: PLC0415

        width, height = probe_image_dimensions(data)
        return data, width, height

    def decode_image(self, msg: DecodedMessage, schema_name: str) -> tuple[Any, int, int]:
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
    ) -> Any:
        from mcap_codec_support.video.ffmpeg import FFmpegVideoEncoder  # noqa: PLC0415

        return FFmpegVideoEncoder(
            width=width,
            height=height,
            codec_name=codec_name,
            quality=quality,
            target_fps=DEFAULT_FPS,
            gop_size=DEFAULT_GOP_SIZE,
            input_pix_fmt=input_pix_fmt,
            scale=scale,
        )


def create_video_compression_backend(
    mode: EncoderMode, codec: str, *, do_video: bool
) -> VideoCompressionBackend:
    """Select the roscompress video backend."""
    if mode is EncoderMode.FFMPEG_CLI:
        return _FfmpegCliCompressionBackend()

    pyav_backend = _PyAVCompressionBackend()
    if mode is EncoderMode.AUTO and do_video:
        try:
            pyav_backend.resolve_encoder(codec)
        except (ImportError, ValueError):
            return _FfmpegCliCompressionBackend()
    return pyav_backend


def prefetch_image_decodes(
    messages: Iterable[DecodedMessage],
    backend: VideoCompressionBackend,
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


def _encode_rgb_array_to_jpeg(rgb_array: Any, quality: int) -> bytes:
    try:
        import imagecodecs  # noqa: PLC0415
    except ImportError as exc:
        raise VideoEncoderError(
            "imagecodecs is required for JPEG image encoding. "
            "Install with: uv add 'mcap-codec-support[image]'"
        ) from exc
    return bytes(imagecodecs.jpeg_encode(rgb_array, level=quality))


def _resize_rgb_array(rgb_array: Any, width: int, height: int) -> Any:
    import av  # noqa: PLC0415

    frame = av.VideoFrame.from_ndarray(rgb_array, format="rgb24")
    return frame.reformat(width=width, height=height, format="rgb24").to_ndarray(format="rgb24")


def encode_raw_image_to_jpeg(
    decoded_message: Any, *, jpeg_quality: int, scale: int | None
) -> tuple[bytes, int, int]:
    """Encode a raw ROS Image message to JPEG using imagecodecs for final encode."""
    rgb_array = raw_image_to_array(decoded_message)
    src_h, src_w = rgb_array.shape[:2]
    if scale is not None:
        target_w, target_h = calculate_downscale_dimensions(src_w, src_h, scale)
    else:
        target_w, target_h = src_w, src_h

    target_w -= target_w % 2
    target_h -= target_h % 2
    if target_w < 2 or target_h < 2:
        raise VideoEncoderError(f"Source frame too small ({target_w}x{target_h}) for JPEG encoding")

    if target_w != src_w or target_h != src_h:
        rgb_array = _resize_rgb_array(rgb_array, target_w, target_h)

    return _encode_rgb_array_to_jpeg(rgb_array, jpeg_quality), target_w, target_h


def decode_compressed_image_to_rgb_array(data: bytes) -> Any:
    """Decode JPEG/PNG compressed image bytes to an RGB numpy array."""
    from mcap_codec_support.video.pyav import decode_compressed_frame  # noqa: PLC0415

    return decode_compressed_frame(data).to_ndarray(format="rgb24")


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
