"""Unified video encode/decode facade for pymcap-cli.

Backend modules hold the low-level PyAV and ffmpeg subprocess primitives. This
module owns backend selection and the CLI-facing workflows.
"""

from __future__ import annotations

from collections import deque
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Protocol, cast

from pymcap_cli.encoding.encoder_common import (
    COMPRESSED_SCHEMAS,
    DEFAULT_FPS,
    DEFAULT_GOP_SIZE,
    IMAGE_SCHEMAS,
    RAW_SCHEMAS,
    EncoderBackend,
    EncoderConfig,
    EncoderMode,
    VideoCodec,
    VideoDecompressorProtocol,
    VideoEncoderError,
    calculate_downscale_dimensions,
    get_encoder_options,
    raw_image_to_array,
    resolve_encoder_for_backend,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from concurrent.futures import Future, ThreadPoolExecutor
    from pathlib import Path

    from small_mcap import DecodedMessage


class VideoCompressionBackend(Protocol):
    """Backend used by roscompress for CompressedVideo output."""

    label: str
    prefetch_supported: bool

    def test_encoder(self, encoder_name: str) -> bool: ...

    def resolve_encoder(self, codec: str) -> str: ...

    def decode_compressed(self, data: bytes) -> tuple[Any, int, int]: ...

    def decode_image(self, msg: DecodedMessage, schema_name: str) -> tuple[Any, int, int]: ...

    def create_encoder(
        self,
        width: int,
        height: int,
        codec_name: str,
        quality: int,
        *,
        input_pix_fmt: str | None = None,
        scale: tuple[int, int] | None = None,
    ) -> Any: ...

    def get_pix_fmt(self, topic: str) -> str | None: ...


class _PyAVCompressionBackend:
    label = "pyav"
    prefetch_supported = True

    def test_encoder(self, encoder_name: str) -> bool:
        from pymcap_cli.encoding.video_pyav import test_encoder  # noqa: PLC0415

        return test_encoder(encoder_name)

    def resolve_encoder(self, codec: str) -> str:
        from pymcap_cli.encoding.video_pyav import resolve_encoder  # noqa: PLC0415

        return resolve_encoder(codec)

    def decode_compressed(self, data: bytes) -> tuple[Any, int, int]:
        from pymcap_cli.encoding.video_pyav import decode_compressed_frame  # noqa: PLC0415

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
        from pymcap_cli.encoding.video_pyav import VideoEncoder  # noqa: PLC0415

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
        from pymcap_cli.encoding.video_ffmpeg import check_encoder_cli  # noqa: PLC0415

        return check_encoder_cli(encoder_name)

    def resolve_encoder(self, codec: str) -> str:
        from pymcap_cli.encoding.video_ffmpeg import resolve_encoder  # noqa: PLC0415

        return resolve_encoder(codec)

    def decode_compressed(self, data: bytes) -> tuple[Any, int, int]:
        from pymcap_cli.encoding.video_ffmpeg import probe_image_dimensions  # noqa: PLC0415

        width, height = probe_image_dimensions(data)
        return data, width, height

    def decode_image(self, msg: DecodedMessage, schema_name: str) -> tuple[Any, int, int]:
        data = bytes(msg.decoded_message.data)
        topic = msg.channel.topic

        if schema_name in COMPRESSED_SCHEMAS:
            self._topic_pix_fmt[topic] = None
            frame, width, height = self.decode_compressed(data)
            return frame, width, height

        from pymcap_cli.encoding.video_ffmpeg import ROS_ENCODING_TO_PIX_FMT  # noqa: PLC0415

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
        from pymcap_cli.encoding.video_ffmpeg import FFmpegVideoEncoder  # noqa: PLC0415

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
        except Exception:  # noqa: BLE001
            return _FfmpegCliCompressionBackend()
    return pyav_backend


def prefetch_image_decodes(
    messages: Iterable[DecodedMessage],
    backend: VideoCompressionBackend,
    pool: ThreadPoolExecutor,
    prefetch: int = 8,
) -> Iterator[tuple[DecodedMessage, Future[Any] | None]]:
    """Wrap message iterator to decode compressed images in background threads."""
    from pymcap_cli.exporters._common import normalize_schema_name  # noqa: PLC0415

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
            "Install with: uv add 'pymcap-cli[image]'"
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
    from pymcap_cli.encoding.video_pyav import decode_compressed_frame  # noqa: PLC0415

    return decode_compressed_frame(data).to_ndarray(format="rgb24")


def create_video_decompressor(
    video_format: str = "compressed",
    jpeg_quality: int = 90,
    *,
    mode: EncoderMode = EncoderMode.AUTO,
) -> VideoDecompressorProtocol:
    """Create a video decompressor using the requested backend."""
    if mode == EncoderMode.PYAV:
        from pymcap_cli.encoding.video_pyav import PyAVVideoDecompressor  # noqa: PLC0415

        return PyAVVideoDecompressor(video_format=video_format, jpeg_quality=jpeg_quality)

    if mode == EncoderMode.FFMPEG_CLI:
        from pymcap_cli.encoding.video_ffmpeg import FFmpegVideoDecompressor  # noqa: PLC0415

        return FFmpegVideoDecompressor(video_format=video_format, jpeg_quality=jpeg_quality)

    try:
        from pymcap_cli.encoding.video_pyav import PyAVVideoDecompressor  # noqa: PLC0415

        return PyAVVideoDecompressor(video_format=video_format, jpeg_quality=jpeg_quality)
    except ImportError:
        from pymcap_cli.encoding.video_ffmpeg import (  # noqa: PLC0415
            FFmpegVideoDecompressor,
            find_ffmpeg,
        )

        if find_ffmpeg():
            return FFmpegVideoDecompressor(video_format=video_format, jpeg_quality=jpeg_quality)
        raise


_TARGET_BITRATE_BY_QUALITY: tuple[tuple[int, int], ...] = (
    (20, 10_000_000),
    (25, 5_000_000),
    (10**6, 2_000_000),
)


def _bitrate_for(quality: int) -> int:
    for threshold, bitrate in _TARGET_BITRATE_BY_QUALITY:
        if quality <= threshold:
            return bitrate
    return _TARGET_BITRATE_BY_QUALITY[-1][1]


_RAW_BYTES_PER_PIXEL: dict[str, int] = {
    "rgb": 3,
    "rgb8": 3,
    "bgr": 3,
    "bgr8": 3,
    "mono": 1,
    "mono8": 1,
    "8uc1": 1,
}


def _pack_raw_image_bytes(decoded: Any, *, width: int, height: int) -> bytes:
    """Pack raw ROS Image bytes to the exact frame size expected by ffmpeg."""
    encoding = str(decoded.encoding).lower()
    bytes_per_pixel = _RAW_BYTES_PER_PIXEL.get(encoding)
    if bytes_per_pixel is None:
        raise VideoEncoderError(f"Unsupported image encoding: {decoded.encoding}")

    src_width = int(decoded.width)
    src_height = int(decoded.height)
    if width > src_width or height > src_height:
        raise VideoEncoderError(
            f"Cannot pack {src_width}x{src_height} frame as larger {width}x{height}"
        )

    src_row_bytes = src_width * bytes_per_pixel
    dst_row_bytes = width * bytes_per_pixel
    step = int(decoded.step)
    if step < src_row_bytes:
        raise VideoEncoderError(f"Image step {step} is smaller than row size {src_row_bytes}")

    data = bytes(decoded.data)
    required = step * src_height
    if len(data) < required:
        raise VideoEncoderError(f"Image data has {len(data)} bytes, expected at least {required}")

    if width == src_width and height == src_height and step == dst_row_bytes:
        return data

    packed = bytearray(dst_row_bytes * height)
    offset = 0
    for row in range(height):
        start = row * step
        end = start + dst_row_bytes
        packed[offset : offset + dst_row_bytes] = data[start:end]
        offset += dst_row_bytes
    return bytes(packed)


class _VideoFileStrategy(Protocol):
    config: EncoderConfig

    def write_compressed(self, data: bytes, log_time_ns: int) -> None: ...

    def write_raw(self, data: bytes, log_time_ns: int) -> None: ...

    def write_rgb(self, rgb: Any, log_time_ns: int) -> None: ...

    def close(self) -> int: ...


class _PyAVMp4Strategy:
    """In-process PyAV MP4 writer."""

    def __init__(
        self,
        path: Path,
        *,
        codec: VideoCodec,
        encoder_backend: EncoderBackend,
        quality: int,
        width: int,
        height: int,
    ) -> None:
        import av  # noqa: PLC0415
        import av.error  # noqa: PLC0415

        from pymcap_cli.encoding.video_pyav import resolve_encoder_for_backend  # noqa: PLC0415

        self.path = path
        self._codec = codec
        self._quality = quality
        self._encoder_name = resolve_encoder_for_backend(codec.value, encoder_backend.value)
        self._first_timestamp_ns: int | None = None
        self._last_pts = -1
        self._frame_count = 0
        self.config = EncoderConfig(width=width, height=height, codec_name=self._encoder_name)

        container = av.open(str(path), "w", format=None, options={"movflags": "faststart"})
        try:
            stream = cast("Any", container.add_stream(codec_name=self._encoder_name))
        except (av.error.FFmpegError, ValueError) as exc:
            container.close()
            raise VideoEncoderError(
                f"Failed to create video stream with encoder '{self._encoder_name}': {exc}"
            ) from exc

        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1_000_000)
        stream.codec_context.time_base = Fraction(1, 1_000_000)
        stream.codec_context.framerate = Fraction(30, 1)
        stream.codec_context.gop_size = 60
        stream.codec_context.bit_rate = _bitrate_for(quality)

        options = get_encoder_options(codec, self._encoder_name)
        if any(s in self._encoder_name for s in ("libx264", "libx265", "videotoolbox")):
            options["bf"] = "0"
        stream.codec_context.options = options

        self._container = container
        self._stream = stream

    def write_compressed(self, data: bytes, log_time_ns: int) -> None:
        self.write_rgb(decode_compressed_image_to_rgb_array(data), log_time_ns)

    def write_raw(self, data: bytes, log_time_ns: int) -> None:
        del data, log_time_ns
        raise VideoEncoderError("PyAV MP4 writer needs decoded RGB for raw frames")

    def write_rgb(self, rgb: Any, log_time_ns: int) -> None:
        import av  # noqa: PLC0415
        import av.error  # noqa: PLC0415

        if self._first_timestamp_ns is None:
            self._first_timestamp_ns = log_time_ns

        try:
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
        except (av.error.FFmpegError, ValueError) as exc:
            raise VideoEncoderError(f"Frame conversion failed: {exc}") from exc

        current_pts = (log_time_ns - self._first_timestamp_ns) // 1000
        if current_pts <= self._last_pts:
            current_pts = self._last_pts + 1
        frame.pts = current_pts
        self._last_pts = current_pts

        try:
            for packet in self._stream.encode(frame):
                self._container.mux(packet)
        except (av.error.FFmpegError, ValueError) as exc:
            raise VideoEncoderError(
                f"PyAV encoding failed at frame {self._frame_count}: {exc}"
            ) from exc
        self._frame_count += 1

    def close(self) -> int:
        for packet in self._stream.encode(None):
            self._container.mux(packet)
        self._container.close()
        return self._frame_count


class _FfmpegMp4Strategy:
    """ffmpeg-subprocess MP4 writer."""

    def __init__(
        self,
        path: Path,
        *,
        codec: VideoCodec,
        encoder_backend: EncoderBackend,
        quality: int,
        width: int,
        height: int,
        input_pix_fmt: str | None,
    ) -> None:
        from pymcap_cli.encoding.video_ffmpeg import (  # noqa: PLC0415
            FFmpegMp4Encoder,
            check_encoder_cli,
        )

        encoder_name = resolve_encoder_for_backend(
            codec.value, encoder_backend.value, test_fn=check_encoder_cli
        )
        self._encoder = FFmpegMp4Encoder(
            path,
            width=width,
            height=height,
            codec_name=encoder_name,
            quality=quality,
            input_pix_fmt=input_pix_fmt,
        )
        self.config = self._encoder.config

    def write_compressed(self, data: bytes, log_time_ns: int) -> None:
        del log_time_ns
        self._encoder.write_frame(data)

    def write_raw(self, data: bytes, log_time_ns: int) -> None:
        del log_time_ns
        self._encoder.write_frame(data)

    def write_rgb(self, rgb: Any, log_time_ns: int) -> None:
        del log_time_ns
        self._encoder.write_frame(rgb.tobytes())

    def close(self) -> int:
        frames = self._encoder.frames_fed
        self._encoder.close()
        return frames


class VideoFileWriterSession:
    """Lazy per-topic MP4 writer with unified backend selection."""

    def __init__(
        self,
        path: Path,
        *,
        codec: VideoCodec,
        encoder_backend: EncoderBackend,
        quality: int,
        mode: EncoderMode,
        on_fallback: Callable[[str], None] | None = None,
    ) -> None:
        self.path = path
        self._codec = codec
        self._encoder_backend = encoder_backend
        self._quality = quality
        self._mode = mode
        self._on_fallback = on_fallback
        self._strategy: _VideoFileStrategy | None = None
        self._input_kind: str | None = None

    def write_message(self, decoded: Any, schema_name: str, log_time_ns: int) -> None:
        if schema_name not in IMAGE_SCHEMAS:
            raise VideoEncoderError(f"Unexpected image schema {schema_name!r}")

        if schema_name in COMPRESSED_SCHEMAS:
            data = bytes(decoded.data)
            first_rgb = self._ensure_open_for_compressed(data)
            assert self._strategy is not None
            if self._input_kind == "pyav":
                rgb = (
                    first_rgb
                    if first_rgb is not None
                    else decode_compressed_image_to_rgb_array(data)
                )
                self._strategy.write_rgb(rgb, log_time_ns)
            else:
                self._strategy.write_compressed(data, log_time_ns)
            return

        if schema_name in RAW_SCHEMAS:
            first_rgb = self._ensure_open_for_raw(decoded)
            assert self._strategy is not None
            if self._input_kind == "pyav":
                rgb = first_rgb if first_rgb is not None else raw_image_to_array(decoded)
                self._strategy.write_rgb(rgb, log_time_ns)
            else:
                data = _pack_raw_image_bytes(
                    decoded,
                    width=self._strategy.config.width,
                    height=self._strategy.config.height,
                )
                self._strategy.write_raw(data, log_time_ns)
            return

        raise VideoEncoderError(f"Unexpected image schema {schema_name!r}")

    def _ensure_open_for_compressed(self, data: bytes) -> Any | None:
        return self._open_with_fallback(
            decode_pyav=lambda: decode_compressed_image_to_rgb_array(data),
            open_ffmpeg=lambda: self._open_ffmpeg_compressed(data),
        )

    def _ensure_open_for_raw(self, decoded: Any) -> Any | None:
        return self._open_with_fallback(
            decode_pyav=lambda: raw_image_to_array(decoded),
            open_ffmpeg=lambda: self._open_ffmpeg_raw(decoded),
        )

    def _open_with_fallback(
        self,
        *,
        decode_pyav: Callable[[], Any],
        open_ffmpeg: Callable[[], None],
    ) -> Any | None:
        if self._strategy is not None:
            return None

        if self._mode is EncoderMode.FFMPEG_CLI:
            open_ffmpeg()
            return None

        try:
            rgb = decode_pyav()
            height, width = rgb.shape[:2]
            self._open_pyav(width, height)
        except (ImportError, VideoEncoderError) as exc:
            if self._mode is EncoderMode.PYAV:
                raise
            if self._on_fallback is not None:
                self._on_fallback(
                    f"PyAV failed to open encoder ({exc}); falling back to ffmpeg-cli."
                )
            open_ffmpeg()
            return None
        return rgb

    def _open_pyav(self, width: int, height: int) -> None:
        width, height = _even_dimensions(width, height)
        self._strategy = _PyAVMp4Strategy(
            self.path,
            codec=self._codec,
            encoder_backend=self._encoder_backend,
            quality=self._quality,
            width=width,
            height=height,
        )
        self._input_kind = "pyav"

    def _open_ffmpeg_compressed(self, data: bytes) -> None:
        from pymcap_cli.encoding.video_ffmpeg import probe_image_dimensions  # noqa: PLC0415

        width, height = probe_image_dimensions(data)
        width, height = _even_dimensions(width, height)
        self._strategy = _FfmpegMp4Strategy(
            self.path,
            codec=self._codec,
            encoder_backend=self._encoder_backend,
            quality=self._quality,
            width=width,
            height=height,
            input_pix_fmt=None,
        )
        self._input_kind = "ffmpeg"

    def _open_ffmpeg_raw(self, decoded: Any) -> None:
        from pymcap_cli.encoding.video_ffmpeg import ROS_ENCODING_TO_PIX_FMT  # noqa: PLC0415

        encoding = str(decoded.encoding).lower()
        pix_fmt = ROS_ENCODING_TO_PIX_FMT.get(encoding)
        if not pix_fmt:
            raise VideoEncoderError(f"Unsupported image encoding: {decoded.encoding}")
        width, height = _even_dimensions(decoded.width, decoded.height)
        self._strategy = _FfmpegMp4Strategy(
            self.path,
            codec=self._codec,
            encoder_backend=self._encoder_backend,
            quality=self._quality,
            width=width,
            height=height,
            input_pix_fmt=pix_fmt,
        )
        self._input_kind = "ffmpeg"

    def close(self) -> int:
        if self._strategy is None:
            return 0
        return self._strategy.close()


def _even_dimensions(width: int, height: int) -> tuple[int, int]:
    width -= width % 2
    height -= height % 2
    if width < 2 or height < 2:
        raise VideoEncoderError(f"Source frame too small ({width}x{height}) for video encoding")
    return width, height


def create_video_file_writer(
    path: Path,
    *,
    codec: VideoCodec,
    encoder_backend: EncoderBackend,
    quality: int,
    mode: EncoderMode,
    on_fallback: Callable[[str], None] | None = None,
) -> VideoFileWriterSession:
    """Create a lazy MP4 writer session."""
    return VideoFileWriterSession(
        path,
        codec=codec,
        encoder_backend=encoder_backend,
        quality=quality,
        mode=mode,
        on_fallback=on_fallback,
    )
