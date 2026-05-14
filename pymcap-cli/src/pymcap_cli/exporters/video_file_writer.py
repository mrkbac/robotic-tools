"""Lazy MP4 writer helpers for the CLI video exporter."""

from __future__ import annotations

import contextlib
import subprocess
import threading
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Protocol, cast

from mcap_codec_support.video.common import (
    EncoderBackend,
    EncoderConfig,
    EncoderMode,
    VideoCodec,
    VideoEncoderError,
    build_encoder_options,
    get_encoder_options,
    raw_image_to_array,
    resolve_encoder_for_backend,
)
from mcap_codec_support.video.compression import decode_compressed_image_to_rgb_array
from mcap_codec_support.video.ffmpeg import (
    ROS_ENCODING_TO_PIX_FMT,
    check_encoder_cli,
    find_ffmpeg,
    probe_image_dimensions,
)
from mcap_codec_support.video.schemas import COMPRESSED_SCHEMAS, IMAGE_SCHEMAS, RAW_SCHEMAS

if TYPE_CHECKING:
    import os
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np

    from mcap_codec_support._protocols import CompressedImageMsg, RawImageMessage


class _VideoFileStrategy(Protocol):
    config: EncoderConfig

    def write_compressed(self, data: bytes, log_time_ns: int) -> None: ...

    def write_raw(self, data: bytes, log_time_ns: int) -> None: ...

    def write_rgb(self, rgb: np.ndarray, log_time_ns: int) -> None: ...

    def close(self) -> int: ...


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


def _pack_raw_image_bytes(decoded: RawImageMessage, *, width: int, height: int) -> bytes:
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


class FFmpegMp4Encoder:
    """Encode a stream of frames to an MP4 file via an ffmpeg subprocess."""

    def __init__(
        self,
        output_path: os.PathLike[str] | str,
        *,
        width: int,
        height: int,
        codec_name: str,
        quality: int = 28,
        target_fps: float = 30.0,
        gop_size: int = 60,
        input_pix_fmt: str | None = "rgb24",
    ) -> None:
        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            raise VideoEncoderError("ffmpeg not found on PATH")

        width, height = _even_dimensions(width, height)
        if not check_encoder_cli(codec_name):
            raise VideoEncoderError(
                f"ffmpeg CLI does not support encoder {codec_name!r}. "
                "Install a fuller ffmpeg build or pick a different --encoder backend."
            )

        options, bit_rate = build_encoder_options(codec_name, quality, width, height)
        fps_int = max(round(target_fps), 1)

        cmd: list[str] = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
        if input_pix_fmt is None:
            cmd.extend(["-f", "image2pipe", "-r", str(fps_int), "-i", "pipe:0"])
        else:
            cmd.extend(
                [
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    input_pix_fmt,
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(fps_int),
                    "-i",
                    "pipe:0",
                ]
            )

        if input_pix_fmt is None:
            cmd.extend(["-vf", f"scale={width}:{height}"])

        cmd.extend(
            [
                "-c:v",
                codec_name,
                "-pix_fmt",
                "yuv420p",
                "-g",
                str(gop_size),
                "-bf",
                "0",
            ]
        )
        if bit_rate is not None:
            cmd.extend(["-b:v", str(bit_rate)])
        for key, value in options.items():
            cmd.extend([f"-{key}", value])

        cmd.extend(["-movflags", "+faststart", "-f", "mp4", str(output_path)])

        self._cmd = cmd
        self.output_path = output_path
        self.config = EncoderConfig(width=width, height=height, codec_name=codec_name)
        self._stderr_lines: list[str] = []
        self._frames_fed = 0

        self._process = subprocess.Popen(  # noqa: S603
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()

    def _read_stderr(self) -> None:
        if self._process.stderr is None:
            return
        for raw in iter(self._process.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                self._stderr_lines.append(line)
        self._process.stderr.close()

    def write_frame(self, frame_bytes: bytes) -> None:
        """Write one input frame to ffmpeg stdin."""
        if self._process.stdin is None or self._process.stdin.closed:
            raise VideoEncoderError("ffmpeg process stdin is closed")
        try:
            self._process.stdin.write(frame_bytes)
        except BrokenPipeError as exc:
            stderr_tail = "\n".join(self._stderr_lines[-5:])
            raise VideoEncoderError(f"ffmpeg subprocess exited mid-stream:\n{stderr_tail}") from exc
        self._frames_fed += 1

    @property
    def frames_fed(self) -> int:
        return self._frames_fed

    def close(self) -> None:
        if self._process.poll() is not None and self._process.stdin is None:
            return
        if self._process.stdin and not self._process.stdin.closed:
            with contextlib.suppress(BrokenPipeError):
                self._process.stdin.close()
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
            raise VideoEncoderError("ffmpeg subprocess did not exit cleanly; killed") from None
        self._stderr_thread.join(timeout=5)
        if self._process.returncode != 0:
            stderr_tail = "\n".join(self._stderr_lines[-10:])
            raise VideoEncoderError(
                f"ffmpeg exited with code {self._process.returncode}:\n{stderr_tail}"
            )

    def __del__(self) -> None:
        try:
            if self._process and self._process.poll() is None:
                self._process.kill()
                self._process.wait(timeout=2)
        except Exception:  # noqa: BLE001, S110
            pass


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
        from mcap_codec_support.video.pyav import resolve_encoder_for_backend  # noqa: PLC0415

        self.path = path
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

    def write_rgb(self, rgb: np.ndarray, log_time_ns: int) -> None:
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

    def write_rgb(self, rgb: np.ndarray, log_time_ns: int) -> None:
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

    def write_message(
        self,
        decoded: RawImageMessage | CompressedImageMsg,
        schema_name: str,
        log_time_ns: int,
    ) -> None:
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
            raw = cast("RawImageMessage", decoded)
            first_rgb = self._ensure_open_for_raw(raw)
            assert self._strategy is not None
            if self._input_kind == "pyav":
                rgb = first_rgb if first_rgb is not None else raw_image_to_array(raw)
                self._strategy.write_rgb(rgb, log_time_ns)
            else:
                data = _pack_raw_image_bytes(
                    raw,
                    width=self._strategy.config.width,
                    height=self._strategy.config.height,
                )
                self._strategy.write_raw(data, log_time_ns)
            return

        raise VideoEncoderError(f"Unexpected image schema {schema_name!r}")

    def _ensure_open_for_compressed(self, data: bytes) -> np.ndarray | None:
        return self._open_with_fallback(
            decode_pyav=lambda: decode_compressed_image_to_rgb_array(data),
            open_ffmpeg=lambda: self._open_ffmpeg_compressed(data),
        )

    def _ensure_open_for_raw(self, decoded: RawImageMessage) -> np.ndarray | None:
        return self._open_with_fallback(
            decode_pyav=lambda: raw_image_to_array(decoded),
            open_ffmpeg=lambda: self._open_ffmpeg_raw(decoded),
        )

    def _open_with_fallback(
        self,
        *,
        decode_pyav: Callable[[], np.ndarray],
        open_ffmpeg: Callable[[], None],
    ) -> np.ndarray | None:
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

    def _open_ffmpeg_raw(self, decoded: RawImageMessage) -> None:
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
