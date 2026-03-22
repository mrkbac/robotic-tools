"""FFmpeg subprocess-based video compression and decompression backend.

All ``ffmpeg`` / ``ffprobe`` subprocess usage is confined to this module.
**No PyAV dependency** — only requires ``ffmpeg`` on PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from queue import Empty, Queue

from pymcap_cli.encoding.encoder_common import (
    EncoderConfig,
    VideoEncoderError,
    build_encoder_options,
)
from pymcap_cli.encoding.encoder_common import (
    resolve_encoder as _resolve_encoder,
)
from pymcap_cli.encoding.video_protocols import DecompressedFrame

# ---------------------------------------------------------------------------
# ffmpeg discovery
# ---------------------------------------------------------------------------


def find_ffmpeg() -> str | None:
    """Return the path to ``ffmpeg`` if it is on PATH, else None."""
    return shutil.which("ffmpeg")


def check_encoder_cli(encoder_name: str) -> bool:
    """Check whether the system ``ffmpeg`` supports *encoder_name*."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    try:
        result = subprocess.run(  # noqa: S603
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == encoder_name:
                return True
    except (subprocess.TimeoutExpired, OSError):
        pass
    return False


def resolve_encoder(codec: str) -> str:
    """Pick the best available encoder for *codec* using ffmpeg CLI to probe."""
    if not find_ffmpeg():
        raise VideoEncoderError("ffmpeg not found on PATH")
    try:
        return _resolve_encoder(codec, test_fn=check_encoder_cli)
    except ValueError as exc:
        raise VideoEncoderError(str(exc)) from exc


# Mapping from ROS image encoding to ffmpeg pixel format.
ROS_ENCODING_TO_PIX_FMT: dict[str, str] = {
    "rgb": "rgb24",
    "rgb8": "rgb24",
    "bgr": "bgr24",
    "bgr8": "bgr24",
    "mono": "gray",
    "mono8": "gray",
    "8uc1": "gray",
}


# ---------------------------------------------------------------------------
# Annex B access-unit splitter
# ---------------------------------------------------------------------------

_START_CODE_4 = b"\x00\x00\x00\x01"
_H264_VCL_TYPES = frozenset({1, 2, 3, 4, 5})
_H265_MAX_VCL_TYPE = 31


class AnnexBParser:
    """Split an Annex B byte stream into per-access-unit chunks."""

    def __init__(self, codec: str) -> None:
        self._is_h265 = "265" in codec or "hevc" in codec
        self._buf = bytearray()
        self._current_au = bytearray()
        self._current_has_vcl = False

    def _is_vcl(self, nal_header: int) -> bool:
        if self._is_h265:
            nal_type = (nal_header >> 1) & 0x3F
            return nal_type <= _H265_MAX_VCL_TYPE
        nal_type = nal_header & 0x1F
        return nal_type in _H264_VCL_TYPES

    def feed(self, data: bytes) -> list[bytes]:
        self._buf.extend(data)
        result: list[bytes] = []

        while True:
            first = self._buf.find(_START_CODE_4)
            if first == -1:
                break
            second = self._buf.find(_START_CODE_4, first + 4)
            if second == -1 or second + 4 >= len(self._buf):
                break

            next_nal_header = self._buf[second + 4]
            next_is_vcl = self._is_vcl(next_nal_header)

            if first + 4 < len(self._buf):
                cur_nal_header = self._buf[first + 4]
                if self._is_vcl(cur_nal_header) and not self._current_has_vcl:
                    self._current_has_vcl = True

            if next_is_vcl and self._current_has_vcl:
                self._current_au.extend(self._buf[:second])
                result.append(bytes(self._current_au))
                self._current_au = bytearray()
                self._current_has_vcl = False
                self._buf = self._buf[second:]
                continue

            if next_is_vcl:
                self._current_has_vcl = True

            self._current_au.extend(self._buf[:second])
            self._buf = self._buf[second:]

        return result

    def flush(self) -> bytes | None:
        self._current_au.extend(self._buf)
        self._buf.clear()
        data = bytes(self._current_au)
        self._current_au.clear()
        self._current_has_vcl = False
        return data or None


# ---------------------------------------------------------------------------
# Codec helpers
# ---------------------------------------------------------------------------

_CODEC_TO_FORMAT: dict[str, str] = {
    "h264": "h264",
    "h265": "hevc",
    "hevc": "hevc",
}


def _codec_family(codec_name: str) -> str:
    lower = codec_name.lower()
    if "264" in lower:
        return "h264"
    if "265" in lower or "hevc" in lower:
        return "h265"
    return "h264"


# ---------------------------------------------------------------------------
# Base ffmpeg encoder (shared stdout/stderr/flush/encode machinery)
# ---------------------------------------------------------------------------


class _BaseFFmpegEncoder:
    """Shared base for ffmpeg subprocess encoders.

    Subclasses build the ffmpeg command; this class handles process I/O,
    Annex B parsing, encode/flush, and cleanup.
    """

    def __init__(self, cmd: list[str], codec_fam: str, config: EncoderConfig) -> None:
        self.config = config

        try:
            self._process = subprocess.Popen(  # noqa: S603
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise VideoEncoderError(f"Failed to start ffmpeg: {exc}") from exc

        self._parser = AnnexBParser(codec_fam)
        self._output_queue: Queue[bytes | None] = Queue()
        self._stderr_lines: list[str] = []

        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _read_stdout(self) -> None:
        if self._process.stdout is None:
            return
        fd = self._process.stdout.fileno()
        os.set_blocking(fd, False)
        try:
            while True:
                try:
                    chunk = os.read(fd, 65536)
                    if not chunk:
                        break
                    for au in self._parser.feed(chunk):
                        self._output_queue.put(au)
                except BlockingIOError:
                    time.sleep(0.005)
                except OSError:
                    break
            remaining = self._parser.flush()
            if remaining:
                self._output_queue.put(remaining)
        finally:
            self._output_queue.put(None)

    def _read_stderr(self) -> None:
        if self._process.stderr is None:
            return
        for raw_line in self._process.stderr:
            text = raw_line.decode(errors="replace").rstrip()
            if text:
                self._stderr_lines.append(text)

    def encode(self, frame: bytes) -> bytes | None:
        """Write *frame* bytes to ffmpeg stdin and return one access unit (or None)."""
        if self._process.stdin is None:
            raise VideoEncoderError("ffmpeg stdin is not available")
        try:
            self._process.stdin.write(frame)
            self._process.stdin.flush()
        except BrokenPipeError as exc:
            stderr_tail = "\n".join(self._stderr_lines[-5:])
            raise VideoEncoderError(f"ffmpeg process died unexpectedly:\n{stderr_tail}") from exc

        time.sleep(0.01)
        try:
            au = self._output_queue.get(timeout=0.1)
        except Empty:
            return None

        if au is None:
            stderr_tail = "\n".join(self._stderr_lines[-5:])
            raise VideoEncoderError(f"ffmpeg exited prematurely:\n{stderr_tail}")
        return au

    def flush(self) -> bytes | None:
        if self._process.stdin and not self._process.stdin.closed:
            self._process.stdin.close()

        self._stdout_thread.join(timeout=10)
        self._stderr_thread.join(timeout=5)
        self._process.wait(timeout=10)

        chunks: list[bytes] = []
        while True:
            try:
                item = self._output_queue.get_nowait()
            except Empty:
                break
            if item is None:
                break
            chunks.append(item)

        if self._process.returncode and self._process.returncode != 0:
            stderr_tail = "\n".join(self._stderr_lines[-5:])
            raise VideoEncoderError(
                f"ffmpeg exited with code {self._process.returncode}:\n{stderr_tail}"
            )

        return b"".join(chunks) if chunks else None

    def __del__(self) -> None:
        try:
            if self._process.poll() is None:
                self._process.kill()
                self._process.wait(timeout=2)
        except Exception:  # noqa: BLE001, S110
            pass


def _build_output_args(
    codec_fam: str, codec_name: str, gop_size: int, options: dict[str, str], bit_rate: int | None
) -> list[str]:
    """Build the shared encoder output arguments."""
    cmd: list[str] = [
        "-c:v",
        codec_name,
        "-g",
        str(gop_size),
        "-bf",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-fflags",
        "+flush_packets",
    ]
    if bit_rate is not None:
        cmd.extend(["-b:v", str(bit_rate)])
    for key, value in options.items():
        cmd.extend([f"-{key}", value])

    output_fmt = _CODEC_TO_FORMAT.get(codec_fam, "h264")
    cmd.extend(["-f", output_fmt, "pipe:1"])
    return cmd


def _require_ffmpeg() -> str:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise VideoEncoderError("ffmpeg not found on PATH")
    return ffmpeg


# ---------------------------------------------------------------------------
# FFmpegVideoEncoder (unified: rawvideo or image2pipe → H.264/H.265)
# ---------------------------------------------------------------------------

# JPEG magic bytes.
_JPEG_MAGIC = b"\xff\xd8"
# PNG magic bytes.
_PNG_MAGIC = b"\x89PNG"


def probe_image_dimensions(data: bytes) -> tuple[int, int]:
    """Extract (width, height) from a JPEG or PNG image header.

    Falls back to ffprobe if the header cannot be parsed.
    """
    if data[:2] == _JPEG_MAGIC:
        dims = parse_jpeg_dimensions(data)
        if dims:
            return dims
    elif data[:4] == _PNG_MAGIC and len(data) >= 24:
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return width, height

    # Fallback: use ffprobe.
    return _ffprobe_image_dimensions(data)


def parse_jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """Parse JPEG SOF marker for dimensions. Returns None on failure."""
    i = 2
    while i < len(data) - 9:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        # SOF0 (0xC0) or SOF2 (0xC2) — baseline or progressive.
        if marker in (0xC0, 0xC2):
            height = (data[i + 5] << 8) | data[i + 6]
            width = (data[i + 7] << 8) | data[i + 8]
            return width, height
        if marker == 0xD9:  # EOI
            return None
        # Skip over marker segment.
        seg_len = (data[i + 2] << 8) | data[i + 3]
        i += 2 + seg_len
    return None


def _ffprobe_image_dimensions(data: bytes) -> tuple[int, int]:
    """Probe image dimensions via ffprobe subprocess."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise VideoEncoderError("ffprobe not found (needed for image dimension probing)")
    cmd = [
        ffprobe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        "-i",
        "pipe:0",
    ]
    try:
        result = subprocess.run(  # noqa: S603
            cmd, input=data, capture_output=True, text=True, timeout=10, check=False
        )
        parts = result.stdout.strip().split(",")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass
    raise VideoEncoderError("Cannot determine image dimensions")


class FFmpegVideoEncoder(_BaseFFmpegEncoder):
    """Encode frames to H.264/H.265 via a single ffmpeg process.

    When *input_pix_fmt* is ``None`` (the default), the encoder accepts
    JPEG/PNG bytes via ``image2pipe``. When set (e.g. ``"rgb24"``), it
    accepts raw pixel data of that format via ``rawvideo``.
    """

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
        input_pix_fmt: str | None = None,
        scale: tuple[int, int] | None = None,
    ) -> None:
        ffmpeg = _require_ffmpeg()
        codec_fam = _codec_family(codec_name)
        fps_int = max(round(target_fps), 1)
        options, bit_rate = build_encoder_options(codec_name, quality, width, height, preset=preset)

        if input_pix_fmt is not None:
            # Raw pixel data (rgb24, bgr24, gray, etc.).
            cmd: list[str] = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
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
        else:
            # Compressed images (JPEG/PNG) — ffmpeg auto-detects format.
            cmd = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "image2pipe",
                "-r",
                str(fps_int),
                "-i",
                "pipe:0",
            ]

        if scale is not None:
            sw, sh = scale
            cmd.extend(["-vf", f"scale={sw}:{sh}"])

        cmd.extend(_build_output_args(codec_fam, codec_name, gop_size, options, bit_rate))

        super().__init__(
            cmd, codec_fam, EncoderConfig(width=width, height=height, codec_name=codec_name)
        )


# ---------------------------------------------------------------------------
# FFmpegVideoDecompressor (H.264/H.265 → Image)
# ---------------------------------------------------------------------------

_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"


class FFmpegVideoDecompressor:
    """Decompresses H.264/H.265 video to JPEG or raw RGB using ffmpeg subprocess.

    Implements ``VideoDecompressorProtocol``. **No PyAV dependency.**
    """

    def __init__(
        self,
        video_format: str = "compressed",
        jpeg_quality: int = 90,
    ) -> None:
        self._video_format = video_format
        self._jpeg_quality = jpeg_quality
        self._process: subprocess.Popen[bytes] | None = None
        self._output_queue: Queue[DecompressedFrame | None] = Queue()
        self._stderr_lines: list[str] = []
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        # For raw mode: need dimensions to know frame size
        self._width: int | None = None
        self._height: int | None = None

    def _start_process(self, codec: str) -> None:
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise VideoEncoderError("ffmpeg not found on PATH")

        input_format = _CODEC_TO_FORMAT.get(codec, "h264")

        cmd: list[str] = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            input_format,
            "-i",
            "pipe:0",
        ]

        if self._video_format == "compressed":
            # Output JPEG stream — one JPEG per frame.
            q = max(1, 31 - self._jpeg_quality * 31 // 100)
            cmd.extend(
                [
                    "-c:v",
                    "mjpeg",
                    "-q:v",
                    str(q),
                    "-f",
                    "image2pipe",
                    "pipe:1",
                ]
            )
        else:
            # Output raw RGB24 frames.
            cmd.extend(
                [
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ]
            )

        try:
            self._process = subprocess.Popen(  # noqa: S603
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise VideoEncoderError(f"Failed to start ffmpeg decoder: {exc}") from exc

        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _detect_dimensions(self, data: bytes, codec: str) -> tuple[int, int]:
        """Detect video dimensions via ffprobe."""
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise VideoEncoderError("ffprobe not found on PATH")

        input_format = _CODEC_TO_FORMAT.get(codec, "h264")
        cmd = [
            ffprobe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            input_format,
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            "-i",
            "pipe:0",
        ]
        try:
            result = subprocess.run(  # noqa: S603
                cmd, input=data, capture_output=True, text=True, timeout=10, check=False
            )
            parts = result.stdout.strip().split(",")
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass
        raise VideoEncoderError("Could not detect video dimensions from bitstream")

    def _read_stdout(self) -> None:
        if self._process is None or self._process.stdout is None:
            return

        if self._video_format == "compressed":
            self._read_jpeg_stream()
        else:
            self._read_raw_stream()

        self._output_queue.put(None)  # sentinel

    def _read_jpeg_stream(self) -> None:
        """Read a stream of concatenated JPEGs, split on SOI/EOI markers."""
        if self._process is None or self._process.stdout is None:
            return
        buf = bytearray()
        while True:
            chunk = self._process.stdout.read(65536)
            if not chunk:
                break
            buf.extend(chunk)

            while True:
                soi = buf.find(_JPEG_SOI)
                if soi == -1:
                    break
                eoi = buf.find(_JPEG_EOI, soi + 2)
                if eoi == -1:
                    break
                jpeg_data = bytes(buf[soi : eoi + 2])
                del buf[: eoi + 2]
                dims = parse_jpeg_dimensions(jpeg_data)
                w, h = dims or (0, 0)
                self._output_queue.put(
                    DecompressedFrame(data=jpeg_data, width=w, height=h, is_jpeg=True)
                )

    def _read_raw_stream(self) -> None:
        """Read raw RGB24 frames of known size."""
        if self._process is None or self._process.stdout is None:
            return

        if self._width is None or self._height is None:
            import logging  # noqa: PLC0415

            logging.getLogger(__name__).warning(
                "FFmpegVideoDecompressor: dimensions unknown, discarding raw output"
            )
            self._process.stdout.read()
            return

        frame_size = self._width * self._height * 3
        buf = bytearray()
        while True:
            chunk = self._process.stdout.read(65536)
            if not chunk:
                break
            buf.extend(chunk)
            while len(buf) >= frame_size:
                frame_bytes = bytes(buf[:frame_size])
                del buf[:frame_size]
                self._output_queue.put(
                    DecompressedFrame(
                        data=frame_bytes,
                        width=self._width,
                        height=self._height,
                        is_jpeg=False,
                    )
                )

    def _read_stderr(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        for line in self._process.stderr:
            self._stderr_lines.append(line.decode(errors="replace").rstrip())

    def decompress(self, video_data: bytes, codec: str) -> DecompressedFrame | None:
        # Start process on first call.
        if self._process is None:
            # For raw mode, detect dimensions before starting.
            if self._video_format != "compressed" and self._width is None:
                import contextlib  # noqa: PLC0415

                with contextlib.suppress(VideoEncoderError):
                    self._width, self._height = self._detect_dimensions(video_data, codec)
            self._start_process(codec)

        if self._process is None or self._process.stdin is None:
            raise VideoEncoderError("ffmpeg process not started")

        self._process.stdin.write(video_data)
        self._process.stdin.flush()

        try:
            frame = self._output_queue.get(timeout=0.2)
        except Empty:
            return None
        else:
            return frame if frame is not None else None

    def flush(self) -> list[DecompressedFrame]:
        if self._process is None:
            return []

        if self._process.stdin and not self._process.stdin.closed:
            self._process.stdin.close()

        frames: list[DecompressedFrame] = []
        while True:
            try:
                frame = self._output_queue.get(timeout=5.0)
                if frame is None:
                    break
                frames.append(frame)
            except Empty:
                break

        self._process.wait(timeout=5)
        return frames

    def __del__(self) -> None:
        try:
            if self._process and self._process.poll() is None:
                self._process.kill()
                self._process.wait(timeout=2)
        except Exception:  # noqa: BLE001, S110
            pass
