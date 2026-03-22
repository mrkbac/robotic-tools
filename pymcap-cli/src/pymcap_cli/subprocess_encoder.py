"""Subprocess-based ffmpeg video encoder backend.

Provides a VideoEncoder-compatible class that shells out to the ``ffmpeg``
binary instead of using PyAV's in-process ``av.CodecContext``. This is useful
on platforms where the system ffmpeg has encoders that PyAV's bundled build
does not (e.g. GStreamer-only systems, custom ffmpeg builds).

**No PyAV dependency** — this module only requires ``ffmpeg`` on PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from queue import Empty, Queue

from pymcap_cli.encoder_common import EncoderConfig, VideoEncoderError, build_encoder_options

# ---------------------------------------------------------------------------
# Annex B access-unit splitter
# ---------------------------------------------------------------------------

_START_CODE_4 = b"\x00\x00\x00\x01"

# H.264 NAL types that carry slice data (VCL NALs).
_H264_VCL_TYPES = frozenset({1, 2, 3, 4, 5})  # non-IDR slices + IDR
# H.265 VCL NAL types are 0..31.
_H265_MAX_VCL_TYPE = 31


class AnnexBParser:
    """Split an Annex B byte stream into per-access-unit chunks.

    An access unit boundary is detected when a new VCL NAL unit appears
    after a previous access unit has been started.  This works without
    AUD markers.
    """

    def __init__(self, codec: str) -> None:
        self._is_h265 = "265" in codec or "hevc" in codec
        self._buf = bytearray()
        # Accumulates NAL units for the current access unit.
        self._current_au = bytearray()
        # True once we've seen at least one VCL NAL in the current AU.
        self._current_has_vcl = False

    def _is_vcl(self, nal_header: int) -> bool:
        """Return True if *nal_header* (first byte after start code) is a VCL NAL."""
        if self._is_h265:
            nal_type = (nal_header >> 1) & 0x3F
            return nal_type <= _H265_MAX_VCL_TYPE
        nal_type = nal_header & 0x1F
        return nal_type in _H264_VCL_TYPES

    def feed(self, data: bytes) -> list[bytes]:
        """Feed raw Annex B data, return list of completed access units."""
        self._buf.extend(data)
        result: list[bytes] = []

        while True:
            # Find the *second* start code (the boundary of the next NAL).
            # The buffer may start with a start code from a prior iteration;
            # we need to look past it.
            first = self._buf.find(_START_CODE_4)
            if first == -1:
                break
            second = self._buf.find(_START_CODE_4, first + 4)
            if second == -1 or second + 4 >= len(self._buf):
                # Only one start code (or none) — need more data to decide
                # if the current NAL is complete.
                break

            # We have a complete NAL from `first` to `second`.
            # Check the *second* start code's NAL type.
            next_nal_header = self._buf[second + 4]
            next_is_vcl = self._is_vcl(next_nal_header)

            # Mark the first NAL as VCL if applicable.
            if first + 4 < len(self._buf):
                cur_nal_header = self._buf[first + 4]
                if self._is_vcl(cur_nal_header) and not self._current_has_vcl:
                    self._current_has_vcl = True

            if next_is_vcl and self._current_has_vcl:
                # The next NAL starts a new AU.  Emit the current one.
                self._current_au.extend(self._buf[:second])
                result.append(bytes(self._current_au))
                self._current_au = bytearray()
                self._current_has_vcl = False
                self._buf = self._buf[second:]
                continue

            if next_is_vcl:
                self._current_has_vcl = True

            # The next NAL is still part of the current AU.
            # Consume everything up to the second start code.
            self._current_au.extend(self._buf[:second])
            self._buf = self._buf[second:]

        return result

    def flush(self) -> bytes | None:
        """Return any remaining data as a final access unit."""
        self._current_au.extend(self._buf)
        self._buf.clear()
        data = bytes(self._current_au)
        self._current_au.clear()
        self._current_has_vcl = False
        return data or None


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
        # Lines look like: " V..... libx264  ..."
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == encoder_name:
                return True
    except (subprocess.TimeoutExpired, OSError):
        pass
    return False


# ---------------------------------------------------------------------------
# ffmpeg image decoding (JPEG/PNG → raw YUV420p)
# ---------------------------------------------------------------------------


def decode_image_to_yuv420p(data: bytes, *, scale: int | None = None) -> tuple[bytes, int, int]:
    """Decode a compressed image (JPEG/PNG) to raw YUV420p bytes.

    Returns ``(yuv420p_bytes, width, height)``.  Dimensions are made even.
    If *scale* is given, caps the largest dimension to that value.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise VideoEncoderError("ffmpeg not found on PATH")

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise VideoEncoderError("ffprobe not found on PATH (required for image dimension probing)")

    # Probe dimensions.
    probe_cmd = [
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
        probe_result = subprocess.run(  # noqa: S603
            probe_cmd, input=data, capture_output=True, text=True, timeout=10, check=False
        )
        parts = probe_result.stdout.strip().split(",")
        if len(parts) != 2:
            raise VideoEncoderError(f"ffprobe dimension parse failed: {probe_result.stdout!r}")
        width, height = int(parts[0]), int(parts[1])
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise VideoEncoderError(f"ffprobe failed: {exc}") from exc

    # Apply scaling / ensure even dimensions.
    if scale is not None:
        if width > height:
            if width > scale:
                height = int(height * scale / width)
                width = scale
        elif height > scale:
            width = int(width * scale / height)
            height = scale
    width -= width % 2
    height -= height % 2
    width = max(width, 2)
    height = max(height, 2)

    # Decode to raw YUV420p at target dimensions.
    decode_cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "image2pipe",
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    try:
        result = subprocess.run(  # noqa: S603
            decode_cmd, input=data, capture_output=True, timeout=10, check=False
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise VideoEncoderError(f"ffmpeg image decode failed: {exc}") from exc

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        raise VideoEncoderError(f"ffmpeg image decode error: {stderr}")

    expected_size = width * height * 3 // 2
    if len(result.stdout) != expected_size:
        raise VideoEncoderError(
            f"Decoded size mismatch: got {len(result.stdout)}, expected {expected_size}"
        )

    return result.stdout, width, height


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


def raw_rgb_to_yuv420p(
    data: bytes, width: int, height: int, encoding: str = "rgb24"
) -> tuple[bytes, int, int]:
    """Convert raw RGB/BGR/mono image bytes to YUV420p using ffmpeg.

    *encoding* is the ffmpeg pixel format name: ``rgb24``, ``bgr24``, or ``gray``.
    Returns ``(yuv420p_bytes, width, height)`` with even dimensions.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise VideoEncoderError("ffmpeg not found on PATH")

    out_w = max(width - (width % 2), 2)
    out_h = max(height - (height % 2), 2)

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        encoding,
        "-s",
        f"{width}x{height}",
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        "-s",
        f"{out_w}x{out_h}",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    try:
        result = subprocess.run(  # noqa: S603
            cmd, input=data, capture_output=True, timeout=10, check=False
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise VideoEncoderError(f"ffmpeg raw conversion failed: {exc}") from exc

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        raise VideoEncoderError(f"ffmpeg raw conversion error: {stderr}")

    return result.stdout, out_w, out_h


# ---------------------------------------------------------------------------
# Subprocess encoder
# ---------------------------------------------------------------------------

_CODEC_TO_FORMAT: dict[str, str] = {
    "h264": "h264",
    "h265": "hevc",
    "hevc": "hevc",
}


def _codec_family(codec_name: str) -> str:
    """Map an ffmpeg encoder name to its codec family (h264 / h265)."""
    lower = codec_name.lower()
    if "264" in lower:
        return "h264"
    if "265" in lower or "hevc" in lower:
        return "h265"
    return "h264"


class SubprocessVideoEncoder:
    """Video encoder that delegates to an ``ffmpeg`` subprocess.

    Only accepts raw ``bytes`` in YUV420p planar format — **no PyAV dependency**.
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
    ) -> None:
        self.config = EncoderConfig(width=width, height=height, codec_name=codec_name)

        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise VideoEncoderError("ffmpeg not found on PATH")

        self._codec_fam = _codec_family(codec_name)
        fps_int = max(round(target_fps), 1)

        options, bit_rate = build_encoder_options(codec_name, quality, width, height, preset=preset)

        cmd: list[str] = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            # Input: raw YUV420p frames on stdin.
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps_int),
            "-i",
            "pipe:0",
            # Encoder settings.
            "-c:v",
            codec_name,
            "-g",
            str(gop_size),
            "-bf",
            "0",
            # Flush output packets immediately.
            "-fflags",
            "+flush_packets",
        ]
        if bit_rate is not None:
            cmd.extend(["-b:v", str(bit_rate)])
        for key, value in options.items():
            cmd.extend([f"-{key}", value])

        output_fmt = _CODEC_TO_FORMAT.get(self._codec_fam, "h264")
        cmd.extend(["-f", output_fmt, "pipe:1"])

        try:
            self._process = subprocess.Popen(  # noqa: S603
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise VideoEncoderError(f"Failed to start ffmpeg: {exc}") from exc

        self._parser = AnnexBParser(self._codec_fam)
        self._output_queue: Queue[bytes | None] = Queue()
        self._stderr_lines: list[str] = []

        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

        self._frame_size = width * height * 3 // 2

    # ------------------------------------------------------------------
    # Background I/O
    # ------------------------------------------------------------------

    def _read_stdout(self) -> None:
        """Read ffmpeg stdout, parse into access units, push to queue."""
        assert self._process.stdout is not None
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
            self._output_queue.put(None)  # sentinel

    def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        for raw_line in self._process.stderr:
            text = (
                raw_line.decode(errors="replace").rstrip()
                if isinstance(raw_line, bytes)
                else raw_line.rstrip()
            )
            if text:
                self._stderr_lines.append(text)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def encode(self, frame: bytes) -> bytes | None:
        """Encode a single raw YUV420p frame.

        *frame* must be raw ``bytes`` in YUV420p planar format
        (width * height * 3 // 2 bytes).

        Returns one access unit as ``bytes``, or ``None`` if the
        encoder has buffered the frame.
        """
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(frame)
            self._process.stdin.flush()
        except BrokenPipeError as exc:
            stderr_tail = "\n".join(self._stderr_lines[-5:])
            raise VideoEncoderError(f"ffmpeg process died unexpectedly:\n{stderr_tail}") from exc

        # Give ffmpeg a moment to produce output, then try the queue.
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
        """Close the encoder and return remaining buffered data."""
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
