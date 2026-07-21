"""FFmpeg subprocess-based video compression and decompression backend.

All ``ffmpeg`` / ``ffprobe`` subprocess usage is confined to this module.
**No PyAV dependency** — only requires ``ffmpeg`` on PATH.
"""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import subprocess
import threading
from functools import lru_cache
from pathlib import Path
from queue import Empty, Queue

from mcap_codec_support.video.common import (
    PROBE_JPEG,
    DecompressedFrame,
    EncoderConfig,
    VideoEncoderError,
    build_encoder_options,
)
from mcap_codec_support.video.common import (
    resolve_encoder as _resolve_encoder,
)

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


@lru_cache(maxsize=32)
def probe_encoder_cli(encoder_name: str, timeout: float = 5.0) -> bool:
    """Check that *encoder_name* can encode a frame on this host.

    FFmpeg may list hardware encoders compiled into the binary even when the
    corresponding device or driver is unavailable. A small real encode keeps
    automatic selection from choosing those unusable encoders.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg or not check_encoder_cli(encoder_name):
        return False
    width = height = 64
    try:
        result = subprocess.run(  # noqa: S603
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-s:v",
                f"{width}x{height}",
                "-r",
                "1",
                "-i",
                "pipe:0",
                "-frames:v",
                "1",
                "-c:v",
                encoder_name,
                "-f",
                "null",
                "-",
            ],
            input=bytes(width * height * 3 // 2),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def resolve_encoder(codec: str) -> str:
    """Pick the best working encoder for *codec* using ffmpeg CLI to probe."""
    if not find_ffmpeg():
        raise VideoEncoderError("ffmpeg not found on PATH")
    try:
        return _resolve_encoder(codec, test_fn=probe_encoder_cli)
    except ValueError as exc:
        raise VideoEncoderError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Hardware MJPEG decode probe (portable JPEG-decode offload)
# ---------------------------------------------------------------------------

# Platform → hardware MJPEG decoders to try, best first. These offload JPEG
# decode off the CPU inside the same ffmpeg process that encodes. Apple's
# VideoToolbox exposes no named MJPEG *decoder* through ffmpeg (H.264/HEVC/ProRes
# only), so macOS has no candidate here and stays on CPU decode.
_HW_MJPEG_DECODERS: dict[str, tuple[str, ...]] = {
    "Linux": ("mjpeg_cuvid", "mjpeg_vaapi", "mjpeg_qsv"),
}


def _hw_mjpeg_candidates() -> tuple[str, ...]:
    """Hardware MJPEG decoders worth probing on this host, best first.

    On Jetson/Tegra ``mjpeg_cuvid`` (the CUDA NVDEC path) *hangs*, and the
    hardware JPEG path there is ``nvjpegdec`` via the ``gstreamer`` backend — so
    it is dropped to avoid paying the probe-timeout to rediscover that every run.
    """
    names = _HW_MJPEG_DECODERS.get(platform.system(), ())
    if Path("/etc/nv_tegra_release").exists():
        names = tuple(n for n in names if n != "mjpeg_cuvid")
    return names


def check_decoder_cli(decoder_name: str) -> bool:
    """Check whether the system ``ffmpeg`` lists *decoder_name*."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    try:
        result = subprocess.run(  # noqa: S603
            [ffmpeg, "-hide_banner", "-decoders"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return any(
        len(parts) >= 2 and parts[1] == decoder_name
        for parts in (line.split() for line in result.stdout.splitlines())
    )


@lru_cache(maxsize=8)
def probe_hw_mjpeg_decoder(timeout: float = 2.5) -> str | None:
    """Return a *working* hardware MJPEG decoder name for this host, or None.

    Probes each platform candidate by actually decoding a tiny embedded JPEG
    under a hard *timeout*. A broken hardware decoder tends to *hang* rather than
    error (``mjpeg_cuvid`` on Jetson Thor wedges in CUDA), which is worse than
    slow — so the deadline, after which the child is killed, is what makes this
    safe to consult by default: callers treat ``None`` as "decode on the CPU".
    Cached, so the (possibly slow) probe runs once per process.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return None
    for name in _hw_mjpeg_candidates():
        if not check_decoder_cli(name):
            continue
        try:
            result = subprocess.run(  # noqa: S603
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "image2pipe",
                    "-c:v",
                    name,
                    "-i",
                    "pipe:0",
                    "-frames:v",
                    "1",
                    "-f",
                    "null",
                    "-",
                ],
                input=PROBE_JPEG,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue  # hung or failed to launch → treat as unavailable
        if result.returncode == 0:
            return name
    return None


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

_START_CODE_3 = b"\x00\x00\x01"
_H264_VCL_TYPES = frozenset({1, 2, 3, 4, 5})
# Non-VCL NAL types that may only appear at the start of an access unit
# (H.264 §7.4.1.2.3): SEI, SPS, PPS, AUD, prefix NAL, subset SPS.
_H264_AU_START_NONVCL = frozenset({6, 7, 8, 9, 14, 15})
_H265_MAX_VCL_TYPE = 31
# H.265 §7.4.2.4.4: VPS, SPS, PPS, AUD, prefix SEI.
_H265_AU_START_NONVCL = frozenset({32, 33, 34, 35, 39})


class AnnexBParser:
    """Split an Annex B byte stream into per-access-unit chunks.

    A new access unit starts at a VCL NAL that begins a picture
    (``first_mb_in_slice == 0`` / ``first_slice_segment_in_pic_flag``) or at a
    non-VCL NAL that may only open an AU (SPS/PPS/SEI/AUD/...), once the
    current AU holds at least one VCL NAL. Both 3- and 4-byte start codes are
    recognized — x264 uses a 4-byte code only on the first NAL of an AU, so
    scanning 4-byte codes alone misses the IDR slices behind SPS/PPS and
    merges each SPS-led IDR AU into the previous frame.
    """

    def __init__(self, codec: str) -> None:
        self._is_h265 = "265" in codec or "hevc" in codec
        self._buf = bytearray()
        self._scan_pos = 0
        self._has_vcl = False

    def _is_vcl(self, nal_header: int) -> bool:
        if self._is_h265:
            nal_type = (nal_header >> 1) & 0x3F
            return nal_type <= _H265_MAX_VCL_TYPE
        nal_type = nal_header & 0x1F
        return nal_type in _H264_VCL_TYPES

    def _starts_new_au(self, header_pos: int) -> bool:
        header = self._buf[header_pos]
        if self._is_h265:
            nal_type = (header >> 1) & 0x3F
            if nal_type <= _H265_MAX_VCL_TYPE:
                # first_slice_segment_in_pic_flag: first bit after 2-byte header.
                return bool(self._buf[header_pos + 2] & 0x80)
            return nal_type in _H265_AU_START_NONVCL
        nal_type = header & 0x1F
        if nal_type in _H264_VCL_TYPES:
            # first_mb_in_slice is Exp-Golomb-coded; a leading 1-bit means value 0.
            return bool(self._buf[header_pos + 1] & 0x80)
        return nal_type in _H264_AU_START_NONVCL

    def _scan(self, *, final: bool) -> list[bytes]:
        """Emit finished AUs from the buffer; the open AU stays at ``buf[0:]``."""
        buf = self._buf
        result: list[bytes] = []
        pos = self._scan_pos
        # Bytes needed past the start code to classify a NAL: header plus the
        # first payload byte (2-byte header for H.265).
        classify_len = 3 if self._is_h265 else 2
        while True:
            idx = buf.find(_START_CODE_3, pos)
            if idx == -1:
                # A start code may straddle the chunk boundary; rescan its tail.
                pos = max(pos, len(buf) - 2)
                break
            header_pos = idx + 3
            if header_pos + classify_len > len(buf) and not final:
                pos = idx
                break
            sc_start = idx - 1 if idx > 0 and buf[idx - 1] == 0 else idx
            classifiable = header_pos + classify_len <= len(buf)
            if sc_start > 0 and self._has_vcl and classifiable and self._starts_new_au(header_pos):
                result.append(bytes(buf[:sc_start]))
                del buf[:sc_start]
                header_pos -= sc_start
                self._has_vcl = False
            if header_pos < len(buf) and self._is_vcl(buf[header_pos]):
                self._has_vcl = True
            pos = header_pos
        self._scan_pos = pos
        return result

    def feed(self, data: bytes) -> list[bytes]:
        self._buf.extend(data)
        return self._scan(final=False)

    def flush(self) -> bytes | None:
        """Return any remaining data as a single final access unit."""
        items = self.flush_list()
        if not items:
            return None
        return b"".join(items)

    def flush_list(self) -> list[bytes]:
        """Return remaining data split into individual access units."""
        result = self._scan(final=True)
        if self._buf:
            result.append(bytes(self._buf))
            self._buf.clear()
        self._scan_pos = 0
        self._has_vcl = False
        return result


# IVF file header: 'DKIF', then version/header-length/fourcc/dimensions/etc.
_IVF_FILE_HEADER_LEN = 32
# Per-frame header: 4-byte little-endian frame size, then an 8-byte timestamp.
_IVF_FRAME_HEADER_LEN = 12


class IVFParser:
    """Split an IVF byte stream (``ffmpeg -f ivf``) into per-frame packets.

    VP9 and AV1 are emitted by ffmpeg in the IVF container rather than a raw
    Annex-B elementary stream, so their frame boundaries come from the 12-byte
    IVF frame headers (a 4-byte little-endian size prefix) instead of NAL start
    codes. Mirrors :class:`AnnexBParser`'s ``feed`` / ``flush_list`` interface so
    :class:`FFmpegVideoEncoder` can swap parsers by codec family.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._file_header_seen = False

    def feed(self, data: bytes) -> list[bytes]:
        self._buf.extend(data)
        return self._drain()

    def flush_list(self) -> list[bytes]:
        return self._drain()

    def _drain(self) -> list[bytes]:
        if not self._file_header_seen:
            if len(self._buf) < _IVF_FILE_HEADER_LEN:
                return []
            del self._buf[:_IVF_FILE_HEADER_LEN]
            self._file_header_seen = True
        frames: list[bytes] = []
        while len(self._buf) >= _IVF_FRAME_HEADER_LEN:
            frame_size = int.from_bytes(self._buf[0:4], "little")
            end = _IVF_FRAME_HEADER_LEN + frame_size
            if len(self._buf) < end:
                break
            frames.append(bytes(self._buf[_IVF_FRAME_HEADER_LEN:end]))
            del self._buf[:end]
        return frames


# ---------------------------------------------------------------------------
# Codec helpers
# ---------------------------------------------------------------------------

_CODEC_TO_FORMAT: dict[str, str] = {
    "h264": "h264",
    "h265": "hevc",
    "hevc": "hevc",
    "vp9": "ivf",
    "av1": "ivf",
}

# Codec families that ffmpeg muxes into IVF (and that IVFParser splits) rather
# than a raw Annex-B elementary stream.
_IVF_FAMILIES = frozenset({"vp9", "av1"})


def _codec_family(codec_name: str) -> str:
    lower = codec_name.lower()
    if "264" in lower:
        return "h264"
    if "265" in lower or "hevc" in lower:
        return "h265"
    if "vp9" in lower:
        return "vp9"
    if "av1" in lower:
        return "av1"
    return "h264"


def _build_output_args(
    codec_fam: str, codec_name: str, gop_size: int, options: dict[str, str], bit_rate: int | None
) -> list[str]:
    """Build the shared encoder output arguments."""
    cmd: list[str] = [
        # Modern replacement for the removed ``-vsync 0``: pass every frame
        # through with its own timestamp. ``-vsync`` errors on ffmpeg 7+.
        "-fps_mode",
        "passthrough",
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
            cmd, input=data, capture_output=True, timeout=10, check=False
        )
        parts = result.stdout.decode("utf-8", errors="replace").strip().split(",")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass
    raise VideoEncoderError("Cannot determine image dimensions")


def probe_image_pipe_decode(data: bytes, timeout: float = 5.0) -> bool:
    """Return whether ffmpeg can decode one image from ``image2pipe``."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    cmd = [
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
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            input=data,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


class FFmpegVideoEncoder:
    """Encode frames to H.264/H.265 via a single ffmpeg process.

    When *input_pix_fmt* is ``None`` (the default), the encoder accepts
    JPEG/PNG bytes via ``image2pipe``. When set (e.g. ``"rgb24"``), it
    accepts raw pixel data of that format via ``rawvideo``.

    *decode_codec* (JPEG path only) forces the input decoder — pass a hardware
    MJPEG decoder from :func:`probe_hw_mjpeg_decoder` to offload JPEG decode off
    the CPU inside this same process; ``None`` lets ffmpeg pick the CPU decoder.
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
        decode_codec: str | None = None,
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
            # Compressed images (JPEG/PNG/etc.) — let ffmpeg detect the image codec,
            # or force a hardware MJPEG decoder to offload JPEG decode off the CPU.
            cmd = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "image2pipe",
                "-probesize",
                "32",
            ]
            if decode_codec is not None:
                cmd.extend(["-c:v", decode_codec])
            cmd.extend(["-r", str(fps_int), "-i", "pipe:0"])

        if scale is not None:
            sw, sh = scale
            cmd.extend(["-vf", f"scale={sw}:{sh}"])

        cmd.extend(_build_output_args(codec_fam, codec_name, gop_size, options, bit_rate))

        self.config = EncoderConfig(width=width, height=height, codec_name=codec_name)

        try:
            self._process = subprocess.Popen(  # noqa: S603
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise VideoEncoderError(f"Failed to start ffmpeg: {exc}") from exc

        self._parser: AnnexBParser | IVFParser = (
            IVFParser() if codec_fam in _IVF_FAMILIES else AnnexBParser(codec_fam)
        )
        self._output_queue: Queue[bytes | None] = Queue()
        self._stderr_lines: list[str] = []

        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

        self._is_image_pipe = input_pix_fmt is None

    def _read_stdout(self) -> None:
        if self._process.stdout is None:
            return
        fd = self._process.stdout.fileno()
        # Dedicated thread: block on read (default pipe mode) instead of
        # spinning on a non-blocking fd with time.sleep — the poll loop burned
        # ~200 wakeups/s per encoder for no benefit and added output latency.
        try:
            while True:
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                for au in self._parser.feed(chunk):
                    self._output_queue.put(au)
            for au in self._parser.flush_list():
                self._output_queue.put(au)
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
        """Write *frame* bytes to ffmpeg stdin and return one access unit (or None).

        The output queue is drained non-blocking: waiting here would serialize
        Python with the encoder, so a frame's access unit is typically returned
        by a later ``encode`` call and the tail by ``flush_packets``.
        """
        if self._process.stdin is None:
            raise VideoEncoderError("ffmpeg stdin is not available")
        try:
            self._process.stdin.write(frame)
            self._process.stdin.flush()
        except BrokenPipeError as exc:
            stderr_tail = "\n".join(self._stderr_lines[-5:])
            raise VideoEncoderError(f"ffmpeg process died unexpectedly:\n{stderr_tail}") from exc

        try:
            au = self._output_queue.get_nowait()
        except Empty:
            return None

        if au is None:
            stderr_tail = "\n".join(self._stderr_lines[-5:])
            raise VideoEncoderError(f"ffmpeg exited prematurely:\n{stderr_tail}")
        return au

    def flush_packets(self) -> list[bytes]:
        """Close the encoder and return all remaining access units."""
        if self._process.stdin and not self._process.stdin.closed:
            with contextlib.suppress(BrokenPipeError):
                self._process.stdin.close()

        self._stdout_thread.join(timeout=10)
        self._stderr_thread.join(timeout=5)
        self._process.wait(timeout=10)

        packets: list[bytes] = []
        while True:
            try:
                item = self._output_queue.get_nowait()
            except Empty:
                break
            if item is None:
                break
            packets.append(item)

        if self._process.returncode and self._process.returncode != 0:
            stderr_tail = "\n".join(self._stderr_lines[-5:])
            raise VideoEncoderError(
                f"ffmpeg exited with code {self._process.returncode}:\n{stderr_tail}"
            )

        return packets

    def close(self) -> None:
        """Terminate the ffmpeg subprocess if it is still running (idempotent)."""
        try:
            if self._process.poll() is None:
                self._process.kill()
                self._process.wait(timeout=2)
        except Exception:  # noqa: BLE001, S110
            pass

    def __del__(self) -> None:
        self.close()


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
        self._probe_buffer = bytearray()

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
                cmd, input=data, capture_output=True, timeout=10, check=False
            )
            parts = result.stdout.decode().strip().split(",")
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
        data_to_write = video_data
        # Start process on first call.
        if self._process is None:
            # For raw mode, detect dimensions before starting.
            if self._video_format != "compressed" and self._width is None:
                self._probe_buffer.extend(video_data)
                try:
                    self._width, self._height = self._detect_dimensions(
                        bytes(self._probe_buffer), codec
                    )
                except VideoEncoderError:
                    return None
                data_to_write = bytes(self._probe_buffer)
                self._probe_buffer.clear()
            self._start_process(codec)

        if self._process is None or self._process.stdin is None:
            raise VideoEncoderError("ffmpeg process not started")

        self._process.stdin.write(data_to_write)
        self._process.stdin.flush()

        try:
            return self._output_queue.get(timeout=0.2)
        except Empty:
            return None

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
