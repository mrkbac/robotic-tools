"""GStreamer-subprocess video backend (NVIDIA Jetson L4T hardware).

Named for the *mechanism* — it drives ``gst-launch-1.0`` — so it can grow other
element sets later (VAAPI, generic software). Today it targets the NVIDIA Jetson
L4T ``nv*`` elements; on a host without them ``resolve_encoder`` fails cleanly and
the backend is skipped.

Encodes via the Jetson V4L2 codec engine and the dedicated hardware JPEG block
through the stock L4T GStreamer ``nv*`` elements — ``nvjpegdec`` (hardware JPEG
decode), ``nvv4l2h264enc`` / ``nvv4l2h265enc`` (hardware H.264/H.265 encode), and
``nvvidconv`` (hardware colourspace/scale). This is the Jetson Multimedia API
surface exposed via GStreamer, distinct from the desktop CUDA NVENC/NVDEC SDK
that PyAV and ffmpeg's ``h264_nvenc`` / ``mjpeg_cuvid`` use — the latter's JPEG
decode (CUVID) hangs on Jetson Thor, whereas ``nvjpegdec`` runs on the separate
JPEG silicon and works.

The differentiator vs the ffmpeg-cli / PyAV backends: JPEG decode is offloaded to
hardware inside the same pipeline as the encode, so the whole camera transcode is
fixed-function (no CPU JPEG decode, no CUDA compute).

No Python GStreamer binding (``gi``) is required: ``gst-launch-1.0`` is driven as
a subprocess exactly like :class:`FFmpegVideoEncoder`, so the backend works under
``uvx`` with only the stock L4T plugins present. The encoded bitstream is read
back on a dedicated inherited fd (not stdout) because the ``nv*`` elements print
status chatter ("Opening in BLOCKING MODE") straight to stdout.

**Colour-fidelity caveat.** ``nvjpegdec`` decodes full-range JFIF JPEGs as if
they were limited/studio-range and expands the levels (measured affine, per
channel: decoded ~= 1.16 * libjpeg - 18.5), which crushes shadow detail toward
black and clips highlights. On bright footage the shift is small; on dark/low-light footage it is
pronounced. libjpeg (PyAV/Pillow) and ffmpeg's ``mjpeg`` decoder both agree on
full-range and do not exhibit this, so the Jetson output can look darker than the
other backends. The nv GStreamer elements expose no range override that fixes it
(the encoder rejects full-range NV12 input, and ``nvjpegdec`` will not renegotiate
its output colorimetry). AUTO may select this backend when the hardware pipeline
probe succeeds because throughput is the priority for roscompress; use explicit
``pyav`` or ``ffmpeg-cli`` when exact shadow fidelity matters more than speed.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import threading
import time
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING

from mcap_codec_support.video.common import (
    DEFAULT_FPS,
    DEFAULT_GOP_SIZE,
    PROBE_JPEG,
    SOFTWARE_CODEC_MAP,
    EncoderConfig,
    VideoEncoderError,
)
from mcap_codec_support.video.ffmpeg import (
    ROS_ENCODING_TO_PIX_FMT,
    AnnexBParser,
    FFmpegVideoEncoder,
    probe_image_dimensions,
)
from mcap_codec_support.video.schemas import COMPRESSED_SCHEMAS

if TYPE_CHECKING:
    from mcap_codec_support._protocols import DecodableImageMessage

# ---------------------------------------------------------------------------
# Element discovery
# ---------------------------------------------------------------------------

# Short codec name -> (nv encoder element, Annex-B parser element, output caps).
_CODEC_TO_NV: dict[str, tuple[str, str, str]] = {
    "h264": ("nvv4l2h264enc", "h264parse", "video/x-h264,stream-format=byte-stream"),
    "h265": ("nvv4l2h265enc", "h265parse", "video/x-h265,stream-format=byte-stream"),
    "hevc": ("nvv4l2h265enc", "h265parse", "video/x-h265,stream-format=byte-stream"),
}

# ROS raw image pix_fmt (ffmpeg name) -> rawvideoparse format name.
_PIX_FMT_TO_RAW: dict[str, str] = {
    "rgb24": "rgb",
    "bgr24": "bgr",
    "gray": "gray8",
}

# Large camera JPEG/raw frames overflow Python's BufferedWriter and reach the
# child without an explicit flush. Tiny probe/live frames need the flush to keep
# latency predictable and avoid buffered cleanup warnings during forced close.
_SMALL_WRITE_FLUSH_BYTES = 64 * 1024


def find_gst_launch() -> str | None:
    """Return the path to ``gst-launch-1.0`` if on PATH, else None."""
    return shutil.which("gst-launch-1.0")


@lru_cache(maxsize=64)
def gst_element_available(name: str) -> bool:
    """Return True if GStreamer knows the element *name* (cached)."""
    inspect = shutil.which("gst-inspect-1.0")
    if not inspect:
        return False
    try:
        result = subprocess.run(  # noqa: S603
            [inspect, name], capture_output=True, timeout=10, check=False
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def _codec_key(codec: str) -> str:
    lower = codec.lower()
    if "264" in lower:
        return "h264"
    if "265" in lower or "hevc" in lower:
        return "h265"
    return lower


def resolve_encoder(codec: str) -> str:
    """Return the Jetson hardware encoder element for *codec*.

    Raises ``VideoEncoderError`` if gst-launch or the element is unavailable — so
    backend auto-selection can skip Jetson on non-Jetson systems.
    """
    if not find_gst_launch():
        raise VideoEncoderError("gst-launch-1.0 not found on PATH")
    key = _codec_key(codec)
    entry = _CODEC_TO_NV.get(key)
    if entry is None:
        raise VideoEncoderError(f"Jetson backend does not support codec {codec!r}")
    element = entry[0]
    if not gst_element_available(element):
        raise VideoEncoderError(f"GStreamer element {element!r} not available")
    return element


def check_encoder(encoder_name: str) -> bool:
    """Return True if *encoder_name* is a usable nv encoder element here."""
    return encoder_name in {v[0] for v in _CODEC_TO_NV.values()} and gst_element_available(
        encoder_name
    )


@lru_cache(maxsize=1)
def _nvjpeg_library_dirs() -> tuple[str, ...]:
    """CUDA package library directories that contain libnvjpeg, if installed."""
    dirs: list[str] = []
    seen: set[str] = set()

    try:
        package = distribution("nvidia-nvjpeg")
    except PackageNotFoundError:
        pass
    else:
        for relative_path in package.files or ():
            if relative_path.name.startswith("libnvjpeg.so"):
                library = Path(str(package.locate_file(relative_path)))
                if library.is_file():
                    path = str(library.parent)
                    if path not in seen:
                        seen.add(path)
                        dirs.append(path)

    system_candidates = [
        *Path("/usr/local").glob("cuda*/targets/*/lib"),
        *Path("/usr/local").glob("cuda*/lib64"),
    ]
    for candidate in system_candidates:
        try:
            has_nvjpeg = any(candidate.glob("libnvjpeg.so*"))
        except OSError:
            continue
        if has_nvjpeg:
            path = str(candidate)
            if path not in seen:
                seen.add(path)
                dirs.append(path)
    return tuple(dirs)


def _gstreamer_env() -> dict[str, str] | None:
    """Add CUDA libnvjpeg dirs to the child loader path when they are not global."""
    nvjpeg_dirs = _nvjpeg_library_dirs()
    if not nvjpeg_dirs:
        return None

    env = os.environ.copy()
    entries = list(nvjpeg_dirs)
    existing = env.get("LD_LIBRARY_PATH")
    if existing:
        entries.extend(existing.split(os.pathsep))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(entry for entry in entries if entry))
    return env


# ---------------------------------------------------------------------------
# GStreamerVideoEncoder
# ---------------------------------------------------------------------------


def _quality_to_qp(quality: int) -> int:
    """Map the user quality knob (CRF-like, lower = better) to an H.264/H.265 QP.

    The nv encoders ignore the ``bitrate`` property on this ``fdsrc → jpegparse``
    path — the buffers carry no framerate, so bits-*per-second* rate control has
    nothing to divide by and the stream comes out near-lossless (bigger than the
    source JPEGs). Constant-QP is both the mode that actually works and the right
    semantic match: the ffmpeg-cli NVENC path uses ``rc=vbr cq=<quality>``.
    Jetson's V4L2 constant-QP scale is consistently less aggressive than NVENC's
    CQ scale here, so apply a small offset to keep the default backend output
    size in the same range.
    """
    return max(0, min(51, quality + 7))


class GStreamerVideoEncoder:
    """Encode frames to H.264/H.265 via a single ``gst-launch-1.0`` process.

    When *input_pix_fmt* is ``None`` the encoder accepts a stream of concatenated
    JPEG bytes (``jpegparse ! nvjpegdec`` — hardware JPEG decode). When set (e.g.
    ``"rgb24"``) it accepts raw pixel data of that format via ``rawvideoparse``.
    In both cases ``nvvidconv`` converts to NVMM ``NV12`` for the hardware encoder.
    """

    def __init__(
        self,
        width: int,
        height: int,
        codec_name: str,
        quality: int = 28,
        target_fps: float = DEFAULT_FPS,
        gop_size: int = DEFAULT_GOP_SIZE,
        *,
        input_pix_fmt: str | None = None,
        scale: tuple[int, int] | None = None,
    ) -> None:
        gst = find_gst_launch()
        if not gst:
            raise VideoEncoderError("gst-launch-1.0 not found on PATH")

        key = _codec_key(codec_name)
        nv = _CODEC_TO_NV.get(key)
        if nv is None or nv[0] != codec_name:
            raise VideoEncoderError(f"GStreamerVideoEncoder cannot use encoder {codec_name!r}")
        enc_element, parse_element, out_caps = nv
        self._codec_family = "h265" if key in {"h265", "hevc"} else "h264"

        fps_int = max(round(target_fps), 1)
        out_w, out_h = scale if scale is not None else (width, height)
        out_w -= out_w % 2
        out_h -= out_h % 2
        qp = _quality_to_qp(quality)

        r_fd, w_fd = os.pipe()
        os.set_inheritable(w_fd, True)
        self._read_fd = r_fd

        if input_pix_fmt is None:
            decode = ["jpegparse", "!", "nvjpegdec", "!"]
        else:
            raw_fmt = _PIX_FMT_TO_RAW.get(input_pix_fmt)
            if raw_fmt is None:
                os.close(r_fd)
                os.close(w_fd)
                raise VideoEncoderError(f"Jetson backend cannot feed raw pix_fmt {input_pix_fmt!r}")
            decode = [
                "rawvideoparse",
                f"format={raw_fmt}",
                f"width={width}",
                f"height={height}",
                f"framerate={fps_int}/1",
                "!",
                "videoconvert",
                "!",
            ]

        nvmm_caps = f"video/x-raw(memory:NVMM),format=NV12,width={out_w},height={out_h}"
        enc_opts = [
            "control-rate=2",  # constant QP (bitrate modes are ignored on this path)
            f"constqp={qp}:{qp}:{qp}",
            f"iframeinterval={gop_size}",
            f"idrinterval={gop_size}",
            "num-B-Frames=0",  # packet order == frame order (like ffmpeg -bf 0)
            "insert-sps-pps=1",  # each IDR is independently decodable
        ]
        pipeline = [
            "fdsrc",
            "fd=0",
            "!",
            *decode,
            "nvvidconv",
            "!",
            nvmm_caps,
            "!",
            enc_element,
            *enc_opts,
            "!",
            parse_element,
            "config-interval=-1",
            "!",
            out_caps,
            "!",
            "fdsink",
            f"fd={w_fd}",
        ]

        self.config = EncoderConfig(width=out_w, height=out_h, codec_name=codec_name)

        try:
            self._process = subprocess.Popen(  # noqa: S603
                [gst, "-q", *pipeline],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,  # nv element chatter goes here
                stderr=subprocess.PIPE,
                pass_fds=(w_fd,),
                env=_gstreamer_env(),
            )
        except OSError as exc:
            os.close(r_fd)
            raise VideoEncoderError(f"Failed to start gst-launch: {exc}") from exc
        finally:
            os.close(w_fd)  # keep only the read end in this process

        self._parser = AnnexBParser(self._codec_family)
        self._output_queue: Queue[bytes | None] = Queue()
        self._stderr_lines: list[str] = []
        self._stdout_thread = threading.Thread(target=self._read_output, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _read_output(self) -> None:
        fd = self._read_fd
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
        """Write *frame* to the pipeline and return one access unit (or None).

        Output is drained non-blocking: a frame's access unit is typically
        returned by a later ``encode`` call and the tail by ``flush_packets``.
        """
        if self._process.stdin is None:
            raise VideoEncoderError("gst-launch stdin is not available")
        try:
            self._process.stdin.write(frame)
            if len(frame) < _SMALL_WRITE_FLUSH_BYTES:
                self._process.stdin.flush()
        except BrokenPipeError as exc:
            stderr_tail = "\n".join(self._stderr_lines[-5:])
            raise VideoEncoderError(f"gst-launch died unexpectedly:\n{stderr_tail}") from exc

        try:
            au = self._output_queue.get_nowait()
        except Empty:
            return None
        if au is None:
            stderr_tail = "\n".join(self._stderr_lines[-5:])
            raise VideoEncoderError(f"gst-launch exited prematurely:\n{stderr_tail}")
        return au

    def flush_packets(self) -> list[bytes]:
        """Close the pipeline and return all remaining access units."""
        if self._process.stdin and not self._process.stdin.closed:
            with contextlib.suppress(BrokenPipeError):
                self._process.stdin.close()

        self._stdout_thread.join(timeout=10)
        self._stderr_thread.join(timeout=5)
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.close()

        packets: list[bytes] = []
        while True:
            try:
                item = self._output_queue.get_nowait()
            except Empty:
                break
            if item is None:
                break
            packets.append(item)

        with contextlib.suppress(OSError):
            os.close(self._read_fd)

        if self._process.returncode:
            stderr_tail = "\n".join(self._stderr_lines[-5:])
            raise VideoEncoderError(
                f"gst-launch exited with code {self._process.returncode}:\n{stderr_tail}"
            )
        return packets

    def close(self) -> None:
        """Terminate the gst-launch subprocess if still running (idempotent)."""
        try:
            if self._process.stdin and not self._process.stdin.closed:
                with contextlib.suppress(BrokenPipeError, OSError, ValueError):
                    self._process.stdin.close()
            if self._process.poll() is None:
                self._process.kill()
                self._process.wait(timeout=2)
        except Exception:  # noqa: BLE001, S110
            pass
        with contextlib.suppress(OSError):
            os.close(self._read_fd)

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Hardware probe (timed, always falls back)
# ---------------------------------------------------------------------------


def probe_hw_jpeg_pipeline(codec: str = "h264", timeout: float = 8.0) -> bool:
    """Return True iff the hardware JPEG-decode → encode pipeline produces output.

    Runs the real ``nvjpegdec → nvv4l2*enc`` path on a tiny embedded JPEG and
    waits at most *timeout* seconds for a first access unit. A broken Jetson
    codec stack *hangs* rather than erroring (as CUVID does on Thor), so this
    hard deadline — after which the process is killed — is what makes the backend
    safe to select automatically: callers treat False as "fall back to CPU/ffmpeg".
    """
    try:
        encoder_name = resolve_encoder(codec)
    except VideoEncoderError:
        return False

    encoder: GStreamerVideoEncoder | None = None
    try:
        # Scale the tiny embedded JPEG up to 256x256: the V4L2 encoder rejects
        # very small frames (S_FMT fails well below ~128px), which would be a
        # false negative for a pipeline that is actually healthy.
        encoder = GStreamerVideoEncoder(
            32, 32, encoder_name, quality=28, gop_size=1, scale=(256, 256)
        )
        deadline = time.monotonic() + timeout
        got = False
        for _ in range(4):
            if encoder.encode(PROBE_JPEG) is not None:
                got = True
                break
        while not got and time.monotonic() < deadline:
            try:
                item = encoder._output_queue.get(timeout=0.1)  # noqa: SLF001
            except Empty:
                continue
            if item is None:
                break
            got = True
    except (VideoEncoderError, OSError):
        return False
    else:
        return got
    finally:
        if encoder is not None:
            encoder.close()


# ---------------------------------------------------------------------------
# Compression backend
# ---------------------------------------------------------------------------


class GStreamerCompressionBackend:
    """CompressedVideo backend using Jetson hardware JPEG decode + V4L2 encode.

    Shape mirrors the ffmpeg-cli backend (``FrameT = bytes``): JPEG/raw bytes are
    passed straight through as the "frame" and decoded inside the encoder
    pipeline, so no separate decode step runs on the CPU. Software fallback (used
    only if the hardware encoder dies mid-stream) is delegated to
    :class:`FFmpegVideoEncoder`.
    """

    label = "gstreamer"
    prefetch_supported = False

    def __init__(self) -> None:
        self._topic_pix_fmt: dict[str, str | None] = {}

    def get_pix_fmt(self, topic: str) -> str | None:
        return self._topic_pix_fmt.get(topic)

    def test_encoder(self, encoder_name: str) -> bool:
        return check_encoder(encoder_name)

    def resolve_encoder(self, codec: str) -> str:
        return resolve_encoder(codec)

    def decode_compressed(self, data: bytes) -> tuple[bytes, int, int]:
        width, height = probe_image_dimensions(data)
        return data, width, height

    def decode_image(self, msg: DecodableImageMessage, schema_name: str) -> tuple[bytes, int, int]:
        data = bytes(msg.decoded_message.data)
        topic = msg.channel.topic
        if schema_name in COMPRESSED_SCHEMAS:
            self._topic_pix_fmt[topic] = None
            return self.decode_compressed(data)

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
    ) -> GStreamerVideoEncoder | FFmpegVideoEncoder:
        if codec_name in set(SOFTWARE_CODEC_MAP.values()):
            # Fallback path (hardware encoder died): reuse the proven CPU encoder.
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
        return GStreamerVideoEncoder(
            width=width,
            height=height,
            codec_name=codec_name,
            quality=quality,
            target_fps=DEFAULT_FPS,
            gop_size=DEFAULT_GOP_SIZE,
            input_pix_fmt=input_pix_fmt,
            scale=scale,
        )
