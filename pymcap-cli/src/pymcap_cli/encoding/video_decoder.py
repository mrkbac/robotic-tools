"""Video decoder backends (PyAV and FFmpeg subprocess).

Provides a unified interface for decoding H.264/H.265 video packets
back to individual frames. Mirrors the encoder pattern in
:mod:`pymcap_cli.encoding.image_utils` and
:mod:`pymcap_cli.encoding.subprocess_encoder`.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Protocol

from pymcap_cli.encoding.encoder_common import EncoderMode, VideoEncoderError

if TYPE_CHECKING:
    from av.video.frame import VideoFrame


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class VideoDecoderProtocol(Protocol):
    """Structural interface shared by VideoDecoder and SubprocessVideoDecoder."""

    def decode(self, data: bytes) -> VideoFrame | None:
        """Decode a single compressed video packet.

        Returns a ``VideoFrame`` or ``None`` if the decoder needs more data
        (e.g. waiting for a keyframe).
        """
        ...

    def flush(self) -> list[VideoFrame]:
        """Flush any buffered frames from the decoder."""
        ...


# ---------------------------------------------------------------------------
# PyAV backend
# ---------------------------------------------------------------------------


class VideoDecoder:
    """PyAV-based H.264/H.265 decoder.

    Maintains a persistent codec context for proper P-frame decoding.
    One instance per video channel.
    """

    def __init__(self, codec: str) -> None:
        import av  # noqa: PLC0415

        codec_name = "h264" if codec == "h264" else "hevc"
        self._ctx: Any = av.CodecContext.create(codec_name, "r")
        self._ctx.open()

    def decode(self, data: bytes) -> VideoFrame | None:
        import av  # noqa: PLC0415

        packet = av.Packet(data)
        frames = self._ctx.decode(packet)
        if not frames:
            return None
        return frames[-1]

    def flush(self) -> list[VideoFrame]:
        packet = None
        frames = self._ctx.decode(packet)
        return list(frames)


# ---------------------------------------------------------------------------
# FFmpeg subprocess backend
# ---------------------------------------------------------------------------

_CODEC_TO_FORMAT: dict[str, str] = {
    "h264": "h264",
    "h265": "hevc",
    "hevc": "hevc",
}


class SubprocessVideoDecoder:
    """FFmpeg subprocess-based H.264/H.265 decoder.

    Pipes compressed video packets into an ``ffmpeg`` process and reads
    raw RGB24 frames from stdout. Wraps the output into ``av.VideoFrame``
    objects for a consistent interface.
    """

    def __init__(self, codec: str) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise VideoEncoderError("ffmpeg not found on PATH")

        input_format = _CODEC_TO_FORMAT.get(codec, "h264")

        cmd: list[str] = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            # Input: raw bitstream on stdin.
            "-f",
            input_format,
            "-i",
            "pipe:0",
            # Output: raw RGB24 frames on stdout.
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]

        try:
            self._process = subprocess.Popen(  # noqa: S603
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise VideoEncoderError(f"Failed to start ffmpeg decoder: {exc}") from exc

        self._codec = codec
        self._width: int | None = None
        self._height: int | None = None
        self._frame_size: int | None = None
        self._output_queue: Queue[VideoFrame | None] = Queue()
        self._stderr_lines: list[str] = []

        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _detect_dimensions(self, data: bytes) -> tuple[int, int]:
        """Detect video dimensions from the bitstream using ffprobe."""
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise VideoEncoderError("ffprobe not found on PATH")

        input_format = _CODEC_TO_FORMAT.get(self._codec, "h264")
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
        assert self._process.stdout is not None
        buf = bytearray()
        try:
            while True:
                chunk = self._process.stdout.read(65536)
                if not chunk:
                    break
                buf.extend(chunk)

                # Wait until we have dimensions before reading frames
                if self._frame_size is None:
                    continue

                while len(buf) >= self._frame_size:
                    frame_bytes = bytes(buf[: self._frame_size])
                    del buf[: self._frame_size]
                    frame = self._bytes_to_frame(frame_bytes)
                    self._output_queue.put(frame)
        finally:
            # Process any remaining complete frames
            if self._frame_size is not None:
                while len(buf) >= self._frame_size:
                    frame_bytes = bytes(buf[: self._frame_size])
                    del buf[: self._frame_size]
                    frame = self._bytes_to_frame(frame_bytes)
                    self._output_queue.put(frame)
            self._output_queue.put(None)  # sentinel

    def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr_lines.append(line.decode(errors="replace").rstrip())

    def _bytes_to_frame(self, data: bytes) -> VideoFrame:
        """Convert raw RGB24 bytes to an av.VideoFrame."""
        import av  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        assert self._width is not None
        assert self._height is not None
        arr = np.frombuffer(data, dtype=np.uint8).reshape(self._height, self._width, 3)
        return av.VideoFrame.from_ndarray(arr, format="rgb24")

    def decode(self, data: bytes) -> VideoFrame | None:
        assert self._process.stdin is not None

        # Detect dimensions from the first packet (which should contain SPS/PPS)
        if self._width is None:
            try:
                self._width, self._height = self._detect_dimensions(data)
                self._frame_size = self._width * self._height * 3
            except VideoEncoderError:
                # Can't detect yet, just pipe data and wait
                self._process.stdin.write(data)
                self._process.stdin.flush()
                return None

        self._process.stdin.write(data)
        self._process.stdin.flush()

        try:
            frame = self._output_queue.get(timeout=0.1)
        except Empty:
            return None
        else:
            return frame if frame is not None else None

    def flush(self) -> list[VideoFrame]:
        if self._process.stdin is not None:
            self._process.stdin.close()

        frames: list[VideoFrame] = []
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_video_decoder(
    codec: str,
    mode: EncoderMode = EncoderMode.AUTO,
) -> VideoDecoderProtocol:
    """Create a video decoder using the requested backend.

    Args:
        codec: Video codec ("h264" or "h265").
        mode: Backend selection (AUTO tries PyAV first, falls back to ffmpeg CLI).
    """
    if mode == EncoderMode.PYAV:
        return VideoDecoder(codec)

    if mode == EncoderMode.FFMPEG_CLI:
        return SubprocessVideoDecoder(codec)

    # AUTO: try PyAV, fall back to subprocess ffmpeg.
    try:
        return VideoDecoder(codec)
    except (ImportError, Exception):
        if shutil.which("ffmpeg"):
            return SubprocessVideoDecoder(codec)
        raise
