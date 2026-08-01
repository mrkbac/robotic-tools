"""Hermetic tests for FFmpeg CLI helpers and process lifecycles."""

import subprocess
from queue import Queue
from types import SimpleNamespace

import mcap_codec_support.video.ffmpeg as ffmpeg
import pytest
from mcap_codec_support.video.common import PROBE_JPEG, DecompressedFrame, VideoEncoderError


def test_ffmpeg_encoder_reports_closed_stdin_instead_of_value_error() -> None:
    class ClosedStdin:
        closed = True

        def write(self, _data: bytes) -> None:
            raise ValueError("I/O operation on closed file")

        def flush(self) -> None:
            raise ValueError("I/O operation on closed file")

    class Process:
        stdin = ClosedStdin()

    encoder = ffmpeg.FFmpegVideoEncoder.__new__(ffmpeg.FFmpegVideoEncoder)
    encoder._process = Process()
    encoder._stderr_lines = []
    encoder._output_queue = Queue()

    with pytest.raises(VideoEncoderError, match="ffmpeg process died unexpectedly"):
        encoder.encode(b"frame")


def test_check_encoder_cli_handles_missing_binary_and_errors(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: None)
    assert ffmpeg.check_encoder_cli("libx264") is False

    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        ffmpeg.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("ffmpeg", 5)),
    )
    assert ffmpeg.check_encoder_cli("libx264") is False


def test_check_encoder_cli_parses_encoder_listing(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        ffmpeg.subprocess,
        "run",
        lambda args, **_kwargs: subprocess.CompletedProcess(args, 0, " V....D libx264 H.264\n", ""),
    )

    assert ffmpeg.check_encoder_cli("libx264") is True
    assert ffmpeg.check_encoder_cli("nope") is False


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("ffmpeg", 5), OSError("no ffmpeg")])
def test_probe_encoder_cli_handles_probe_errors(monkeypatch, error) -> None:
    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(ffmpeg, "check_encoder_cli", lambda _name: True)
    monkeypatch.setattr(
        ffmpeg.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    ffmpeg.probe_encoder_cli.cache_clear()
    try:
        assert ffmpeg.probe_encoder_cli("libx264") is False
    finally:
        ffmpeg.probe_encoder_cli.cache_clear()


def test_resolve_encoder_requires_ffmpeg_and_wraps_value_error(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: None)
    with pytest.raises(VideoEncoderError, match="ffmpeg not found"):
        ffmpeg.resolve_encoder("h264")

    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        ffmpeg,
        "_resolve_encoder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad")),
    )
    with pytest.raises(VideoEncoderError, match="bad"):
        ffmpeg.resolve_encoder("h264")


def test_hw_mjpeg_candidates_skip_cuvid_on_jetson(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg, "_HW_MJPEG_DECODERS", {"Linux": ("mjpeg_cuvid", "mjpeg_qsv")})
    monkeypatch.setattr(ffmpeg.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ffmpeg, "Path", lambda _path: SimpleNamespace(exists=lambda: True))

    assert ffmpeg._hw_mjpeg_candidates() == ("mjpeg_qsv",)


def test_check_decoder_cli_handles_missing_binary_and_errors(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: None)
    assert ffmpeg.check_decoder_cli("mjpeg") is False

    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        ffmpeg.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("failed")),
    )
    assert ffmpeg.check_decoder_cli("mjpeg") is False


def test_probe_hw_mjpeg_decoder_handles_failures_and_success(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(ffmpeg, "_hw_mjpeg_candidates", lambda: ("first", "second"))
    monkeypatch.setattr(ffmpeg, "check_decoder_cli", lambda _name: True)
    calls = []

    def run(args, **_kwargs):
        calls.append(args)
        if args[args.index("-c:v") + 1] == "first":
            raise subprocess.TimeoutExpired("ffmpeg", 2)
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(ffmpeg.subprocess, "run", run)
    ffmpeg.probe_hw_mjpeg_decoder.cache_clear()
    try:
        assert ffmpeg.probe_hw_mjpeg_decoder() == "second"
        assert len(calls) == 2
    finally:
        ffmpeg.probe_hw_mjpeg_decoder.cache_clear()


def test_probe_hw_mjpeg_decoder_skips_unlisted_candidates(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(ffmpeg, "_hw_mjpeg_candidates", lambda: ("missing", "working"))
    monkeypatch.setattr(ffmpeg, "check_decoder_cli", lambda name: name == "working")
    monkeypatch.setattr(
        ffmpeg.subprocess,
        "run",
        lambda args, **_kwargs: subprocess.CompletedProcess(args, 0, b"", b""),
    )
    ffmpeg.probe_hw_mjpeg_decoder.cache_clear()
    try:
        assert ffmpeg.probe_hw_mjpeg_decoder() == "working"
    finally:
        ffmpeg.probe_hw_mjpeg_decoder.cache_clear()


def test_probe_hw_mjpeg_decoder_returns_none_when_binary_missing(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: None)
    ffmpeg.probe_hw_mjpeg_decoder.cache_clear()
    try:
        assert ffmpeg.probe_hw_mjpeg_decoder() is None
    finally:
        ffmpeg.probe_hw_mjpeg_decoder.cache_clear()


def test_frame_sync_args_falls_back_on_subprocess_error(monkeypatch) -> None:
    monkeypatch.setattr(
        ffmpeg.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("failed")),
    )
    ffmpeg._frame_sync_args.cache_clear()
    try:
        assert ffmpeg._frame_sync_args("ffmpeg") == ["-vsync", "0"]
    finally:
        ffmpeg._frame_sync_args.cache_clear()


def test_codec_helpers_and_output_arguments(monkeypatch) -> None:
    assert ffmpeg._codec_family("h264_nvenc") == "h264"
    assert ffmpeg._codec_family("hevc") == "h265"
    assert ffmpeg._codec_family("libvpx-vp9") == "vp9"
    assert ffmpeg._codec_family("libaom-av1") == "av1"
    assert ffmpeg._codec_family("unknown") == "h264"
    monkeypatch.setattr(ffmpeg, "_frame_sync_args", lambda _: [])

    args = ffmpeg._build_output_args("ffmpeg", "h264", "libx264", 12, {"crf": "28"}, 123)
    assert args[-7:] == ["-b:v", "123", "-crf", "28", "-f", "h264", "pipe:1"]


def test_require_ffmpeg(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: None)
    with pytest.raises(VideoEncoderError, match="ffmpeg not found"):
        ffmpeg._require_ffmpeg()
    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    assert ffmpeg._require_ffmpeg() == "/usr/bin/ffmpeg"


def test_image_dimension_and_ffprobe_helpers(monkeypatch) -> None:
    png = b"\x89PNG" + bytes(12) + (7).to_bytes(4, "big") + (5).to_bytes(4, "big")
    assert ffmpeg.probe_image_dimensions(png) == (7, 5)
    assert ffmpeg.parse_jpeg_dimensions(b"not jpeg") is None

    monkeypatch.setattr(ffmpeg.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        ffmpeg.subprocess,
        "run",
        lambda args, **_kwargs: subprocess.CompletedProcess(args, 0, b"11,9\n", b""),
    )
    assert ffmpeg.probe_image_dimensions(b"unknown") == (11, 9)

    assert ffmpeg.parse_jpeg_dimensions(b"\xff\xd8" + bytes(10)) is None
    assert ffmpeg.parse_jpeg_dimensions(b"\xff\xd8\xff\xd9" + bytes(8)) is None


def test_image_probe_errors_are_reported(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda _name: None)
    with pytest.raises(VideoEncoderError, match="ffprobe not found"):
        ffmpeg.probe_image_dimensions(b"unknown")

    monkeypatch.setattr(ffmpeg.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        ffmpeg.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("ffprobe", 10)),
    )
    with pytest.raises(VideoEncoderError, match="Cannot determine"):
        ffmpeg.probe_image_dimensions(b"unknown")


def test_probe_image_pipe_decode_handles_missing_binary_and_errors(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: None)
    assert ffmpeg.probe_image_pipe_decode(b"image") is False
    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        ffmpeg.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("failed")),
    )
    assert ffmpeg.probe_image_pipe_decode(b"image") is False

    monkeypatch.setattr(
        ffmpeg.subprocess,
        "run",
        lambda args, **_kwargs: subprocess.CompletedProcess(args, 0, b"", b""),
    )
    assert ffmpeg.probe_image_pipe_decode(b"image") is True


class _FakeThread:
    def __init__(self, target, daemon: bool) -> None:
        self.target = target
        self.daemon = daemon

    def start(self) -> None:
        pass

    def join(self, timeout: float) -> None:
        pass


class _FakeStdin:
    def __init__(self) -> None:
        self.closed = False
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FakePipe:
    def __init__(self, lines: list[bytes] | None = None) -> None:
        self.lines = lines or []

    def __iter__(self):
        return iter(self.lines)

    def read(self, _size: int = -1) -> bytes:
        return b""

    def fileno(self) -> int:
        return 42


class _FakeProcess:
    def __init__(self, *, returncode: int = 0) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakePipe()
        self.stderr = _FakePipe()
        self.returncode = returncode
        self.waited = False
        self.killed = False

    def wait(self, timeout: float) -> None:
        del timeout
        self.waited = True

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.killed = True


def test_ffmpeg_encoder_builds_raw_and_image_pipe_commands(monkeypatch) -> None:
    processes: list[_FakeProcess] = []
    commands: list[list[str]] = []

    def popen(command, **_kwargs):
        commands.append(command)
        process = _FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(ffmpeg.subprocess, "Popen", popen)
    monkeypatch.setattr(ffmpeg.threading, "Thread", _FakeThread)
    monkeypatch.setattr(ffmpeg, "_frame_sync_args", lambda _: [])

    raw = ffmpeg.FFmpegVideoEncoder(16, 8, "libx264", input_pix_fmt="rgb24", scale=(8, 4))
    image = ffmpeg.FFmpegVideoEncoder(16, 8, "libx264", decode_codec="mjpeg")
    try:
        assert "rawvideo" in commands[0]
        assert commands[0][commands[0].index("-vf") : commands[0].index("-vf") + 2] == [
            "-vf",
            "scale=8:4",
        ]
        assert "image2pipe" in commands[1]
        assert commands[1][commands[1].index("-c:v") + 1] == "mjpeg"
        assert raw._is_image_pipe is False
        assert image._is_image_pipe is True
    finally:
        raw.close()
        image.close()


def test_ffmpeg_encoder_reports_popen_error(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(ffmpeg, "_frame_sync_args", lambda _: [])
    monkeypatch.setattr(
        ffmpeg.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot start")),
    )

    with pytest.raises(VideoEncoderError, match="Failed to start ffmpeg"):
        ffmpeg.FFmpegVideoEncoder(16, 8, "libx264")


def test_ffmpeg_encoder_reads_stdout_and_stderr(monkeypatch) -> None:
    encoder = ffmpeg.FFmpegVideoEncoder.__new__(ffmpeg.FFmpegVideoEncoder)
    process = _FakeProcess()
    encoder._process = process
    encoder._output_queue = Queue()
    encoder._stderr_lines = []

    class Parser:
        def feed(self, data: bytes):
            assert data == b"chunk"
            return [b"au"]

        def flush_list(self):
            return [b"tail"]

    encoder._parser = Parser()
    reads = iter([b"chunk", b""])
    monkeypatch.setattr(ffmpeg.os, "read", lambda _fd, _size: next(reads))
    process.stdout = _FakePipe()
    process.stderr = _FakePipe([b"warning\n"])

    encoder._read_stdout()
    encoder._read_stderr()

    assert encoder._output_queue.get_nowait() == b"au"
    assert encoder._output_queue.get_nowait() == b"tail"
    assert encoder._output_queue.get_nowait() is None
    assert encoder._stderr_lines == ["warning"]


def test_ffmpeg_encoder_read_stdout_handles_missing_pipe_and_oserror(monkeypatch) -> None:
    encoder = ffmpeg.FFmpegVideoEncoder.__new__(ffmpeg.FFmpegVideoEncoder)
    encoder._process = SimpleNamespace(stdout=None)
    encoder._output_queue = Queue()
    encoder._read_stdout()

    process = _FakeProcess()
    encoder._process = process
    encoder._parser = SimpleNamespace(feed=lambda _data: [], flush_list=list)
    monkeypatch.setattr(
        ffmpeg.os,
        "read",
        lambda *_args: (_ for _ in ()).throw(OSError("closed")),
    )
    encoder._read_stdout()
    assert encoder._output_queue.get_nowait() is None


def test_ffmpeg_encoder_read_stderr_handles_missing_pipe() -> None:
    encoder = ffmpeg.FFmpegVideoEncoder.__new__(ffmpeg.FFmpegVideoEncoder)
    encoder._process = SimpleNamespace(stderr=None)
    encoder._stderr_lines = []
    encoder._read_stderr()
    assert encoder._stderr_lines == []


def test_ffmpeg_encoder_encode_and_flush_error_paths() -> None:
    encoder = ffmpeg.FFmpegVideoEncoder.__new__(ffmpeg.FFmpegVideoEncoder)
    process = _FakeProcess()
    encoder._process = process
    encoder._stderr_lines = ["last error"]
    encoder._output_queue = Queue()

    assert encoder.encode(b"frame") is None
    encoder._output_queue.put(b"au")
    assert encoder.encode(b"frame") == b"au"
    encoder._output_queue.put(None)
    with pytest.raises(VideoEncoderError, match="exited prematurely"):
        encoder.encode(b"frame")

    encoder._process.stdin = None
    with pytest.raises(VideoEncoderError, match="stdin is not available"):
        encoder.encode(b"frame")

    process = _FakeProcess(returncode=0)
    encoder._process = process
    encoder._output_queue = Queue()
    encoder._stdout_thread = _FakeThread(lambda: None, True)
    encoder._stderr_thread = _FakeThread(lambda: None, True)
    encoder._output_queue.put(b"tail")
    encoder._output_queue.put(None)
    assert encoder.flush_packets() == [b"tail"]

    process = _FakeProcess(returncode=1)
    encoder._process = process
    encoder._output_queue = Queue()
    encoder._stdout_thread = _FakeThread(lambda: None, True)
    encoder._stderr_thread = _FakeThread(lambda: None, True)
    with pytest.raises(VideoEncoderError, match="exited with code 1"):
        encoder.flush_packets()


def test_ffmpeg_encoder_close_is_idempotent() -> None:
    encoder = ffmpeg.FFmpegVideoEncoder.__new__(ffmpeg.FFmpegVideoEncoder)
    process = _FakeProcess(returncode=None)
    encoder._process = process

    encoder.close()
    encoder.close()
    assert process.killed is True


def test_ffmpeg_decompressor_reports_dead_process_pipe() -> None:
    class DeadStdin:
        closed = False

        def write(self, _data: bytes) -> None:
            raise BrokenPipeError("decoder exited")

        def flush(self) -> None:
            pass

    decompressor = ffmpeg.FFmpegVideoDecompressor()
    decompressor._process = SimpleNamespace(stdin=DeadStdin())
    decompressor._codec_family = "h264"

    with pytest.raises(VideoEncoderError, match="ffmpeg decoder process died"):
        decompressor.decompress(b"packet", "h264")


class _ChunkPipe:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)

    def read(self, _size: int = -1) -> bytes:
        return self.chunks.pop(0) if self.chunks else b""

    def __iter__(self):
        return iter(self.chunks)


def test_ffmpeg_decompressor_starts_compressed_and_raw_processes(monkeypatch) -> None:
    processes: list[_FakeProcess] = []
    commands: list[list[str]] = []

    def popen(command, **_kwargs):
        commands.append(command)
        process = _FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(ffmpeg.subprocess, "Popen", popen)
    monkeypatch.setattr(ffmpeg.threading, "Thread", _FakeThread)

    compressed = ffmpeg.FFmpegVideoDecompressor("compressed", jpeg_quality=80)
    compressed._start_process("h265")
    raw = ffmpeg.FFmpegVideoDecompressor("raw")
    raw._start_process("h264")

    assert commands[0][commands[0].index("-f") + 1] == "hevc"
    assert "image2pipe" in commands[0]
    assert "rawvideo" in commands[1]
    assert processes[0].stdin is not None


def test_ffmpeg_decompressor_reports_start_error(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        ffmpeg.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot start")),
    )

    with pytest.raises(VideoEncoderError, match="Failed to start ffmpeg decoder"):
        ffmpeg.FFmpegVideoDecompressor()._start_process("h264")

    monkeypatch.setattr(ffmpeg, "find_ffmpeg", lambda: None)
    with pytest.raises(VideoEncoderError, match="ffmpeg not found"):
        ffmpeg.FFmpegVideoDecompressor()._start_process("h264")


def test_ffmpeg_decompressor_dimension_probe_paths(monkeypatch) -> None:
    decoder = ffmpeg.FFmpegVideoDecompressor("raw")
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        ffmpeg.subprocess,
        "run",
        lambda args, **_kwargs: subprocess.CompletedProcess(args, 0, b"4,3\n", b""),
    )
    assert decoder._detect_dimensions(b"stream", "h264") == (4, 3)

    monkeypatch.setattr(ffmpeg.shutil, "which", lambda _name: None)
    with pytest.raises(VideoEncoderError, match="ffprobe not found"):
        decoder._detect_dimensions(b"stream", "h264")

    monkeypatch.setattr(ffmpeg.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        ffmpeg.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess("ffprobe", 0, b"bad", b""),
    )
    with pytest.raises(VideoEncoderError, match="Could not detect"):
        decoder._detect_dimensions(b"stream", "h264")

    monkeypatch.setattr(
        ffmpeg.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess("ffprobe", 0, b"a,b", b""),
    )
    with pytest.raises(VideoEncoderError, match="Could not detect"):
        decoder._detect_dimensions(b"stream", "h264")


def test_ffmpeg_decompressor_reads_jpeg_and_raw_streams() -> None:
    jpeg_decoder = ffmpeg.FFmpegVideoDecompressor("compressed")
    jpeg_decoder._output_queue = Queue()
    jpeg_process = _FakeProcess()
    jpeg_process.stdout = _ChunkPipe([PROBE_JPEG + PROBE_JPEG, b""])
    jpeg_decoder._process = jpeg_process
    jpeg_decoder._read_jpeg_stream()
    first = jpeg_decoder._output_queue.get_nowait()
    second = jpeg_decoder._output_queue.get_nowait()
    assert (first.width, first.height, first.is_jpeg) == (32, 32, True)
    assert (second.width, second.height, second.is_jpeg) == (32, 32, True)

    raw_decoder = ffmpeg.FFmpegVideoDecompressor("raw")
    raw_decoder._output_queue = Queue()
    raw_decoder._width = 2
    raw_decoder._height = 1
    raw_process = _FakeProcess()
    raw_process.stdout = _ChunkPipe([b"abc", b"def", b""])
    raw_decoder._process = raw_process
    raw_decoder._read_raw_stream()
    assert raw_decoder._output_queue.get_nowait() == DecompressedFrame(
        data=b"abcdef", width=2, height=1, is_jpeg=False
    )


def test_ffmpeg_decompressor_handles_missing_raw_dimensions_and_stderr(monkeypatch) -> None:
    decoder = ffmpeg.FFmpegVideoDecompressor("raw")
    decoder._output_queue = Queue()
    process = _FakeProcess()
    process.stdout = _ChunkPipe([b"discarded", b""])
    process.stderr = _ChunkPipe([b"error\n"])
    decoder._process = process

    decoder._read_raw_stream()
    decoder._read_stderr()
    assert decoder._stderr_lines == ["error"]

    decoder._process = SimpleNamespace(stdout=None, stderr=None)
    decoder._read_stdout()
    decoder._read_stderr()

    decoder = ffmpeg.FFmpegVideoDecompressor("compressed")
    decoder._output_queue = Queue()
    decoder._process = _FakeProcess()
    jpeg_called = []
    monkeypatch.setattr(decoder, "_read_jpeg_stream", lambda: jpeg_called.append(True))
    decoder._read_stdout()
    assert jpeg_called == [True]
    assert decoder._output_queue.get_nowait() is None

    raw_decoder = ffmpeg.FFmpegVideoDecompressor("raw")
    raw_decoder._output_queue = Queue()
    raw_decoder._process = _FakeProcess()
    raw_called = []
    monkeypatch.setattr(raw_decoder, "_read_raw_stream", lambda: raw_called.append(True))
    raw_decoder._read_stdout()
    assert raw_called == [True]
    assert raw_decoder._output_queue.get_nowait() is None


def test_ffmpeg_decompressor_decompresses_and_probes_raw_stream(monkeypatch) -> None:
    frame = DecompressedFrame(data=b"rgb", width=2, height=1, is_jpeg=False)
    decoder = ffmpeg.FFmpegVideoDecompressor("raw")
    started: list[str] = []

    def start(codec: str) -> None:
        started.append(codec)
        decoder._process = _FakeProcess()
        decoder._codec_family = "h264"
        decoder._output_queue.put(frame)

    monkeypatch.setattr(decoder, "_detect_dimensions", lambda _data, _codec: (2, 1))
    monkeypatch.setattr(decoder, "_start_process", start)

    assert decoder.decompress(b"packet", "h264") == frame
    assert started == ["h264"]
    assert decoder._width == 2
    assert decoder._height == 1

    decoder = ffmpeg.FFmpegVideoDecompressor("compressed")

    def start_compressed(codec: str) -> None:
        started.append(codec)
        decoder._process = _FakeProcess()
        decoder._codec_family = "h264"
        decoder._output_queue.put(frame)

    monkeypatch.setattr(decoder, "_start_process", start_compressed)
    assert decoder.decompress(b"packet", "h264") == frame


def test_ffmpeg_decompressor_waits_for_more_raw_data_and_reports_start_failure(monkeypatch) -> None:
    decoder = ffmpeg.FFmpegVideoDecompressor("raw")
    monkeypatch.setattr(
        decoder,
        "_detect_dimensions",
        lambda _data, _codec: (_ for _ in ()).throw(VideoEncoderError("not enough")),
    )
    assert decoder.decompress(b"partial", "h264") is None
    assert decoder._process is None

    decoder = ffmpeg.FFmpegVideoDecompressor("compressed")
    monkeypatch.setattr(decoder, "_start_process", lambda _codec: None)
    with pytest.raises(VideoEncoderError, match="process not started"):
        decoder.decompress(b"packet", "h264")

    decoder = ffmpeg.FFmpegVideoDecompressor()
    decoder._process = _FakeProcess()
    decoder._codec_family = "h264"
    decoder._output_queue = Queue()
    assert decoder.decompress(b"packet", "h264") is None


def test_ffmpeg_decompressor_read_helpers_ignore_missing_process() -> None:
    decoder = ffmpeg.FFmpegVideoDecompressor()
    decoder._read_jpeg_stream()
    decoder._read_raw_stream()

    decoder._process = _FakeProcess()
    decoder._process.stdout = _ChunkPipe([b"\xff\xd8partial", b""])
    decoder._read_jpeg_stream()


def test_ffmpeg_decompressor_returns_pending_frame_before_queue() -> None:
    decoder = ffmpeg.FFmpegVideoDecompressor()
    decoder._process = _FakeProcess()
    decoder._codec_family = "h264"
    frame = DecompressedFrame(data=b"pending", width=1, height=1, is_jpeg=True)
    decoder._pending_frames.append(frame)

    assert decoder.decompress(b"packet", "h264") == frame


def test_ffmpeg_decompressor_flush_returns_pending_and_resets() -> None:
    decoder = ffmpeg.FFmpegVideoDecompressor()
    pending = DecompressedFrame(data=b"pending", width=1, height=1, is_jpeg=True)
    decoder._pending_frames.append(pending)
    assert decoder.flush() == [pending]

    process = _FakeProcess()
    decoder._process = process
    decoder._output_queue.put(DecompressedFrame(data=b"frame", width=1, height=1, is_jpeg=True))
    decoder._output_queue.put(None)
    decoder._stdout_thread = _FakeThread(lambda: None, True)
    decoder._stderr_thread = _FakeThread(lambda: None, True)
    assert decoder.flush() == [DecompressedFrame(data=b"frame", width=1, height=1, is_jpeg=True)]
    assert decoder._process is None


def test_ffmpeg_decompressor_destructor_kills_running_process() -> None:
    decoder = ffmpeg.FFmpegVideoDecompressor()
    process = _FakeProcess(returncode=None)
    decoder._process = process

    decoder.__del__()

    assert process.killed is True
