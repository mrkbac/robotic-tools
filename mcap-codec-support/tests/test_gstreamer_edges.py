"""Hermetic tests for the optional Jetson GStreamer backend."""

from queue import Queue
from types import SimpleNamespace

import mcap_codec_support.video.gstreamer as gst
import pytest
from mcap_codec_support.video.common import VideoEncoderError


class _Thread:
    def __init__(self, target, daemon: bool) -> None:
        self.target = target
        self.daemon = daemon

    def start(self) -> None:
        pass

    def join(self, timeout: float) -> None:
        del timeout


class _Stdin:
    def __init__(self) -> None:
        self.closed = False
        self.writes: list[bytes] = []
        self.flushes = 0

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.closed = True


class _Process:
    def __init__(self, returncode: int | None = 0) -> None:
        self.stdin = _Stdin()
        self.stdout = object()
        self.stderr = []
        self.returncode = returncode
        self.killed = False
        self.waited = False

    def wait(self, timeout: float) -> None:
        del timeout
        self.waited = True

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.killed = True


def test_gst_element_available_handles_missing_tool_and_errors(monkeypatch) -> None:
    monkeypatch.setattr(gst, "shutil", SimpleNamespace(which=lambda _: None))
    gst.gst_element_available.cache_clear()
    assert gst.gst_element_available("nvjpegdec") is False

    class Shutil:
        @staticmethod
        def which(_name):
            return "/usr/bin/gst-inspect-1.0"

    monkeypatch.setattr(gst, "shutil", Shutil)
    monkeypatch.setattr(
        gst.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("failed")),
    )
    gst.gst_element_available.cache_clear()
    try:
        assert gst.gst_element_available("nvjpegdec") is False
    finally:
        gst.gst_element_available.cache_clear()


def test_gst_element_available_checks_return_code(monkeypatch) -> None:
    class Shutil:
        @staticmethod
        def which(_name):
            return "/usr/bin/gst-inspect-1.0"

    monkeypatch.setattr(gst, "shutil", Shutil)
    monkeypatch.setattr(
        gst.subprocess,
        "run",
        lambda args, **_kwargs: SimpleNamespace(returncode=0 if args[-1] == "good" else 1),
    )
    gst.gst_element_available.cache_clear()
    try:
        assert gst.gst_element_available("good") is True
        assert gst.gst_element_available("bad") is False
    finally:
        gst.gst_element_available.cache_clear()


@pytest.mark.parametrize(
    ("codec", "expected"),
    [
        ("h264", "h264"),
        ("nvv4l2h264enc", "h264"),
        ("h265", "h265"),
        ("hevc", "h265"),
        ("vp9", "vp9"),
    ],
)
def test_codec_key(codec: str, expected: str) -> None:
    assert gst._codec_key(codec) == expected


def test_resolve_encoder_and_check_encoder(monkeypatch) -> None:
    monkeypatch.setattr(gst, "find_gst_launch", lambda: None)
    with pytest.raises(VideoEncoderError, match="gst-launch"):
        gst.resolve_encoder("h264")

    monkeypatch.setattr(gst, "find_gst_launch", lambda: "/usr/bin/gst-launch-1.0")
    monkeypatch.setattr(gst, "gst_element_available", lambda name: name == "nvv4l2h264enc")
    assert gst.resolve_encoder("h264") == "nvv4l2h264enc"
    with pytest.raises(VideoEncoderError, match="does not support"):
        gst.resolve_encoder("vp9")
    with pytest.raises(VideoEncoderError, match="not available"):
        gst.resolve_encoder("h265")
    assert gst.check_encoder("nvv4l2h264enc") is True
    assert gst.check_encoder("libx264") is False


def test_nvjpeg_library_dirs_handles_missing_distribution(monkeypatch) -> None:
    monkeypatch.setattr(
        gst, "distribution", lambda _: (_ for _ in ()).throw(gst.PackageNotFoundError())
    )
    monkeypatch.setattr(gst.Path, "glob", lambda *_args: ())
    gst._nvjpeg_library_dirs.cache_clear()
    try:
        assert gst._nvjpeg_library_dirs() == ()
    finally:
        gst._nvjpeg_library_dirs.cache_clear()


def test_nvjpeg_library_dirs_scans_system_candidates(monkeypatch) -> None:
    class Candidate:
        def glob(self, pattern: str):
            if pattern == "libnvjpeg.so*":
                return ["libnvjpeg.so"]
            raise OSError("broken glob")

        def __str__(self) -> str:
            return "/usr/local/cuda/lib64"

    class Root:
        def glob(self, pattern: str):
            return [Candidate()] if pattern == "cuda*/targets/*/lib" else []

    monkeypatch.setattr(
        gst, "distribution", lambda _: (_ for _ in ()).throw(gst.PackageNotFoundError())
    )
    monkeypatch.setattr(gst, "Path", lambda _path: Root())
    gst._nvjpeg_library_dirs.cache_clear()
    try:
        assert gst._nvjpeg_library_dirs() == ("/usr/local/cuda/lib64",)
    finally:
        gst._nvjpeg_library_dirs.cache_clear()


def test_nvjpeg_library_dirs_ignores_broken_system_candidate(monkeypatch) -> None:
    class BrokenCandidate:
        def glob(self, _pattern: str):
            raise OSError("permission denied")

    class Root:
        def glob(self, _pattern: str):
            return [BrokenCandidate()]

    monkeypatch.setattr(
        gst, "distribution", lambda _: (_ for _ in ()).throw(gst.PackageNotFoundError())
    )
    monkeypatch.setattr(gst, "Path", lambda _path: Root())
    gst._nvjpeg_library_dirs.cache_clear()
    try:
        assert gst._nvjpeg_library_dirs() == ()
    finally:
        gst._nvjpeg_library_dirs.cache_clear()


def test_gstreamer_env_returns_none_without_extra_libraries(monkeypatch) -> None:
    monkeypatch.setattr(gst, "_nvjpeg_library_dirs", lambda: ())
    assert gst._gstreamer_env() is None


def _patch_encoder_process(monkeypatch, *, process: _Process | None = None):
    process = process or _Process()
    commands: list[list[str]] = []
    closed: list[int] = []

    monkeypatch.setattr(gst, "find_gst_launch", lambda: "/usr/bin/gst-launch-1.0")
    monkeypatch.setattr(gst.os, "pipe", lambda: (10000, 10001))
    monkeypatch.setattr(gst.os, "set_inheritable", lambda *_args: None)
    monkeypatch.setattr(gst.os, "close", closed.append)
    monkeypatch.setattr(
        gst.subprocess, "Popen", lambda command, **_kwargs: commands.append(command) or process
    )
    monkeypatch.setattr(gst.threading, "Thread", _Thread)
    monkeypatch.setattr(gst, "_gstreamer_env", lambda: {"PATH": "test"})
    return process, commands, closed


def test_gstreamer_encoder_builds_jpeg_and_raw_pipelines(monkeypatch) -> None:
    _process, commands, closed = _patch_encoder_process(monkeypatch)
    jpeg = gst.GStreamerVideoEncoder(5, 3, "nvv4l2h264enc", scale=(5, 3))
    assert jpeg.config.width == 4
    assert jpeg.config.height == 2
    assert "nvjpegdec" in commands[0]
    assert any("width=4" in item for item in commands[0])
    jpeg.close()

    _process, commands, closed = _patch_encoder_process(monkeypatch, process=_Process())
    raw = gst.GStreamerVideoEncoder(8, 6, "nvv4l2h265enc", input_pix_fmt="rgb24")
    assert "rawvideoparse" in commands[0]
    assert "format=rgb" in commands[0]
    assert raw._codec_family == "h265"
    raw.close()
    assert closed


def test_gstreamer_encoder_rejects_missing_or_invalid_configuration(monkeypatch) -> None:
    monkeypatch.setattr(gst, "find_gst_launch", lambda: None)
    with pytest.raises(VideoEncoderError, match="gst-launch"):
        gst.GStreamerVideoEncoder(4, 4, "nvv4l2h264enc")

    monkeypatch.setattr(gst, "find_gst_launch", lambda: "/usr/bin/gst-launch-1.0")
    with pytest.raises(VideoEncoderError, match="cannot use encoder"):
        gst.GStreamerVideoEncoder(4, 4, "libx264")

    _patch_encoder_process(monkeypatch)
    with pytest.raises(VideoEncoderError, match="cannot feed raw"):
        gst.GStreamerVideoEncoder(4, 4, "nvv4l2h264enc", input_pix_fmt="yuv420p")


def test_gstreamer_encoder_reports_process_start_error(monkeypatch) -> None:
    monkeypatch.setattr(gst, "find_gst_launch", lambda: "/usr/bin/gst-launch-1.0")
    monkeypatch.setattr(gst.os, "pipe", lambda: (10000, 10001))
    monkeypatch.setattr(gst.os, "set_inheritable", lambda *_args: None)
    monkeypatch.setattr(gst.os, "close", lambda _fd: None)
    monkeypatch.setattr(
        gst.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot start")),
    )

    with pytest.raises(VideoEncoderError, match="Failed to start gst-launch"):
        gst.GStreamerVideoEncoder(4, 4, "nvv4l2h264enc")


def test_gstreamer_encoder_reads_output_and_stderr(monkeypatch) -> None:
    encoder = gst.GStreamerVideoEncoder.__new__(gst.GStreamerVideoEncoder)
    encoder._read_fd = 10000
    encoder._output_queue = Queue()
    encoder._stderr_lines = []
    encoder._process = SimpleNamespace(stderr=[b"warning\n"])

    class Parser:
        def feed(self, data: bytes):
            assert data == b"chunk"
            return [b"au"]

        def flush_list(self):
            return [b"tail"]

    encoder._parser = Parser()
    reads = iter([b"chunk", b""])
    monkeypatch.setattr(gst.os, "read", lambda *_args: next(reads))
    encoder._read_output()
    encoder._read_stderr()
    assert encoder._output_queue.get_nowait() == b"au"
    assert encoder._output_queue.get_nowait() == b"tail"
    assert encoder._output_queue.get_nowait() is None
    assert encoder._stderr_lines == ["warning"]


def test_gstreamer_encoder_read_output_handles_oserror_and_missing_stderr(monkeypatch) -> None:
    encoder = gst.GStreamerVideoEncoder.__new__(gst.GStreamerVideoEncoder)
    encoder._read_fd = 10000
    encoder._output_queue = Queue()
    encoder._parser = SimpleNamespace(feed=lambda _data: [], flush_list=list)
    encoder._process = SimpleNamespace(stderr=None)
    monkeypatch.setattr(gst.os, "read", lambda *_args: (_ for _ in ()).throw(OSError("closed")))
    encoder._read_output()
    encoder._read_stderr()
    assert encoder._output_queue.get_nowait() is None

    missing_fd = gst.GStreamerVideoEncoder.__new__(gst.GStreamerVideoEncoder)
    missing_fd._read_fd = None
    missing_fd._output_queue = Queue()
    missing_fd._read_output()
    assert missing_fd._output_queue.get_nowait() is None


def test_gstreamer_encoder_encode_and_flush_lifecycle(monkeypatch) -> None:
    encoder = gst.GStreamerVideoEncoder.__new__(gst.GStreamerVideoEncoder)
    process = _Process()
    encoder._process = process
    encoder._stderr_lines = []
    encoder._output_queue = Queue()
    encoder._stdout_thread = _Thread(lambda: None, True)
    encoder._stderr_thread = _Thread(lambda: None, True)
    encoder._read_fd = 10000

    assert encoder.encode(b"small") is None
    assert process.stdin.flushes == 1
    encoder._output_queue.put(b"au")
    assert encoder.encode(b"small") == b"au"
    encoder._output_queue.put(None)
    with pytest.raises(VideoEncoderError, match="exited prematurely"):
        encoder.encode(b"small")

    process = _Process()
    encoder._process = process
    encoder._output_queue = Queue()
    encoder._output_queue.put(b"tail")
    encoder._output_queue.put(None)
    closed: list[int] = []
    monkeypatch.setattr(gst.os, "close", closed.append)
    assert encoder.flush_packets() == [b"tail"]
    assert process.waited is True
    assert closed == [10000]

    process = _Process(returncode=1)
    encoder._process = process
    encoder._output_queue = Queue()
    with pytest.raises(VideoEncoderError, match="exited with code 1"):
        encoder.flush_packets()


def test_gstreamer_encoder_flush_handles_timeout_and_close(monkeypatch) -> None:
    process = _Process(returncode=0)

    def timeout(*, timeout: float) -> None:
        del timeout
        raise gst.subprocess.TimeoutExpired("gst-launch", 10)

    process.wait = timeout
    encoder = gst.GStreamerVideoEncoder.__new__(gst.GStreamerVideoEncoder)
    encoder._process = process
    encoder._stderr_lines = []
    encoder._output_queue = Queue()
    encoder._stdout_thread = _Thread(lambda: None, True)
    encoder._stderr_thread = _Thread(lambda: None, True)
    encoder._read_fd = 10000
    killed = []
    encoder.close = lambda: killed.append(True)
    monkeypatch.setattr(gst.os, "close", lambda _fd: None)

    assert encoder.flush_packets() == []
    assert killed == [True]

    process = _Process(returncode=None)
    encoder._process = process
    encoder.close = gst.GStreamerVideoEncoder.close.__get__(encoder)
    encoder.close()
    encoder.close()
    assert process.killed is True

    unavailable = gst.GStreamerVideoEncoder.__new__(gst.GStreamerVideoEncoder)
    unavailable._process = None
    with pytest.raises(VideoEncoderError, match="process is not available"):
        unavailable.flush_packets()


def test_gstreamer_encoder_reports_missing_or_dead_stdin() -> None:
    encoder = gst.GStreamerVideoEncoder.__new__(gst.GStreamerVideoEncoder)
    encoder._process = SimpleNamespace(stdin=None)
    with pytest.raises(VideoEncoderError, match="stdin is not available"):
        encoder.encode(b"frame")

    class DeadStdin:
        closed = True

        def write(self, _data: bytes) -> None:
            raise BrokenPipeError("dead")

        def flush(self) -> None:
            pass

    encoder._process = SimpleNamespace(stdin=DeadStdin())
    encoder._stderr_lines = []
    with pytest.raises(VideoEncoderError, match="gst-launch died"):
        encoder.encode(b"frame")


def test_gstreamer_encoder_close_is_safe_during_partial_initialization() -> None:
    encoder = gst.GStreamerVideoEncoder.__new__(gst.GStreamerVideoEncoder)

    encoder.close()


def test_gstreamer_encoder_reads_large_frames_without_flush() -> None:
    encoder = gst.GStreamerVideoEncoder.__new__(gst.GStreamerVideoEncoder)
    process = _Process()
    encoder._process = process
    encoder._stderr_lines = []
    encoder._output_queue = Queue()

    assert encoder.encode(bytes(64 * 1024)) is None
    assert process.stdin.flushes == 0


def test_probe_hw_jpeg_pipeline_falls_back_and_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(gst, "resolve_encoder", lambda _codec: "nvv4l2h264enc")
    monkeypatch.setattr(gst, "probe_image_dimensions", lambda _data: (256, 256))
    monkeypatch.setattr(gst.time, "monotonic", lambda: 0.0)

    class Encoder:
        def __init__(self, *_args, **_kwargs) -> None:
            self._output_queue = Queue()
            self.closed = False

        def encode(self, _data: bytes):
            return b"au"

        def close(self):
            self.closed = True

    monkeypatch.setattr(gst, "GStreamerVideoEncoder", Encoder)
    assert gst.probe_hw_jpeg_pipeline("h264") is True

    class QueuedEncoder(Encoder):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__(*_args, **_kwargs)
            self.calls = 0

        def encode(self, _data: bytes):
            self.calls += 1
            if self.calls == 4:
                self._output_queue.put(b"queued au")

    monkeypatch.setattr(gst, "GStreamerVideoEncoder", QueuedEncoder)
    monkeypatch.setattr(gst.time, "monotonic", iter([0.0, 1.0]).__next__)
    assert gst.probe_hw_jpeg_pipeline("h264", timeout=2.0) is True

    monkeypatch.setattr(
        gst, "resolve_encoder", lambda _codec: (_ for _ in ()).throw(VideoEncoderError("bad"))
    )
    assert gst.probe_hw_jpeg_pipeline("h264") is False


def test_probe_hw_jpeg_pipeline_handles_encoder_errors(monkeypatch) -> None:
    monkeypatch.setattr(gst, "resolve_encoder", lambda _codec: "nvv4l2h264enc")
    monkeypatch.setattr(gst, "probe_image_dimensions", lambda _data: (256, 256))
    monkeypatch.setattr(
        gst,
        "GStreamerVideoEncoder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bad")),
    )
    assert gst.probe_hw_jpeg_pipeline("h264") is False


def test_probe_hw_jpeg_pipeline_times_out_without_output(monkeypatch) -> None:
    monkeypatch.setattr(gst, "resolve_encoder", lambda _codec: "nvv4l2h264enc")
    monkeypatch.setattr(gst, "probe_image_dimensions", lambda _data: (256, 256))

    class Encoder:
        def __init__(self, *_args, **_kwargs) -> None:
            self._output_queue = Queue()

        def encode(self, _data: bytes):
            return None

        def close(self):
            pass

    monkeypatch.setattr(gst, "GStreamerVideoEncoder", Encoder)
    monkeypatch.setattr(gst.time, "monotonic", iter([0.0, 0.0, 2.0]).__next__)
    assert gst.probe_hw_jpeg_pipeline("h264", timeout=1.0) is False


def test_probe_hw_jpeg_pipeline_stops_on_output_sentinel(monkeypatch) -> None:
    monkeypatch.setattr(gst, "resolve_encoder", lambda _codec: "nvv4l2h264enc")
    monkeypatch.setattr(gst, "probe_image_dimensions", lambda _data: (256, 256))

    class Encoder:
        def __init__(self, *_args, **_kwargs) -> None:
            self._output_queue = Queue()
            self.calls = 0

        def encode(self, _data: bytes):
            self.calls += 1
            if self.calls == 4:
                self._output_queue.put(None)

        def close(self):
            pass

    monkeypatch.setattr(gst, "GStreamerVideoEncoder", Encoder)
    monkeypatch.setattr(gst.time, "monotonic", iter([0.0, 0.0]).__next__)
    assert gst.probe_hw_jpeg_pipeline("h264", timeout=1.0) is False


def test_compression_backend_delegates_and_falls_back(monkeypatch) -> None:
    backend = gst.GStreamerCompressionBackend()
    monkeypatch.setattr(gst, "check_encoder", lambda name: name == "good")
    monkeypatch.setattr(gst, "resolve_encoder", lambda _codec: "resolved")
    monkeypatch.setattr(gst, "probe_image_dimensions", lambda _data: (4, 3))
    assert backend.test_encoder("good") is True
    assert backend.resolve_encoder("h264") == "resolved"
    assert backend.decode_compressed(b"image") == (b"image", 4, 3)

    compressed = SimpleNamespace(
        decoded_message=SimpleNamespace(data=b"image"), channel=SimpleNamespace(topic="/camera")
    )
    assert backend.decode_image(compressed, "sensor_msgs/CompressedImage") == (b"image", 4, 3)
    assert backend.get_pix_fmt("/camera") is None

    raw = SimpleNamespace(
        decoded_message=SimpleNamespace(data=b"abcdef", width=2, height=1, encoding="rgb8", step=6),
        channel=SimpleNamespace(topic="/raw"),
    )
    assert backend.decode_image(raw, "sensor_msgs/Image") == (b"abcdef", 2, 1)
    assert backend.get_pix_fmt("/raw") == "rgb24"

    monkeypatch.setattr(
        gst,
        "raw_image_to_buffer",
        lambda _message: SimpleNamespace(
            encoding="unknown", step=1, row_bytes=1, height=1, width=1, data=b"x"
        ),
    )
    with pytest.raises(VideoEncoderError, match="Unsupported image encoding"):
        backend.decode_image(raw, "sensor_msgs/Image")

    software = gst.GStreamerCompressionBackend()
    monkeypatch.setattr(gst, "FFmpegVideoEncoder", lambda **kwargs: ("ffmpeg", kwargs))
    assert software.create_encoder(2, 2, "libx264", 28)[0] == "ffmpeg"
    monkeypatch.setattr(gst, "GStreamerVideoEncoder", lambda **kwargs: ("gst", kwargs))
    assert software.create_encoder(2, 2, "nvv4l2h264enc", 28)[0] == "gst"


def test_compression_backend_rejects_ffmpeg_arguments() -> None:
    backend = gst.GStreamerCompressionBackend()

    with pytest.raises(VideoEncoderError, match="require the ffmpeg-cli backend"):
        backend.create_encoder(4, 4, "nvv4l2h264enc", 28, extra_args=("-preset", "slow"))
