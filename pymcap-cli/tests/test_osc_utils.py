"""Tests for cross-platform OSC utilities."""

import builtins
import os
from pathlib import Path

from pymcap_cli.osc_utils import Osc94States, set_progress, set_window_title, write_osc


class TestWriteOsc:
    """Test write_osc cross-platform writer."""

    def test_ctermid_path(self, monkeypatch):
        """Test writing via os.ctermid() on POSIX."""
        written_data = []

        # Mock os.ctermid to return a test path
        monkeypatch.setattr("os.ctermid", lambda: "/dev/pts/0")

        original_open = os.open
        original_write = os.write
        original_close = os.close

        def mock_os_open(path, flags):
            if path == "/dev/pts/0":
                return 888  # Fake fd
            return original_open(path, flags)

        def mock_os_write(fd, data):
            if fd == 888:
                written_data.append(data.decode("utf-8"))
                return len(data)
            return original_write(fd, data)

        def mock_os_close(fd):
            if fd == 888:
                return None
            return original_close(fd)

        monkeypatch.setattr("os.open", mock_os_open)
        monkeypatch.setattr("os.write", mock_os_write)
        monkeypatch.setattr("os.close", mock_os_close)

        write_osc("\x1b]9;4;1;50\x1b\\")

        assert len(written_data) == 1
        assert "\x1b]9;4;1;50\x1b\\" in written_data[0]

    def test_dev_tty_path(self, monkeypatch):
        """Test writing via /dev/tty on POSIX."""
        written_data = []

        # Remove os.ctermid to test /dev/tty fallback
        if hasattr(os, "ctermid"):
            monkeypatch.delattr("os.ctermid")

        # Mock Path.exists for /dev/tty
        original_exists = Path.exists

        def mock_exists(self):
            if str(self) == "/dev/tty":
                return True
            return original_exists(self)

        monkeypatch.setattr(Path, "exists", mock_exists)

        original_open = os.open
        original_write = os.write
        original_close = os.close

        def mock_os_open(path, flags):
            path_str = str(path) if hasattr(path, "__fspath__") else path
            if path_str == "/dev/tty":
                return 999  # Fake fd
            return original_open(path, flags)

        def mock_os_write(fd, data):
            if fd == 999:
                written_data.append(data.decode("utf-8"))
                return len(data)
            return original_write(fd, data)

        def mock_os_close(fd):
            if fd == 999:
                return None
            return original_close(fd)

        monkeypatch.setattr("os.open", mock_os_open)
        monkeypatch.setattr("os.write", mock_os_write)
        monkeypatch.setattr("os.close", mock_os_close)

        write_osc("\x1b]9;4;1;50\x1b\\")

        assert len(written_data) == 1
        assert "\x1b]9;4;1;50\x1b\\" in written_data[0]

    def test_windows_conout(self, monkeypatch):
        """Test writing via CONOUT$ on Windows."""
        written_data = []

        # Mock os.name to be Windows
        monkeypatch.setattr("os.name", "nt")

        # Remove os.ctermid (doesn't exist on Windows)
        if hasattr(os, "ctermid"):
            monkeypatch.delattr("os.ctermid")

        # Mock Path.exists to return False for /dev/tty
        original_exists = Path.exists

        def mock_exists(self):
            if str(self) == "/dev/tty":
                return False
            return original_exists(self)

        monkeypatch.setattr(Path, "exists", mock_exists)

        # Mock builtin open for CONOUT$
        original_open_builtin = builtins.open

        def mock_open_builtin(file, mode="r", *args, **kwargs):
            if file == "CONOUT$" and "b" in mode:

                class MockFile:
                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        pass

                    def write(self, data):
                        written_data.append(data.decode("utf-8"))
                        return len(data)

                return MockFile()
            return original_open_builtin(file, mode, *args, **kwargs)

        monkeypatch.setattr("builtins.open", mock_open_builtin)

        write_osc("\x1b]9;4;1;50\x1b\\")

        assert len(written_data) == 1
        assert "\x1b]9;4;1;50\x1b\\" in written_data[0]

    def test_stdout_fallback(self, monkeypatch):
        """Test fallback to stdout when file-based methods fail."""
        written_data = []

        # Remove os.ctermid
        if hasattr(os, "ctermid"):
            monkeypatch.delattr("os.ctermid")

        # Mock Path.exists to return False
        monkeypatch.setattr(Path, "exists", lambda _: False)

        # Mock os.name to not be Windows
        monkeypatch.setattr("os.name", "posix")

        # Mock stdout.isatty() to return True
        class MockStdout:
            def isatty(self):
                return True

            def write(self, data):
                written_data.append(data)

            def flush(self):
                pass

        monkeypatch.setattr("sys.stdout", MockStdout())

        write_osc("\x1b]9;4;1;50\x1b\\")

        assert len(written_data) == 1
        assert "\x1b]9;4;1;50\x1b\\" in written_data[0]

    def test_stderr_fallback(self, monkeypatch):
        """Test fallback to stderr when stdout is not a TTY."""
        written_data = []

        # Remove os.ctermid
        if hasattr(os, "ctermid"):
            monkeypatch.delattr("os.ctermid")

        # Mock Path.exists to return False
        monkeypatch.setattr(Path, "exists", lambda _: False)

        # Mock os.name to not be Windows
        monkeypatch.setattr("os.name", "posix")

        # Mock stdout.isatty() to return False
        class MockStdoutNotTTY:
            def isatty(self):
                return False

        # Mock stderr.isatty() to return True
        class MockStderr:
            def isatty(self):
                return True

            def write(self, data):
                written_data.append(data)

            def flush(self):
                pass

        monkeypatch.setattr("sys.stdout", MockStdoutNotTTY())
        monkeypatch.setattr("sys.stderr", MockStderr())

        write_osc("\x1b]9;4;1;50\x1b\\")

        assert len(written_data) == 1
        assert "\x1b]9;4;1;50\x1b\\" in written_data[0]

    def test_graceful_failure(self, monkeypatch):
        """Test graceful failure when all write methods fail."""
        # Remove os.ctermid
        if hasattr(os, "ctermid"):
            monkeypatch.delattr("os.ctermid")

        # Mock Path.exists to return False
        monkeypatch.setattr(Path, "exists", lambda _: False)

        # Mock os.name to not be Windows
        monkeypatch.setattr("os.name", "posix")

        # Mock stdout/stderr.isatty() to return False
        class MockStream:
            def isatty(self):
                return False

        monkeypatch.setattr("sys.stdout", MockStream())
        monkeypatch.setattr("sys.stderr", MockStream())

        # Should not raise an exception
        write_osc("\x1b]9;4;1;50\x1b\\")


class TestSetWindowTitle:
    """Test set_window_title helper."""

    def test_set_window_title(self, monkeypatch):
        """Test setting window title."""
        written_sequences = []

        def mock_write(sequence: str) -> None:
            written_sequences.append(sequence)

        monkeypatch.setattr("pymcap_cli.osc_utils.write_osc", mock_write)

        set_window_title("Processing data.mcap")

        assert len(written_sequences) == 1
        assert "Processing data.mcap" in written_sequences[0]
        assert written_sequences[0].startswith("\x1b]0;")
        assert written_sequences[0].endswith("\x1b\\")

    def test_set_window_title_error_handling(self, monkeypatch):
        """Test error handling in set_window_title."""

        def mock_write_error(_sequence: str) -> None:
            raise OSError("Write failed")

        monkeypatch.setattr("pymcap_cli.osc_utils.write_osc", mock_write_error)

        # Should not raise an exception
        set_window_title("Test")


class TestSetProgress:
    """Test set_progress helper."""

    def test_set_progress_states(self, monkeypatch):
        """Test setting various progress states."""
        written_sequences = []

        def mock_write(sequence: str) -> None:
            written_sequences.append(sequence)

        monkeypatch.setattr("pymcap_cli.osc_utils.write_osc", mock_write)

        # Test progress state (1)
        set_progress(Osc94States.PROGRESS, 50)
        assert "\x1b]9;4;1;50\x1b\\" in written_sequences

        # Test reset state (0)
        set_progress(Osc94States.RESET, 0)
        assert "\x1b]9;4;0;0\x1b\\" in written_sequences

        # Test error state (2)
        set_progress(Osc94States.ERROR, 0)
        assert "\x1b]9;4;2;0\x1b\\" in written_sequences

        # Test indeterminate state (3)
        set_progress(Osc94States.INDETERMINATE, 0)
        assert "\x1b]9;4;3;0\x1b\\" in written_sequences

    def test_set_progress_error_handling(self, monkeypatch):
        """Test error handling in set_progress."""

        def mock_write_error(_sequence: str) -> None:
            raise OSError("Write failed")

        monkeypatch.setattr("pymcap_cli.osc_utils.write_osc", mock_write_error)

        # Should not raise an exception
        set_progress(Osc94States.PROGRESS, 50)
