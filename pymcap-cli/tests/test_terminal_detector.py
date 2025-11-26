"""Tests for terminal capability detection."""

from pymcap_cli.osc_utils import supports_osc_9_4


class TestTerminalDetection:
    """Test terminal capability detection."""

    def test_windows_terminal_detection(self, monkeypatch):
        """Test Windows Terminal detection via WT_SESSION."""
        monkeypatch.setenv("WT_SESSION", "12345-test-session")
        monkeypatch.delenv("ConEmuPID", raising=False)
        monkeypatch.delenv("ConEmuBuild", raising=False)
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)

        assert supports_osc_9_4() is True

    def test_conemu_detection_via_pid(self, monkeypatch):
        """Test ConEmu detection via ConEmuPID."""
        monkeypatch.delenv("WT_SESSION", raising=False)
        monkeypatch.setenv("ConEmuPID", "9876")
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)

        assert supports_osc_9_4() is True

    def test_conemu_detection_via_build(self, monkeypatch):
        """Test ConEmu detection via ConEmuBuild."""
        monkeypatch.delenv("WT_SESSION", raising=False)
        monkeypatch.delenv("ConEmuPID", raising=False)
        monkeypatch.setenv("ConEmuBuild", "200123")
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)

        assert supports_osc_9_4() is True

    def test_ghostty_detection(self, monkeypatch):
        """Test Ghostty detection via TERM_PROGRAM."""
        monkeypatch.delenv("WT_SESSION", raising=False)
        monkeypatch.delenv("ConEmuPID", raising=False)
        monkeypatch.delenv("ConEmuBuild", raising=False)
        monkeypatch.setenv("TERM_PROGRAM", "ghostty")
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)

        assert supports_osc_9_4() is True

    def test_fallback_terminal(self, monkeypatch):
        """Test fallback for unsupported terminals."""
        monkeypatch.delenv("WT_SESSION", raising=False)
        monkeypatch.delenv("ConEmuPID", raising=False)
        monkeypatch.delenv("ConEmuBuild", raising=False)
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)

        assert supports_osc_9_4() is False

    def test_non_tty_detection(self, monkeypatch):
        """Test detection when stdout is not a TTY."""
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)

        assert supports_osc_9_4() is False

    def test_priority_windows_terminal_over_conemu(self, monkeypatch):
        """Test that Windows Terminal detection takes priority."""
        monkeypatch.setenv("WT_SESSION", "test")
        monkeypatch.setenv("ConEmuPID", "1234")
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)

        # Windows Terminal should be detected (appears first in checks)
        assert supports_osc_9_4() is True

    def test_other_term_program_not_detected(self, monkeypatch):
        """Test that other TERM_PROGRAM values don't trigger Ghostty detection."""
        monkeypatch.delenv("WT_SESSION", raising=False)
        monkeypatch.delenv("ConEmuPID", raising=False)
        monkeypatch.delenv("ConEmuBuild", raising=False)
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)

        # Should not support OSC 9;4 (not a recognized terminal)
        assert supports_osc_9_4() is False
