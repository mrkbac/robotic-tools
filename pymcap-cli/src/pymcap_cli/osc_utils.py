"""Cross-platform OSC (Operating System Command) escape sequence utilities.

This module provides utilities for writing OSC escape sequences to the terminal
in a cross-platform manner, supporting Unix-like systems (macOS, Linux) and Windows.
"""

import contextlib
import os
import sys
from enum import Enum
from pathlib import Path


def supports_osc_9_4() -> bool:
    """Detect terminal capabilities for progress display.

    Checks for supported terminals:
    - Windows Terminal: WT_SESSION env var
    - ConEmu: ConEmuPID or ConEmuBuild env vars
    - Ghostty: TERM_PROGRAM=ghostty
    """
    # Check if stdout is a TTY first
    if not sys.stdout.isatty():
        return False

    # Windows Terminal detection
    if os.environ.get("WT_SESSION"):
        return True

    # ConEmu detection
    # ruff: noqa: SIM112  # ConEmu uses mixed case env vars
    if os.environ.get("ConEmuPID") or os.environ.get("ConEmuBuild"):
        return True

    # Ghostty detection
    return os.environ.get("TERM_PROGRAM") == "ghostty"


def write_osc(sequence: str) -> None:
    """Write OSC sequence to the best available terminal device, cross-platform.

    Try, in order:
    1. Controlling terminal on POSIX (os.ctermid).
    2. /dev/tty on POSIX.
    3. CONOUT$ on Windows.
    4. Fallback to stdout / stderr if they are TTYs.

    Args:
        sequence: The OSC escape sequence to write (already formatted with ESC codes)

    Example:
        >>> write_osc("\\x1b]0;My Window Title\\x1b\\\\")  # Set window title
        >>> write_osc("\\x1b]9;4;1;50\\x1b\\\\")  # Progress: 50%
    """
    data = sequence.encode("utf-8", "replace")

    # 1) POSIX controlling terminal via ctermid
    if hasattr(os, "ctermid"):
        try:
            ctermid_path = os.ctermid()
            if ctermid_path:
                fd = os.open(ctermid_path, os.O_WRONLY | getattr(os, "O_CLOEXEC", 0))
                try:
                    os.write(fd, data)
                finally:
                    os.close(fd)
        except OSError:
            pass
        else:
            return

    # 2) /dev/tty on POSIX-like systems
    tty_path = Path("/dev/tty")
    if tty_path.exists():
        try:
            fd = os.open(str(tty_path), os.O_WRONLY | getattr(os, "O_CLOEXEC", 0))
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
        except OSError:
            pass
        else:
            return

    # 3) Windows console device
    if os.name == "nt":
        try:
            # CONOUT$ is a device, not a file - cannot use Path.open()
            with open("CONOUT$", "wb", buffering=0) as f:  # noqa: PTH123
                f.write(data)
        except OSError:
            pass
        else:
            return

    # 4) Fallback to stdout / stderr if they are TTYs
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "isatty") and stream.isatty():
                stream.write(sequence)
                stream.flush()
                return
        except (OSError, AttributeError):  # Narrow exception scope
            pass


def set_window_title(title: str) -> None:
    """Set terminal window title using OSC 0 or OSC 2.

    Args:
        title: The window title to set

    Example:
        >>> set_window_title("Processing data.mcap")
    """
    # OSC 0 sets both icon and window title; OSC 2 sets just window title
    sequence = f"\x1b]0;{title}\x1b\\"
    # Silent failure - OSC is optional enhancement
    with contextlib.suppress(Exception):
        write_osc(sequence)


class Osc94States(int, Enum):
    UNKNOWN = -1
    RESET = 0
    PROGRESS = 1
    ERROR = 2
    INDETERMINATE = 3


def set_progress(state: Osc94States, progress: int) -> None:
    """Set terminal progress indicator using OSC 9;4.

    Supported by Windows Terminal, ConEmu, and Ghostty.

    Args:
        state: Progress state (0=reset, 1=progress, 2=error, 3=indeterminate)
        progress: Progress percentage (0-100), ignored for non-progress states

    Example:
        >>> set_progress(Osc94States.PROGRESS, 50)  # 50% progress
        >>> set_progress(Osc94States.RESET, 0)   # Reset/complete
        >>> set_progress(Osc94States.ERROR, 0)   # Error
    """
    sequence = f"\x1b]9;4;{state.value};{progress}\x1b\\"
    # Silent failure - OSC is optional enhancement
    with contextlib.suppress(Exception):
        write_osc(sequence)
