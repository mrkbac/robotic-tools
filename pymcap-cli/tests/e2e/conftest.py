"""E2E test configuration — ensure static ffmpeg is available."""

from portable_ffmpeg import add_to_path

# Prepend static ffmpeg/ffprobe to PATH so tests get a build with libx264,
# even when the system ffmpeg lacks it (e.g. GitHub Actions Ubuntu runner).
add_to_path()
