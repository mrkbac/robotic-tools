from __future__ import annotations

import io

import pytest
from mcap_codec_support.video import (
    EncoderBackend,
    EncoderMode,
    VideoCodec,
    VideoEncoderError,
)
from pymcap_cli.exporters import video_exporter
from pymcap_cli.exporters.video_exporter import _VideoTopicWriter
from rich.console import Console


def test_video_topic_writer_close_reraises_flush_failure(monkeypatch, tmp_path) -> None:
    class FailingSession:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> int:
            self.closed = True
            raise VideoEncoderError("ffmpeg failed")

    def fake_create_video_file_writer(*args, **kwargs) -> FailingSession:
        del args, kwargs
        return FailingSession()

    monkeypatch.setattr(
        video_exporter,
        "create_video_file_writer",
        fake_create_video_file_writer,
    )
    writer = _VideoTopicWriter(
        tmp_path / "bad.mp4",
        codec=VideoCodec.H264,
        encoder_backend=EncoderBackend.SOFTWARE,
        quality=28,
        mode=EncoderMode.FFMPEG_CLI,
        console=Console(file=io.StringIO()),
    )

    with pytest.raises(VideoEncoderError, match=r"Flush failed .*ffmpeg failed"):
        writer.close()
