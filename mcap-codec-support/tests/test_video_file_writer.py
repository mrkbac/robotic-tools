from __future__ import annotations

from types import SimpleNamespace

from mcap_codec_support.video import (
    EncoderBackend,
    EncoderConfig,
    EncoderMode,
    VideoCodec,
    VideoFileWriterSession,
)
from mcap_codec_support.video import file_writer as video_file_writer_module


def test_ffmpeg_cli_raw_odd_dimensions_are_cropped_to_even_frame(monkeypatch, tmp_path) -> None:
    opened: dict[str, int | str] = {}
    written: list[bytes] = []

    class FakeFfmpegMp4Strategy:
        def __init__(
            self,
            path,
            *,
            codec,
            encoder_backend,
            quality,
            width,
            height,
            input_pix_fmt,
        ) -> None:
            del path, codec, encoder_backend, quality
            opened.update(width=width, height=height, input_pix_fmt=input_pix_fmt)
            self.config = EncoderConfig(width=width, height=height, codec_name="fake")
            self._closed = False

        def write_raw(self, data, log_time_ns) -> None:
            del log_time_ns
            written.append(data)
            self._closed = False

        def close(self) -> int:
            self._closed = True
            return len(written)

    monkeypatch.setattr(video_file_writer_module, "_FfmpegMp4Strategy", FakeFfmpegMp4Strategy)

    decoded = SimpleNamespace(
        width=5,
        height=5,
        encoding="rgb8",
        step=5 * 3,
        data=bytes(range(5 * 5 * 3)),
    )
    writer = VideoFileWriterSession(
        tmp_path / "out.mp4",
        codec=VideoCodec.H264,
        encoder_backend=EncoderBackend.SOFTWARE,
        quality=28,
        mode=EncoderMode.FFMPEG_CLI,
    )

    writer.write_message(decoded, "sensor_msgs/Image", 0)

    assert opened == {"width": 4, "height": 4, "input_pix_fmt": "rgb24"}
    expected = b"".join(bytes(range(row * 15, row * 15 + 12)) for row in range(4))
    assert written == [expected]
    assert writer.close() == 1
