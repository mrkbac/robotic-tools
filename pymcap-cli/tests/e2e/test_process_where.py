"""End-to-end tests for ``process --where``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mcap_ros2_support_fast.decoder import DecoderFactory as Ros2DecoderFactory
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from pymcap_cli.cli import app
from small_mcap import (
    CompressionType,
    JSONDecoderFactory,
    McapWriter,
    read_message_decoded,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_json_mcap(path: Path) -> None:
    with path.open("wb") as stream:
        writer = McapWriter(
            stream,
            chunk_size=128,
            compression=CompressionType.NONE,
        )
        writer.start()
        writer.add_schema(1, "example/msg/Event", "jsonschema", b"{}")
        writer.add_channel(1, "/events", "json", 1)
        writer.add_channel(2, "/other", "json", 1)
        writer.add_message(1, 1, b'{"kind":"alarm","score":1}', 1)
        writer.add_message(2, 2, b'{"value":"untouched"}', 2)
        writer.add_message(1, 3, b'{"kind":"normal","score":12}', 3)
        writer.add_message(1, 4, b'{"kind":"normal","score":1}', 4)
        writer.finish()


@pytest.mark.e2e
def test_process_where_ors_repeated_paths_and_preserves_other_topics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mcap"
    output = tmp_path / "output.mcap"
    _write_json_mcap(source)

    with pytest.raises(SystemExit) as exc_info:
        app(
            [
                "process",
                str(source),
                "-o",
                str(output),
                "--where",
                '/events{kind == "alarm"}',
                "--where",
                "/events{score >= $minimum}",
                "--var",
                "minimum=10",
            ],
            exit_on_error=False,
        )

    assert exc_info.value.code == 0
    with output.open("rb") as stream:
        messages = [
            (decoded.channel.topic, decoded.decoded_message)
            for decoded in read_message_decoded(
                stream,
                decoder_factories=[JSONDecoderFactory()],
            )
        ]

    assert messages == [
        ("/events", {"kind": "alarm", "score": 1}),
        ("/other", {"value": "untouched"}),
        ("/events", {"kind": "normal", "score": 12}),
    ]


@pytest.mark.e2e
def test_process_where_filters_ros2_cdr_messages(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    output = tmp_path / "output.mcap"
    with source.open("wb") as stream:
        writer = McapWriter(
            stream,
            chunk_size=128,
            compression=CompressionType.NONE,
            encoder_factory=ROS2EncoderFactory(),
        )
        writer.start()
        writer.add_schema(
            1,
            "example_msgs/msg/Event",
            "ros2msg",
            b"string kind\nint32 score",
        )
        writer.add_channel(1, "/events", "cdr", 1)
        writer.add_message_encode(1, 1, {"kind": "alarm", "score": 12}, 1)
        writer.add_message_encode(1, 2, {"kind": "alarm", "score": 1}, 2)
        writer.add_message_encode(1, 3, {"kind": "normal", "score": 12}, 3)
        writer.finish()

    with pytest.raises(SystemExit) as exc_info:
        app(
            [
                "process",
                str(source),
                "-o",
                str(output),
                "--where",
                '/events{kind == "alarm" && score >= 10}',
            ],
            exit_on_error=False,
        )

    assert exc_info.value.code == 0
    with output.open("rb") as stream:
        messages = list(
            read_message_decoded(
                stream,
                decoder_factories=[Ros2DecoderFactory()],
            )
        )
    assert len(messages) == 1
    assert messages[0].decoded_message.kind == "alarm"
    assert messages[0].decoded_message.score == 12
