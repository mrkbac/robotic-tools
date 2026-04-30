"""Tests for ``roscompress --image-format jpeg``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mcap_ros2_support_fast.decoder import DecoderFactory as Ros2DecoderFactory
from pymcap_cli.cmd.roscompress_cmd import roscompress
from pymcap_cli.encoding.encoder_common import EncoderMode
from small_mcap import read_message_decoded

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("av")


def _decoded_messages(path: Path):
    with path.open("rb") as stream:
        return list(read_message_decoded(stream, decoder_factories=[Ros2DecoderFactory()]))


def test_jpeg_mode_converts_raw_images(image_rgb_mcap: Path, tmp_path: Path) -> None:
    output = tmp_path / "jpeg.mcap"

    rc = roscompress(
        file=str(image_rgb_mcap),
        output=output,
        force=True,
        image_format="jpeg",
        backend=EncoderMode.PYAV,
        pointcloud=False,
    )

    assert rc == 0
    messages = _decoded_messages(output)
    assert messages
    for msg in messages:
        assert msg.schema is not None
        assert msg.schema.name == "sensor_msgs/msg/CompressedImage"
        assert msg.decoded_message.format == "jpeg"
        assert bytes(msg.decoded_message.data).startswith(b"\xff\xd8")


def test_jpeg_mode_copies_compressed_images_unchanged(
    image_compressed_mcap: Path, tmp_path: Path
) -> None:
    input_messages = _decoded_messages(image_compressed_mcap)
    input_payloads = [bytes(msg.decoded_message.data) for msg in input_messages]
    output = tmp_path / "jpeg-copy.mcap"

    rc = roscompress(
        file=str(image_compressed_mcap),
        output=output,
        force=True,
        image_format="jpeg",
        backend=EncoderMode.PYAV,
        pointcloud=False,
    )

    assert rc == 0
    output_messages = _decoded_messages(output)
    assert [bytes(msg.decoded_message.data) for msg in output_messages] == input_payloads
    assert [msg.decoded_message.format for msg in output_messages] == ["jpeg"] * len(input_payloads)
