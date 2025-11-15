import itertools
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any

import pytest
from mcap_ros2._dynamic import EncoderFunction, serialize_dynamic
from mcap_ros2.decoder import DecoderFactory
from mcap_ros2_support_fast.decoder import DecoderFactory as DecoderFactoryFast
from mcap_ros2_support_fast.writer import ROS2EncoderFactory as ROS2EncoderFactoryFast
from small_mcap import DecodedMessage, McapWriter, read_message_decoded


class ROS2EncoderFactory:
    """
    Encoder factory for ROS2 messages that implements EncoderFactoryProtocol.
    Caches encoders by schema ID for efficient repeated encoding.
    """

    profile = "ros2"
    encoding = "ros2msg"  # Schema encoding format
    message_encoding = "cdr"  # Message data encoding format

    def __init__(self) -> None:
        self._encoders: dict[int, EncoderFunction] = {}
        self.library = "mcap-ros2-support-benchmark; small-mcap"

    def encoder_for(self, schema: Any | None) -> Callable[[object], bytes] | None:
        if schema is None:
            return None
        encoder = self._encoders.get(schema.id)
        if encoder is None:
            if schema.encoding != "ros2msg":
                raise RuntimeError(f'can\'t parse schema with encoding "{schema.encoding}"')
            type_dict = serialize_dynamic(schema.name, schema.data.decode())
            # Check if schema.name is in type_dict
            if schema.name not in type_dict:
                raise RuntimeError(f'schema parsing failed for "{schema.name}"')
            encoder = type_dict[schema.name]
            self._encoders[schema.id] = encoder

        return encoder


def _read_all(factory, msgs: int):
    file = (
        Path(__file__).parent.parent.parent
        / "data"
        / "data"
        / "nuScenes-v1.0-mini-scene-0061-ros2.mcap"
    )

    with file.open("rb") as f:
        for _ in itertools.islice(read_message_decoded(f, decoder_factories=[factory]), msgs):
            pass


def _read_and_write(
    decoder_factory: DecoderFactory | DecoderFactoryFast,
    encoder_factory: ROS2EncoderFactory | ROS2EncoderFactoryFast,
    msgs: int,
):
    """Read messages using appropriate decoder and write them back using specified writer."""
    mcap_file = (
        Path(__file__).parent.parent.parent
        / "data"
        / "data"
        / "nuScenes-v1.0-mini-scene-0061-ros2.mcap"
    )

    # First pass: read messages and collect data
    messages_data: list[DecodedMessage] = []
    with mcap_file.open("rb") as f:
        messages_data.extend(
            itertools.islice(read_message_decoded(f, decoder_factories=[decoder_factory]), msgs)
        )

    # Second pass: write messages using specified writer
    with TemporaryFile() as temp_file:
        writer = McapWriter(temp_file, encoder_factory=encoder_factory)
        writer.start()

        # Track schema and channel IDs
        schema_ids: dict[str, int] = {}
        channel_ids: dict[str, int] = {}
        next_schema_id = 1
        next_channel_id = 1

        for msg_data in messages_data:
            assert msg_data.schema is not None

            # Register schema if needed
            if msg_data.schema.name not in schema_ids:
                schema_id = next_schema_id
                next_schema_id += 1
                writer.add_schema(schema_id, msg_data.schema.name, msg_data.schema.encoding, msg_data.schema.data)
                schema_ids[msg_data.schema.name] = schema_id
            else:
                schema_id = schema_ids[msg_data.schema.name]

            # Register channel if needed
            if msg_data.channel.topic not in channel_ids:
                channel_id = next_channel_id
                next_channel_id += 1
                writer.add_channel(channel_id, msg_data.channel.topic, msg_data.channel.message_encoding, schema_id, msg_data.channel.metadata)
                channel_ids[msg_data.channel.topic] = channel_id
            else:
                channel_id = channel_ids[msg_data.channel.topic]

            # Write the message
            writer.add_message_encode(
                channel_id=channel_id,
                log_time=msg_data.message.log_time,
                data=msg_data.decoded_message,
                publish_time=msg_data.message.publish_time,
                sequence=msg_data.message.sequence,
            )

        writer.finish()


@pytest.mark.parametrize(
    ("factory", "msgs"),
    [
        pytest.param(factory, msgs, id=f"{name}-{msgs}")
        for factory, name in [
            (DecoderFactory(), "mcap_ros2"),
            (DecoderFactoryFast(), "mcap_ros2_fast"),
        ]
        for msgs in [10, 100, 1_000]
    ],
)
@pytest.mark.benchmark(group="msgs-")
def test_benchmark_decoder(benchmark, factory, msgs):
    benchmark.group += str(msgs)
    benchmark(_read_all, factory, msgs)


@pytest.mark.parametrize(
    ("decoder_factory", "encoder_factory", "msgs"),
    [
        pytest.param(encoder_factory, decoder_factory, msgs, id=f"{name}-{msgs}")
        for encoder_factory, decoder_factory, name in [
            (DecoderFactory(), ROS2EncoderFactory(), "mcap_ros2_writer"),
            (DecoderFactoryFast(), ROS2EncoderFactoryFast(), "mcap_ros2_fast_writer"),
        ]
        for msgs in [10, 100, 1_000]
    ],
)
@pytest.mark.benchmark(group="write-msgs-")
def test_benchmark_read_and_write(benchmark, decoder_factory, encoder_factory, msgs):
    benchmark.group += str(msgs)
    benchmark(_read_and_write, decoder_factory, encoder_factory, msgs)
