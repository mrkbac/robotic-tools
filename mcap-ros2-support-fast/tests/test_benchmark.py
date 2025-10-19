import itertools
from pathlib import Path
from tempfile import TemporaryFile

import pytest
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from mcap_ros2.writer import Writer as OriginalWriter
from mcap_ros2_support_fast.decoder import DecoderFactory as DecoderFactoryFast
from mcap_ros2_support_fast.writer import Writer as FastWriter


def _read_all(factory, msgs: int):
    file = (
        Path(__file__).parent.parent.parent
        / "data"
        / "data"
        / "nuScenes-v1.0-mini-scene-0061-ros2.mcap"
    )

    with file.open("rb") as f:
        reader = make_reader(f, decoder_factories=[factory])
        for _ in itertools.islice(reader.iter_decoded_messages(), msgs):
            pass


def _read_and_write(writer_class, msgs: int):
    """Read messages using appropriate decoder and write them back using specified writer."""
    mcap_file = (
        Path(__file__).parent.parent.parent
        / "data"
        / "data"
        / "nuScenes-v1.0-mini-scene-0061-ros2.mcap"
    )

    # Use appropriate decoder based on writer to avoid compatibility issues
    decoder_factory = DecoderFactory() if writer_class == OriginalWriter else DecoderFactoryFast()

    # First pass: read messages and collect data
    messages_data = []
    with mcap_file.open("rb") as f:
        reader = make_reader(f, decoder_factories=[decoder_factory])
        for decoded_msg in itertools.islice(reader.iter_decoded_messages(), msgs):
            messages_data.append(
                {
                    "topic": decoded_msg.channel.topic,
                    "schema": decoded_msg.schema,
                    "message": decoded_msg.decoded_message,
                    "log_time": decoded_msg.message.log_time,
                    "publish_time": decoded_msg.message.publish_time,
                    "sequence": decoded_msg.message.sequence,
                }
            )

    # Second pass: write messages using specified writer
    with TemporaryFile() as temp_file:
        writer = writer_class(temp_file)

        # Keep track of registered schemas to avoid duplicates
        registered_schemas = {}

        for msg_data in messages_data:
            schema = msg_data["schema"]

            # Register schema if not already done
            if schema.name not in registered_schemas:
                registered_schema = writer.register_msgdef(schema.name, schema.data.decode())
                registered_schemas[schema.name] = registered_schema
            else:
                registered_schema = registered_schemas[schema.name]

            # Write the message
            writer.write_message(
                topic=msg_data["topic"],
                schema=registered_schema,
                message=msg_data["message"],
                log_time=msg_data["log_time"],
                publish_time=msg_data["publish_time"],
                sequence=msg_data["sequence"],
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
    ("writer_class", "msgs"),
    [
        pytest.param(writer_class, msgs, id=f"{name}-{msgs}")
        for writer_class, name in [
            (OriginalWriter, "mcap_ros2_writer"),
            (FastWriter, "mcap_ros2_fast_writer"),
        ]
        for msgs in [10, 100, 1_000]
    ],
)
@pytest.mark.benchmark(group="write-msgs-")
def test_benchmark_read_and_write(benchmark, writer_class, msgs):
    benchmark.group += str(msgs)
    benchmark(_read_and_write, writer_class, msgs)
