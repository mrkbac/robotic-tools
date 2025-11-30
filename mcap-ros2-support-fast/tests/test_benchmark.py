from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any

import pytest
from mcap_ros2._dynamic import EncoderFunction, serialize_dynamic
from mcap_ros2.decoder import DecoderFactory
from mcap_ros2_support_fast.decoder import DecoderFactory as DecoderFactoryFast
from mcap_ros2_support_fast.writer import ROS2EncoderFactory as ROS2EncoderFactoryFast
from pybag.encoding.cdr import CdrDecoder, CdrEncoder
from pybag.schema.compiler import compile_schema, compile_serializer
from pybag.schema.ros2msg import Ros2MsgSchemaDecoder
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
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


class RosbagsDecoderFactory:
    """Decoder factory for rosbags library with adhoc type generation."""

    def __init__(self) -> None:
        # Start with empty typestore for fair comparison
        self._typestore = get_typestore(Stores.EMPTY)

    def decoder_for(
        self, message_encoding: str, schema: Any | None
    ) -> Callable[[bytes | memoryview], Any] | None:
        """Get decoder for the given schema."""
        if message_encoding != "cdr" or schema is None or schema.encoding != "ros2msg":
            return None

        # Normalize typename: rosbags uses "package/msg/Type" format
        typename = schema.name
        if typename.count("/") == 1:
            # Convert "package/Type" to "package/msg/Type"
            parts = typename.split("/")
            typename = f"{parts[0]}/msg/{parts[1]}"

        # Check if type already registered
        if typename not in self._typestore.types:
            # Parse schema and register type adhoc
            schema_text = schema.data.decode()
            types = get_types_from_msg(schema_text, typename)
            self._typestore.register(types)

        def decode(data: bytes | memoryview) -> Any:
            """Decode CDR data using rosbags."""
            if isinstance(data, memoryview):
                data = bytes(data)
            return self._typestore.deserialize_cdr(data, typename)

        return decode


class RosbagsEncoderFactory:
    """Encoder factory for rosbags library with adhoc type generation."""

    profile = "ros2"
    encoding = "ros2msg"
    message_encoding = "cdr"

    def __init__(self) -> None:
        # Start with empty typestore for fair comparison
        self._typestore = get_typestore(Stores.EMPTY)
        self.library = "rosbags; small-mcap"

    def encoder_for(self, schema: Any | None) -> Callable[[object], bytes] | None:
        """Get encoder for the given schema."""
        if schema is None or schema.encoding != "ros2msg":
            return None

        # Normalize typename: rosbags uses "package/msg/Type" format
        typename = schema.name
        if typename.count("/") == 1:
            # Convert "package/Type" to "package/msg/Type"
            parts = typename.split("/")
            typename = f"{parts[0]}/msg/{parts[1]}"

        # Check if type already registered
        if typename not in self._typestore.types:
            # Parse schema and register type adhoc
            schema_text = schema.data.decode()
            types = get_types_from_msg(schema_text, typename)
            self._typestore.register(types)

        def encode(msg: Any) -> bytes:
            """Encode message to CDR using rosbags."""
            return self._typestore.serialize_cdr(msg, typename)

        return encode


class PyBagDecoderFactory:
    """Decoder factory for pybag library."""

    def __init__(self) -> None:
        self._schema_decoder = Ros2MsgSchemaDecoder()
        # Cache compiled decoders by schema ID
        self._compiled: dict[int, Callable[[CdrDecoder], Any]] = {}

    def decoder_for(
        self, message_encoding: str, schema: Any | None
    ) -> Callable[[bytes | memoryview], Any] | None:
        """Get decoder for the given schema."""
        if message_encoding != "cdr" or schema is None or schema.encoding != "ros2msg":
            return None

        # Get or compile decoder
        if schema.id not in self._compiled:
            msg_schema, schema_msgs = self._schema_decoder.parse_schema(schema)
            self._compiled[schema.id] = compile_schema(msg_schema, schema_msgs)

        compiled_decoder = self._compiled[schema.id]

        def decode(data: bytes | memoryview) -> Any:
            """Decode CDR data using pybag."""
            if isinstance(data, memoryview):
                data = bytes(data)
            decoder = CdrDecoder(data)
            return compiled_decoder(decoder)

        return decode


class PyBagEncoderFactory:
    """Encoder factory for pybag library.

    Note: PyBag's encoder requires dataclass instances. For fair benchmarking,
    this must be paired with PyBagDecoderFactory which produces compatible dataclasses.
    """

    profile = "ros2"
    encoding = "ros2msg"
    message_encoding = "cdr"

    def __init__(self) -> None:
        self._schema_decoder = Ros2MsgSchemaDecoder()
        # Cache compiled encoders by schema ID
        self._compiled: dict[int, Callable[[CdrEncoder, Any], None]] = {}
        # Cache message type info by schema ID
        self._schema_info: dict[int, tuple[Any, dict[str, Any]]] = {}
        self.library = "pybag; small-mcap"

    def encoder_for(self, schema: Any | None) -> Callable[[object], bytes] | None:
        """Get encoder for the given schema."""
        if schema is None or schema.encoding != "ros2msg":
            return None

        # Get or compile encoder
        if schema.id not in self._compiled:
            # Parse schema and compile serializer
            msg_schema, schema_msgs = self._schema_decoder.parse_schema(schema)
            self._compiled[schema.id] = compile_serializer(msg_schema, schema_msgs)
            self._schema_info[schema.id] = (msg_schema, schema_msgs)

        compiled_encoder = self._compiled[schema.id]

        def encode(msg: Any) -> bytes:
            """Encode message to CDR using pybag.

            Expects a dataclass instance produced by PyBagDecoderFactory.
            """
            encoder = CdrEncoder(little_endian=True)
            compiled_encoder(encoder, msg)
            return encoder.save()

        return encode


def _read_all(factory, msgs: int):
    file = (
        Path(__file__).parent.parent.parent
        / "data"
        / "data"
        / "nuScenes-v1.0-mini-scene-0061-ros2.mcap"
    )

    with file.open("rb") as f:
        for count, msg in enumerate(read_message_decoded(f, decoder_factories=[factory]), 1):
            _ = msg.decoded_message  # Force decoding by accessing the property
            if count >= msgs:
                break


def _read_and_write(
    decoder_factory: DecoderFactory
    | DecoderFactoryFast
    | RosbagsDecoderFactory
    | PyBagDecoderFactory,
    encoder_factory: ROS2EncoderFactory
    | ROS2EncoderFactoryFast
    | RosbagsEncoderFactory
    | PyBagEncoderFactory,
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
        for msg in read_message_decoded(f, decoder_factories=[decoder_factory]):
            _ = msg.decoded_message  # Force decoding by checking the property
            messages_data.append(msg)
            if len(messages_data) >= msgs:
                break

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
                writer.add_schema(
                    schema_id, msg_data.schema.name, msg_data.schema.encoding, msg_data.schema.data
                )
                schema_ids[msg_data.schema.name] = schema_id
            else:
                schema_id = schema_ids[msg_data.schema.name]

            # Register channel if needed
            if msg_data.channel.topic not in channel_ids:
                channel_id = next_channel_id
                next_channel_id += 1
                writer.add_channel(
                    channel_id,
                    msg_data.channel.topic,
                    msg_data.channel.message_encoding,
                    schema_id,
                    msg_data.channel.metadata,
                )
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


def _write_only(
    encoder_factory: ROS2EncoderFactory
    | ROS2EncoderFactoryFast
    | RosbagsEncoderFactory
    | PyBagEncoderFactory,
    messages_data: list[DecodedMessage],
):
    """Write pre-loaded messages using specified encoder.

    Measures pure write performance (encoding + MCAP writing) without read overhead.
    """
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
                writer.add_schema(
                    schema_id,
                    msg_data.schema.name,
                    msg_data.schema.encoding,
                    msg_data.schema.data,
                )
                schema_ids[msg_data.schema.name] = schema_id
            else:
                schema_id = schema_ids[msg_data.schema.name]

            # Register channel if needed
            if msg_data.channel.topic not in channel_ids:
                channel_id = next_channel_id
                next_channel_id += 1
                writer.add_channel(
                    channel_id,
                    msg_data.channel.topic,
                    msg_data.channel.message_encoding,
                    schema_id,
                    msg_data.channel.metadata,
                )
                channel_ids[msg_data.channel.topic] = channel_id
            else:
                channel_id = channel_ids[msg_data.channel.topic]

            # Write the message (THIS IS WHAT WE BENCHMARK)
            writer.add_message_encode(
                channel_id=channel_id,
                log_time=msg_data.message.log_time,
                data=msg_data.decoded_message,
                publish_time=msg_data.message.publish_time,
                sequence=msg_data.message.sequence,
            )

        writer.finish()


def _preload_messages(decoder_factory, msgs: int) -> list[DecodedMessage]:
    """Helper to pre-load messages from MCAP file."""
    mcap_file = (
        Path(__file__).parent.parent.parent
        / "data"
        / "data"
        / "nuScenes-v1.0-mini-scene-0061-ros2.mcap"
    )

    messages_data: list[DecodedMessage] = []
    with mcap_file.open("rb") as f:
        for msg in read_message_decoded(f, decoder_factories=[decoder_factory]):
            _ = msg.decoded_message  # Force decoding
            messages_data.append(msg)
            if len(messages_data) >= msgs:
                break

    return messages_data


# Decoder factories for pre-loading messages
_DECODER_FACTORIES = {
    "mcap_ros2": DecoderFactory,
    "mcap_ros2_fast": DecoderFactoryFast,
    "rosbags": RosbagsDecoderFactory,
    "pybag": PyBagDecoderFactory,
}

# Cache for pre-loaded messages to avoid re-loading
_preloaded_cache: dict[tuple[str, int], list[DecodedMessage]] = {}


def _get_preloaded_messages(library_name: str, msgs: int) -> list[DecodedMessage]:
    """Get or create pre-loaded messages for a library and message count."""
    cache_key = (library_name, msgs)
    if cache_key not in _preloaded_cache:
        decoder_factory_class = _DECODER_FACTORIES[library_name]
        _preloaded_cache[cache_key] = _preload_messages(decoder_factory_class(), msgs)
    return _preloaded_cache[cache_key]


@pytest.mark.parametrize(
    ("factory", "msgs"),
    [
        pytest.param(factory, msgs, id=f"{name}-{msgs}")
        for factory, name in [
            (DecoderFactory(), "mcap_ros2"),
            (DecoderFactoryFast(), "mcap_ros2_fast"),
            (RosbagsDecoderFactory(), "rosbags"),
        ]
        for msgs in [10, 100, 1_000]
    ]
    + [
        # pybag can only decode up to ~100 messages due to missing schema support
        pytest.param(PyBagDecoderFactory(), msgs, id=f"pybag-{msgs}")
        for msgs in [10, 100]
    ]
    + [
        pytest.param(
            PyBagDecoderFactory(),
            1_000,
            id="pybag-1000",
            marks=pytest.mark.skip(reason="pybag cannot decode diagnostic_msgs/DiagnosticArray"),
        ),
    ],
)
@pytest.mark.benchmark(group="msgs-")
def test_benchmark_decoder(benchmark, factory, msgs):
    benchmark.group += str(msgs)
    benchmark(_read_all, factory, msgs)


@pytest.mark.parametrize(
    ("decoder_factory", "encoder_factory", "msgs"),
    [
        pytest.param(decoder_factory, encoder_factory, msgs, id=f"{name}-{msgs}")
        for decoder_factory, encoder_factory, name in [
            (DecoderFactory(), ROS2EncoderFactory(), "mcap_ros2_writer"),
            (DecoderFactoryFast(), ROS2EncoderFactoryFast(), "mcap_ros2_fast_writer"),
            (RosbagsDecoderFactory(), RosbagsEncoderFactory(), "rosbags_writer"),
        ]
        for msgs in [10, 100, 1_000]
    ]
    + [
        # pybag can only decode up to ~100 messages due to missing schema support
        pytest.param(PyBagDecoderFactory(), PyBagEncoderFactory(), msgs, id=f"pybag_writer-{msgs}")
        for msgs in [10, 100]
    ]
    + [
        pytest.param(
            PyBagDecoderFactory(),
            PyBagEncoderFactory(),
            1_000,
            id="pybag_writer-1000",
            marks=pytest.mark.skip(reason="pybag cannot decode diagnostic_msgs/DiagnosticArray"),
        ),
    ],
)
@pytest.mark.benchmark(group="write-msgs-")
def test_benchmark_read_and_write(benchmark, decoder_factory, encoder_factory, msgs):
    benchmark.group += str(msgs)
    benchmark(_read_and_write, decoder_factory, encoder_factory, msgs)


@pytest.mark.parametrize(
    ("encoder_factory", "msgs", "library_name"),
    [
        pytest.param(encoder_factory, msgs, library_name, id=f"{library_name}-{msgs}")
        for encoder_factory, library_name in [
            (ROS2EncoderFactory(), "mcap_ros2"),
            (ROS2EncoderFactoryFast(), "mcap_ros2_fast"),
            (RosbagsEncoderFactory(), "rosbags"),
        ]
        for msgs in [10, 100, 1_000]
    ]
    + [
        # pybag can only decode up to ~100 messages due to missing schema support
        pytest.param(PyBagEncoderFactory(), msgs, "pybag", id=f"pybag-{msgs}")
        for msgs in [10, 100]
    ]
    + [
        pytest.param(
            PyBagEncoderFactory(),
            1_000,
            "pybag",
            id="pybag-1000",
            marks=pytest.mark.skip(reason="pybag cannot decode diagnostic_msgs/DiagnosticArray"),
        ),
    ],
)
@pytest.mark.benchmark(group="write-only-msgs-")
def test_benchmark_write_only(benchmark, encoder_factory, msgs, library_name):
    """Benchmark pure write performance (pre-loaded messages)."""
    benchmark.group += str(msgs)

    # Get pre-loaded messages (each library needs its own decoder's format)
    messages_data = _get_preloaded_messages(library_name, msgs)

    benchmark(_write_only, encoder_factory, messages_data)


# ============================================================================
# SLOW TESTS - Full File (30,900 messages)
# Run with: pytest -m slow --benchmark-only
# ============================================================================


@pytest.mark.slow
@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(factory, id=name)
        for factory, name in [
            (DecoderFactory(), "mcap_ros2"),
            (DecoderFactoryFast(), "mcap_ros2_fast"),
            (RosbagsDecoderFactory(), "rosbags"),
        ]
    ]
    + [
        pytest.param(
            PyBagDecoderFactory(),
            id="pybag",
            marks=pytest.mark.skip(reason="pybag cannot decode diagnostic_msgs/DiagnosticArray"),
        ),
    ],
)
@pytest.mark.benchmark(group="full-file-read")
def test_benchmark_decoder_full_file(benchmark, factory):
    """Benchmark reading ALL messages (30,900) from nuScenes MCAP."""
    benchmark(_read_all, factory, 30_900)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("decoder_factory", "encoder_factory"),
    [
        pytest.param(decoder_factory, encoder_factory, id=name)
        for decoder_factory, encoder_factory, name in [
            (DecoderFactory(), ROS2EncoderFactory(), "mcap_ros2"),
            (DecoderFactoryFast(), ROS2EncoderFactoryFast(), "mcap_ros2_fast"),
            (RosbagsDecoderFactory(), RosbagsEncoderFactory(), "rosbags"),
        ]
    ]
    + [
        pytest.param(
            PyBagDecoderFactory(),
            PyBagEncoderFactory(),
            id="pybag",
            marks=pytest.mark.skip(reason="pybag cannot decode diagnostic_msgs/DiagnosticArray"),
        ),
    ],
)
@pytest.mark.benchmark(group="full-file-read-write")
def test_benchmark_read_and_write_full_file(benchmark, decoder_factory, encoder_factory):
    """Benchmark reading and writing ALL messages (30,900) from nuScenes MCAP."""
    benchmark(_read_and_write, decoder_factory, encoder_factory, 30_900)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("encoder_factory", "library_name"),
    [
        pytest.param(encoder_factory, library_name, id=library_name)
        for encoder_factory, library_name in [
            (ROS2EncoderFactory(), "mcap_ros2"),
            (ROS2EncoderFactoryFast(), "mcap_ros2_fast"),
            (RosbagsEncoderFactory(), "rosbags"),
        ]
    ]
    + [
        pytest.param(
            PyBagEncoderFactory(),
            "pybag",
            id="pybag",
            marks=pytest.mark.skip(reason="pybag cannot decode diagnostic_msgs/DiagnosticArray"),
        ),
    ],
)
@pytest.mark.benchmark(group="full-file-write-only")
def test_benchmark_write_only_full_file(benchmark, encoder_factory, library_name):
    """Benchmark writing ALL messages (30,900) with pre-loaded data."""
    messages_data = _get_preloaded_messages(library_name, 30_900)
    benchmark(_write_only, encoder_factory, messages_data)
