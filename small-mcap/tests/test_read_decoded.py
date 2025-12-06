"""Tests for read_message_decoded() function."""

import io

import pytest
from small_mcap import (
    JSONDecoderFactory,
    JSONEncoderFactory,
    McapWriter,
    read_message_decoded,
)


def _create_mcap_with_json_messages(messages: list[dict]) -> io.BytesIO:
    """Create an in-memory MCAP file with JSON-encoded messages using JSONEncoderFactory."""
    buffer = io.BytesIO()
    writer = McapWriter(buffer, encoder_factory=JSONEncoderFactory())
    writer.start()

    writer.add_channel(channel_id=1, topic="/test", message_encoding="json", schema_id=0)

    for i, msg in enumerate(messages):
        writer.add_message_encode(
            channel_id=1,
            log_time=i * 1_000_000,
            publish_time=i * 1_000_000,
            data=msg,
        )

    writer.finish()
    buffer.seek(0)
    return buffer


def test_read_message_decoded_with_json():
    """read_message_decoded should decode JSON messages using JSONDecoderFactory."""
    original = {"key": "value", "number": 42}
    buffer = _create_mcap_with_json_messages([original])

    results = list(read_message_decoded(buffer, decoder_factories=[JSONDecoderFactory()]))

    assert len(results) == 1
    decoded_msg = results[0]
    assert decoded_msg.channel.topic == "/test"
    assert decoded_msg.decoded_message == original


def test_read_message_decoded_multiple_messages():
    """read_message_decoded should decode multiple messages."""
    messages = [{"id": 1}, {"id": 2}, {"id": 3}]
    buffer = _create_mcap_with_json_messages(messages)

    results = list(read_message_decoded(buffer, decoder_factories=[JSONDecoderFactory()]))

    assert len(results) == 3
    for i, result in enumerate(results):
        assert result.decoded_message == messages[i]


def test_read_message_decoded_schemaless():
    """Schemaless messages should return raw data bytes."""
    buffer = io.BytesIO()
    writer = McapWriter(buffer)
    writer.start()

    # Schemaless channel (schema_id=0)
    writer.add_channel(channel_id=1, topic="/raw", message_encoding="raw", schema_id=0)

    raw_data = b"raw binary data"
    writer.add_message(channel_id=1, log_time=0, publish_time=0, data=raw_data)

    writer.finish()
    buffer.seek(0)

    results = list(read_message_decoded(buffer, decoder_factories=[]))

    assert len(results) == 1
    # Schemaless returns raw message.data
    assert bytes(results[0].decoded_message) == raw_data


def test_read_message_decoded_no_decoder_raises():
    """Should raise ValueError when no decoder factory matches."""
    buffer = io.BytesIO()
    writer = McapWriter(buffer)
    writer.start()

    writer.add_schema(schema_id=1, name="custom", encoding="custom", data=b"")
    writer.add_channel(channel_id=1, topic="/test", message_encoding="custom", schema_id=1)
    writer.add_message(channel_id=1, log_time=0, publish_time=0, data=b"data")

    writer.finish()
    buffer.seek(0)

    results = read_message_decoded(buffer, decoder_factories=[])

    with pytest.raises(ValueError, match=r"no decoder factory.*custom"):
        # Access decoded_message to trigger the error
        _ = next(iter(results)).decoded_message


def test_read_message_decoded_decoder_caching():
    """Decoder should be cached and reused for subsequent messages with same schema."""
    messages = [{"msg": i} for i in range(5)]
    buffer = _create_mcap_with_json_messages(messages)

    # Track how many times decoder_for is called
    call_count = 0
    original_factory = JSONDecoderFactory()

    class CountingFactory:
        def decoder_for(self, message_encoding, schema):
            nonlocal call_count
            call_count += 1
            return original_factory.decoder_for(message_encoding, schema)

    results = list(read_message_decoded(buffer, decoder_factories=[CountingFactory()]))

    # Access all decoded messages to ensure decoder is used
    decoded = [r.decoded_message for r in results]

    assert len(decoded) == 5
    # Decoder should only be looked up once, then cached
    assert call_count == 1
