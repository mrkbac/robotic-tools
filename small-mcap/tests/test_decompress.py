# ruff: noqa: ARG002
"""Tests for the ChannelDecoderFactoryProtocol and caching in read_message_decoded."""

import io

from small_mcap import McapWriter, read_message_decoded


class TestChannelAwareDecoding:
    """Tests for the ChannelDecoderFactoryProtocol and caching in read_message_decoded."""

    def test_channel_aware_factory_gets_per_channel_decoders(self):
        """Channel-aware factory should create separate decoders per channel."""
        buf = io.BytesIO()
        writer = McapWriter(buf)
        writer.start()

        # Same schema, two channels
        writer.add_schema(schema_id=1, name="test", encoding="raw", data=b"")
        writer.add_channel(channel_id=1, topic="/a", message_encoding="raw", schema_id=1)
        writer.add_channel(channel_id=2, topic="/b", message_encoding="raw", schema_id=1)
        writer.add_message(channel_id=1, log_time=1, publish_time=1, data=b"msg_a")
        writer.add_message(channel_id=2, log_time=2, publish_time=2, data=b"msg_b")
        writer.finish()
        buf.seek(0)

        # Track which channels the factory sees
        channels_seen: list[int] = []

        class ChannelAwareFactory:
            channel_aware = True

            def decoder_for(self, message_encoding, schema, channel):
                channels_seen.append(channel.id)

                def _decode(data):
                    return f"decoded_{channel.id}_{data!r}"

                return _decode

        results = list(read_message_decoded(buf, decoder_factories=[ChannelAwareFactory()]))

        assert len(results) == 2
        # Access decoded messages to trigger lazy evaluation
        decoded = [r.decoded_message for r in results]
        # Factory should be called once per channel (not once per schema)
        assert sorted(channels_seen) == [1, 2]
        # Each channel gets its own decoder
        assert "decoded_1_" in decoded[0]
        assert "decoded_2_" in decoded[1]

    def test_channel_aware_caching(self):
        """Channel-aware decoders should be cached per (schema_id, channel_id)."""
        buf = io.BytesIO()
        writer = McapWriter(buf)
        writer.start()

        writer.add_schema(schema_id=1, name="test", encoding="raw", data=b"")
        writer.add_channel(channel_id=1, topic="/a", message_encoding="raw", schema_id=1)
        # Multiple messages on same channel
        for i in range(5):
            writer.add_message(channel_id=1, log_time=i, publish_time=i, data=f"msg_{i}".encode())
        writer.finish()
        buf.seek(0)

        call_count = 0

        class ChannelAwareFactory:
            channel_aware = True

            def decoder_for(self, message_encoding, schema, channel):
                nonlocal call_count
                call_count += 1
                return lambda data: data

        results = list(read_message_decoded(buf, decoder_factories=[ChannelAwareFactory()]))
        _ = [r.decoded_message for r in results]

        assert len(results) == 5
        # Factory called only once for the channel, then cached
        assert call_count == 1

    def test_mixed_factory_types(self):
        """Regular and channel-aware factories should work together."""
        buf = io.BytesIO()
        writer = McapWriter(buf)
        writer.start()

        writer.add_schema(schema_id=1, name="type_a", encoding="raw", data=b"")
        writer.add_schema(schema_id=2, name="type_b", encoding="raw", data=b"")
        writer.add_channel(channel_id=1, topic="/a", message_encoding="raw", schema_id=1)
        writer.add_channel(channel_id=2, topic="/b", message_encoding="raw", schema_id=2)
        writer.add_message(channel_id=1, log_time=1, publish_time=1, data=b"a")
        writer.add_message(channel_id=2, log_time=2, publish_time=2, data=b"b")
        writer.finish()
        buf.seek(0)

        class RegularFactory:
            def decoder_for(self, message_encoding, schema):
                if schema and schema.name == "type_a":
                    return lambda data: f"regular_{data!r}"
                return None

        class ChannelAwareFactory:
            channel_aware = True

            def decoder_for(self, message_encoding, schema, channel):
                if schema and schema.name == "type_b":
                    return lambda data: f"channel_{channel.id}_{data!r}"
                return None

        results = list(
            read_message_decoded(buf, decoder_factories=[RegularFactory(), ChannelAwareFactory()])
        )

        assert len(results) == 2
        assert results[0].decoded_message.startswith("regular_")
        assert results[1].decoded_message.startswith("channel_2_")
