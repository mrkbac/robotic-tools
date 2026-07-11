from __future__ import annotations

import io
from dataclasses import dataclass

from pymcap_cli.core.mcap_processor import (
    InputFile,
    InputOptions,
    McapProcessor,
    OutputOptions,
    ProcessingOptions,
    ProcessingStats,
)
from small_mcap import CompressionType, McapWriter, read_message


class CountingBytesIO(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.bytes_read = 0
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        data = super().read(size)
        self.bytes_read += len(data)
        return data


class NonClosingBytesIO(io.BytesIO):
    def close(self) -> None:
        self.flush()


@dataclass(slots=True)
class ProcessorRun:
    stats: ProcessingStats
    output: bytes
    bytes_read: int
    read_sizes: list[int]


def _run_processor(data: bytes, input_options: InputOptions) -> ProcessorRun:
    input_stream = CountingBytesIO(data)
    output_stream = NonClosingBytesIO()
    options = ProcessingOptions(
        inputs=[InputFile(stream=input_stream, size=len(data), options=input_options)],
        input_options=InputOptions.from_args(),
        output_options=OutputOptions(compression="none"),
    )
    stats = McapProcessor(options).process(output_stream)
    return ProcessorRun(
        stats=stats,
        output=output_stream.getvalue(),
        bytes_read=input_stream.bytes_read,
        read_sizes=input_stream.read_sizes,
    )


def _read_output(data: bytes) -> list[tuple[str, int, bytes]]:
    with io.BytesIO(data) as stream:
        return [
            (channel.topic, message.log_time, bytes(message.data))
            for _schema, channel, message in read_message(stream)
        ]


def _build_unchunked(
    events: list[tuple[int, int, bytes]],
    *,
    channels: dict[int, str] | None = None,
) -> bytes:
    output = io.BytesIO()
    writer = McapWriter(output, use_chunking=False, compression=CompressionType.NONE)
    writer.start()
    writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")
    for channel_id, topic in (channels or {1: "/test"}).items():
        writer.add_channel(channel_id, topic=topic, message_encoding="json", schema_id=1)
    for log_time, channel_id, payload in events:
        writer.add_message(
            channel_id=channel_id,
            log_time=log_time,
            publish_time=log_time,
            data=payload,
        )
    writer.finish()
    return output.getvalue()


def _build_chunked(messages: list[tuple[int, bytes]]) -> bytes:
    output = io.BytesIO()
    writer = McapWriter(
        output,
        use_chunking=True,
        chunk_size=128,
        compression=CompressionType.NONE,
    )
    writer.start()
    writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")
    writer.add_channel(channel_id=1, topic="/test", message_encoding="json", schema_id=1)
    for log_time, payload in messages:
        writer.add_message(
            channel_id=1,
            log_time=log_time,
            publish_time=log_time,
            data=payload,
        )
    writer.finish()
    return output.getvalue()


def test_unchunked_time_filter_seeks_over_rejected_payloads() -> None:
    payload_size = 200_000
    payload = b"x" * payload_size
    data = _build_unchunked(
        [
            (0, 1, payload),
            (10, 1, payload),
            (100, 1, payload),
            (200, 1, payload),
            (300, 1, payload),
        ]
    )

    result = _run_processor(
        data,
        InputOptions.from_args(
            end_nsecs=50,
            include_metadata=False,
            include_attachments=False,
        ),
    )

    assert [log_time for _topic, log_time, _payload in _read_output(result.output)] == [0, 10]
    assert result.bytes_read < len(data) - (2 * payload_size)


def test_unchunked_accepted_message_reads_body_once() -> None:
    payload_size = 200_000
    data = _build_unchunked([(10, 1, b"x" * payload_size)])

    result = _run_processor(
        data,
        InputOptions.from_args(include_metadata=False, include_attachments=False),
    )

    assert _read_output(result.output) == [("/test", 10, b"x" * payload_size)]
    assert payload_size + 22 in result.read_sizes
    assert payload_size not in result.read_sizes


def test_unchunked_time_filter_does_not_stop_on_non_monotonic_time() -> None:
    data = _build_unchunked(
        [
            (0, 1, b"first"),
            (100, 1, b"outside"),
            (10, 1, b"late-inside"),
        ]
    )

    result = _run_processor(
        data,
        InputOptions.from_args(
            end_nsecs=50,
            include_metadata=False,
            include_attachments=False,
        ),
    )

    assert [(log_time, payload) for _topic, log_time, payload in _read_output(result.output)] == [
        (0, b"first"),
        (10, b"late-inside"),
    ]


def test_unchunked_time_filter_early_bail_stops_at_end() -> None:
    payload_size = 100_000
    payload = b"x" * payload_size
    data = _build_unchunked(
        [
            (0, 1, payload),
            (10, 1, payload),
            (100, 1, payload),
            (200, 1, payload),
            (300, 1, payload),
        ]
    )

    result = _run_processor(
        data,
        InputOptions.from_args(
            end_nsecs=50,
            include_metadata=False,
            include_attachments=False,
            is_early_bail_enabled=True,
        ),
    )

    assert [log_time for _topic, log_time, _payload in _read_output(result.output)] == [0, 10]
    assert result.stats.messages_processed == 3


def test_unchunked_time_filter_early_bail_trusts_monotonic_time() -> None:
    data = _build_unchunked(
        [
            (0, 1, b"first"),
            (100, 1, b"outside"),
            (10, 1, b"late-inside"),
        ]
    )

    result = _run_processor(
        data,
        InputOptions.from_args(
            end_nsecs=50,
            include_metadata=False,
            include_attachments=False,
            is_early_bail_enabled=True,
        ),
    )

    assert [(log_time, payload) for _topic, log_time, payload in _read_output(result.output)] == [
        (0, b"first"),
    ]


def test_unchunked_latched_message_is_read_before_time_filter_skip() -> None:
    data = _build_unchunked(
        [
            (0, 1, b'{"latched": true}'),
            (5_000_000_000, 2, b'{"scan": 1}'),
        ],
        channels={1: "/tf_static", 2: "/scan"},
    )

    result = _run_processor(
        data,
        InputOptions.from_args(
            start="5000000000",
            latch_topics=["/tf_static"],
            include_metadata=False,
            include_attachments=False,
        ),
    )

    assert [(topic, log_time) for topic, log_time, _payload in _read_output(result.output)] == [
        ("/tf_static", 0),
        ("/scan", 5_000_000_000),
    ]


def test_indexed_chunk_time_filter_reads_only_overlapping_chunks() -> None:
    payload_size = 150_000
    payload = b"x" * payload_size
    data = _build_chunked(
        [
            (0, payload),
            (100, payload),
            (200, payload),
            (300, payload),
        ]
    )

    result = _run_processor(
        data,
        InputOptions.from_args(
            end_nsecs=50,
            include_metadata=False,
            include_attachments=False,
        ),
    )

    assert [log_time for _topic, log_time, _payload in _read_output(result.output)] == [0]
    assert result.stats.writer_statistics.message_count == 1
    assert result.bytes_read < len(data) - payload_size
