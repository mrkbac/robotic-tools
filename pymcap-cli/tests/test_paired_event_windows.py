from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from mcap_ros2_support_fast.decoder import DecoderFactory as Ros2DecoderFactory
from pymcap_cli.cmd import split_cmd
from pymcap_cli.cmd.split_cmd import split
from pymcap_cli.core.processors.paired_event_window import (
    BoundaryEvent,
    BoundaryMatcher,
    _pair_events,
)
from small_mcap import Channel, CompressionType, McapWriter, Message, Schema, stream_reader

if TYPE_CHECKING:
    from pathlib import Path


def _write_events(path: Path, events: list[tuple[str, int, str]]) -> None:
    topics = list(dict.fromkeys(topic for topic, _, _ in events))
    channel_ids = {topic: index for index, topic in enumerate(topics, start=1)}
    with path.open("wb") as stream:
        writer = McapWriter(stream, chunk_size=128, compression=CompressionType.NONE)
        writer.start()
        writer.add_schema(1, "example/msg/Event", "jsonschema", b"{}")
        for topic, channel_id in channel_ids.items():
            writer.add_channel(channel_id, topic, "json", 1)
        for sequence, (topic, log_time, payload) in enumerate(events):
            writer.add_message(
                channel_ids[topic],
                log_time,
                payload.encode(),
                log_time,
                sequence,
            )
        writer.finish()


def _write_one_message_chunks(path: Path, events: list[tuple[str, int, str]]) -> None:
    topics = list(dict.fromkeys(topic for topic, _, _ in events))
    channel_ids = {topic: index for index, topic in enumerate(topics, start=1)}
    with path.open("wb") as stream:
        writer = McapWriter(stream, chunk_size=1024 * 1024, compression=CompressionType.NONE)
        writer.start()
        writer.add_schema(1, "example/msg/Event", "jsonschema", b"{}")
        for topic, channel_id in channel_ids.items():
            writer.add_channel(channel_id, topic, "json", 1)
        for sequence, (topic, log_time, payload) in enumerate(events):
            writer.add_message(channel_ids[topic], log_time, payload.encode(), log_time, sequence)
            writer._submit_or_write_chunk()
            writer.chunk_builder.reset()
        writer.finish()


def _messages(path: Path) -> list[tuple[str, int, int]]:
    channels: dict[int, Channel] = {}
    messages: list[tuple[str, int, int]] = []
    with path.open("rb") as stream:
        for record in stream_reader(stream):
            if isinstance(record, Channel):
                channels[record.id] = record
            elif isinstance(record, Message):
                messages.append(
                    (channels[record.channel_id].topic, record.log_time, record.sequence)
                )
    return messages


def test_split_paired_windows_writes_disjoint_inclusive_outputs(tmp_path: Path) -> None:
    source = tmp_path / "input.mcap"
    _write_events(
        source,
        [
            ("/data", 0, "{}"),
            ("/events/start", 1, '{"data": true}'),
            ("/data", 2, "{}"),
            ("/events/stop", 3, '{"data": true}'),
            ("/data", 4, "{}"),
            ("/events/start", 5, '{"data": true}'),
            ("/data", 6, "{}"),
            ("/events/stop", 7, '{"data": true}'),
            ("/data", 8, "{}"),
        ],
    )

    result = split(
        str(source),
        window_start="/events/start{data == true}",
        window_end="/events/stop{data == true}",
        output_template=str(tmp_path / "window_{index:03d}.mcap"),
        compression="none",
    )

    assert result == 0
    assert _messages(tmp_path / "window_000.mcap") == [
        ("/events/start", 1, 1),
        ("/data", 2, 2),
        ("/events/stop", 3, 3),
    ]
    assert _messages(tmp_path / "window_001.mcap") == [
        ("/events/start", 5, 5),
        ("/data", 6, 6),
        ("/events/stop", 7, 7),
    ]


def test_split_paired_windows_same_topic_and_timestamp_preserves_file_order(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.mcap"
    _write_events(
        source,
        [
            ("/events", 1, '{"kind": "outside-before"}'),
            ("/events", 1, '{"kind": "start"}'),
            ("/data", 1, "{}"),
            ("/events", 1, '{"kind": "stop"}'),
            ("/events", 1, '{"kind": "outside-after"}'),
        ],
    )

    result = split(
        str(source),
        window_start='/events{kind == "start"}',
        window_end='/events{kind == "stop"}',
        output_template=str(tmp_path / "same.mcap"),
        compression="none",
    )

    assert result == 0
    assert [sequence for _topic, _time, sequence in _messages(tmp_path / "same.mcap")] == [
        1,
        2,
        3,
    ]


def test_split_paired_windows_exposes_window_template_fields(tmp_path: Path) -> None:
    source = tmp_path / "input.mcap"
    _write_events(
        source,
        [
            ("/start", 10, '{"data": true}'),
            ("/stop", 20, '{"data": true}'),
        ],
    )

    result = split(
        str(source),
        window_start="/start{data == true}",
        window_end="/stop{data == true}",
        output_template=str(tmp_path / "window_{window_start}_{window_end}.mcap"),
        compression="none",
    )

    assert result == 0
    assert (tmp_path / "window_10_20.mcap").is_file()


def test_pair_events_policies_are_explicit() -> None:
    orphan = [BoundaryEvent("stop", 1, "/stop")]
    nested = [
        BoundaryEvent("start", 1, "/start"),
        BoundaryEvent("start", 2, "/start"),
        BoundaryEvent("stop", 3, "/stop"),
    ]
    unclosed = [BoundaryEvent("start", 1, "/start")]

    with pytest.raises(ValueError, match="orphan"):
        _pair_events(
            orphan,
            minimum_duration_ns=None,
            maximum_duration_ns=None,
            orphan_stop="error",
            nested_start="error",
            unclosed_window="error",
            invalid_window="error",
        )
    assert (
        _pair_events(
            orphan,
            minimum_duration_ns=None,
            maximum_duration_ns=None,
            orphan_stop="ignore",
            nested_start="error",
            unclosed_window="error",
            invalid_window="error",
        ).windows
        == ()
    )
    with pytest.raises(ValueError, match="nested"):
        _pair_events(
            nested,
            minimum_duration_ns=None,
            maximum_duration_ns=None,
            orphan_stop="error",
            nested_start="error",
            unclosed_window="error",
            invalid_window="error",
        )
    ignored = _pair_events(
        nested,
        minimum_duration_ns=None,
        maximum_duration_ns=None,
        orphan_stop="error",
        nested_start="ignore",
        unclosed_window="error",
        invalid_window="error",
    )
    dropped = _pair_events(
        nested,
        minimum_duration_ns=None,
        maximum_duration_ns=None,
        orphan_stop="error",
        nested_start="drop",
        unclosed_window="error",
        invalid_window="error",
    )
    assert (ignored.windows[0].start_time, ignored.windows[0].end_time) == (1, 3)
    assert (dropped.windows[0].start_time, dropped.windows[0].end_time) == (2, 3)
    with pytest.raises(ValueError, match="unclosed"):
        _pair_events(
            unclosed,
            minimum_duration_ns=None,
            maximum_duration_ns=None,
            orphan_stop="error",
            nested_start="error",
            unclosed_window="error",
            invalid_window="error",
        )


def test_pair_events_duration_error_or_drop() -> None:
    events = [BoundaryEvent("start", 0, "/start"), BoundaryEvent("stop", 5, "/stop")]

    with pytest.raises(ValueError, match="invalid duration"):
        _pair_events(
            events,
            minimum_duration_ns=10,
            maximum_duration_ns=None,
            orphan_stop="error",
            nested_start="error",
            unclosed_window="error",
            invalid_window="error",
        )
    assert (
        _pair_events(
            events,
            minimum_duration_ns=10,
            maximum_duration_ns=None,
            orphan_stop="error",
            nested_start="error",
            unclosed_window="error",
            invalid_window="drop",
        ).windows
        == ()
    )

    with pytest.raises(ValueError, match="invalid duration"):
        _pair_events(
            events,
            minimum_duration_ns=None,
            maximum_duration_ns=4,
            orphan_stop="error",
            nested_start="error",
            unclosed_window="error",
            invalid_window="error",
        )


def test_boundary_matcher_rejects_non_boolean_primitive() -> None:
    matcher = BoundaryMatcher("/start.data", "/stop.data")

    with pytest.raises(ValueError, match="must evaluate to true or false"):
        matcher.match("/start", {"data": "yes"}, 1, 1)


def test_split_paired_windows_rejects_unsupported_event_encoding(tmp_path: Path) -> None:
    source = tmp_path / "input.mcap"
    output = tmp_path / "window.mcap"
    with source.open("wb") as stream:
        writer = McapWriter(stream, compression=CompressionType.NONE)
        writer.start()
        writer.add_schema(1, "example/Event", "opaque", b"")
        writer.add_channel(1, "/start", "opaque", 1)
        writer.add_message(1, 1, b"event", 1)
        writer.finish()

    result = split(
        str(source),
        window_start="/start{data == true}",
        window_end="/stop{data == true}",
        output_template=str(output),
    )

    assert result == 1
    assert not output.exists()


def test_split_paired_windows_removes_new_outputs_when_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.mcap"
    output = tmp_path / "window.mcap"
    _write_events(
        source,
        [
            ("/start", 1, '{"data": true}'),
            ("/data", 2, "{}"),
            ("/stop", 3, '{"data": true}'),
        ],
    )
    identity = split_cmd._source_identity(source)
    calls = 0

    def changing_identity(path: Path) -> split_cmd._SourceIdentity:
        nonlocal calls
        assert path == source
        calls += 1
        return identity if calls < 3 else replace(identity, size=identity.size + 1)

    monkeypatch.setattr(split_cmd, "_source_identity", changing_identity)

    result = split_cmd.split(
        str(source),
        window_start="/start{data == true}",
        window_end="/stop{data == true}",
        output_template=str(output),
        compression="none",
    )

    assert result == 1
    assert calls == 3
    assert not output.exists()


def test_split_paired_windows_copies_inside_chunks_and_skips_outside_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.mcap"
    output = tmp_path / "window.mcap"
    _write_one_message_chunks(
        source,
        [
            ("/data", 0, "{}"),
            ("/start", 1, '{"data": true}'),
            ("/data", 2, "{}"),
            ("/stop", 3, '{"data": true}'),
            ("/data", 4, "{}"),
        ],
    )
    run_processor_multi = split_cmd.run_processor_multi
    results = []

    def capture_result(**kwargs):
        result = run_processor_multi(**kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(split_cmd, "run_processor_multi", capture_result)

    result = split_cmd.split(
        str(source),
        window_start="/start{data == true}",
        window_end="/stop{data == true}",
        output_template=str(output),
        compression="none",
    )

    assert result == 0
    assert _messages(output) == [("/start", 1, 1), ("/data", 2, 2), ("/stop", 3, 3)]
    stats = results[0].stats
    assert stats.chunks_processed == 5
    assert stats.chunks_decoded == 2
    assert stats.chunks_copied == 1


def test_split_paired_windows_rejects_duplicate_output_mapping(tmp_path: Path) -> None:
    source = tmp_path / "input.mcap"
    output = tmp_path / "window.mcap"
    _write_events(
        source,
        [
            ("/start", 1, '{"data": true}'),
            ("/stop", 2, '{"data": true}'),
            ("/start", 3, '{"data": true}'),
            ("/stop", 4, '{"data": true}'),
        ],
    )

    result = split(
        str(source),
        window_start="/start{data == true}",
        window_end="/stop{data == true}",
        output_template=str(output),
        compression="none",
        force=True,
    )

    assert result == 1
    assert not output.exists()


def test_split_paired_windows_rejects_reported_processing_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.mcap"
    output = tmp_path / "window.mcap"
    _write_events(
        source,
        [
            ("/start", 1, '{"data": true}'),
            ("/stop", 2, '{"data": true}'),
        ],
    )
    run_processor_multi = split_cmd.run_processor_multi

    def report_error(**kwargs):
        result = run_processor_multi(**kwargs)
        result.stats.errors_encountered += 1
        return result

    monkeypatch.setattr(split_cmd, "run_processor_multi", report_error)

    result = split_cmd.split(
        str(source),
        window_start="/start{data == true}",
        window_end="/stop{data == true}",
        output_template=str(output),
        compression="none",
    )

    assert result == 1
    assert not output.exists()


def test_split_paired_windows_preserves_existing_output_on_late_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.mcap"
    output = tmp_path / "window.mcap"
    output.write_bytes(b"existing")
    _write_events(
        source,
        [
            ("/start", 1, '{"data": true}'),
            ("/stop", 2, '{"data": true}'),
        ],
    )
    run_processor_multi = split_cmd.run_processor_multi

    def report_error(**kwargs):
        result = run_processor_multi(**kwargs)
        result.stats.errors_encountered += 1
        return result

    monkeypatch.setattr(split_cmd, "run_processor_multi", report_error)

    result = split_cmd.split(
        str(source),
        window_start="/start{data == true}",
        window_end="/stop{data == true}",
        output_template=str(output),
        compression="none",
        force=True,
    )

    assert result == 1
    assert output.read_bytes() == b"existing"
    assert list(tmp_path.glob("*.pymcap-partial-*")) == []


def test_split_paired_windows_preserves_decodable_ros2_payloads(tmp_path: Path) -> None:
    source = tmp_path / "input.mcap"
    output = tmp_path / "window.mcap"
    cdr_true = b"\x00\x01\x00\x00\x01"
    cdr_false = b"\x00\x01\x00\x00\x00"
    with source.open("wb") as stream:
        writer = McapWriter(stream, compression=CompressionType.ZSTD)
        writer.start()
        writer.add_schema(1, "example/msg/Event", "ros2msg", b"bool data")
        writer.add_channel(1, "/start", "cdr", 1)
        writer.add_channel(2, "/data", "cdr", 1)
        writer.add_channel(3, "/stop", "cdr", 1)
        writer.add_message(1, 1, cdr_true, 1, 1)
        writer.add_message(2, 2, cdr_false, 2, 2)
        writer.add_message(3, 3, cdr_true, 3, 3)
        writer.finish()

    result = split(
        str(source),
        window_start="/start{data == true}",
        window_end="/stop{data == true}",
        output_template=str(output),
    )

    assert result == 0
    schemas: dict[int, Schema] = {}
    channels: dict[int, Channel] = {}
    decoded_values: list[bool] = []
    with output.open("rb") as stream:
        for record in stream_reader(stream):
            if isinstance(record, Schema):
                schemas[record.id] = record
            elif isinstance(record, Channel):
                channels[record.id] = record
            elif isinstance(record, Message):
                channel = channels[record.channel_id]
                decoder = Ros2DecoderFactory().decoder_for(
                    channel.message_encoding,
                    schemas[channel.schema_id],
                )
                assert decoder is not None
                decoded_values.append(decoder(record.data).data)

    assert decoded_values == [True, False, True]
