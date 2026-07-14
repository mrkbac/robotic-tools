"""E2E tests for the split command."""

from pathlib import Path

import pytest
from pymcap_cli.constants import DEFAULT_CHUNK_SIZE
from pymcap_cli.core.mcap_processor import (
    InputFile,
    InputOptions,
    McapProcessor,
    OutputOptions,
    ProcessingOptions,
)
from pymcap_cli.core.processors.chunk_groupers import PerChannelGrouper
from pymcap_cli.core.processors.duration_split import DurationSplitProcessor
from pymcap_cli.core.processors.expression_split import ExpressionSplitProcessor
from pymcap_cli.core.processors.size_split import SizeSplitProcessor
from pymcap_cli.core.processors.timestamp_split import TimestampSplitProcessor
from small_mcap import stream_reader


def _write_mcap(data: bytes, path: Path) -> None:
    path.write_bytes(data)


def _read_mcap(path: Path) -> dict:
    """Read an MCAP file and return message counts per channel."""
    channel_topics = {}
    message_counts = {}
    with path.open("rb") as f:
        for record in stream_reader(f):
            if hasattr(record, "topic"):
                channel_topics[record.id] = record.topic
                message_counts[record.id] = 0
            elif hasattr(record, "channel_id"):
                message_counts[record.channel_id] = message_counts.get(record.channel_id, 0) + 1
    return {"channel_topics": channel_topics, "message_counts": message_counts}


@pytest.mark.e2e
class TestDurationSplit:
    """Test duration-based splitting."""

    def test_split_into_multiple_files(self, multi_topic_mcap: Path, tmp_path: Path):
        """Test that duration split creates multiple output files."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=file_size,
                    options=InputOptions.from_args(),
                )
            ]

            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    routers=[DurationSplitProcessor(duration_ns=100_000_000)],
                    output_template=str(tmp_path / "output_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=DEFAULT_CHUNK_SIZE,
                ),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream=None)

        # Verify multiple segments were created
        assert len(processor.output_manager.segments) > 1

        # Verify total messages match
        assert stats.writer_statistics.message_count > 0

    def test_split_preserves_all_messages(self, multi_topic_mcap: Path, tmp_path: Path):
        """Test that all messages are preserved across split files."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=file_size,
                    options=InputOptions.from_args(),
                )
            ]

            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    routers=[DurationSplitProcessor(duration_ns=50_000_000)],
                    output_template=str(tmp_path / "seg_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=DEFAULT_CHUNK_SIZE,
                ),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream=None)

        # Count messages in each output file
        total_messages = 0
        for segment in processor.output_manager.segments.values():
            total_messages += segment.writer.statistics.message_count

        # All messages should be preserved
        assert total_messages == stats.writer_statistics.message_count

    def test_split_file_naming(self, multi_topic_mcap: Path, tmp_path: Path):
        """Test that output files are named correctly."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=file_size,
                    options=InputOptions.from_args(),
                )
            ]

            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    routers=[DurationSplitProcessor(duration_ns=100_000_000)],
                    output_template=str(tmp_path / "output_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=DEFAULT_CHUNK_SIZE,
                ),
            )

            processor = McapProcessor(options)
            processor.process(output_stream=None)

        # Verify files exist with expected naming
        for segment in processor.output_manager.segments.values():
            assert Path(segment.path).exists()


@pytest.mark.e2e
class TestTimestampSplit:
    """Test timestamp-based splitting."""

    def test_split_at_timestamps(self, multi_topic_mcap: Path, tmp_path: Path):
        """Test splitting at specific timestamps."""
        file_size = multi_topic_mcap.stat().st_size

        # Split at 50ms and 100ms (50_000_000 and 100_000_000 ns)
        with multi_topic_mcap.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=file_size,
                    options=InputOptions.from_args(),
                )
            ]

            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    routers=[TimestampSplitProcessor(split_points=[50_000_000, 100_000_000])],
                    output_template=str(tmp_path / "seg_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=DEFAULT_CHUNK_SIZE,
                ),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream=None)

        # Should create 3 segments (before 50ms, 50-100ms, after 100ms)
        assert len(processor.output_manager.segments) == 3
        assert stats.writer_statistics.message_count > 0

    def test_combined_duration_and_timestamp_split(self, multi_topic_mcap: Path, tmp_path: Path):
        """Combined split processors should route by the intersection of both boundary sets."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=file_size,
                    options=InputOptions.from_args(),
                )
            ]

            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    routers=[
                        DurationSplitProcessor(duration_ns=100_000_000),
                        TimestampSplitProcessor(split_points=[125_000_000]),
                    ],
                    output_template=str(tmp_path / "seg_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=DEFAULT_CHUNK_SIZE,
                ),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream=None)

        assert len(processor.output_manager.segments) == 3
        assert (
            sum(
                segment.writer.statistics.message_count
                for segment in processor.output_manager.segments.values()
            )
            == stats.writer_statistics.message_count
        )


@pytest.mark.e2e
class TestSizeSplit:
    """Test size-budget-based splitting."""

    def test_size_split_fast_copy_routes_chunks_to_multiple_segments(self, tmp_path: Path) -> None:
        # Build a small fixture with several distinct chunks so fast-copy
        # segment routing has something to split on.
        from tests.fixtures.mcap_generator import create_multi_topic_mcap  # noqa: PLC0415

        data = create_multi_topic_mcap(
            topics=["/a", "/b", "/c"],
            messages_per_topic=200,
            chunk_size=8192,
        )
        source = tmp_path / "input.mcap"
        source.write_bytes(data)

        # Each input chunk uncompresses to ~8KB; a 5KB budget forces a new
        # segment after each chunk.
        budget = 5000

        with source.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=source.stat().st_size,
                    options=InputOptions.from_args(),
                )
            ]
            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    routers=[SizeSplitProcessor(budget)],
                    output_template=str(tmp_path / "shard_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=DEFAULT_CHUNK_SIZE,
                ),
            )
            processor = McapProcessor(options)
            stats = processor.process(output_stream=None)

        assert len(processor.output_manager.segments) > 1
        total = sum(
            segment.writer.statistics.message_count
            for segment in processor.output_manager.segments.values()
        )
        assert total == stats.writer_statistics.message_count


@pytest.mark.e2e
class TestExpressionSplit:
    """Test expression-based splitting via ros-parser message paths."""

    def test_split_by_field_value(self, multi_topic_mcap: Path, tmp_path: Path):
        """Splitting by a JSON field value produces one segment per distinct value."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=file_size,
                    options=InputOptions.from_args(),
                )
            ]

            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    routers=[ExpressionSplitProcessor("/camera/front.msg")],
                    output_template=str(tmp_path / "msg_{key}.mcap"),
                    compression="zstd",
                    chunk_size=DEFAULT_CHUNK_SIZE,
                ),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream=None)

        # /camera/front carries 50 distinct "msg" values (0..49).
        assert len(processor.output_manager.segments) > 1
        assert stats.writer_statistics.message_count > 0

    def test_split_preserves_all_messages(self, multi_topic_mcap: Path, tmp_path: Path):
        """Sticky routing keeps every message: sum across segments == total."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=file_size,
                    options=InputOptions.from_args(),
                )
            ]

            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    routers=[ExpressionSplitProcessor("/camera/front.msg")],
                    output_template=str(tmp_path / "ch_{key}.mcap"),
                    compression="zstd",
                    chunk_size=DEFAULT_CHUNK_SIZE,
                ),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream=None)

        total = sum(
            s.writer.statistics.message_count for s in processor.output_manager.segments.values()
        )
        assert total == stats.writer_statistics.message_count

    def _write_noisy_bool_mcap(self, path: Path, *, samples: list[tuple[int, str]]) -> int:
        """Write a tiny JSON MCAP at ``path`` with one ``/state`` topic.

        ``samples`` is a list of ``(log_time_ns, value)`` tuples where ``value``
        is the string written into the ``msg`` field.
        Returns the file size in bytes.
        """
        import io  # noqa: PLC0415

        from small_mcap import CompressionType, McapWriter  # noqa: PLC0415

        buf = io.BytesIO()
        writer = McapWriter(buf, compression=CompressionType.NONE)
        writer.start()
        writer.add_schema(schema_id=1, name="t", encoding="json", data=b"{}")
        writer.add_channel(channel_id=1, topic="/state", message_encoding="json", schema_id=1)
        for log_time_ns, value in samples:
            payload = f'{{"msg": "{value}"}}'.encode()
            writer.add_message(
                channel_id=1, log_time=log_time_ns, publish_time=log_time_ns, data=payload
            )
        writer.finish()
        path.write_bytes(buf.getvalue())
        return path.stat().st_size

    def _write_direction_mcap(self, path: Path, *, samples: list[tuple[int, int]]) -> int:
        import io  # noqa: PLC0415

        from small_mcap import CompressionType, McapWriter  # noqa: PLC0415

        buf = io.BytesIO()
        writer = McapWriter(buf, compression=CompressionType.NONE)
        writer.start()
        writer.add_schema(schema_id=1, name="t", encoding="json", data=b"{}")
        writer.add_channel(channel_id=1, topic="/state", message_encoding="json", schema_id=1)
        for log_time_ns, value in samples:
            writer.add_message(
                channel_id=1,
                log_time=log_time_ns,
                publish_time=log_time_ns,
                data=f'{{"direction":{value}}}'.encode(),
            )
        writer.finish()
        path.write_bytes(buf.getvalue())
        return path.stat().st_size

    def test_skip_value_and_typed_value_filename(self, tmp_path: Path):
        from small_mcap import read_message  # noqa: PLC0415

        src = tmp_path / "in.mcap"
        size = self._write_direction_mcap(
            src,
            samples=[(0, 0), (1, 1), (2, 1), (3, 0), (4, -1), (5, -1)],
        )

        with src.open("rb") as input_stream:
            options = ProcessingOptions(
                inputs=[
                    InputFile(stream=input_stream, size=size, options=InputOptions.from_args())
                ],
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    routers=[
                        ExpressionSplitProcessor(
                            "/state.direction",
                            skip_values=(0,),
                            require_value=True,
                        )
                    ],
                    output_template=str(tmp_path / "drive_{value:+d}_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=DEFAULT_CHUNK_SIZE,
                ),
            )
            processor = McapProcessor(options)
            processor.process(output_stream=None)

        paths = sorted(tmp_path.glob("drive_*.mcap"))
        assert [path.name for path in paths] == ["drive_+1_000.mcap", "drive_-1_001.mcap"]

        times_by_name: dict[str, list[int]] = {}
        for path in paths:
            with path.open("rb") as stream:
                times_by_name[path.name] = [
                    message.log_time for _schema, _channel, message in read_message(stream)
                ]
        assert times_by_name == {
            "drive_+1_000.mcap": [1, 2],
            "drive_-1_001.mcap": [4, 5],
        }

    def test_hysteresis_count_suppresses_flapping(self, tmp_path: Path):
        """A flapping signal under the count threshold produces a single segment."""
        src = tmp_path / "in.mcap"
        # 3-count threshold; the value flaps A→B→A→B→A so no run of 3 betas.
        size = self._write_noisy_bool_mcap(
            src,
            samples=[
                (0, "alpha"),
                (1, "beta"),
                (2, "alpha"),
                (3, "beta"),
                (4, "alpha"),
            ],
        )

        with src.open("rb") as input_stream:
            options = ProcessingOptions(
                inputs=[
                    InputFile(stream=input_stream, size=size, options=InputOptions.from_args())
                ],
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    routers=[ExpressionSplitProcessor("/state.msg", hysteresis_count=3)],
                    output_template=str(tmp_path / "seg_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=DEFAULT_CHUNK_SIZE,
                ),
            )
            processor = McapProcessor(options)
            stats = processor.process(output_stream=None)

        # All 5 messages collapse into a single segment.
        assert len(processor.output_manager.segments) == 1
        assert stats.writer_statistics.message_count == 5

    def test_hysteresis_count_commits_after_threshold(self, tmp_path: Path):
        """A sustained value run beyond the threshold creates a new segment."""
        src = tmp_path / "in.mcap"
        size = self._write_noisy_bool_mcap(
            src,
            samples=[
                (0, "alpha"),
                (1, "beta"),
                (2, "beta"),
                (3, "beta"),  # threshold met → segment 1
                (4, "beta"),
            ],
        )

        with src.open("rb") as input_stream:
            options = ProcessingOptions(
                inputs=[
                    InputFile(stream=input_stream, size=size, options=InputOptions.from_args())
                ],
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    routers=[ExpressionSplitProcessor("/state.msg", hysteresis_count=3)],
                    output_template=str(tmp_path / "seg_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=DEFAULT_CHUNK_SIZE,
                ),
            )
            McapProcessor(options).process(output_stream=None)

        assert len(list(tmp_path.glob("seg_*.mcap"))) == 2

    def test_trailing_context_duplicates_target_messages(self, tmp_path: Path):
        """``trailing_context_count`` duplicates target msgs into the prev segment."""
        from small_mcap import read_message  # noqa: PLC0415

        src = tmp_path / "in.mcap"
        # Transition at t=2 (first beta) → segment 1. The next two beta
        # messages (t=2 and t=3 themselves, not subsequent ones) duplicate
        # back into segment 0 via extra output routes.
        size = self._write_noisy_bool_mcap(
            src,
            samples=[
                (0, "alpha"),
                (1, "alpha"),
                (2, "beta"),
                (3, "beta"),
                (4, "beta"),
                (5, "beta"),
                (6, "beta"),
            ],
        )

        with src.open("rb") as input_stream:
            options = ProcessingOptions(
                inputs=[
                    InputFile(stream=input_stream, size=size, options=InputOptions.from_args())
                ],
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    routers=[ExpressionSplitProcessor("/state.msg", trailing_context_count=2)],
                    output_template=str(tmp_path / "seg_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=DEFAULT_CHUNK_SIZE,
                ),
            )
            McapProcessor(options).process(output_stream=None)

        seg_paths = sorted(tmp_path.glob("seg_*.mcap"))
        assert len(seg_paths) == 2

        def times(p: Path) -> list[int]:
            with p.open("rb") as f:
                return [m.log_time for _s, _c, m in read_message(f)]

        # Segment 0: alphas 0, 1 + the first two betas duplicated as context.
        assert times(seg_paths[0]) == [0, 1, 2, 3]
        # Segment 1: all betas land naturally in the new segment.
        assert times(seg_paths[1]) == [2, 3, 4, 5, 6]

    def test_trailing_context_combines_with_duration_split_key(self, tmp_path: Path):
        """Trailing context replaces only the expression component of tuple keys."""
        from small_mcap import read_message  # noqa: PLC0415

        src = tmp_path / "in.mcap"
        size = self._write_noisy_bool_mcap(
            src,
            samples=[
                (0, "alpha"),
                (1, "alpha"),
                (2, "beta"),
                (3, "beta"),
                (4, "beta"),
            ],
        )

        with src.open("rb") as input_stream:
            options = ProcessingOptions(
                inputs=[
                    InputFile(stream=input_stream, size=size, options=InputOptions.from_args())
                ],
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    routers=[
                        DurationSplitProcessor(duration_ns=10),
                        ExpressionSplitProcessor("/state.msg", trailing_context_count=2),
                    ],
                    output_template=str(tmp_path / "seg_{key}.mcap"),
                    compression="zstd",
                    chunk_size=DEFAULT_CHUNK_SIZE,
                ),
            )
            processor = McapProcessor(options)
            processor.process(output_stream=None)

        assert processor.output_manager is not None
        assert set(processor.output_manager.segments) == {(0, 0), (0, 1)}

        def times(key: tuple[int, int]) -> list[int]:
            with Path(processor.output_manager.segments[key].path).open("rb") as f:
                return [m.log_time for _s, _c, m in read_message(f)]

        assert times((0, 0)) == [0, 1, 2, 3]
        assert times((0, 1)) == [2, 3, 4]


@pytest.mark.e2e
class TestSplitWithRechunking:
    """Test splitting combined with rechunking."""

    def test_split_and_rechunk(self, multi_topic_mcap: Path, tmp_path: Path):
        """Test that splitting works with rechunking active."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=file_size,
                    options=InputOptions.from_args(),
                )
            ]

            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    routers=[DurationSplitProcessor(duration_ns=100_000_000)],
                    output_template=str(tmp_path / "seg_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    output_processors=[PerChannelGrouper()],
                ),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream=None)

        # Should create multiple segments with rechunking
        assert len(processor.output_manager.segments) > 1
        assert stats.writer_statistics.message_count > 0


@pytest.mark.e2e
class TestSplitWithFiltering:
    """Test splitting combined with topic/time filtering."""

    def test_split_with_topic_filter(self, multi_topic_mcap: Path, tmp_path: Path):
        """Test that splitting works with topic filtering."""
        file_size = multi_topic_mcap.stat().st_size

        with multi_topic_mcap.open("rb") as input_stream:
            input_files = [
                InputFile(
                    stream=input_stream,
                    size=file_size,
                    options=InputOptions.from_args(include_topic_regex=["/camera/.*"]),
                )
            ]

            options = ProcessingOptions(
                inputs=input_files,
                input_options=InputOptions.from_args(),
                output_options=OutputOptions(
                    routers=[DurationSplitProcessor(duration_ns=100_000_000)],
                    output_template=str(tmp_path / "camera_{index:03d}.mcap"),
                    compression="zstd",
                    chunk_size=DEFAULT_CHUNK_SIZE,
                ),
            )

            processor = McapProcessor(options)
            stats = processor.process(output_stream=None)

        # Should only include camera topics
        assert stats.writer_statistics.channel_count <= 2
        assert stats.writer_statistics.message_count > 0
