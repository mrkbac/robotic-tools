from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from pymcap_cli.core.processors.base import (
    ChannelContext,
    ChunkContext,
    InputContext,
    MessageContext,
    PipelineContext,
)
from pymcap_cli.utils import read_info
from small_mcap import InvalidMagicError, LazyChunk, McapError

if TYPE_CHECKING:
    from pathlib import Path


def validate_mcap_output(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            read_info(stream)
    except (McapError, InvalidMagicError, OSError, AssertionError):
        return False
    return True


def mcap_message_count(path: Path) -> int | None:
    try:
        with path.open("rb") as stream:
            statistics = read_info(stream).summary.statistics
    except (McapError, InvalidMagicError, OSError, AssertionError):
        return None
    return statistics.message_count if statistics is not None else None


def lazy_chunk(start: int, end: int) -> LazyChunk:
    return LazyChunk(
        message_start_time=start,
        message_end_time=end,
        uncompressed_size=0,
        uncompressed_crc=0,
        compression="none",
        record_start=0,
        data_len=0,
    )


def channel_context(channel, *, stream_id: int = 0) -> ChannelContext:
    return ChannelContext(
        input=input_context(stream_id=stream_id),
        input_channel_id=channel.id,
    )


def chunk_context(indexes=(), *, stream_id: int = 0) -> ChunkContext:
    return ChunkContext(
        input=input_context(stream_id=stream_id),
        message_indexes=tuple(indexes) if indexes else None,
    )


def message_context(message, *, stream_id: int = 0) -> MessageContext:
    return MessageContext(
        input=input_context(stream_id=stream_id),
        input_channel_id=message.channel_id,
    )


def input_context(*, stream_id: int = 0, summary=None) -> InputContext:
    return InputContext(
        stream_id=stream_id,
        summary=summary,
        statistics=summary.statistics if summary is not None else None,
        chunk_indexes=tuple(summary.chunk_indexes)
        if summary is not None and summary.chunk_indexes
        else None,
        remap_channel=lambda channel: channel,
        remap_message=lambda message: message,
        register_channel=lambda channel, _source_channel_id=None: channel,
        register_schema=lambda *_: 0,
    )


def pipeline_context(summaries=()) -> PipelineContext:
    return PipelineContext(
        inputs=tuple(
            input_context(stream_id=i, summary=summary) for i, summary in enumerate(summaries)
        ),
        output_segments=(),
    )


def empty_processor_result(
    segments: dict[int, SimpleNamespace] | None = None,
    *,
    message_count: int = 0,
    errors_encountered: int = 0,
    validation_errors: int = 0,
) -> SimpleNamespace:
    output_segments = segments if segments is not None else {0: SimpleNamespace(chunk_groups=[])}
    return SimpleNamespace(
        stats=SimpleNamespace(
            messages_processed=message_count,
            errors_encountered=errors_encountered,
            validation_errors=validation_errors,
            writer_statistics=SimpleNamespace(
                message_count=message_count,
                message_start_time=0,
                message_end_time=0,
            ),
        ),
        processor=SimpleNamespace(
            output_manager=SimpleNamespace(segments=output_segments),
        ),
    )
