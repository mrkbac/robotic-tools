from __future__ import annotations

from types import SimpleNamespace

from pymcap_cli.core.processors.base import (
    ChannelContext,
    ChunkContext,
    InputContext,
    MessageContext,
    PipelineContext,
)
from small_mcap import LazyChunk


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
        register_channel=lambda channel: channel,
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
) -> SimpleNamespace:
    output_segments = segments if segments is not None else {0: SimpleNamespace(chunk_groups=[])}
    return SimpleNamespace(
        stats=SimpleNamespace(
            messages_processed=0,
            writer_statistics=SimpleNamespace(
                message_count=0,
                message_start_time=0,
                message_end_time=0,
            ),
        ),
        processor=SimpleNamespace(
            output_manager=SimpleNamespace(segments=output_segments),
        ),
    )
