from __future__ import annotations

from types import SimpleNamespace

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


def empty_processor_result(
    segments: dict[int, SimpleNamespace] | None = None,
) -> SimpleNamespace:
    output_segments = segments if segments is not None else {0: SimpleNamespace(rechunk_groups=[])}
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
            large_channels=[],
            output_manager=SimpleNamespace(segments=output_segments),
        ),
    )
