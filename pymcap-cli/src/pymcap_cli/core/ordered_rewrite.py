"""Message-ordered MCAP rewrite shared by the ``sort`` command and ``--order``.

The rewrite buffers every message in memory, reorders by the requested key with a
stable sort (so equal keys keep their stored order), and writes a fresh file.
Schemas, channels, attachments, and metadata are preserved; attachment and
metadata records keep their original relative order and are written after the
messages. This mirrors the semantics of ``mcap sort``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from small_mcap import (
    Attachment,
    Channel,
    Message,
    Metadata,
    Schema,
    get_header,
)
from small_mcap.reader import stream_reader

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.utils import McapWriterOptions, create_mcap_writer

if TYPE_CHECKING:
    from pathlib import Path

    from pymcap_cli.types.types_manual import CompressionName, OrderName


def rewrite_ordered(
    *,
    input_path: str,
    output_path: Path,
    order: OrderName,
    compression: CompressionName,
    chunk_size: int,
    enable_crcs: bool = True,
    use_chunking: bool = True,
) -> int:
    """Rewrite ``input_path`` to ``output_path`` with messages reordered.

    Returns the number of messages written.
    """
    schemas: dict[int, Schema] = {}
    channels: dict[int, Channel] = {}
    messages: list[Message] = []
    attachments: list[Attachment] = []
    metadata: list[Metadata] = []

    with open_input(input_path) as (stream, _size):
        header = get_header(stream)
        stream.seek(0)
        for record in stream_reader(stream):
            if isinstance(record, Message):
                messages.append(record)
            elif isinstance(record, Channel):
                channels[record.id] = record
            elif isinstance(record, Schema):
                schemas[record.id] = record
            elif isinstance(record, Attachment):
                attachments.append(record)
            elif isinstance(record, Metadata):
                metadata.append(record)

    topic_by_channel = {cid: channel.topic for cid, channel in channels.items()}
    if order == "log_time":
        messages.sort(key=lambda m: m.log_time)
    elif order == "topic":
        messages.sort(key=lambda m: (topic_by_channel.get(m.channel_id, ""), m.log_time))

    with output_path.open("wb") as out:
        writer = create_mcap_writer(
            out,
            McapWriterOptions(
                chunk_size=chunk_size,
                compression=compression,
                enable_crcs=enable_crcs,
                use_chunking=use_chunking,
            ),
        )
        writer.start(profile=header.profile, library=header.library)
        for schema in schemas.values():
            writer.add_schema(schema.id, schema.name, schema.encoding, schema.data)
        for channel in channels.values():
            writer.add_channel(
                channel.id,
                channel.topic,
                channel.message_encoding,
                channel.schema_id,
                channel.metadata,
            )
        for message in messages:
            writer.add_message(
                message.channel_id,
                message.log_time,
                message.data,
                message.publish_time,
                message.sequence,
            )
        for attachment in attachments:
            writer.add_attachment(
                attachment.log_time,
                attachment.create_time,
                attachment.name,
                attachment.media_type,
                attachment.data,
            )
        for meta in metadata:
            writer.add_metadata(meta.name, meta.metadata)
        writer.finish()

    return len(messages)


def reorder_output(
    path: Path,
    *,
    order: OrderName,
    compression: CompressionName,
    chunk_size: int,
    enable_crcs: bool = True,
    use_chunking: bool = True,
) -> int:
    """Reorder an existing MCAP file in place via a sibling temp file."""
    tmp = path.with_name(f"{path.name}.reorder.tmp")
    try:
        count = rewrite_ordered(
            input_path=str(path),
            output_path=tmp,
            order=order,
            compression=compression,
            chunk_size=chunk_size,
            enable_crcs=enable_crcs,
            use_chunking=use_chunking,
        )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return count
