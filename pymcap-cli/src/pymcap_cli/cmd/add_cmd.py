"""Add command for pymcap-cli — attach attachments or metadata to an MCAP file."""

import logging
import mimetypes
import time
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from small_mcap import (
    Attachment,
    Channel,
    Message,
    Metadata,
    Schema,
    get_header,
)
from small_mcap.reader import stream_reader

from pymcap_cli.cmd._cli_options import (
    ChunkSizeOption,
    CompressionOption,
    ForceOverwriteOption,
    OutputPathOption,
)
from pymcap_cli.cmd._run_processor import finalize_replace_source
from pymcap_cli.constants import DEFAULT_CHUNK_SIZE, DEFAULT_COMPRESSION
from pymcap_cli.core.input_handler import open_input
from pymcap_cli.types.types_manual import CompressionName
from pymcap_cli.utils import (
    McapWriterOptions,
    confirm_output_overwrite,
    create_mcap_writer,
    output_overwrites_input,
)

logger = logging.getLogger(__name__)

add_app = App(help="Add an attachment or metadata record to an MCAP file")


def _rewrite_with_extra(
    input_path: str,
    output_path: Path,
    *,
    extra_attachment: Attachment | None = None,
    extra_metadata: Metadata | None = None,
    compression: CompressionName,
    chunk_size: int,
) -> None:
    """Copy every record from ``input_path`` in stored order, then append one record."""
    with open_input(input_path) as (stream, _size):
        header = get_header(stream)
        stream.seek(0)
        with output_path.open("wb") as out:
            writer = create_mcap_writer(
                out,
                McapWriterOptions(chunk_size=chunk_size, compression=compression),
            )
            writer.start(profile=header.profile, library=header.library)
            written_schemas: set[int] = set()
            written_channels: set[int] = set()
            for record in stream_reader(stream):
                if isinstance(record, Message):
                    writer.add_message(
                        record.channel_id,
                        record.log_time,
                        record.data,
                        record.publish_time,
                        record.sequence,
                    )
                elif isinstance(record, Channel):
                    if record.id not in written_channels:
                        writer.add_channel(
                            record.id,
                            record.topic,
                            record.message_encoding,
                            record.schema_id,
                            record.metadata,
                        )
                        written_channels.add(record.id)
                elif isinstance(record, Schema):
                    if record.id not in written_schemas:
                        writer.add_schema(record.id, record.name, record.encoding, record.data)
                        written_schemas.add(record.id)
                elif isinstance(record, Attachment):
                    writer.add_attachment(
                        record.log_time,
                        record.create_time,
                        record.name,
                        record.media_type,
                        record.data,
                    )
                elif isinstance(record, Metadata):
                    writer.add_metadata(record.name, record.metadata)
            if extra_attachment is not None:
                writer.add_attachment(
                    extra_attachment.log_time,
                    extra_attachment.create_time,
                    extra_attachment.name,
                    extra_attachment.media_type,
                    extra_attachment.data,
                )
            if extra_metadata is not None:
                writer.add_metadata(extra_metadata.name, extra_metadata.metadata)
            writer.finish()


def _resolve_target(input_path: str, output: Path | None, force: bool) -> tuple[Path, bool] | None:
    """Return (write_path, is_in_place) or None if the target cannot be written."""
    if output is not None:
        if output_overwrites_input(input_path, output):
            logger.error(
                "Output path is the same file as the input. "
                "Omit --output to add the record in place safely."
            )
            return None
        if output.exists() and not force:
            confirm_output_overwrite(output, force=False)
        return output, False
    source = Path(input_path)
    if not source.exists():
        logger.error(f"File not found: {source}")
        return None
    return source.with_name(f"{source.name}.add.tmp"), True


def attachment(
    mcap: str,
    *,
    data: Annotated[
        Path,
        Parameter(name=["--file"], help="Path to the file whose bytes become the attachment."),
    ],
    name: Annotated[
        str | None,
        Parameter(name=["--name"], help="Attachment name (defaults to the source filename)."),
    ] = None,
    media_type: Annotated[
        str | None,
        Parameter(name=["--media-type"], help="MIME type (guessed from the name if omitted)."),
    ] = None,
    log_time: Annotated[
        int | None,
        Parameter(name=["--log-time"], help="Attachment log time in nanoseconds (default: now)."),
    ] = None,
    create_time: Annotated[
        int | None,
        Parameter(
            name=["--create-time"],
            help="Attachment creation time in nanoseconds (default: --log-time).",
        ),
    ] = None,
    output: OutputPathOption | None = None,
    force: ForceOverwriteOption = False,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
) -> int:
    """Add an attachment to an MCAP file.

    Without --output the file is modified in place (stored message order is
    preserved). The attachment record is written after the existing records.

    Parameters
    ----------
    mcap
        Path to the MCAP file to modify (local file only).
    """
    if not data.exists():
        logger.error(f"Attachment source not found: {data}")
        return 1

    resolved = _resolve_target(mcap, output, force)
    if resolved is None:
        return 1
    write_path, is_in_place = resolved

    now = time.time_ns()
    att_log_time = log_time if log_time is not None else now
    att_create_time = create_time if create_time is not None else att_log_time
    att_name = name if name is not None else data.name
    att_media_type = media_type or mimetypes.guess_type(att_name)[0] or "application/octet-stream"

    record = Attachment(
        log_time=att_log_time,
        create_time=att_create_time,
        name=att_name,
        media_type=att_media_type,
        data=data.read_bytes(),
    )

    try:
        _rewrite_with_extra(
            mcap,
            write_path,
            extra_attachment=record,
            compression=compression,
            chunk_size=chunk_size,
        )
    except Exception:
        logger.exception("Error while adding attachment")
        write_path.unlink(missing_ok=True)
        return 1

    if is_in_place:
        result = finalize_replace_source(source=Path(mcap), tmp_output=write_path)
        if result != 0:
            return result
    logger.info(f"[green]✓ Added attachment {att_name!r}[/green]")
    return 0


def metadata(
    mcap: str,
    *,
    name: Annotated[str, Parameter(name=["--name"], help="Metadata record name.")],
    key: Annotated[
        list[str] | None,
        Parameter(name=["--key"], help="A key=value pair (repeatable)."),
    ] = None,
    output: OutputPathOption | None = None,
    force: ForceOverwriteOption = False,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
) -> int:
    """Add a metadata record to an MCAP file.

    Without --output the file is modified in place (stored message order is
    preserved). The metadata record is written after the existing records.

    Parameters
    ----------
    mcap
        Path to the MCAP file to modify (local file only).
    """
    pairs: dict[str, str] = {}
    for entry in key or []:
        if "=" not in entry:
            logger.error(f"Invalid --key {entry!r}; expected key=value.")
            return 1
        k, v = entry.split("=", 1)
        pairs[k] = v
    if not pairs:
        logger.error("At least one --key key=value is required.")
        return 1

    resolved = _resolve_target(mcap, output, force)
    if resolved is None:
        return 1
    write_path, is_in_place = resolved

    record = Metadata(name=name, metadata=pairs)

    try:
        _rewrite_with_extra(
            mcap,
            write_path,
            extra_metadata=record,
            compression=compression,
            chunk_size=chunk_size,
        )
    except Exception:
        logger.exception("Error while adding metadata")
        write_path.unlink(missing_ok=True)
        return 1

    if is_in_place:
        result = finalize_replace_source(source=Path(mcap), tmp_output=write_path)
        if result != 0:
            return result
    logger.info(f"[green]✓ Added metadata {name!r}[/green]")
    return 0


add_app.command(attachment, name="attachment")
add_app.command(metadata, name="metadata")
