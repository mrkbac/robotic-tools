import argparse
import io
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from small_mcap import InvalidMagicError

from pymcap_cli.rebuild import Info, read_info, rebuild_info
from pymcap_cli.utils import bytes_to_human

console = Console()


def add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Add the list command and its subcommands to the subparsers."""
    parser = subparsers.add_parser(
        "list",
        help="List records in an MCAP file",
        description="List records in an MCAP file",
    )

    list_subparsers = parser.add_subparsers(dest="list_subcommand", help="Record type to list")

    # list channels
    channels_parser = list_subparsers.add_parser(
        "channels",
        help="List channels in an MCAP file",
    )
    channels_parser.add_argument("file", help="Path to the MCAP file", type=str)

    # list chunks
    chunks_parser = list_subparsers.add_parser(
        "chunks",
        help="List chunks in an MCAP file",
    )
    chunks_parser.add_argument("file", help="Path to the MCAP file", type=str)

    # list schemas
    schemas_parser = list_subparsers.add_parser(
        "schemas",
        help="List schemas in an MCAP file",
    )
    schemas_parser.add_argument("file", help="Path to the MCAP file", type=str)

    # list attachments
    attachments_parser = list_subparsers.add_parser(
        "attachments",
        help="List attachments in an MCAP file",
    )
    attachments_parser.add_argument("file", help="Path to the MCAP file", type=str)

    # list metadata
    metadata_parser = list_subparsers.add_parser(
        "metadata",
        help="List metadata in an MCAP file",
    )
    metadata_parser.add_argument("file", help="Path to the MCAP file", type=str)

    return parser


def handle_command(args: argparse.Namespace) -> None:
    """Handle the list command execution by routing to appropriate subcommand."""
    if args.list_subcommand == "channels":
        handle_channels(args)
    elif args.list_subcommand == "chunks":
        handle_chunks(args)
    elif args.list_subcommand == "schemas":
        handle_schemas(args)
    elif args.list_subcommand == "attachments":
        handle_attachments(args)
    elif args.list_subcommand == "metadata":
        handle_metadata(args)
    else:
        console.print("[red]No subcommand specified. Use --help for options.[/red]")


def _read_mcap_info(file_path: str) -> Info:
    """Read MCAP file info, with automatic rebuild on invalid magic."""
    file = Path(file_path)
    file_size = file.stat().st_size

    with file.open("rb", buffering=0) as f_raw:
        f_buffered = io.BufferedReader(f_raw, buffer_size=1024)
        try:
            info = read_info(f_buffered)
        except InvalidMagicError:
            console.print("[yellow]Invalid MCAP magic, rebuilding info...[/yellow]")
            info = rebuild_info(f_buffered, file_size)

    return info


def handle_channels(args: argparse.Namespace) -> None:
    """List all channels in the MCAP file."""
    info = _read_mcap_info(args.file)
    summary = info.summary

    if not summary.channels:
        console.print("[yellow]No channels found[/yellow]")
        return

    table = Table()
    table.add_column("ID", style="green", justify="right")
    table.add_column("Schema ID", style="cyan", justify="right")
    table.add_column("Topic", style="bold white")
    table.add_column("Encoding", style="yellow")
    table.add_column("Metadata", style="blue")

    for channel_id in sorted(summary.channels.keys()):
        channel = summary.channels[channel_id]
        metadata_str = str(channel.metadata) if channel.metadata else "{}"
        table.add_row(
            str(channel.id),
            str(channel.schema_id),
            channel.topic,
            channel.message_encoding,
            metadata_str,
        )

    console.print(table)


def handle_chunks(args: argparse.Namespace) -> None:
    """List all chunks in the MCAP file."""
    info = _read_mcap_info(args.file)
    summary = info.summary

    if not summary.chunk_indexes:
        console.print("[yellow]No chunks found[/yellow]")
        return

    table = Table()
    table.add_column("Offset", style="cyan", justify="right")
    table.add_column("Length", style="yellow", justify="right")
    table.add_column("Start", style="blue", no_wrap=True)
    table.add_column("End", style="blue", no_wrap=True)
    table.add_column("Compression", style="green")
    table.add_column("Compressed Size", style="yellow", justify="right")
    table.add_column("Uncompressed Size", style="yellow", justify="right")
    table.add_column("Ratio", style="magenta", justify="right")
    table.add_column("Channel IDs", style="green")

    for chunk in summary.chunk_indexes:
        # Convert timestamps to human readable format (time only, not date)
        start_time = datetime.fromtimestamp(chunk.message_start_time / 1_000_000_000).strftime(
            "%H:%M:%S.%f"
        )[:-3]  # Trim to milliseconds
        end_time = datetime.fromtimestamp(chunk.message_end_time / 1_000_000_000).strftime(
            "%H:%M:%S.%f"
        )[:-3]

        # Calculate ratio as percentage
        ratio = (
            (chunk.compressed_size / chunk.uncompressed_size) * 100
            if chunk.uncompressed_size > 0
            else 0.0
        )

        # Extract and format channel IDs from message_index_offsets
        channel_ids = sorted(chunk.message_index_offsets.keys())
        channel_ids_str = ", ".join(str(cid) for cid in channel_ids)

        table.add_row(
            str(chunk.chunk_start_offset),
            str(chunk.chunk_length),
            start_time,
            end_time,
            chunk.compression,
            bytes_to_human(chunk.compressed_size),
            bytes_to_human(chunk.uncompressed_size),
            f"{ratio:.1f}%",
            channel_ids_str,
        )

    console.print(table)


def handle_schemas(args: argparse.Namespace) -> None:
    """List all schemas in the MCAP file."""
    info = _read_mcap_info(args.file)
    summary = info.summary

    if not summary.schemas:
        console.print("[yellow]No schemas found[/yellow]")
        return

    table = Table()
    table.add_column("ID", style="green", justify="right")
    table.add_column("Name", style="bold cyan")
    table.add_column("Encoding", style="yellow")
    table.add_column("Data", style="blue")

    schema_ids = sorted(summary.schemas.keys())
    for idx, schema_id in enumerate(schema_ids):
        schema = summary.schemas[schema_id]
        # Decode schema data if it's ROS2 message format
        try:
            schema_data_str = schema.data.decode("utf-8") if schema.data else ""
            # Format multiline data with proper indentation
            if "\n" in schema_data_str:
                schema_data_lines = schema_data_str.split("\n")
                schema_data_str = "\n".join(f"\t{line}" for line in schema_data_lines)
        except UnicodeDecodeError:
            schema_data_str = f"<binary data: {len(schema.data)} bytes>"

        # Add separator line after each schema except the last one
        is_last = idx == len(schema_ids) - 1
        table.add_row(
            str(schema.id),
            schema.name,
            schema.encoding,
            schema_data_str,
            end_section=not is_last,
        )

    console.print(table)


def handle_attachments(args: argparse.Namespace) -> None:
    """List all attachments in the MCAP file."""
    info = _read_mcap_info(args.file)
    summary = info.summary

    if not summary.attachment_indexes:
        console.print("[yellow]No attachments found[/yellow]")
        return

    table = Table()
    table.add_column("Name", style="bold white")
    table.add_column("Media Type", style="cyan")
    table.add_column("Log Time", style="yellow", justify="right")
    table.add_column("Create Time", style="yellow", justify="right")
    table.add_column("Size", style="green", justify="right")
    table.add_column("Offset", style="blue", justify="right")

    for attachment in summary.attachment_indexes:
        log_time = (
            datetime.fromtimestamp(attachment.log_time / 1_000_000_000).isoformat()
            if attachment.log_time
            else ""
        )
        create_time = (
            datetime.fromtimestamp(attachment.create_time / 1_000_000_000).isoformat()
            if attachment.create_time
            else ""
        )
        table.add_row(
            attachment.name,
            attachment.media_type,
            log_time,
            create_time,
            str(attachment.data_size),
            str(attachment.offset),
        )

    console.print(table)


def handle_metadata(args: argparse.Namespace) -> None:
    """List all metadata records in the MCAP file."""
    info = _read_mcap_info(args.file)
    summary = info.summary

    if not summary.metadata_indexes:
        console.print("[yellow]No metadata found[/yellow]")
        return

    table = Table()
    table.add_column("Name", style="bold white")
    table.add_column("Offset", style="cyan", justify="right")
    table.add_column("Length", style="yellow", justify="right")
    table.add_column("Metadata", style="blue")

    for metadata_idx in summary.metadata_indexes:
        # Note: metadata_indexes only contain name, offset, length
        # To get the actual metadata dict, we'd need to read the full Metadata record
        # For now, just show what's available in the index
        table.add_row(
            metadata_idx.name,
            str(metadata_idx.offset),
            str(metadata_idx.length),
            "",  # Metadata dict not available in index
        )

    console.print(table)
