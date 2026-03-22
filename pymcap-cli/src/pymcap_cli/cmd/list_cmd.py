"""List command for pymcap-cli - list various MCAP file records."""

import re
from datetime import datetime
from typing import Annotated

from cyclopts import App, Parameter
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from ros_parser import parse_schema_to_definitions
from ros_parser.models import Constant, MessageDefinition
from small_mcap import RebuildInfo

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.display.display_utils import _create_ros_docs_url
from pymcap_cli.utils import bytes_to_human, read_or_rebuild_info

console = Console()

# Create the list sub-app
list_app = App(help="List records in an MCAP file")


def _read_mcap_info(file_path: str) -> RebuildInfo:
    """Read MCAP file info, with automatic rebuild on invalid magic."""
    with open_input(file_path) as (f, file_size):
        return read_or_rebuild_info(f, file_size)


def channels(
    file: str,
) -> int:
    """List channels in an MCAP file.

    Parameters
    ----------
    file : str
        Path to the MCAP file (local file or HTTP/HTTPS URL).
    """
    info = _read_mcap_info(file)
    summary = info.summary

    if not summary.channels:
        console.print("[yellow]No channels found[/yellow]")
        return 0

    table = Table()
    table.add_column("ID", style="green", justify="right")
    table.add_column("Schema ID", style="cyan", justify="right")
    table.add_column("Topic", style="bold white")
    table.add_column("Encoding", style="yellow")
    table.add_column("Metadata", style="blue")

    for channel_id in sorted(summary.channels):
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

    return 0


def chunks(
    file: str,
) -> int:
    """List chunks in an MCAP file.

    Parameters
    ----------
    file : str
        Path to the MCAP file (local file or HTTP/HTTPS URL).
    """
    info = _read_mcap_info(file)
    summary = info.summary

    if not summary.chunk_indexes:
        console.print("[yellow]No chunks found[/yellow]")
        return 0

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

    return 0


def schemas(
    file: str,
) -> int:
    """List schemas in an MCAP file.

    Parameters
    ----------
    file : str
        Path to the MCAP file (local file or HTTP/HTTPS URL).
    """
    info = _read_mcap_info(file)
    summary = info.summary

    if not summary.schemas:
        console.print("[yellow]No schemas found[/yellow]")
        return 0

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

    return 0


def attachments(
    file: str,
) -> int:
    """List attachments in an MCAP file.

    Parameters
    ----------
    file : str
        Path to the MCAP file (local file or HTTP/HTTPS URL).
    """
    info = _read_mcap_info(file)
    summary = info.summary

    if not summary.attachment_indexes:
        console.print("[yellow]No attachments found[/yellow]")
        return 0

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

    return 0


def metadata(
    file: str,
) -> int:
    """List metadata records in an MCAP file.

    Parameters
    ----------
    file : str
        Path to the MCAP file (local file or HTTP/HTTPS URL).
    """
    info = _read_mcap_info(file)
    summary = info.summary

    if not summary.metadata_indexes:
        console.print("[yellow]No metadata found[/yellow]")
        return 0

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

    return 0


def _render_fields(
    tree: Tree,
    definition: MessageDefinition,
    all_defs: dict[str, MessageDefinition],
    ancestors: set[str],
    expanded: set[str],
) -> None:
    """Recursively render message fields into a Rich Tree.

    Uses uv-tree-style deduplication: complex types are fully expanded
    on first occurrence, then shown with (*) on subsequent appearances.
    ``ancestors`` tracks the current recursion stack (cycle prevention),
    ``expanded`` tracks types already expanded anywhere in the tree.
    """
    for item in definition.fields_all:
        if isinstance(item, Constant):
            tree.add(f"[magenta]{item.type} {item.name}={item.value}[/magenta]")
            continue

        field_type = item.type

        # Format array suffix with escaped brackets for Rich markup
        array_suffix = ""
        if field_type.is_array:
            if field_type.array_size and not field_type.is_upper_bound:
                array_suffix = f"[yellow]\\[{field_type.array_size}][/yellow]"
            elif field_type.array_size and field_type.is_upper_bound:
                array_suffix = f"[yellow]\\[<={field_type.array_size}][/yellow]"
            else:
                array_suffix = "[yellow]\\[][/yellow]"

        if field_type.is_primitive:
            tree.add(f"[green]{field_type.type_name}[/green]{array_suffix} {item.name}")
        else:
            # Complex type — try to recurse
            # Build lookup key using base name (without array brackets)
            base_type = (
                f"{field_type.package_name}/{field_type.type_name}"
                if field_type.package_name
                else field_type.type_name
            )
            lookup_keys = [base_type]
            if field_type.package_name:
                lookup_keys.append(f"{field_type.package_name}/msg/{field_type.type_name}")

            child_def = None
            for key in lookup_keys:
                if key in all_defs:
                    child_def = all_defs[key]
                    break

            # Format label with link
            docs_url = _create_ros_docs_url(base_type)
            if docs_url:
                type_markup = f"[link={docs_url}][cyan]{base_type}[/cyan][/link]"
            else:
                type_markup = f"[cyan]{base_type}[/cyan]"

            if child_def and base_type not in ancestors:
                if base_type in expanded:
                    # Already expanded elsewhere — show (*) like uv tree
                    tree.add(f"{type_markup}{array_suffix} {item.name} [dim](*)[/dim]")
                else:
                    branch = tree.add(f"{type_markup}{array_suffix} {item.name}")
                    expanded.add(base_type)
                    ancestors.add(base_type)
                    _render_fields(branch, child_def, all_defs, ancestors, expanded)
                    ancestors.discard(base_type)
            else:
                tree.add(f"{type_markup}{array_suffix} {item.name}")


def schema(
    file: str,
    *,
    name: Annotated[
        str | None,
        Parameter(name=["--name"]),
    ] = None,
) -> int:
    """Inspect schema structure with nested field display.

    Parse and display ROS2 message schemas as a tree, showing nested
    field types recursively. Complex types are expanded inline.

    Parameters
    ----------
    file
        Path to the MCAP file (local file or HTTP/HTTPS URL).
    name
        Filter schemas by regex pattern on schema name.

    Examples
    --------
    ```
    # Show all schemas
    pymcap-cli list schema recording.mcap

    # Filter by name
    pymcap-cli list schema recording.mcap --name Image
    ```
    """
    info = _read_mcap_info(file)
    summary = info.summary

    if not summary.schemas:
        console.print("[yellow]No schemas found[/yellow]")
        return 0

    matched = False
    for schema_id in sorted(summary.schemas.keys()):
        s = summary.schemas[schema_id]

        if name and not re.search(name, s.name):
            continue

        # Try to parse schema data
        if not s.data:
            console.print(f"[yellow]Schema {s.name} (ID: {s.id}) has no data[/yellow]")
            continue

        try:
            all_defs = parse_schema_to_definitions(s.name, s.data)
        except Exception:  # noqa: BLE001
            console.print(f"[yellow]Could not parse schema {s.name} (ID: {s.id})[/yellow]")
            continue

        # Find root definition
        root_def = all_defs.get(s.name)
        if root_def is None:
            parts = s.name.split("/")
            short_name = f"{parts[0]}/{parts[-1]}"
            root_def = all_defs.get(short_name)

        if root_def is None:
            console.print(f"[yellow]No root definition found for {s.name}[/yellow]")
            continue

        matched = True

        # Build tree
        docs_url = _create_ros_docs_url(s.name)
        if docs_url:
            label = (
                f"[link={docs_url}][bold cyan]{s.name}[/bold cyan][/link]  [dim](ID: {s.id})[/dim]"
            )
        else:
            label = f"[bold cyan]{s.name}[/bold cyan]  [dim](ID: {s.id})[/dim]"

        tree = Tree(label)
        _render_fields(tree, root_def, all_defs, set(), set())
        console.print(tree)
        console.print()

    if not matched:
        if name:
            console.print(f"[yellow]No schemas matching '{name}'[/yellow]")
        else:
            console.print("[yellow]No parseable schemas found[/yellow]")

    return 0


# Register commands with the app
list_app.command(channels, name="channels")
list_app.command(chunks, name="chunks")
list_app.command(schemas, name="schemas")
list_app.command(schema, name="schema")
list_app.command(attachments, name="attachments")
list_app.command(metadata, name="metadata")
