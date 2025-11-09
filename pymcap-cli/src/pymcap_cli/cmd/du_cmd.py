"""DU command - report space usage within an MCAP file."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from pymcap_cli.cmd.info_json_cmd import info_to_dict
from pymcap_cli.display_utils import ChannelTableColumn, display_channels_table
from pymcap_cli.utils import rebuild_info

console = Console()

app = typer.Typer()


@app.command()
def du(
    file: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Path to the MCAP file to analyze"
    ),
    exact_sizes: bool = typer.Option(
        False, "--exact-sizes", "-e", help="Use exact sizes for message data (may be slower)"
    ),
) -> None:
    """Report space usage within an MCAP file.

    This command reports space usage within an mcap file. Space usage for messages is
    calculated using the uncompressed size.

    Note: This command will scan and uncompress the entire file.
    """
    file_size = file.stat().st_size
    with file.open("rb") as f:
        # Use rebuild_info to get channel sizes (du always rebuilds)
        info = rebuild_info(f, file_size, exact_sizes=exact_sizes)

        # Transform to JSON structure (shared logic with info-json command)
        data = info_to_dict(info, str(file), file_size)

        # Display channels table sorted by size
        display_channels_table(
            data,
            console,
            sort_key="size",
            reverse=True,
            columns=(
                ChannelTableColumn.TOPIC
                | ChannelTableColumn.MSGS
                | ChannelTableColumn.HZ
                | ChannelTableColumn.SIZE
                | ChannelTableColumn.PERCENT
                | ChannelTableColumn.BPS
                | ChannelTableColumn.B_PER_MSG
            ),
            responsive=False,
            index_duration=False,
        )
