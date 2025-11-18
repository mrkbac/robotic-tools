"""DU command - report space usage within an MCAP file."""

from typing import Annotated

from cyclopts import Parameter
from rich.console import Console

from pymcap_cli.cmd.info_json_cmd import info_to_dict
from pymcap_cli.display_utils import ChannelTableColumn, display_channels_table
from pymcap_cli.input_handler import open_input
from pymcap_cli.utils import rebuild_info

console = Console()


def du(
    file: str,
    *,
    exact_sizes: Annotated[
        bool,
        Parameter(
            name=["-e", "--exact-sizes"],
        ),
    ] = False,
) -> None:
    """Report space usage within an MCAP file.

    This command reports space usage within an mcap file. Space usage for messages is
    calculated using the uncompressed size.

    Note: This command will scan and uncompress the entire file.

    Parameters
    ----------
    file
        Path to the MCAP file to analyze (local file or HTTP/HTTPS URL).
    exact_sizes
        Use exact sizes for message data (may be slower).
    """
    with open_input(file) as (f, file_size):
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
                ChannelTableColumn.MSGS
                | ChannelTableColumn.HZ
                | ChannelTableColumn.SIZE
                | ChannelTableColumn.PERCENT
                | ChannelTableColumn.BPS
                | ChannelTableColumn.B_PER_MSG
            ),
            responsive=False,
            index_duration=False,
        )
