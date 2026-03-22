"""DU command - report space usage within an MCAP file."""

from typing import Annotated

from cyclopts import Parameter
from rich.console import Console

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.display.display_utils import ChannelTableColumn, display_channels_table
from pymcap_cli.types.info_data import info_to_dict
from pymcap_cli.utils import read_or_rebuild_info

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
) -> int:
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
        info = read_or_rebuild_info(f, file_size, rebuild=True, exact_sizes=exact_sizes)

    data = info_to_dict(info, str(file), file_size)

    console.print(
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
    )

    return 0
