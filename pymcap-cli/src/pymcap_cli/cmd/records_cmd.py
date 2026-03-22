"""Records command for pymcap-cli - print all MCAP records."""

import sys

from rich.console import Console
from small_mcap.reader import stream_reader

from pymcap_cli.core.input_handler import open_input

console_err = Console(stderr=True)


def records(
    file: str,
) -> int:
    """Print all MCAP records in file order.

    Prints one line per record using the record's repr. Useful for inspecting
    the raw structure of an MCAP file.

    Parameters
    ----------
    file : str
        Path to the MCAP file (local file or HTTP/HTTPS URL).
    """
    try:
        with open_input(file) as (input_stream, _):
            for record in stream_reader(input_stream):
                print(record, file=sys.stdout)  # noqa: T201

    except KeyboardInterrupt:
        console_err.print("\n[yellow]Interrupted by user[/yellow]")
        return 0

    except Exception as e:  # noqa: BLE001
        console_err.print(f"[red]Error reading MCAP: {e}[/red]")
        return 1

    return 0
