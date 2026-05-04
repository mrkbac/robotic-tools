"""Records command for pymcap-cli - print all MCAP records."""

import logging
import sys

from small_mcap import stream_reader

from pymcap_cli.core.input_handler import open_input

logger = logging.getLogger(__name__)


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
        logger.warning("Interrupted by user")
        return 0

    except Exception:
        logger.exception("Error reading MCAP")
        return 1

    return 0
