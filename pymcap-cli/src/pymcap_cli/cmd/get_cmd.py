"""Get command for pymcap-cli — extract attachments and metadata records."""

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from small_mcap import get_summary, read_attachment, read_metadata

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)

get_app = App(help="Get an attachment or metadata record from an MCAP file")


def attachment(
    file: str,
    *,
    name: Annotated[
        str,
        Parameter(name=["--name", "-n"], help="Name of the attachment to extract."),
    ],
    offset: Annotated[
        int | None,
        Parameter(
            name=["--offset"],
            help="Byte offset to disambiguate when multiple attachments share a name.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        Parameter(
            name=["--output", "-o"],
            help="Output file path. Defaults to stdout (refuses to write to a TTY).",
        ),
    ] = None,
) -> int:
    """Extract a single attachment by name and write its bytes.

    Parameters
    ----------
    file
        Path to the MCAP file (local file or HTTP/HTTPS URL).
    """
    try:
        with open_input(file) as (stream, _size):
            summary = get_summary(stream)
            if summary is None:
                ERR.print("[red]Error:[/red] no summary section in file")
                return 1

            matches = [idx for idx in summary.attachment_indexes if idx.name == name]
            if not matches:
                ERR.print(f"[red]Error:[/red] no attachment named {name!r}")
                return 1

            if offset is not None:
                matches = [idx for idx in matches if idx.offset == offset]
                if not matches:
                    ERR.print(f"[red]Error:[/red] no attachment named {name!r} at offset {offset}")
                    return 1
            elif len(matches) > 1:
                offsets = ", ".join(str(idx.offset) for idx in matches)
                ERR.print(
                    f"[red]Error:[/red] multiple attachments named {name!r} exist; "
                    f"specify --offset (offsets: {offsets})"
                )
                return 1

            chosen = matches[0]
            record = read_attachment(stream, chosen)

            if output is not None:
                output.write_bytes(record.data)
                return 0

            if sys.stdout.isatty():
                ERR.print(
                    "[red]Error:[/red] refusing to write binary attachment to a "
                    "terminal; use --output PATH or redirect stdout"
                )
                return 1

            sys.stdout.buffer.write(record.data)
            sys.stdout.buffer.flush()
            return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 0
    except Exception:
        logger.exception("get attachment failed")
        return 1


def metadata(
    file: str,
    *,
    name: Annotated[
        str,
        Parameter(name=["--name", "-n"], help="Name of the metadata record to extract."),
    ],
) -> int:
    """Extract a metadata record by name and print as JSON.

    When multiple metadata records share the same name, their key/value maps
    are merged into a single dict (later records win on key collision), matching
    the official `mcap get metadata` behavior.

    Parameters
    ----------
    file
        Path to the MCAP file (local file or HTTP/HTTPS URL).
    """
    try:
        with open_input(file) as (stream, _size):
            summary = get_summary(stream)
            if summary is None:
                ERR.print("[red]Error:[/red] no summary section in file")
                return 1

            matches = [idx for idx in summary.metadata_indexes if idx.name == name]
            if not matches:
                ERR.print(f"[red]Error:[/red] no metadata record named {name!r}")
                return 1

            merged: dict[str, str] = {}
            for idx in matches:
                record = read_metadata(stream, idx)
                merged.update(record.metadata)

            json.dump(merged, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            sys.stdout.flush()
            return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 0
    except Exception:
        logger.exception("get metadata failed")
        return 1


get_app.command(attachment, name="attachment")
get_app.command(metadata, name="metadata")
