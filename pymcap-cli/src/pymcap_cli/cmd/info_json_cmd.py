"""Info-JSON command - output MCAP file statistics as JSON.

Thin wrapper around ``info --json``. Keeps its own parameter signature so that
display-only flags (--sort, --tree, etc.) don't appear in ``info-json --help``.
"""

from typing import Annotated

from cyclopts import Parameter

from pymcap_cli.cmd.info_cmd import info


def info_json(
    files: list[str],
    *,
    rebuild: Annotated[
        bool,
        Parameter(
            name=["-r", "--rebuild"],
        ),
    ] = False,
    exact_sizes: Annotated[
        bool,
        Parameter(
            name=["-e", "--exact-sizes"],
        ),
    ] = False,
    debug: Annotated[
        bool,
        Parameter(
            name=["--debug"],
        ),
    ] = False,
    compress: Annotated[
        bool,
        Parameter(
            name=["--compress"],
        ),
    ] = False,
) -> int:
    """Output MCAP file(s) statistics as JSON with all available data.

    Parameters
    ----------
    files
        Path(s) to MCAP file(s) to analyze (local files or HTTP/HTTPS URLs).
    rebuild
        Rebuild the MCAP file from scratch.
    exact_sizes
        Use exact sizes for message data (may be slower, requires --rebuild).
    debug
        Enable debug mode.
    compress
        Compressed output using gzip and outputs it as base64.
    """
    return info(
        files,
        rebuild=rebuild,
        exact_sizes=exact_sizes,
        debug=debug,
        json_output=True,
        compress=compress,
    )
