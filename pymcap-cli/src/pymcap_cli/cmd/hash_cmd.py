"""Compression-independent structural hash for MCAP recordings."""

import logging

from small_mcap import InvalidMagicError, McapError

from pymcap_cli.core.mcap_compare import read_structural_identity_file
from pymcap_cli.log_setup import OUT

logger = logging.getLogger(__name__)

HASH_SCHEME = "mcap-index-v1"


def hash_mcap(file: str) -> int:
    """Print a stable, compression-independent hash of an MCAP file.

    The hash covers the normalized profile, schemas, channels, channel metadata,
    message counts, and every message log timestamp. It reuses the footer summary
    and message indexes when available, and rebuilds or reads them when needed,
    so chunking, chunk size, and compression do not change the result.

    This deliberately does not hash message payload bytes, publish timestamps,
    sequences, attachments, or metadata bodies. Use it as a fast structural
    fingerprint, not as proof of byte-for-byte or payload equality.

    Parameters
    ----------
    file
        Path to the MCAP file to hash (local file or HTTP/HTTPS URL).

    Examples
    --------
    ```
    pymcap-cli hash recording.mcap
    ```
    """
    try:
        result = read_structural_identity_file(file)
    except (McapError, InvalidMagicError, OSError) as exc:
        logger.error(f"Error hashing {file}: {exc}")  # noqa: TRY400
        return 1

    if result is None:
        logger.error(f"Could not read every message log time for {file}")
        return 1

    OUT.print(f"{HASH_SCHEME}:{result.identity.digest}")
    return 0
