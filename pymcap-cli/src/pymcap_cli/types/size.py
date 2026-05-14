"""Byte-size string parser shared by the CLI commands."""

from __future__ import annotations

import re

_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([KMGTP]i?B?|B)?\s*$", re.IGNORECASE)

_FACTORS_BYTES: dict[str | None, int] = {
    None: 1,
    "B": 1,
    "K": 1_000,
    "KB": 1_000,
    "M": 1_000_000,
    "MB": 1_000_000,
    "G": 1_000_000_000,
    "GB": 1_000_000_000,
    "T": 1_000_000_000_000,
    "TB": 1_000_000_000_000,
    "P": 1_000_000_000_000_000,
    "PB": 1_000_000_000_000_000,
    "KI": 1024,
    "KIB": 1024,
    "MI": 1024**2,
    "MIB": 1024**2,
    "GI": 1024**3,
    "GIB": 1024**3,
    "TI": 1024**4,
    "TIB": 1024**4,
    "PI": 1024**5,
    "PIB": 1024**5,
}


def parse_size_bytes(value: str) -> int:
    """Parse ``1G`` / ``500MB`` / ``2GiB`` / bare-bytes into an integer byte count.

    Decimal suffixes (`K`, `KB`, `M`, `MB`, …) use 1000 as the base; IEC
    suffixes (`Ki`, `KiB`, `Mi`, `MiB`, …) use 1024. Bare numbers are bytes.
    """
    match = _SIZE_RE.match(value)
    if not match:
        msg = f"Invalid size {value!r}; expected forms like '1G', '500MB', '2GiB', '1024'"
        raise ValueError(msg)
    number = float(match.group(1))
    suffix = match.group(2)
    factor = _FACTORS_BYTES[suffix.upper() if suffix else None]
    result = int(number * factor)
    if result <= 0:
        msg = f"Size must be positive, got {value!r}"
        raise ValueError(msg)
    return result
