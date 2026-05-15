"""Duration string parser shared by the CLI commands."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pymcap_cli.constants import NS_TO_MS, NS_TO_SEC

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cyclopts.token import Token

_DURATION_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(ns|us|ms|s|m|h)?\s*$")
_FACTORS_NS: dict[str | None, int] = {
    "ns": 1,
    "us": 1_000,
    "ms": NS_TO_MS,
    "s": NS_TO_SEC,
    "m": 60 * NS_TO_SEC,
    "h": 3600 * NS_TO_SEC,
    None: NS_TO_SEC,  # Bare number = seconds.
}


def parse_duration_ns(value: str) -> int:
    """Parse ``500ms`` / ``2s`` / ``1.5m`` / bare-seconds into nanoseconds."""
    match = _DURATION_RE.match(value)
    if not match:
        raise ValueError(f"Invalid duration {value!r}; expected forms like '500ms', '2s', '30s'")
    return int(float(match.group(1)) * _FACTORS_NS[match.group(2)])


def duration_ns_token_converter(_type: type, tokens: Sequence[Token]) -> int:
    """Cyclopts converter for a single duration token (e.g. ``--split-duration 2s``)."""
    if len(tokens) != 1:
        msg = "Expected exactly one duration."
        raise ValueError(msg)
    return parse_duration_ns(tokens[0].value)
