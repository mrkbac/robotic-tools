"""CLI-level rechunking strategy enum.

Lives in the cmd layer because it is purely a user-facing selector that the
``rechunk`` and ``process`` commands translate into concrete
``OutputProcessor`` instances. The core processor itself is strategy-agnostic.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pymcap_cli.core.processors.chunk_groupers import (
    PatternGrouper,
    PerChannelGrouper,
)

if TYPE_CHECKING:
    from re import Pattern

    from pymcap_cli.core.processors.base import OutputProcessor


class RechunkStrategy(str, Enum):
    """User-facing rechunking strategy."""

    NONE = "none"  # No rechunking — fast-copy optimization when possible
    PATTERN = "pattern"  # Group by topic / schema regex
    ALL = "all"  # Each channel in its own chunk group


def build_output_processors(
    strategy: RechunkStrategy,
    topic_patterns: list[Pattern[str]] | None = None,
    schema_patterns: list[Pattern[str]] | None = None,
) -> list[OutputProcessor]:
    """Translate a CLI rechunking strategy into concrete OutputProcessors."""
    if strategy == RechunkStrategy.NONE:
        return []
    if strategy == RechunkStrategy.ALL:
        return [PerChannelGrouper()]
    return [PatternGrouper(topic_patterns or [], schema_patterns or [])]
