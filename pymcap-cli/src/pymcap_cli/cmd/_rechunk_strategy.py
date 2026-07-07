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
    SchemaCompressionGrouper,
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
    incompressible_schema_patterns: list[Pattern[str]] | None = None,
) -> list[OutputProcessor]:
    """Translate a CLI rechunking strategy into concrete OutputProcessors.

    ``incompressible_schema_patterns`` is independent of ``strategy`` — it
    always appends a grouper that pulls matching-schema channels into their
    own uncompressed group, on top of whatever grouping ``strategy`` selects
    (or, with ``strategy=none``, on its own, activating chunk grouping just
    for that split).
    """
    processors: list[OutputProcessor]
    if strategy == RechunkStrategy.NONE:
        processors = []
    elif strategy == RechunkStrategy.ALL:
        processors = [PerChannelGrouper()]
    else:
        processors = [PatternGrouper(topic_patterns or [], schema_patterns or [])]

    if incompressible_schema_patterns:
        processors.append(SchemaCompressionGrouper(incompressible_schema_patterns))

    return processors
