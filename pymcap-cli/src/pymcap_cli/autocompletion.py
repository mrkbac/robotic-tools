"""Autocompletion utilities for pymcap-cli commands."""

from collections.abc import Iterable
from pathlib import Path

import typer
from small_mcap import get_summary


def complete_all_topics(ctx: typer.Context, incomplete: str) -> list[str]:
    """Autocomplete function for all topics in the MCAP file.

    Args:
        ctx: Typer context containing already-parsed parameters
        incomplete: The incomplete value the user has typed so far

    Returns:
        List of topic names that match the incomplete value
    """
    # Get the file parameter from ctx.args (positional arguments)
    # Note: ctx.params doesn't contain positional args during option completion
    if not ctx.args or len(ctx.args) < 1:
        return []

    try:
        file_path = Path(ctx.args[0])
    except (IndexError, TypeError):
        return []

    if not file_path.exists():
        return []

    try:
        # Read MCAP summary to get topics
        with file_path.open("rb") as f:
            summary = get_summary(f)

        if summary is None:
            return []

        # Get all topics from all channels
        all_topics = [channel.topic for channel in summary.channels.values()]

        # Filter by what user has typed
        return [topic for topic in all_topics if topic.startswith(incomplete)]

    except Exception:  # noqa: BLE001
        # Silently fail on errors during autocompletion
        return []


def complete_topic_by_schema(
    ctx: typer.Context, incomplete: str, *, schemas: Iterable[str]
) -> list[str]:
    """Autocomplete function for topics in the MCAP file filtered by schema names.

    Args:
        ctx: Typer context containing already-parsed parameters
        incomplete: The incomplete value the user has typed so far

    Returns:
        List of topic names that match the incomplete value
    """
    # Get the file parameter from ctx.args (positional arguments)
    # Note: ctx.params doesn't contain positional args during option completion
    if not ctx.args or len(ctx.args) < 1:
        return []

    already_selected = ctx.params.get("topics", []) or []

    try:
        file_path = Path(ctx.args[0])
    except (IndexError, TypeError):
        return []

    if not file_path.exists():
        return []

    try:
        # Read MCAP summary to get topics
        with file_path.open("rb") as f:
            summary = get_summary(f)

        if summary is None:
            return []

        schema_ids = {schema.id for schema in summary.schemas.values() if schema.name in schemas}

        return [
            channel.topic
            for channel in summary.channels.values()
            if channel.schema_id in schema_ids
            and channel.topic.startswith(incomplete)
            and channel.topic not in already_selected
        ]

    except Exception:  # noqa: BLE001
        # Silently fail on errors during autocompletion
        return []
