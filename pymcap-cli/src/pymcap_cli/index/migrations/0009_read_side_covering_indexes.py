"""V9 schema - add covering indexes for large topic/schema fanout."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

description = "add covering indexes for large topic/schema fanout"


def apply(conn: sqlite3.Connection) -> None:
    """Add covering indexes used by large topic/schema queries."""
    conn.execute(
        "CREATE INDEX IF NOT EXISTS content_channel_sig_content_msg "
        "ON content_channel(channel_signature_id, content_id, message_count)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS content_schema_schema_content "
        "ON content_schema(schema_id, content_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS channel_signature_schema_id ON channel_signature(schema_id)"
    )
