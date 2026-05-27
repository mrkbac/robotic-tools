"""``pymcap-cli index migrate`` — apply pending schema migrations."""

import sqlite3
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from rich.table import Table

from pymcap_cli.cmd.index._helpers import _resolve_db, console
from pymcap_cli.index.db import CURRENT_SCHEMA_VERSION, connect


def _read_user_version(db_path: Path) -> int:
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    finally:
        conn.close()


def migrate_cmd(
    *,
    db: Annotated[
        Path | None,
        Parameter(name=["--db"], help="Override the sidecar DB path."),
    ] = None,
) -> int:
    """Apply any pending schema migrations to the index DB."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        console.print(f"[red]Error:[/] no index DB at {db_path}")
        return 1

    before = _read_user_version(db_path)
    if before == CURRENT_SCHEMA_VERSION:
        console.print(
            f"Index DB at [cyan]{db_path}[/] is already at schema "
            f"[green]v{CURRENT_SCHEMA_VERSION}[/]; nothing to do."
        )
        return 0
    if before > CURRENT_SCHEMA_VERSION:
        console.print(
            f"[yellow]Index DB at {db_path} is at v{before}, newer than this CLI's "
            f"v{CURRENT_SCHEMA_VERSION}; refusing to downgrade.[/]"
        )
        return 1

    conn = connect(db_path, read_only=False)
    try:
        rows = conn.execute(
            "SELECT version, applied_at, description "
            "FROM schema_migrations "
            "WHERE version > ? "
            "ORDER BY version",
            (before,),
        ).fetchall()
    finally:
        conn.close()

    console.print(
        f"Migrated [cyan]{db_path}[/] from "
        f"[yellow]v{before}[/] to [green]v{CURRENT_SCHEMA_VERSION}[/]."
    )
    if rows:
        table = Table(title=f"Applied migrations ({len(rows)})")
        table.add_column("Version", justify="right", style="cyan")
        table.add_column("Description")
        for version, _applied_at, description in rows:
            table.add_row(f"v{version:04d}", str(description or "-"))
        console.print(table)
    return 0
