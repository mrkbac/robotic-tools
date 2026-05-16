"""Numbered DB migrations applied in ``user_version`` order.

Each migration is a ``NNNN.py`` file with an ``apply(conn)`` function and a
``description`` string. Frozen once written — schema changes ship as a new
numbered file.
"""

from __future__ import annotations

import importlib
import pkgutil
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable


def _discover() -> list[tuple[int, Callable[[sqlite3.Connection], None], str]]:
    """Return ``(version, apply, description)`` tuples for every numbered migration."""
    found: list[tuple[int, Callable[[sqlite3.Connection], None], str]] = []
    seen: dict[int, str] = {}
    for info in pkgutil.iter_modules(__path__):
        if len(info.name) < 4 or not info.name[:4].isdigit():
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        version = int(info.name[:4])
        if version in seen:
            raise RuntimeError(
                f"duplicate index DB migration version {version:04d}: "
                f"{seen[version]} and {info.name}"
            )
        seen[version] = info.name
        description = getattr(module, "description", info.name)
        found.append((version, module.apply, description))
    found.sort(key=lambda v: v[0])
    return found


def _ensure_transaction_active(conn: sqlite3.Connection, version: int) -> None:
    if conn.in_transaction:
        return
    raise RuntimeError(f"index DB migration {version:04d} ended its transaction early")


def apply_pending(conn: sqlite3.Connection) -> None:
    """Apply every migration whose version is greater than ``user_version``.

    Migrations frequently drop / rebuild tables that are FK targets of other
    tables (e.g. ``content_channel`` → ``channel_sig`` in 0004). SQLite blocks
    ``DROP TABLE`` on an FK-target as long as enforcement is on, even with
    ``PRAGMA defer_foreign_keys``, so the canonical pattern is to disable FK
    enforcement around the whole migration window and re-validate afterwards
    with ``PRAGMA foreign_key_check``.
    """
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    pending = [v for v, _, _ in _discover() if v > current]
    if not pending:
        return

    fk_was_on = bool(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    if fk_was_on:
        # Must toggle outside a transaction; ``apply_pending`` is called on an
        # autocommit connection so this works.
        conn.execute("PRAGMA foreign_keys=OFF")
    try:
        for version, apply, description in _discover():
            if version <= current:
                continue
            conn.execute("BEGIN IMMEDIATE")
            try:
                apply(conn)
                _ensure_transaction_active(conn, version)
                conn.execute(
                    "INSERT INTO schema_migrations(version, applied_at, description) "
                    "VALUES (?, ?, ?)",
                    (version, time.time_ns(), description),
                )
                conn.execute(f"PRAGMA user_version = {version}")
                conn.execute("COMMIT")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            current = version
        # Migrations that drop tables (e.g. 0002) leave a lot of free pages.
        # ``VACUUM`` reclaims them so the file actually shrinks. Must run
        # outside any transaction.
        conn.execute("VACUUM")
    finally:
        if fk_was_on:
            conn.execute("PRAGMA foreign_keys=ON")
            violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise RuntimeError(f"foreign-key violations after migrations: {violations[:5]}")
