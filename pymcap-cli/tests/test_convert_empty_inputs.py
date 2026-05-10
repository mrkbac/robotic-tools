from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from pymcap_cli.cmd import bag2mcap_cmd, convert_cmd

from tests.fixtures.bag_generator import generate_bag

if TYPE_CHECKING:
    from pathlib import Path


def test_bag2mcap_empty_bag_writes_no_output(tmp_path: Path) -> None:
    bag_path = tmp_path / "empty.bag"
    output = tmp_path / "empty.mcap"
    bag_path.write_bytes(generate_bag([], []))

    exit_code = bag2mcap_cmd.bag2mcap(str(bag_path), output, force=True)

    assert exit_code == 0
    assert not output.exists()


def test_convert_empty_db3_writes_no_output(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.db3"
    output = tmp_path / "empty.mcap"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE topics (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                serialization_format TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                topic_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                data BLOB NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    exit_code = convert_cmd.convert(str(db_path), output, force=True)

    assert exit_code == 0
    assert not output.exists()
