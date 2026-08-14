"""Tests for the compression-independent MCAP hash command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cli import app
from pymcap_cli.cmd import decompress_cmd, hash_cmd
from small_mcap import CompressionType, McapWriter

if TYPE_CHECKING:
    from pathlib import Path


def _write_mcap(path: Path, *, payloads: list[bytes] | None = None) -> None:
    messages = payloads or [b"one", b"two", b"three"]
    with path.open("wb") as stream:
        writer = McapWriter(stream, compression=CompressionType.ZSTD)
        writer.start(profile="ros2", library="hash-test")
        writer.add_schema(1, "std_msgs/msg/String", "ros2msg", b"string data\n")
        writer.add_channel(1, "/chatter", "cdr", 1)
        for index, payload in enumerate(messages):
            timestamp = (index + 1) * 1_000_000_000
            writer.add_message(1, timestamp, payload, timestamp, index)
        writer.finish()


def test_hash_is_stable_after_decompression(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    compressed = tmp_path / "compressed.mcap"
    uncompressed = tmp_path / "uncompressed.mcap"
    _write_mcap(compressed)

    assert decompress_cmd.decompress(str(compressed), uncompressed) == 0
    capsys.readouterr()

    assert hash_cmd.hash_mcap(str(compressed)) == 0
    compressed_hash = capsys.readouterr().out
    assert hash_cmd.hash_mcap(str(uncompressed)) == 0
    uncompressed_hash = capsys.readouterr().out

    assert compressed_hash == uncompressed_hash
    scheme, digest = compressed_hash.strip().split(":", 1)
    assert scheme == "mcap-index-v1"
    assert len(digest) == 64


def test_hash_changes_when_message_timestamps_change(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    left = tmp_path / "left.mcap"
    right = tmp_path / "right.mcap"
    _write_mcap(left)
    _write_mcap(right)

    with right.open("wb") as stream:
        writer = McapWriter(stream, compression=CompressionType.NONE)
        writer.start(profile="ros2", library="another-writer")
        writer.add_schema(1, "std_msgs/msg/String", "ros2msg", b"string data\n")
        writer.add_channel(1, "/chatter", "cdr", 1)
        for index, payload in enumerate((b"one", b"two", b"three")):
            timestamp = (index + 1) * 1_000_000_000 + (1 if index == 1 else 0)
            writer.add_message(1, timestamp, payload, timestamp, index)
        writer.finish()

    assert hash_cmd.hash_mcap(str(left)) == 0
    left_hash = capsys.readouterr().out
    assert hash_cmd.hash_mcap(str(right)) == 0
    right_hash = capsys.readouterr().out

    assert left_hash != right_hash


def test_hash_does_not_claim_to_cover_payload_bytes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    left = tmp_path / "left.mcap"
    right = tmp_path / "right.mcap"
    _write_mcap(left, payloads=[b"one", b"two", b"three"])
    _write_mcap(right, payloads=[b"one", b"changed", b"three"])

    assert hash_cmd.hash_mcap(str(left)) == 0
    left_hash = capsys.readouterr().out
    assert hash_cmd.hash_mcap(str(right)) == 0
    right_hash = capsys.readouterr().out

    assert left_hash == right_hash


def test_hash_is_registered_in_top_level_cli_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        app(["hash", "--help"])

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert exc_info.value.code == 0
    assert "Usage: pymcap-cli hash" in output
    assert "compression-independent" in output
    assert "does not hash message payload bytes" in output
