"""Tests for the head+tail+size byte probe."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.index.fingerprint import (
    HEAD_BYTES,
    TAIL_BYTES,
    fingerprint_path,
    fingerprint_stream,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def test_fingerprint_is_deterministic(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    _write(a, b"hello world" * 100)
    fp1, size1 = fingerprint_path(a)
    fp2, size2 = fingerprint_path(a)
    assert fp1 == fp2
    assert size1 == size2 == a.stat().st_size


def test_fingerprint_changes_with_head(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    _write(a, b"AAA" + b"x" * 1024)
    _write(b, b"BBB" + b"x" * 1024)
    fp_a, _ = fingerprint_path(a)
    fp_b, _ = fingerprint_path(b)
    assert fp_a != fp_b


def test_fingerprint_changes_with_tail(tmp_path: Path) -> None:
    """Files identical in the head region but different at the tail."""
    body = b"x" * (HEAD_BYTES + TAIL_BYTES + 1024)
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(body + b"end_a")
    b.write_bytes(body + b"end_b")
    fp_a, _ = fingerprint_path(a)
    fp_b, _ = fingerprint_path(b)
    assert fp_a != fp_b


def test_fingerprint_is_bounded_head_tail_probe(tmp_path: Path) -> None:
    """Middle-only differences are intentionally invisible to avoid full-file reads."""
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"h" * HEAD_BYTES + b"a" * 1024 + b"t" * TAIL_BYTES)
    b.write_bytes(b"h" * HEAD_BYTES + b"b" * 1024 + b"t" * TAIL_BYTES)

    fp_a, _ = fingerprint_path(a)
    fp_b, _ = fingerprint_path(b)

    assert fp_a == fp_b


def test_fingerprint_changes_with_size(tmp_path: Path) -> None:
    """Two files share head+tail bytes but differ in length."""
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"y" * 100)
    b.write_bytes(b"y" * 200)
    fp_a, _ = fingerprint_path(a)
    fp_b, _ = fingerprint_path(b)
    assert fp_a != fp_b


def test_fingerprint_stream_handles_small_files(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    a.write_bytes(b"tiny")
    with a.open("rb") as f:
        fp = fingerprint_stream(f, 4)
    assert isinstance(fp, str)
    assert len(fp) > 0


def test_fingerprint_independent_of_path(tmp_path: Path) -> None:
    """Same bytes under different paths give the same fingerprint."""
    payload = b"same content" * 1024
    a = tmp_path / "a.bin"
    b = tmp_path / "sub" / "b.bin"
    b.parent.mkdir()
    a.write_bytes(payload)
    b.write_bytes(payload)
    fp_a, _ = fingerprint_path(a)
    fp_b, _ = fingerprint_path(b)
    assert fp_a == fp_b
