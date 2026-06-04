"""Round-trip and wire-format tests for MessageIndex serialization.

The codec uses ``array``-based bulk pack/unpack for speed; these tests pin the
exact byte layout against a hand-written struct oracle so the fast path can
never silently diverge from the MCAP spec.
"""

from __future__ import annotations

import io
import struct

import pytest
from small_mcap import MessageIndex
from small_mcap.records import Opcode

_ENTRY = struct.Struct("<QQ")
_OPCODE_LEN = struct.Struct("<BQ")
_HEADER = struct.Struct("<HI")


def _expected_bytes(channel_id: int, timestamps: list[int], offsets: list[int]) -> bytes:
    """Hand-written serializer used as an independent oracle."""
    records = b"".join(_ENTRY.pack(t, o) for t, o in zip(timestamps, offsets, strict=True))
    content = _HEADER.pack(channel_id, len(records)) + records
    return _OPCODE_LEN.pack(Opcode.MESSAGE_INDEX, len(content)) + content


def _serialize(index: MessageIndex) -> bytes:
    out = io.BytesIO()
    written = index.write_record_to(out)
    data = out.getvalue()
    assert written == len(data)
    return data


def test_write_matches_struct_oracle() -> None:
    idx = MessageIndex(channel_id=7, timestamps=[10, 20, 30], offsets=[0, 64, 128])
    assert _serialize(idx) == _expected_bytes(7, [10, 20, 30], [0, 64, 128])


def test_write_empty_matches_oracle() -> None:
    idx = MessageIndex(channel_id=3, timestamps=[], offsets=[])
    assert _serialize(idx) == _expected_bytes(3, [], [])


def test_read_inverts_write() -> None:
    idx = MessageIndex(channel_id=42, timestamps=[1, 2, 3, 4], offsets=[0, 16, 48, 96])
    # `.read` consumes the content after the 9-byte opcode/length prefix.
    parsed = MessageIndex.read(_serialize(idx)[9:])
    assert parsed.channel_id == 42
    assert parsed.timestamps == [1, 2, 3, 4]
    assert parsed.offsets == [0, 16, 48, 96]


def test_read_empty() -> None:
    idx = MessageIndex(channel_id=5, timestamps=[], offsets=[])
    parsed = MessageIndex.read(_serialize(idx)[9:])
    assert parsed.channel_id == 5
    assert parsed.timestamps == []
    assert parsed.offsets == []


def test_round_trip_large_index() -> None:
    n = 50_000
    timestamps = list(range(0, n * 1000, 1000))
    offsets = list(range(0, n * 16, 16))
    idx = MessageIndex(channel_id=1, timestamps=timestamps, offsets=offsets)
    parsed = MessageIndex.read(_serialize(idx)[9:])
    assert parsed.timestamps == timestamps
    assert parsed.offsets == offsets


def test_round_trip_max_uint64_values() -> None:
    big = (1 << 64) - 1
    idx = MessageIndex(channel_id=9, timestamps=[0, big, 1], offsets=[big, 0, big])
    parsed = MessageIndex.read(_serialize(idx)[9:])
    assert parsed.timestamps == [0, big, 1]
    assert parsed.offsets == [big, 0, big]


def test_read_then_write_reproduces_exact_bytes() -> None:
    # A round-tripped index must serialize back to the identical record bytes,
    # whether it rebuilds from arrays or re-emits the cached raw content.
    idx = MessageIndex(channel_id=11, timestamps=[5, 6, 7], offsets=[0, 32, 64])
    original = _serialize(idx)
    parsed = MessageIndex.read(original[9:])
    assert parsed._raw_content is not None  # read() cached the content
    assert _serialize(parsed) == original


def test_constructed_index_has_no_cached_content() -> None:
    # Indexes built incrementally (writer/rebuild) must not carry a stale cache.
    idx = MessageIndex(channel_id=1, timestamps=[1], offsets=[0])
    assert idx._raw_content is None


def test_read_write_round_trip_matches_oracle_via_raw() -> None:
    # The raw-content fast path must produce the same bytes as the struct oracle.
    expected = _expected_bytes(8, [100, 200], [0, 16])
    parsed = MessageIndex.read(expected[9:])
    assert _serialize(parsed) == expected


def test_read_rejects_malformed_records_length() -> None:
    # channel_id=1, records_len=8 (not a multiple of 16) → only half an entry.
    payload = _HEADER.pack(1, 8) + b"\x00" * 8
    with pytest.raises(struct.error):
        MessageIndex.read(payload)
