"""Tests for IVFParser: splitting ffmpeg's IVF output into per-frame packets."""

from __future__ import annotations

from mcap_codec_support.video.ffmpeg import IVFParser


def _ivf_file_header() -> bytes:
    # 'DKIF', version, header length, fourcc, dims, rates, frame count — the
    # exact field values do not matter to the parser, only the 32-byte length.
    return b"DKIF" + bytes(28)


def _ivf_frame(payload: bytes, timestamp: int = 0) -> bytes:
    return len(payload).to_bytes(4, "little") + timestamp.to_bytes(8, "little") + payload


def test_ivf_parser_splits_two_frames() -> None:
    stream = _ivf_file_header() + _ivf_frame(b"AAAA", 0) + _ivf_frame(b"BB", 1)
    parser = IVFParser()
    assert parser.feed(stream) == [b"AAAA", b"BB"]
    assert parser.flush_list() == []


def test_ivf_parser_reassembles_across_chunk_boundaries() -> None:
    stream = _ivf_file_header() + _ivf_frame(b"HELLO", 0) + _ivf_frame(b"WORLD!", 1)
    parser = IVFParser()
    out: list[bytes] = []
    # Feed one byte at a time to exercise partial file/frame headers and payloads.
    for i in range(len(stream)):
        out.extend(parser.feed(stream[i : i + 1]))
    out.extend(parser.flush_list())
    assert out == [b"HELLO", b"WORLD!"]


def test_ivf_parser_holds_incomplete_frame() -> None:
    parser = IVFParser()
    # File header plus a frame header claiming 8 bytes but only 3 delivered.
    assert parser.feed(_ivf_file_header() + (8).to_bytes(4, "little") + bytes(8) + b"abc") == []
    # The remaining 5 bytes complete the frame.
    assert parser.feed(b"defgh") == [b"abcdefgh"]
