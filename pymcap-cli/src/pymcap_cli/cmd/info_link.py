"""Generate a shareable URL for the MCAP web inspector."""

import base64
import json
import re
import zlib
from pathlib import Path
from typing import Literal, TypedDict

from pymcap_cli.info_types import McapInfoOutput

ScanMode = Literal["summary", "rebuild", "exact"]

BASE_URL = "https://mrkbac.github.io/robotic-tools/view#"


class _UrlPayload(TypedDict):
    mode: ScanMode
    fileId: str
    data: McapInfoOutput


def _create_file_id(file_path: str, file_size: int) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", Path(file_path).name)
    size_b36 = _int_to_base36(file_size)
    return f"{sanitized}-{size_b36}"


def _int_to_base36(n: int) -> str:
    if n == 0:
        return "0"
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    result: list[str] = []
    while n:
        n, remainder = divmod(n, 36)
        result.append(chars[remainder])
    return "".join(reversed(result))


def _compress_to_base64url(json_str: str) -> str:
    compressor = zlib.compressobj(wbits=-15)
    compressed = compressor.compress(json_str.encode("utf-8"))
    compressed += compressor.flush()
    b64 = base64.urlsafe_b64encode(compressed).decode("ascii")
    return b64.rstrip("=")


def _encode(payload: _UrlPayload) -> str:
    json_str = json.dumps(payload, separators=(",", ":"))
    return _compress_to_base64url(json_str)


def _strip_for_url(data: McapInfoOutput) -> McapInfoOutput:
    """Create a lean copy of data suitable for URL encoding.

    Keeps overview info (file, header, statistics, schemas) and slim channels.
    Strips per-channel distributions, chunks detail, metadata, attachments, thumbnail.
    """
    stripped = McapInfoOutput(**data)

    # Clear schema data content (keep id, name, encoding)
    for s in stripped.get("schemas", []):
        s["data"] = ""

    # Slim channels: strip message_distribution (large arrays)
    for ch in stripped.get("channels", []):
        ch.pop("message_distribution", None)

    # Keep only chunk overlaps, drop by_compression
    chunks = stripped.get("chunks")
    if chunks:
        stripped["chunks"] = {
            "by_compression": {},
            "overlaps": chunks["overlaps"],
        }

    # Strip heavy top-level fields
    stripped.pop("metadata", None)
    stripped.pop("attachments", None)
    stripped.pop("thumbnail", None)

    return stripped


def generate_link(data: McapInfoOutput, file_path: str, file_size: int, mode: ScanMode) -> str:
    file_id = _create_file_id(file_path, file_size)
    stripped = _strip_for_url(data)
    payload = _UrlPayload(mode=mode, fileId=file_id, data=stripped)
    return f"{BASE_URL}{_encode(payload)}"
