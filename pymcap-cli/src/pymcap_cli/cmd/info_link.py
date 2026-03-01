"""Generate a shareable URL for the MCAP web inspector."""

import base64
import json
import re
import zlib
from pathlib import Path
from typing import Literal, TypedDict

from pymcap_cli.info_types import McapInfoOutput, SchemaInfo

ScanMode = Literal["summary", "rebuild", "exact"]

BASE_URL = "https://mrkbac.github.io/robotic-tools/view#"
MAX_HASH_BYTES = 8 * 1024


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


def _progressive_strip(payload: _UrlPayload) -> str:
    encoded = _encode(payload)
    if len(encoded) <= MAX_HASH_BYTES:
        return encoded

    channels = payload["data"].get("channels", [])

    # Strip per-channel message distributions
    for ch in channels:
        ch["message_distribution"] = []
    encoded = _encode(payload)
    if len(encoded) <= MAX_HASH_BYTES:
        return encoded

    # Strip bytes_per_second_stats
    for ch in channels:
        ch["bytes_per_second_stats"] = None
    encoded = _encode(payload)
    if len(encoded) <= MAX_HASH_BYTES:
        return encoded

    # Strip metadata and attachments (present in web inspector payloads)
    data = payload["data"]
    data.pop("metadata", None)  # type: ignore[misc]
    data.pop("attachments", None)  # type: ignore[misc]
    return _encode(payload)


def generate_link(data: McapInfoOutput, file_path: str, file_size: int, mode: ScanMode) -> str:
    file_id = _create_file_id(file_path, file_size)

    # Strip schema data for URL (keep only id and name)
    stripped = McapInfoOutput(**data)
    if "schemas" in stripped:
        stripped["schemas"] = [SchemaInfo(id=s["id"], name=s["name"]) for s in stripped["schemas"]]

    payload = _UrlPayload(mode=mode, fileId=file_id, data=stripped)
    return f"{BASE_URL}{_progressive_strip(payload)}"
