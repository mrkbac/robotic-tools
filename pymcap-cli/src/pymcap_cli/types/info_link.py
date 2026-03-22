"""Generate a shareable URL for the MCAP web inspector."""

import base64
import json
import re
import zlib
from pathlib import Path
from typing import Literal, TypedDict

from pymcap_cli.types.info_types import ChannelInfo, ChunksInfo, McapInfoOutput, SchemaInfo

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

    Builds new dicts instead of mutating, so the caller's data is never modified.
    """
    schemas: list[SchemaInfo] = []
    for s in data.get("schemas", []):
        copy = SchemaInfo(**s)
        copy["data"] = ""
        schemas.append(copy)

    channels: list[ChannelInfo] = []
    for ch in data.get("channels", []):
        copy = ChannelInfo(**ch)
        copy.pop("message_distribution", None)
        channels.append(copy)

    return McapInfoOutput(
        file=data["file"],
        header=data["header"],
        statistics=data["statistics"],
        schemas=schemas,
        channels=channels,
        message_distribution=data["message_distribution"],
        chunks=ChunksInfo(
            by_compression={},
            overlaps=data["chunks"]["overlaps"],
        ),
    )


def generate_link(data: McapInfoOutput, file_path: str, file_size: int, mode: ScanMode) -> str:
    file_id = _create_file_id(file_path, file_size)
    stripped = _strip_for_url(data)
    payload = _UrlPayload(mode=mode, fileId=file_id, data=stripped)
    return f"{BASE_URL}{_encode(payload)}"
