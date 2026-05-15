"""Semantic fingerprint of an MCAP summary.

This identifies a file by **what's in it** (schemas, channels, per-channel
message counts, time range, attachment index entries, metadata index entries)
rather than by raw byte layout. Two MCAPs written by different writers but
carrying the same logical content yield the same fingerprint.

The fingerprint is the catalog's primary content identity (``content`` PK).
For byte-identity (rename detection, rebuild cache) see
:mod:`pymcap_cli.index.fingerprint`.

The hex output is prefixed with the scheme version (``s1:``). If the
canonicalisation ever changes, bumping the version prevents silent collisions
between old and new fingerprints in the same DB.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import xxhash

from pymcap_cli.rihs01 import compute_rihs01

if TYPE_CHECKING:
    from small_mcap.rebuild import RebuildInfo

SCHEME_VERSION = "s1"


def _schema_hash(encoding: str, name: str, data: bytes) -> str:
    """Stable schema identifier: RIHS01 for ros2msg, xxh3_64 content hash otherwise."""
    if encoding == "ros2msg":
        try:
            return compute_rihs01(name, data)
        except Exception:  # noqa: BLE001 — ros parser raises a wide variety of exceptions
            return _fallback_schema_hash(encoding, name, data)
    return _fallback_schema_hash(encoding, name, data)


def _fallback_schema_hash(encoding: str, name: str, data: bytes) -> str:
    payload = json.dumps(
        {"encoding": encoding, "name": name, "data_hex": data.hex()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"schema_xxh3_128_{xxhash.xxh3_128_hexdigest(payload)}"


def _canonical_metadata(meta: dict[str, str] | None) -> list[list[str]]:
    if not meta:
        return []
    return [[k, v] for k, v in sorted(meta.items())]


def summary_fingerprint(info: RebuildInfo) -> str:
    """Compute the semantic summary fingerprint for an MCAP file.

    Requires ``info.summary`` to carry a Statistics record — without it,
    per-channel message counts are missing and the hash would collide
    across unrelated recordings. Caller must trigger a full rebuild for
    such files.

    Raises:
        ValueError: if the summary's statistics is missing.
    """
    header = info.header
    summary = info.summary
    stats = summary.statistics
    if stats is None:
        raise ValueError("summary_fingerprint requires Statistics in the Summary")

    # Schemas — keyed by their canonical hash (RIHS01 for ros2msg, content
    # hash otherwise) rather than the writer's local ``schema_id`` (which is
    # not stable across writers).
    schema_hash_by_id: dict[int, str] = {}
    schema_entries: list[dict[str, object]] = []
    for sc in summary.schemas.values():
        h = _schema_hash(sc.encoding, sc.name, sc.data)
        schema_hash_by_id[sc.id] = h
        schema_entries.append(
            {
                "h": h,
                "name": sc.name,
                "encoding": sc.encoding,
                "size": len(sc.data),
            }
        )
    schema_entries.sort(key=lambda e: (e["h"], e["name"], e["encoding"]))

    # Channels — sorted by topic + referenced schema hash.
    channel_entries: list[dict[str, object]] = [
        {
            "topic": ch.topic,
            "encoding": ch.message_encoding,
            "schema_h": schema_hash_by_id.get(ch.schema_id, ""),
            "metadata": _canonical_metadata(ch.metadata),
            "messages": stats.channel_message_counts.get(ch.id, 0),
        }
        for ch in summary.channels.values()
    ]
    channel_entries.sort(key=lambda e: (e["topic"], e["schema_h"], e["encoding"]))

    # AttachmentIndex records — name + media_type + data_size identify
    # the attachment without depending on file-layout offsets or
    # writer-set timestamps.
    attachment_entries = sorted(
        (
            {"name": a.name, "media_type": a.media_type, "data_size": a.data_size}
            for a in summary.attachment_indexes
        ),
        key=lambda e: (e["name"], e["media_type"], e["data_size"]),
    )

    # MetadataIndex records — name is the only stable identity.
    metadata_names = sorted(m.name for m in summary.metadata_indexes)

    payload = {
        "v": SCHEME_VERSION,
        "profile": header.profile or "",
        "schemas": schema_entries,
        "channels": channel_entries,
        "msg_start": stats.message_start_time,
        "msg_end": stats.message_end_time,
        "msg_count": stats.message_count,
        "attachments": attachment_entries,
        "metadata": metadata_names,
    }
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = xxhash.xxh3_128_hexdigest(payload_bytes)
    return f"{SCHEME_VERSION}:{digest}"
