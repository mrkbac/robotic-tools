"""Export a versioned recording catalog from the sidecar index."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, TypedDict
from urllib.parse import quote, urlsplit

from cyclopts import Parameter
from rich.console import Console

from pymcap_cli.cmd._cli_options import IndexDbOption
from pymcap_cli.cmd.index._helpers import (
    _EFF_END_SQL,
    _EFF_START_SQL,
    _path_prefix_bounds,
    _print_db_needs_migration,
    _resolve_db,
    _stdout_line,
)
from pymcap_cli.index.db import IndexDbNeedsMigrationError, open_db


class _Source(TypedDict):
    type: Literal["remote-file"]
    url: str


class _TopicNames(TypedDict):
    name: str
    schema: str


class _Topic(_TopicNames, total=False):
    messages: int


class _RecordingFields(TypedDict):
    id: str
    name: str
    source: _Source
    dataset: str
    size: int
    messages: int
    topics: list[_Topic]


class _Recording(_RecordingFields, total=False):
    start: str
    end: str


def _rfc3339(time_ns: int) -> str:
    seconds, nanos = divmod(time_ns, 1_000_000_000)
    stamp = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{stamp}.{nanos:09d}Z"


def catalog_cmd(
    root: Annotated[Path, Parameter(help="Local root of the hosted files.")],
    *,
    name: Annotated[str | None, Parameter(help="Display name of the catalog.")] = None,
    base_url: Annotated[
        str | None, Parameter(help="HTTP(S) URL of the file root. Defaults to relative file URLs.")
    ] = None,
    db: IndexDbOption = None,
) -> int:
    """Write a version 1 recording catalog as JSON on stdout."""
    errors = Console(stderr=True)
    root = root.expanduser().resolve()
    db_path = _resolve_db(db)
    prefix = ""
    if base_url is not None:
        try:
            parsed = urlsplit(base_url)
            port = parsed.port
        except ValueError:
            parsed = None
            port = None
        if (
            parsed is None
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or port == 0
            or parsed.query
            or parsed.fragment
        ):
            errors.print("[red]Error:[/] --base-url must be an HTTP(S) directory URL.")
            return 1
        prefix = parsed.geturl().rstrip("/") + "/"

    recordings: dict[str, _Recording] = {}
    try:
        with open_db(db_path, read_only=True) as conn:
            rows = conn.execute(
                f"""
                SELECT cf.abs_path, cf.size_bytes, c.message_count,
                       COALESCE({_EFF_START_SQL}, c.message_start_time_ns),
                       COALESCE({_EFF_END_SQL}, c.message_end_time_ns),
                       t.name, s.name, SUM(cc.message_count)
                FROM current_file cf
                JOIN content c ON c.id = cf.content_id
                LEFT JOIN content_channel cc ON cc.content_id = c.id
                LEFT JOIN channel_signature sig ON sig.id = cc.channel_signature_id
                LEFT JOIN topic t ON t.id = sig.topic_id
                LEFT JOIN schema s ON s.id = sig.schema_id
                WHERE cf.abs_path >= ? AND cf.abs_path < ?
                GROUP BY cf.abs_path, t.name, s.name
                ORDER BY cf.abs_path, t.name, s.name
                """,  # noqa: S608
                _path_prefix_bounds(root),
            )
            for path, size, messages, start, end, topic, schema, topic_messages in rows:
                absolute = Path(path)
                if not absolute.is_relative_to(root):
                    continue
                relative = absolute.relative_to(root)
                recording_id = relative.as_posix()
                recording = recordings.get(recording_id)
                if recording is None:
                    dataset = relative.parent.as_posix() if relative.parent.parts else root.name
                    new_recording: _Recording = {
                        "id": recording_id,
                        "name": relative.name,
                        "source": {"type": "remote-file", "url": prefix + quote(recording_id)},
                        "dataset": dataset,
                        "size": size,
                        "messages": messages,
                        "topics": [],
                    }
                    recording = new_recording
                    if messages > 0 and start is not None and end is not None and start <= end:
                        recording["start"] = _rfc3339(start)
                        recording["end"] = _rfc3339(end)
                    recordings[recording_id] = recording
                if topic is not None:
                    channel: _Topic = {"name": topic, "schema": schema or ""}
                    if topic_messages is not None:
                        channel["messages"] = topic_messages
                    recording["topics"].append(channel)
    except FileNotFoundError:
        errors.print(f"[red]Error:[/] no index DB at {db_path}")
        return 1
    except IndexDbNeedsMigrationError as exc:
        _print_db_needs_migration(exc)
        return 1

    catalog = {"version": 1, "name": name or root.name, "recordings": list(recordings.values())}
    _stdout_line(json.dumps(catalog))
    return 0
