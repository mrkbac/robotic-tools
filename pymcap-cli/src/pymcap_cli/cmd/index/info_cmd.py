"""``pymcap-cli index info`` — everything the index knows about a target."""

import json as _json
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from cyclopts import Parameter
from rich.table import Table

if TYPE_CHECKING:
    from pymcap_cli.types.qos import QosProfile

from pymcap_cli.cmd.index._helpers import (
    _describe_error_kind,
    _format_compression_cell,
    _format_count,
    _format_duration_ns,
    _format_ts_ns,
    _IndexedTopicPayload,
    _print_db_needs_migration,
    _resolve_db,
    _resolve_target_to_summary_fp,
    _safe_duration_ns,
    _short_id_from_fingerprint,
    _stdout_line,
    _topics_to_channel_table_data,
    console,
)
from pymcap_cli.core.qos import parse_qos_profiles
from pymcap_cli.display.display_utils import _format_parts_with_colors, display_channels_table
from pymcap_cli.index.db import IndexDbNeedsMigrationError, open_db
from pymcap_cli.index.scanner import unpack_distribution_blob
from pymcap_cli.utils import bytes_to_human


def info_cmd(
    target: str,
    *,
    format: Annotated[
        Literal["table", "json"],
        Parameter(name=["--format"], help="Output as Rich tables or JSON."),
    ] = "table",
    db: Annotated[
        Path | None,
        Parameter(name=["--db"], help="Override the sidecar DB path."),
    ] = None,
) -> int:
    """Show everything the index knows about TARGET (path, fingerprint, or short id)."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        console.print(f"[red]Error:[/] no index DB at {db_path}")
        return 1

    try:
        with open_db(db_path, read_only=True) as conn:
            summary_fp, abs_path, err = _resolve_target_to_summary_fp(conn, target)
            if err is not None or summary_fp is None:
                console.print(f"[red]Error:[/] {err}")
                return 1

            content = conn.execute(
                "SELECT c.size_bytes, lib.name, prof.name, "
                "       c.message_count, c.schema_count, c.channel_count, "
                "       c.attachment_count, c.metadata_count, c.chunk_count, "
                "       c.message_start_time_ns, c.message_end_time_ns, c.first_seen_at_ns, "
                "       c.compression, c.compressed_size_bytes, c.uncompressed_size_bytes "
                "FROM content c "
                "LEFT JOIN library lib  ON lib.id  = c.library_id "
                "LEFT JOIN profile prof ON prof.id = c.profile_id "
                "WHERE c.summary_fingerprint = ?",
                (summary_fp,),
            ).fetchone()
            topic_rows = conn.execute(
                "SELECT cc.mcap_channel_id, t.name AS topic, "
                "       sig.schema_id, sig.message_encoding, "
                "       cc.message_count, cc.uncompressed_size_bytes, "
                "       cc.message_start_time_ns, cc.message_end_time_ns, "
                "       cc.distribution_blob, "
                "       cm.metadata_json_zlib "
                "FROM content_channel cc "
                "JOIN content c       ON c.id        = cc.content_id "
                "JOIN channel_signature sig ON sig.id  = cc.channel_signature_id "
                "JOIN topic t         ON t.id          = sig.topic_id "
                "LEFT JOIN channel_metadata cm ON cm.id = sig.channel_metadata_id "
                "WHERE c.summary_fingerprint = ? "
                "ORDER BY cc.message_count DESC NULLS LAST, t.name",
                (summary_fp,),
            ).fetchall()
            schema_dim_rows = conn.execute(
                "SELECT DISTINCT s.id, s.name, s.encoding, s.size_bytes "
                "FROM content_channel cc "
                "JOIN content c       ON c.id        = cc.content_id "
                "JOIN channel_signature sig ON sig.id  = cc.channel_signature_id "
                "JOIN schema s        ON s.id      = sig.schema_id "
                "WHERE c.summary_fingerprint = ?",
                (summary_fp,),
            ).fetchall()
            observation_rows = conn.execute(
                "SELECT fp.value AS abs_path, obs.observed_at_ns, obs.scan_session_id, "
                "       obs.file_fingerprint, c.summary_fingerprint "
                "FROM file_observation obs "
                "JOIN file_path fp ON fp.id = obs.file_path_id "
                "LEFT JOIN content c ON c.id = obs.content_id "
                "WHERE c.summary_fingerprint = ? OR fp.value = ? "
                "ORDER BY obs.observed_at_ns DESC LIMIT 20",
                (summary_fp, abs_path or ""),
            ).fetchall()
            error_rows = conn.execute(
                "SELECT fp.value AS abs_path, se.observed_at_ns, se.error_kind, se.error_message "
                "FROM scan_error se "
                "JOIN file_path fp ON fp.id = se.file_path_id "
                "WHERE fp.value = ? "
                "ORDER BY se.observed_at_ns DESC LIMIT 10",
                (abs_path or "",),
            ).fetchall()
    except IndexDbNeedsMigrationError as exc:
        _print_db_needs_migration(exc)
        return 1

    (
        size_bytes,
        library,
        profile,
        message_count,
        schema_count,
        channel_count,
        attachment_count,
        metadata_count,
        chunk_count,
        start_ns,
        end_ns,
        first_seen_at,
        compression,
        compressed_size_bytes,
        uncompressed_size_bytes,
    ) = content

    identity = {
        "summary_fingerprint": summary_fp,
        "short_id": _short_id_from_fingerprint(summary_fp),
        "path": abs_path,
        "size_bytes": size_bytes,
        "library": library,
        "profile": profile,
        "message_count": message_count,
        "schema_count": schema_count,
        "channel_count": channel_count,
        "attachment_count": attachment_count,
        "metadata_count": metadata_count,
        "chunk_count": chunk_count,
        "message_start_time_ns": start_ns,
        "message_end_time_ns": end_ns,
        "duration_ns": (duration_ns := _safe_duration_ns(start_ns, end_ns)),
        "compression": compression,
        "compressed_size_bytes": compressed_size_bytes,
        "uncompressed_size_bytes": uncompressed_size_bytes,
        "first_seen_at_ns": first_seen_at,
    }
    schema_name_by_pk_id: dict[int, str | None] = {
        pk_id: name for pk_id, name, _enc, _sz in schema_dim_rows
    }
    topics_payload: list[_IndexedTopicPayload] = [
        {
            "channel_id": ch_id,
            "topic": topic,
            "schema_pk_id": schema_pk_id,
            "schema": schema_name_by_pk_id.get(schema_pk_id) if schema_pk_id is not None else None,
            "encoding": encoding,
            "message_count": msg_count,
            "uncompressed_size_bytes": ch_bytes,
            "message_start_time_ns": ch_start,
            "message_end_time_ns": ch_end,
            "duration_ns": _safe_duration_ns(ch_start, ch_end),
            "distribution": unpack_distribution_blob(dist_blob),
        }
        for (
            ch_id,
            topic,
            schema_pk_id,
            encoding,
            msg_count,
            ch_bytes,
            ch_start,
            ch_end,
            dist_blob,
            _metadata_zlib,
        ) in topic_rows
    ]
    qos_by_channel_id: dict[int, list[QosProfile]] = {}
    for row in topic_rows:
        channel_id = row[0]
        metadata_zlib = row[-1]
        if not metadata_zlib:
            continue
        try:
            metadata = _json.loads(zlib.decompress(metadata_zlib))
        except (zlib.error, _json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        profiles = parse_qos_profiles(metadata)
        if profiles:
            qos_by_channel_id[channel_id] = profiles
    observations_payload = [
        {
            "path": obs_path,
            "observed_at_ns": observed_at,
            "session_id": session_id,
            "file_fingerprint": file_fp,
            "summary_fingerprint": obs_summary_fp,
        }
        for obs_path, observed_at, session_id, file_fp, obs_summary_fp in observation_rows
    ]
    errors_payload = [
        {
            "path": err_path,
            "observed_at_ns": observed_at,
            "kind": kind,
            "message": message,
        }
        for err_path, observed_at, kind, message in error_rows
    ]

    if format == "json":
        _stdout_line(
            _json.dumps(
                {
                    "identity": identity,
                    "topics": topics_payload,
                    "observations": observations_payload,
                    "errors": errors_payload,
                },
                default=str,
            )
        )
        return 0

    identity_table = Table.grid(padding=(0, 1))
    identity_table.add_column(style="bold blue")
    identity_table.add_column()
    identity_table.add_row("Summary fingerprint:", f"[dim]{summary_fp}[/]")
    short_id = _short_id_from_fingerprint(summary_fp)
    if short_id:
        identity_table.add_row("Short ID:", f"[bold green]{short_id}[/]")
    if abs_path is not None:
        identity_table.add_row("Path:", _format_parts_with_colors(abs_path))
    if size_bytes is not None:
        size_cell = f"[green]{bytes_to_human(size_bytes)}[/] [dim]({size_bytes:,} B)[/]"
        if isinstance(duration_ns, int) and duration_ns > 0:
            bps = size_bytes / (duration_ns / 1_000_000_000)
            size_cell += (
                f" [red]{bytes_to_human(bps)}/s[/] [orange1]{bytes_to_human(bps * 3600)}/h[/]"
            )
        identity_table.add_row("Size:", size_cell)
    else:
        identity_table.add_row("Size:", "-")
    identity_table.add_row("Library:", f"[yellow]{library or '-'}[/]")
    identity_table.add_row("Profile:", f"[yellow]{profile or '-'}[/]")
    if isinstance(message_count, int):
        identity_table.add_row("Messages:", f"[green]{_format_count(message_count)}[/]")
    else:
        identity_table.add_row("Messages:", "-")
    identity_table.add_row(
        "Schemas / Channels / Chunks:",
        f"[green]{schema_count or 0:,}[/] / "
        f"[green]{channel_count or 0:,}[/] / "
        f"[cyan]{chunk_count or 0:,}[/]",
    )
    identity_table.add_row(
        "Attachments / Metadata:",
        f"[yellow]{attachment_count or 0:,}[/] / [cyan]{metadata_count or 0:,}[/]",
    )
    identity_table.add_row("Start:", _format_ts_ns(start_ns))
    identity_table.add_row("End:", _format_ts_ns(end_ns))
    identity_table.add_row(
        "Duration:",
        f"[cyan]{_format_duration_ns(0, duration_ns) if duration_ns is not None else '-'}[/]",
    )
    identity_table.add_row(
        "Compression:",
        _format_compression_cell(
            compression,
            compressed_size_bytes,
            uncompressed_size_bytes,
        ),
    )
    console.print("[bold cyan]Identity[/]")
    console.print(identity_table)

    if topics_payload:
        console.print(f"[bold cyan]Topics ({len(topics_payload):,})[/]")
        console.print(
            display_channels_table(
                _topics_to_channel_table_data(topics_payload, schema_dim_rows, duration_ns),
                console,
                responsive=False,
                index_duration=True,
                qos_by_channel_id=qos_by_channel_id,
            )
        )

    if observations_payload:
        obs_table = Table(title=f"Observations ({len(observations_payload):,})")
        obs_table.add_column("Path", overflow="fold")
        obs_table.add_column("Observed at (UTC)", style="cyan")
        obs_table.add_column("Session", justify="right", style="green")
        obs_table.add_column("File fp", style="dim")
        obs_table.add_column("Summary fp", style="dim")
        for entry in observations_payload:
            observed_at = entry["observed_at_ns"]
            obs_table.add_row(
                _format_parts_with_colors(str(entry["path"])),
                _format_ts_ns(observed_at if isinstance(observed_at, int) else None),
                str(entry["session_id"]),
                str(entry["file_fingerprint"] or "-"),
                str(entry["summary_fingerprint"] or "-"),
            )
        console.print(obs_table)

    if errors_payload:
        err_table = Table(title=f"Errors ({len(errors_payload):,})")
        err_table.add_column("Path", overflow="fold")
        err_table.add_column("Observed at (UTC)", style="cyan")
        err_table.add_column("Kind", style="red")
        err_table.add_column("Message", overflow="fold", style="dim")
        for entry in errors_payload:
            observed_at = entry["observed_at_ns"]
            err_table.add_row(
                _format_parts_with_colors(str(entry["path"])),
                _format_ts_ns(observed_at if isinstance(observed_at, int) else None),
                _describe_error_kind(str(entry["kind"])),
                str(entry["message"] or "-"),
            )
        console.print(err_table)

    return 0
