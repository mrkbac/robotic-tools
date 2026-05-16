"""Recursive scanner that populates the sidecar index database.

Single writer thread (owns the SQLite connection) drains scan results from a
pool of I/O workers. Worker work per file:

  1. ``stat`` -> (size, mtime_ns, inode).
  2. Stat hit: existing ``file_observation`` row with same (path, size, mtime_ns)
     => skip without any file I/O.
  3. Error suppression: last ``scan_error`` row with matching (size, mtime_ns)
     and ``--retry-errors`` not set => skip.
  4. ``file_fingerprint`` (bounded head+tail+size byte probe).
  5. Byte-cache hit: ``file_fingerprint`` already seen with a known
     ``summary_fingerprint`` => alias to the existing content row without
     opening / rebuilding the summary.
  6. Otherwise: load the MCAP summary natively. Files without usable summary
     statistics are skipped unless ``rebuild_missing`` is enabled.
  7. Content hit: ``summary_fingerprint`` already in ``content`` => alias;
     otherwise insert content + child rows in one transaction.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import stat
import threading
import time
import zlib
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING

import xxhash
from small_mcap import McapError, get_header, get_summary, rebuild_summary

from pymcap_cli.index.db import finish_session, start_session
from pymcap_cli.index.fingerprint import fingerprint_stream
from pymcap_cli.index.summary_fingerprint import compute_schema_hash_map, summary_fingerprint

if TYPE_CHECKING:
    import sqlite3

    from small_mcap.rebuild import RebuildInfo

logger = logging.getLogger(__name__)


class _SummaryUnavailableError(Exception):
    """Raised when indexing would need a full data-section rebuild."""


# Per-process state for worker pool — populated once by ``_worker_init`` in
# each child process, then read by ``_process_file_task`` for every job.
_WORKER_FILE_FP_CACHE: dict[str, str] = {}
_WORKER_KNOWN_SUMMARY_FPS: set[str] = set()


def _worker_init(file_fp_cache: dict[str, str], known_summary_fps: set[str]) -> None:
    global _WORKER_FILE_FP_CACHE, _WORKER_KNOWN_SUMMARY_FPS
    _WORKER_FILE_FP_CACHE = file_fp_cache
    _WORKER_KNOWN_SUMMARY_FPS = known_summary_fps


def _process_file_task(inp: "_ScanInput", rebuild_missing: bool) -> "_ScanResult":
    return _process_file(
        inp,
        file_fp_cache=_WORKER_FILE_FP_CACHE,
        known_summary_fps=_WORKER_KNOWN_SUMMARY_FPS,
        rebuild_missing=rebuild_missing,
    )


@dataclass(slots=True)
class ScanStats:
    discovered: int = 0
    dirs_walked: int = 0
    stat_skipped: int = 0
    error_skipped: int = 0
    fingerprint_reused: int = 0
    indexed: int = 0
    deleted: int = 0
    errored: int = 0
    errored_by_kind: dict[str, int] = field(default_factory=dict)


def _bump_error_kind(stats: ScanStats, kind: str) -> None:
    stats.errored_by_kind[kind] = stats.errored_by_kind.get(kind, 0) + 1


ScanProgressCallback = Callable[[ScanStats], None]


@dataclass(slots=True)
class _ScanInput:
    path: Path
    size_bytes: int
    mtime_ns: int
    inode: int


@dataclass(slots=True)
class _ContentRow:
    summary_fingerprint: str
    size_bytes: int
    library: str
    profile: str
    statistics_present: bool
    message_count: int
    schema_count: int
    channel_count: int
    attachment_count: int
    metadata_count: int
    chunk_count: int
    message_start_time: int
    message_end_time: int
    # MIN/MAX over chunks whose own start clears ``_SANE_EPOCH_NS``. ``None``
    # when no chunk qualifies, in which case the duration is unknown.
    sane_message_start_time: int | None
    sane_message_end_time: int | None
    scan_kind: str
    channels: list[dict] = field(default_factory=list)
    schemas: list[dict] = field(default_factory=list)


# 2000-01-01T00:00:00Z in ns. Mirrors ``cmd.index_cmd._SANE_EPOCH_NS`` and
# ``migrations.0002_normalise_and_drop_chunks._SANE_EPOCH_NS`` — keep in sync.
_SANE_EPOCH_NS = 946_684_800 * 1_000_000_000


@dataclass(slots=True)
class _ScanResult:
    inp: _ScanInput
    file_fingerprint: str | None = None
    summary_fingerprint: str | None = None
    content: _ContentRow | None = None
    error_kind: str | None = None
    error_message: str | None = None


_DEFAULT_WALKER_WORKERS = 8


def _iter_mcap_files(
    root: Path,
    *,
    recurse: bool,
    on_dir_done: Callable[[int], None] | None = None,
    walker_workers: int = _DEFAULT_WALKER_WORKERS,
) -> Iterator[tuple[Path, os.stat_result | None]]:
    """Yield ``(path, stat_result | None)`` for every ``.mcap`` under ``root``.

    Recursive walks spread ``os.scandir()`` and ``entry.stat()`` across
    ``walker_workers`` threads. On slow mounts (NFS, FUSE, SSHFS) most of the
    per-directory cost is round-trip latency rather than CPU work, so even
    a small thread pool overlaps those round-trips and cuts walker
    wall-clock dramatically. ``walker_workers <= 1`` falls back to a serial
    BFS that uses no threads.

    The walker hands the caller the ``stat_result`` from each entry, so the
    main loop avoids a second ``stat`` per file. ``follow_symlinks=False``
    on directory descent matches :meth:`Path.rglob`'s default and skips
    venv / cache symlinks.

    ``on_dir_done`` is called with the running directory count after every
    finished directory, so the caller can drive a progress UI even before
    the first ``.mcap`` is yielded.
    """
    if root.is_file():
        if root.suffix == ".mcap":
            yield root.resolve(), None
        return
    if not root.is_dir():
        return

    root_str = str(root.resolve())

    if not recurse:
        try:
            scan_it = os.scandir(root_str)
        except OSError:
            return
        with scan_it:
            for entry in scan_it:
                if not entry.name.endswith(".mcap"):
                    continue
                try:
                    st = entry.stat()
                except OSError:
                    continue
                if stat.S_ISREG(st.st_mode):
                    yield Path(entry.path), st
        if on_dir_done is not None:
            on_dir_done(1)
        return

    if walker_workers <= 1:
        yield from _iter_mcap_files_serial(root_str, on_dir_done=on_dir_done)
    else:
        yield from _iter_mcap_files_parallel(
            root_str, walker_workers=walker_workers, on_dir_done=on_dir_done
        )


def _iter_mcap_files_serial(
    root_str: str,
    *,
    on_dir_done: Callable[[int], None] | None,
) -> Iterator[tuple[Path, os.stat_result]]:
    """Single-threaded BFS used when ``walker_workers <= 1``."""
    stack: list[str] = [root_str]
    dirs_done = 0
    while stack:
        directory = stack.pop()
        try:
            scan_it = os.scandir(directory)
        except OSError:
            continue
        with scan_it:
            for entry in scan_it:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                        continue
                    if not entry.name.endswith(".mcap"):
                        continue
                    st = entry.stat()
                except OSError:
                    continue
                if stat.S_ISREG(st.st_mode):
                    yield Path(entry.path), st
        dirs_done += 1
        if on_dir_done is not None and dirs_done % 200 == 0:
            on_dir_done(dirs_done)
    if on_dir_done is not None:
        on_dir_done(dirs_done)


def _iter_mcap_files_parallel(
    root_str: str,
    *,
    walker_workers: int,
    on_dir_done: Callable[[int], None] | None,
) -> Iterator[tuple[Path, os.stat_result]]:
    """Walk the tree with a small thread pool issuing scandir() in parallel.

    Each worker pulls a directory path off ``dir_q``, scandirs it, pushes
    subdirectories back onto ``dir_q``, and feeds ``(path, stat)`` tuples to
    ``out_q`` for the main thread to yield. An ``in_flight`` counter (under
    ``state_lock``) tracks how many directories are still pending — when it
    hits zero the workers send a sentinel and exit.
    """
    out_q: queue.Queue[tuple[Path, os.stat_result] | None] = queue.Queue(maxsize=8192)
    dir_q: queue.Queue[str] = queue.Queue()
    dir_q.put(root_str)

    state_lock = threading.Lock()
    state = {"in_flight": 1, "dirs_done": 0, "shutdown": False}
    sentinels_sent = [False]

    def _worker() -> None:
        while True:
            try:
                directory = dir_q.get(timeout=0.25)
            except queue.Empty:
                with state_lock:
                    if state["shutdown"] or state["in_flight"] == 0:
                        return
                continue
            new_dirs: list[str] = []
            try:
                with os.scandir(directory) as it:
                    for entry in it:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                new_dirs.append(entry.path)
                                continue
                            if not entry.name.endswith(".mcap"):
                                continue
                            st = entry.stat()
                        except OSError:
                            continue
                        if stat.S_ISREG(st.st_mode):
                            out_q.put((Path(entry.path), st))
            except OSError:
                pass
            with state_lock:
                state["in_flight"] += len(new_dirs)
                state["in_flight"] -= 1
                state["dirs_done"] += 1
                local_dirs_done = state["dirs_done"]
                drained = state["in_flight"] == 0
            for d in new_dirs:
                dir_q.put(d)
            if on_dir_done is not None and local_dirs_done % 200 == 0:
                # Rich progress's update() is thread-safe; the only shared
                # state we mutate from the callback is ScanStats.dirs_walked,
                # whose monotonic-int set is harmless under a small race.
                on_dir_done(local_dirs_done)
            if drained:
                with state_lock:
                    if not sentinels_sent[0]:
                        sentinels_sent[0] = True
                        for _ in range(walker_workers):
                            out_q.put(None)
                return

    threads = [
        threading.Thread(target=_worker, name=f"walker-{i}", daemon=True)
        for i in range(walker_workers)
    ]
    for t in threads:
        t.start()

    pending_sentinels = walker_workers
    try:
        while pending_sentinels > 0:
            item = out_q.get()
            if item is None:
                pending_sentinels -= 1
                continue
            yield item
    finally:
        with state_lock:
            state["shutdown"] = True
        for t in threads:
            t.join(timeout=1.0)
        if on_dir_done is not None:
            on_dir_done(state["dirs_done"])


def _stat_skip(conn: sqlite3.Connection, inp: _ScanInput) -> bool:
    row = conn.execute(
        """SELECT size_bytes, mtime_ns, content_id FROM file_observation
           WHERE abs_path=? ORDER BY id DESC LIMIT 1""",
        (str(inp.path),),
    ).fetchone()
    if row is None:
        return False
    size_bytes, mtime_ns, content_id = row
    return size_bytes == inp.size_bytes and mtime_ns == inp.mtime_ns and content_id is not None


def _error_skip(
    conn: sqlite3.Connection,
    inp: _ScanInput,
    *,
    rebuild_missing: bool,
) -> bool:
    row = conn.execute(
        """SELECT error_kind FROM scan_error
           WHERE abs_path=? AND size_bytes=? AND mtime_ns=?
           ORDER BY id DESC LIMIT 1""",
        (str(inp.path), inp.size_bytes, inp.mtime_ns),
    ).fetchone()
    if row is None:
        return False
    return not (rebuild_missing and row[0] == "no_summary")


def _content_exists(conn: sqlite3.Connection, summary_fp: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM content WHERE summary_fingerprint=? LIMIT 1",
        (summary_fp,),
    ).fetchone()
    return row is not None


def _current_paths_for_root(conn: sqlite3.Connection, root: Path, *, recurse: bool) -> set[str]:
    root = root.resolve()
    if root.is_file():
        rows = conn.execute(
            "SELECT abs_path FROM current_file WHERE abs_path = ?",
            (str(root),),
        ).fetchall()
        return {row[0] for row in rows}

    root_str = str(root)
    child_prefix = root_str if root_str.endswith(os.sep) else f"{root_str}{os.sep}"
    rows = conn.execute(
        "SELECT abs_path FROM current_file WHERE abs_path = ? OR substr(abs_path, 1, ?) = ?",
        (root_str, len(child_prefix), child_prefix),
    ).fetchall()
    paths = {row[0] for row in rows}
    if recurse:
        return paths
    return {path for path in paths if Path(path).parent == root}


def _load_summary_info(f: IO[bytes], *, rebuild_missing: bool) -> tuple[RebuildInfo, str]:
    """Return (info, scan_kind). ``info.summary.statistics`` is guaranteed non-None.

    Tries the native summary first (fast). Rebuilds the full data section only
    when the caller opts in with ``rebuild_missing``.
    """
    from small_mcap.rebuild import RebuildInfo as _RebuildInfo  # noqa: PLC0415

    f.seek(0)
    header = get_header(f)
    try:
        summary = get_summary(f)
    except (McapError, AssertionError):
        summary = None

    if summary is not None and summary.statistics is not None and header is not None:
        return _RebuildInfo(header=header, summary=summary), "summary"

    if not rebuild_missing:
        if summary is not None and summary.statistics is None:
            raise _SummaryUnavailableError("MCAP summary is missing Statistics")
        raise _SummaryUnavailableError("MCAP has no readable summary section")

    f.seek(0)
    result = rebuild_summary(
        f,
        validate_crc=False,
        calculate_channel_sizes=False,
        exact_sizes=False,
    )
    return result, "rebuilt"


def _build_content_row(
    summary_fp: str,
    info: RebuildInfo,
    size_bytes: int,
    scan_kind: str,
    schema_hash_by_id: dict[int, str],
) -> _ContentRow:
    header = info.header
    summary = info.summary
    stats = summary.statistics
    channel_message_counts = stats.channel_message_counts if stats is not None else {}

    channels = [
        {
            "channel_id": ch.id,
            "topic": ch.topic,
            "schema_id": ch.schema_id,
            "message_encoding": ch.message_encoding,
            "metadata": json.dumps(ch.metadata) if ch.metadata else None,
            "message_count": channel_message_counts.get(ch.id),
        }
        for ch in summary.channels.values()
    ]

    schemas = [
        {
            "schema_id": sc.id,
            "name": sc.name,
            "encoding": sc.encoding,
            "schema_size": len(sc.data),
            "schema_hash": schema_hash_by_id[sc.id],
        }
        for sc in summary.schemas.values()
    ]

    sane_starts = [
        ci.message_start_time
        for ci in summary.chunk_indexes
        if ci.message_start_time >= _SANE_EPOCH_NS
    ]
    sane_ends = [
        ci.message_end_time
        for ci in summary.chunk_indexes
        if ci.message_start_time >= _SANE_EPOCH_NS
    ]

    return _ContentRow(
        summary_fingerprint=summary_fp,
        size_bytes=size_bytes,
        library=header.library,
        profile=header.profile,
        statistics_present=stats is not None,
        message_count=(stats.message_count if stats is not None else 0),
        schema_count=(stats.schema_count if stats is not None else len(summary.schemas)),
        channel_count=(stats.channel_count if stats is not None else len(summary.channels)),
        attachment_count=(stats.attachment_count if stats is not None else 0),
        metadata_count=(stats.metadata_count if stats is not None else 0),
        chunk_count=(stats.chunk_count if stats is not None else len(summary.chunk_indexes)),
        message_start_time=(stats.message_start_time if stats is not None else 0),
        message_end_time=(stats.message_end_time if stats is not None else 0),
        sane_message_start_time=min(sane_starts) if sane_starts else None,
        sane_message_end_time=max(sane_ends) if sane_ends else None,
        scan_kind=scan_kind,
        channels=channels,
        schemas=schemas,
    )


def _process_file(
    inp: _ScanInput,
    *,
    file_fp_cache: dict[str, str],
    known_summary_fps: set[str],
    rebuild_missing: bool,
) -> _ScanResult:
    """Worker step. Computes file_fingerprint, then either reuses the cached
    summary_fingerprint (byte-cache hit) or loads/rebuilds the summary and
    computes a new summary_fingerprint.
    """
    file_fp: str | None = None
    try:
        with inp.path.open("rb") as f:
            file_fp = fingerprint_stream(f, inp.size_bytes)
        cached_summary_fp = file_fp_cache.get(file_fp)
        if cached_summary_fp is not None:
            return _ScanResult(
                inp=inp,
                file_fingerprint=file_fp,
                summary_fingerprint=cached_summary_fp,
            )

        with inp.path.open("rb") as f:
            info, scan_kind = _load_summary_info(f, rebuild_missing=rebuild_missing)
        schema_hash_by_id = compute_schema_hash_map(info.summary)
        summary_fp = summary_fingerprint(info, schema_hash_by_id=schema_hash_by_id)

        if summary_fp in known_summary_fps:
            # Same logical content as another file we've already inserted —
            # no need to build a full content row.
            return _ScanResult(
                inp=inp,
                file_fingerprint=file_fp,
                summary_fingerprint=summary_fp,
            )

        content = _build_content_row(
            summary_fp, info, inp.size_bytes, scan_kind, schema_hash_by_id
        )
        return _ScanResult(
            inp=inp,
            file_fingerprint=file_fp,
            summary_fingerprint=summary_fp,
            content=content,
        )
    except OSError as exc:
        return _ScanResult(
            inp=inp,
            file_fingerprint=file_fp,
            error_kind="io",
            error_message=str(exc),
        )
    except _SummaryUnavailableError as exc:
        return _ScanResult(
            inp=inp,
            file_fingerprint=file_fp,
            error_kind="no_summary",
            error_message=str(exc),
        )
    except McapError as exc:
        return _ScanResult(
            inp=inp,
            file_fingerprint=file_fp,
            error_kind="corrupt",
            error_message=str(exc),
        )


def _record_observation(
    conn: sqlite3.Connection,
    inp: _ScanInput,
    file_fp: str,
    content_id: int | None,
    session_id: int,
) -> None:
    conn.execute(
        """INSERT INTO file_observation
           (abs_path, size_bytes, mtime_ns, inode, file_fingerprint,
            content_id, session_id, observed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(inp.path),
            inp.size_bytes,
            inp.mtime_ns,
            inp.inode,
            file_fp,
            content_id,
            session_id,
            time.time_ns(),
        ),
    )


def _record_error(
    conn: sqlite3.Connection,
    inp: _ScanInput,
    kind: str,
    message: str,
    session_id: int,
) -> None:
    conn.execute(
        """INSERT INTO scan_error
           (abs_path, size_bytes, mtime_ns, session_id, observed_at, error_kind, error_message)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            str(inp.path),
            inp.size_bytes,
            inp.mtime_ns,
            session_id,
            time.time_ns(),
            kind,
            message,
        ),
    )


def _record_deletion(conn: sqlite3.Connection, abs_path: str, session_id: int) -> None:
    conn.execute(
        """INSERT INTO file_observation
           (abs_path, size_bytes, mtime_ns, inode, file_fingerprint,
            content_id, is_deleted, session_id, observed_at)
           VALUES (?, 0, 0, NULL, '', NULL, 1, ?, ?)""",
        (abs_path, session_id, time.time_ns()),
    )


def _insert_content(
    conn: sqlite3.Connection,
    content: _ContentRow,
    session_id: int,
    *,
    topic_id_cache: dict[str, int],
    schema_pk_id_cache: dict[str, int],
    channel_sig_cache: dict[tuple[int, int, str, int], int],
    metadata_id_cache: dict[str, int],
) -> int:
    """Insert ``content`` + child rows. Returns the new ``content_id``.

    Mutates the caches so repeated topic / schema / channel-signature lookups
    within one scan don't re-roundtrip to SQLite.
    """
    cur = conn.execute(
        """INSERT INTO content
           (summary_fingerprint, size_bytes, library, profile, message_count, schema_count,
            channel_count, attachment_count, metadata_count, chunk_count,
            message_start_time, message_end_time,
            sane_message_start_time, sane_message_end_time,
            scan_kind, first_seen_at, first_seen_session)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            content.summary_fingerprint,
            content.size_bytes,
            content.library,
            content.profile,
            content.message_count,
            content.schema_count,
            content.channel_count,
            content.attachment_count,
            content.metadata_count,
            content.chunk_count,
            content.message_start_time,
            content.message_end_time,
            content.sane_message_start_time,
            content.sane_message_end_time,
            content.scan_kind,
            time.time_ns(),
            session_id,
        ),
    )
    content_id = cur.lastrowid
    assert content_id is not None

    # ``schema`` dimension table — one row per canonical schema hash.
    for sc in content.schemas:
        sh = sc["schema_hash"]
        if sh in schema_pk_id_cache:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO schema(schema_hash, name, encoding, schema_size) "
            "VALUES (?, ?, ?, ?)",
            (sh, sc["name"], sc["encoding"], sc["schema_size"]),
        )
        row = conn.execute(
            "SELECT schema_pk_id FROM schema WHERE schema_hash = ?", (sh,)
        ).fetchone()
        schema_pk_id_cache[sh] = row[0]

    # Map file-local schema_id → canonical schema_pk_id (NULL when the
    # channel declares no schema, i.e. schema_id == 0 in MCAP).
    schema_pk_id_by_local: dict[int, int | None] = {
        sc["schema_id"]: schema_pk_id_cache[sc["schema_hash"]] for sc in content.schemas
    }

    # ``topic`` dimension table — one row per channel topic string.
    for ch in content.channels:
        topic = ch["topic"]
        if topic in topic_id_cache:
            continue
        conn.execute("INSERT OR IGNORE INTO topic(name) VALUES (?)", (topic,))
        row = conn.execute(
            "SELECT topic_id FROM topic WHERE name = ?", (topic,)
        ).fetchone()
        topic_id_cache[topic] = row[0]

    # ``channel_metadata`` dim — same JSON metadata repeats heavily across
    # files. Intern + zlib-compress so the channel_sig row carries only an
    # INTEGER FK.
    def _intern_metadata(raw: str | None) -> int | None:
        if raw is None:
            return None
        cached = metadata_id_cache.get(raw)
        if cached is not None:
            return cached
        encoded = raw.encode("utf-8")
        content_hash = xxhash.xxh3_128_hexdigest(encoded)
        conn.execute(
            "INSERT OR IGNORE INTO channel_metadata(content_hash, blob_zlib) VALUES (?, ?)",
            (content_hash, zlib.compress(encoded, 6)),
        )
        row = conn.execute(
            "SELECT metadata_id FROM channel_metadata WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        metadata_id_cache[raw] = row[0]
        return row[0]

    # ``channel_sig`` dimension — collapses the
    # (topic, schema, encoding, metadata) tuple that repeats heavily across
    # files. Now keyed by the small INTEGER ``metadata_id`` from
    # ``channel_metadata`` rather than the raw blob.
    channel_rows: list[tuple[int, int, int, int | None]] = []
    for ch in content.channels:
        topic_id = topic_id_cache[ch["topic"]]
        schema_pk_id = schema_pk_id_by_local.get(ch["schema_id"])
        encoding = ch["message_encoding"]
        metadata_id = _intern_metadata(ch["metadata"])
        sig_key = (
            topic_id,
            schema_pk_id if schema_pk_id is not None else 0,
            encoding or "",
            metadata_id if metadata_id is not None else 0,
        )
        sig_id = channel_sig_cache.get(sig_key)
        if sig_id is None:
            conn.execute(
                "INSERT OR IGNORE INTO channel_sig"
                " (topic_id, schema_pk_id, message_encoding, channel_metadata_id)"
                " VALUES (?, ?, ?, ?)",
                (topic_id, schema_pk_id, encoding, metadata_id),
            )
            row = conn.execute(
                "SELECT channel_sig_id FROM channel_sig"
                " WHERE topic_id = ?"
                "   AND COALESCE(schema_pk_id, 0) = ?"
                "   AND COALESCE(message_encoding, '') = ?"
                "   AND COALESCE(channel_metadata_id, 0) = ?",
                sig_key,
            ).fetchone()
            sig_id = row[0]
            channel_sig_cache[sig_key] = sig_id
        channel_rows.append((content_id, ch["channel_id"], sig_id, ch["message_count"]))

    conn.executemany(
        """INSERT INTO content_channel
           (content_id, channel_id, channel_sig_id, message_count)
           VALUES (?, ?, ?, ?)""",
        channel_rows,
    )
    conn.executemany(
        """INSERT INTO content_schema
           (content_id, schema_id, schema_pk_id)
           VALUES (?, ?, ?)""",
        [
            (content_id, sc["schema_id"], schema_pk_id_cache[sc["schema_hash"]])
            for sc in content.schemas
        ],
    )
    return content_id


def _existing_summary_fingerprints(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT summary_fingerprint FROM content")}


def _existing_file_to_summary(conn: sqlite3.Connection) -> dict[str, str]:
    """Map file_fingerprint → summary_fingerprint from prior scans.

    Used to short-circuit the rebuild step when a file with the same bounded
    byte probe is encountered again under a new path.
    """
    return {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT DISTINCT fo.file_fingerprint, c.summary_fingerprint "
            "FROM file_observation fo "
            "JOIN content c ON c.content_id = fo.content_id "
            "WHERE fo.content_id IS NOT NULL"
        )
    }


def scan(
    root: Path,
    conn: sqlite3.Connection,
    *,
    pymcap_cli_version: str,
    jobs: int = 8,
    recurse: bool = True,
    retry_errors: bool = False,
    rebuild_missing: bool = False,
    progress: ScanProgressCallback | None = None,
) -> ScanStats:
    """Scan ``root`` and update ``conn``.

    The connection is used exclusively by this function; do not share it with
    concurrent writers.
    """
    stats = ScanStats()
    current_paths = _current_paths_for_root(conn, root, recurse=recurse)
    session_id: int | None = None

    def _ensure_session() -> int:
        nonlocal session_id
        if session_id is None:
            session_id = start_session(conn, root, pymcap_cli_version)
        return session_id

    # Per-scan caches.
    content_id_by_fp: dict[str, int] = {}
    topic_id_cache: dict[str, int] = {}
    schema_pk_id_cache: dict[str, int] = {}
    channel_sig_cache: dict[tuple[int, int, str, int], int] = {}
    metadata_id_cache: dict[str, int] = {}

    def _resolve_content_id(summary_fp: str) -> int:
        cid = content_id_by_fp.get(summary_fp)
        if cid is not None:
            return cid
        row = conn.execute(
            "SELECT content_id FROM content WHERE summary_fingerprint = ?",
            (summary_fp,),
        ).fetchone()
        assert row is not None, f"content row missing for {summary_fp!r}"
        content_id_by_fp[summary_fp] = row[0]
        return row[0]

    def _record_scan_error(inp: _ScanInput, kind: str, message: str, file_fp: str = "") -> None:
        active_session_id = _ensure_session()
        conn.execute("BEGIN")
        try:
            _record_error(conn, inp, kind, message, active_session_id)
            _record_observation(conn, inp, file_fp, None, active_session_id)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def _record_scan_result(result: _ScanResult) -> None:
        """Apply one scan result. Caller wraps a batch in a single BEGIN/COMMIT."""
        active_session_id = _ensure_session()
        if result.error_kind is not None:
            _record_error(
                conn,
                result.inp,
                result.error_kind,
                result.error_message or "",
                active_session_id,
            )
            _record_observation(
                conn,
                result.inp,
                result.file_fingerprint or "",
                None,
                active_session_id,
            )
            stats.errored += 1
            _bump_error_kind(stats, result.error_kind)
        elif result.summary_fingerprint is not None and result.content is None:
            content_id = _resolve_content_id(result.summary_fingerprint)
            _record_observation(
                conn,
                result.inp,
                result.file_fingerprint or "",
                content_id,
                active_session_id,
            )
            file_fp_cache[result.file_fingerprint or ""] = result.summary_fingerprint
            stats.fingerprint_reused += 1
        elif result.summary_fingerprint is not None and result.content is not None:
            if _content_exists(conn, result.summary_fingerprint):
                content_id = _resolve_content_id(result.summary_fingerprint)
                _record_observation(
                    conn,
                    result.inp,
                    result.file_fingerprint or "",
                    content_id,
                    active_session_id,
                )
                stats.fingerprint_reused += 1
            else:
                content_id = _insert_content(
                    conn,
                    result.content,
                    active_session_id,
                    topic_id_cache=topic_id_cache,
                    schema_pk_id_cache=schema_pk_id_cache,
                    channel_sig_cache=channel_sig_cache,
                    metadata_id_cache=metadata_id_cache,
                )
                content_id_by_fp[result.summary_fingerprint] = content_id
                _record_observation(
                    conn,
                    result.inp,
                    result.file_fingerprint or "",
                    content_id,
                    active_session_id,
                )
                known_summary_fps.add(result.summary_fingerprint)
                stats.indexed += 1
            file_fp_cache[result.file_fingerprint or ""] = result.summary_fingerprint

        if progress is not None:
            progress(stats)

    try:
        known_summary_fps = _existing_summary_fingerprints(conn)
        file_fp_cache = _existing_file_to_summary(conn)
        file_fp_cache_snapshot = dict(file_fp_cache)
        known_summary_fps_snapshot = set(known_summary_fps)

        max_workers = max(1, jobs)
        max_pending = max_workers * 4
        pending: set[Future[_ScanResult]] = set()

        def _drain_one_or_more() -> None:
            nonlocal pending
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            if not done:
                return
            conn.execute("BEGIN")
            try:
                for future in done:
                    _record_scan_result(future.result())
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        def _on_dir_done(dirs: int) -> None:
            stats.dirs_walked = dirs
            if progress is not None:
                progress(stats)

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(file_fp_cache_snapshot, known_summary_fps_snapshot),
        ) as pool:
            for path, walker_st in _iter_mcap_files(
                root, recurse=recurse, on_dir_done=_on_dir_done
            ):
                stats.discovered += 1
                _ensure_session()
                abs_path = str(path)
                current_paths.discard(abs_path)
                if walker_st is not None:
                    st: os.stat_result | None = walker_st
                else:
                    try:
                        st = path.stat()
                    except OSError as exc:
                        stats.errored += 1
                        _bump_error_kind(stats, "io")
                        inp = _ScanInput(path=path, size_bytes=0, mtime_ns=0, inode=0)
                        _record_scan_error(inp, "io", str(exc))
                        if progress is not None:
                            progress(stats)
                        continue
                inp = _ScanInput(
                    path=path,
                    size_bytes=st.st_size,
                    mtime_ns=st.st_mtime_ns,
                    inode=st.st_ino,
                )
                if _stat_skip(conn, inp):
                    stats.stat_skipped += 1
                    if progress is not None:
                        progress(stats)
                    continue
                if not retry_errors and _error_skip(conn, inp, rebuild_missing=rebuild_missing):
                    stats.error_skipped += 1
                    if progress is not None:
                        progress(stats)
                    continue
                pending.add(
                    pool.submit(_process_file_task, inp, rebuild_missing)
                )
                if len(pending) >= max_pending:
                    _drain_one_or_more()

            while pending:
                _drain_one_or_more()

        for abs_path in sorted(current_paths):
            active_session_id = _ensure_session()
            conn.execute("BEGIN")
            try:
                _record_deletion(conn, abs_path, active_session_id)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            stats.deleted += 1
            if progress is not None:
                progress(stats)

        return stats
    finally:
        if session_id is not None:
            finish_session(conn, session_id)
