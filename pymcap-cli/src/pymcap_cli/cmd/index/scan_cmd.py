"""``pymcap-cli index scan`` — walk a folder and index every .mcap."""

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from pymcap_cli.cmd.index._helpers import (
    _describe_error_kind,
    _pymcap_cli_version,
    _resolve_db,
    console,
)
from pymcap_cli.index.db import open_db
from pymcap_cli.index.scanner import ScanStats, scan


def scan_cmd(
    folder: Path,
    *,
    db: Annotated[
        Path | None,
        Parameter(name=["--db"], help="Override the sidecar DB path."),
    ] = None,
    jobs: Annotated[
        int,
        Parameter(name=["-j", "--jobs"], help="Parallel I/O workers."),
    ] = 8,
    retry_errors: Annotated[
        bool,
        Parameter(
            name=["--retry-errors"],
            help="Retry files previously recorded in scan_error.",
        ),
    ] = False,
    rebuild_missing: Annotated[
        bool,
        Parameter(
            help=(
                "Rebuild summaries in memory for files without usable summary data. "
                "This can read entire MCAP files."
            ),
        ),
    ] = False,
    read_message_indexes: Annotated[
        bool,
        Parameter(
            name=["--read-message-indexes"],
            help=(
                "Read per-chunk MessageIndex records for per-channel size, time, "
                "and distribution metrics. Slower on files with many chunks."
            ),
        ),
    ] = False,
    force_rebuild: Annotated[
        bool,
        Parameter(
            name=["--force-rebuild"],
            help=(
                "Re-read every discovered file and refresh the derived columns "
                "on its ``content`` row (compression, sane times, …). Combine "
                "with --read-message-indexes to refresh per-channel metrics."
            ),
        ),
    ] = False,
    root_only: Annotated[
        bool,
        Parameter(name=["--root-only"], help="Do not recurse into subfolders."),
    ] = False,
) -> int:
    """Scan FOLDER recursively and index every .mcap into the sidecar DB.

    Unchanged files (matching path + size + mtime) are skipped without any
    file I/O. Files that moved or were duplicated can reuse an existing
    summary snapshot via the cheap byte probe.
    """
    folder = folder.expanduser().resolve()
    if not folder.exists():
        console.print(f"[red]Error:[/] {folder} does not exist")
        return 1

    db_path = _resolve_db(db)
    console.print(f"[dim]DB:[/dim] {db_path}")
    console.print(f"[dim]Root:[/dim] {folder}")

    with Progress(
        TextColumn("[bold blue]Scanning"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn(
            "[dim]walked={task.fields[walked]} dirs[/] "
            "[green]indexed={task.fields[indexed]}[/] "
            "[cyan]reused={task.fields[reused]}[/] "
            "[dim]skip={task.fields[skip]}[/] "
            "[red]err={task.fields[errored]}[/]"
        ),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress_bar:
        task_id = progress_bar.add_task(
            "scan", total=None, walked=0, indexed=0, reused=0, skip=0, errored=0
        )
        seen = 0

        def _progress(stats: ScanStats) -> None:
            nonlocal seen
            total = stats.discovered + stats.deleted
            if progress_bar.tasks[task_id].total != total:
                progress_bar.update(task_id, total=total)
            done = (
                stats.deleted
                + stats.stat_skipped
                + stats.error_skipped
                + stats.indexed
                + stats.fingerprint_reused
                + stats.errored
            )
            progress_bar.update(
                task_id,
                advance=done - seen,
                walked=stats.dirs_walked,
                indexed=stats.indexed,
                reused=stats.fingerprint_reused,
                skip=stats.stat_skipped + stats.error_skipped,
                errored=stats.errored,
            )
            seen = done

        with open_db(db_path) as conn:
            stats = scan(
                folder,
                conn,
                pymcap_cli_version=_pymcap_cli_version(),
                jobs=jobs,
                recurse=not root_only,
                retry_errors=retry_errors,
                rebuild_missing=rebuild_missing,
                read_message_indexes=read_message_indexes,
                force_rebuild=force_rebuild,
                progress=_progress,
            )

    table = Table(title="Scan summary", show_header=False)
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_row("Discovered", f"{stats.discovered:,}")
    table.add_row("Stat-skipped (unchanged)", f"{stats.stat_skipped:,}")
    table.add_row("Error-skipped (previously failed)", f"{stats.error_skipped:,}")
    table.add_row("Fingerprint-reused (moved/dup)", f"{stats.fingerprint_reused:,}")
    table.add_row("Indexed (new content)", f"{stats.indexed:,}")
    table.add_row("Deleted/stale paths", f"{stats.deleted:,}")
    table.add_row("Errored", f"{stats.errored:,}")
    for kind, count in sorted(stats.errored_by_kind.items()):
        table.add_row(f"  [dim]└ {_describe_error_kind(kind)}[/]", f"{count:,}")
    console.print(table)
    return 0
