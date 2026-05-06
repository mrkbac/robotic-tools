"""Doctor command - validate MCAP container structure and indexes."""

from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import ExitStack
from typing import IO, Annotated

from cyclopts import Parameter
from rich.console import Console
from rich.table import Table
from rich.text import Text

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.doctor import DoctorReport, Finding, Severity, examine_mcap
from pymcap_cli.log_setup import ERR
from pymcap_cli.utils import ProgressTrackingIO, file_progress

logger = logging.getLogger(__name__)
console = Console()


def doctor(
    file: str,
    *,
    strict_message_order: Annotated[
        bool,
        Parameter(
            name=["--strict-message-order"],
            help="Treat non-monotonic message log_time as an error instead of info.",
        ),
    ] = False,
    show_all: Annotated[
        bool,
        Parameter(
            name=["--show-all"],
            help="Print every finding individually instead of the grouped summary.",
        ),
    ] = False,
) -> int:
    """Check an MCAP file structure against the MCAP container specification."""
    try:
        with open_input(file, buffering=0) as (stream, size), ExitStack() as stack:
            scanned: IO[bytes] = stream
            if size:
                progress = file_progress("[bold blue]MCAP doctor scanning...")
                progress.start()
                stack.callback(progress.stop)
                task = progress.add_task("Examining", total=size)
                scanned = ProgressTrackingIO(stream, task, progress, stream.tell())
            report = examine_mcap(
                scanned,
                size,
                str(file),
                strict_message_order=strict_message_order,
            )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 0
    except Exception as exc:
        ERR.print(f"[red]Doctor command failed:[/red] {exc}")
        logger.exception("Doctor command failed")
        return 1

    _print_report(report, show_all=show_all)
    return 1 if report.error_count else 0


def _print_report(report: DoctorReport, *, show_all: bool) -> None:
    if not report.findings:
        console.print(
            f"[green]OK[/green] {report.path} passed MCAP doctor checks "
            f"({report.record_count:,} records, {report.chunk_count:,} chunks, "
            f"{report.message_count:,} messages)."
        )
        return

    if show_all:
        console.print(_findings_table(report.findings))
    else:
        console.print(_summary_table(report.findings))

    if report.error_count == 0 and report.warning_count == 0:
        console.print(
            f"[green]OK[/green] {report.path} passed MCAP doctor checks with "
            f"[cyan]{report.info_count} info[/cyan] "
            f"({report.record_count:,} records, {report.chunk_count:,} chunks, "
            f"{report.message_count:,} messages)."
        )
        return

    console.print(
        f"{report.path}: "
        f"[red]{report.error_count} errors[/red], "
        f"[yellow]{report.warning_count} warnings[/yellow], "
        f"[cyan]{report.info_count} info[/cyan], "
        f"{report.record_count:,} records, "
        f"{report.chunk_count:,} chunks, "
        f"{report.message_count:,} messages"
    )


def _findings_table(findings: list[Finding]) -> Table:
    table = Table(title="MCAP Doctor Findings")
    table.add_column("Severity", no_wrap=True)
    table.add_column("Code", no_wrap=True)
    table.add_column("Offset", justify="right", no_wrap=True)
    table.add_column("Section", no_wrap=True)
    table.add_column("Record", no_wrap=True)
    table.add_column("Message")
    for finding in findings:
        table.add_row(
            _severity_text(finding),
            finding.code,
            "" if finding.offset is None else str(finding.offset),
            finding.section.value,
            finding.record,
            finding.message,
        )
    return table


def _summary_table(findings: list[Finding]) -> Table:
    grouped: dict[tuple[str, str], list[Finding]] = defaultdict(list)
    for finding in findings:
        grouped[(finding.severity, finding.code)].append(finding)

    table = Table(title="MCAP Doctor Summary")
    table.add_column("Severity", no_wrap=True)
    table.add_column("Code", no_wrap=True)
    table.add_column("Count", justify="right", no_wrap=True)
    table.add_column("First Offset", justify="right", no_wrap=True)
    table.add_column("Sample Message")

    severity_rank = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    for (_severity, code), group in sorted(
        grouped.items(), key=lambda item: (severity_rank.get(item[0][0], 99), -len(item[1]))
    ):
        first = group[0]
        offsets = [f.offset for f in group if f.offset is not None]
        first_offset = "" if not offsets else str(min(offsets))
        table.add_row(
            _severity_text(first),
            code,
            f"{len(group):,}",
            first_offset,
            first.message,
        )
    return table


def _severity_text(finding: Finding) -> Text:
    if finding.severity is Severity.ERROR:
        return Text("error", style="red")
    if finding.severity is Severity.WARNING:
        return Text("warning", style="yellow")
    return Text("info", style="cyan")
