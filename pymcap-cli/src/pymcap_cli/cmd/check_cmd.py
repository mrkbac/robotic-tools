"""Check an MCAP recording against a YAML contract."""

import logging

from rich.console import Console
from rich.table import Table
from rich.text import Text

from pymcap_cli.check import (
    OK,
    CheckReport,
    CheckResult,
    CheckSpecError,
    check_mcap,
    load_check_spec,
)
from pymcap_cli.cmd._cli_options import CheckSpecOption, NumWorkersOption
from pymcap_cli.core.diagnostics import LEVEL_NAMES, LEVEL_STYLES
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)
console = Console()


def check(
    file: str,
    *,
    spec: CheckSpecOption,
    num_workers: NumWorkersOption = 4,
) -> int:
    """Check an MCAP recording against expected topics, timing, schemas, and values."""
    try:
        report = check_mcap(file, load_check_spec(spec), num_workers=num_workers)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 1
    except CheckSpecError as exc:
        ERR.print(f"[red]Invalid check spec:[/red] {exc}")
        return 1
    except Exception as exc:
        ERR.print(f"[red]Check command failed:[/red] {exc}")
        logger.exception("Check command failed")
        return 1

    print_check_report(report)
    return 1 if report.error_count else 0


# Keys that only restate the spec or locate a failure — noise on passing rows.
_OK_HIDDEN_KEYS = frozenset({"timeout_ns", "window_ns"})


def _fmt_num(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _observed_cell(result: CheckResult) -> str:
    """Render the Observed column: compact for OK rows, full detail on failures."""
    if result.level != OK:
        return ", ".join(f"{key}={value}" for key, value in result.values.items())

    values = {
        key: value
        for key, value in result.values.items()
        if not key.startswith("required_")
        and key not in _OK_HIDDEN_KEYS
        and not key.endswith("window_start_ns")
    }
    parts: list[str] = []
    if "topics" in values:
        parts.append(f"topics={values.pop('topics')}")
    if "minimum_hz" in values and "maximum_hz" in values:
        parts.append(
            f"{_fmt_num(values.pop('minimum_hz'))}-{_fmt_num(values.pop('maximum_hz'))} Hz"
        )
    if "observed_count" in values:
        entry = f"n={values.pop('observed_count')}"
        if "observed_minimum" in values and "observed_maximum" in values:
            low = _fmt_num(values.pop("observed_minimum"))
            high = _fmt_num(values.pop("observed_maximum"))
            entry += f" in [{low}, {high}]" if low != high else f", all {low}"
        parts.append(entry)
    gap_ns = values.pop("maximum_gap_ns", None)
    if isinstance(gap_ns, int):
        parts.append(f"max gap {gap_ns / 1e9:.3f}s")
    elif gap_ns is not None:
        parts.append(f"maximum_gap_ns={gap_ns}")
    if "message_count" in values:
        parts.append(f"n={values.pop('message_count')}")
    for key, value in values.items():
        if key.endswith("_ns") and isinstance(value, int):
            parts.append(f"{key.removesuffix('_ns')}={value / 1e9:.3f}s")
        else:
            parts.append(f"{key}={_fmt_num(value)}")
    return ", ".join(parts)


def print_check_report(report: CheckReport) -> None:
    table = Table(title="Contract Check")
    table.add_column("Level", no_wrap=True)
    table.add_column("Check")
    table.add_column("Summary")
    table.add_column("Observed", overflow="fold")
    for result in report.results:
        level_name = LEVEL_NAMES.get(result.level, f"L{result.level}")
        level = Text(level_name, style=LEVEL_STYLES.get(result.level, "dim"))
        table.add_row(level, result.name, result.summary, _observed_cell(result))
    console.print(table)
    console.print(
        f"{report.path}: "
        f"[green]{report.ok_count} OK[/green], "
        f"[yellow]{report.warning_count} WARN[/yellow], "
        f"[red]{report.error_count} ERROR[/red]"
    )
