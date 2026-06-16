"""Shared processor pipeline for transform commands."""

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse

from small_mcap import InvalidMagicError, McapError

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.core.mcap_processor import (
    InputFile,
    InputOptions,
    McapProcessor,
    OutputOptions,
    OverwriteCollisionPolicy,
    ProcessingOptions,
    ProcessingStats,
)
from pymcap_cli.utils import confirm_output_overwrite, read_info

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProcessorResult:
    """Result of a processor run."""

    stats: ProcessingStats
    processor: McapProcessor


def resolve_overwrite_policy(*, force: bool, no_clobber: bool) -> OverwriteCollisionPolicy | None:
    """Map CLI overwrite flags to the processor overwrite policy."""
    if force and no_clobber:
        return None
    if force:
        return OverwriteCollisionPolicy.OVERWRITE
    if no_clobber:
        return OverwriteCollisionPolicy.ERROR
    return OverwriteCollisionPolicy.ASK


def _open_output_stream(output: Path, overwrite_policy: OverwriteCollisionPolicy) -> BinaryIO:
    """Open a single-output destination with the configured overwrite policy."""
    if overwrite_policy == OverwriteCollisionPolicy.ASK:
        confirm_output_overwrite(output, force=False)
    elif overwrite_policy == OverwriteCollisionPolicy.ERROR and output.exists():
        raise FileExistsError(f"Output file '{output}' already exists.")

    return output.open("wb")


def run_processor(
    *,
    files: list[str],
    output: Path,
    input_options: InputOptions,
    output_options: OutputOptions,
) -> ProcessorResult:
    """Open files, build ProcessingOptions, run McapProcessor, return results.

    Raises any exception from McapProcessor.process() to the caller.
    """
    with contextlib.ExitStack() as stack:
        input_files: list[InputFile] = []

        for f in files:
            stream, size = stack.enter_context(open_input(f))
            input_files.append(InputFile(stream=stream, size=size, options=input_options))

        output_stream = stack.enter_context(
            _open_output_stream(output, output_options.overwrite_policy)
        )

        processing_options = ProcessingOptions(
            inputs=input_files,
            input_options=InputOptions.from_args(),
            output_options=output_options,
        )

        processor = McapProcessor(processing_options)
        stats = processor.process(output_stream)

    return ProcessorResult(stats=stats, processor=processor)


def validate_mcap_output(path: Path) -> bool:
    """Return True iff the MCAP at ``path`` has a readable header and summary."""
    try:
        with path.open("rb") as f:
            read_info(f)
    except (McapError, InvalidMagicError, OSError, AssertionError) as e:
        logger.debug(f"Output validation failed for {path}: {e}")
        return False
    return True


def mcap_message_count(path: Path) -> int | None:
    """Return the message count from an MCAP summary, or None if it can't be read."""
    try:
        with path.open("rb") as f:
            info = read_info(f)
    except (McapError, InvalidMagicError, OSError, AssertionError) as e:
        logger.debug(f"Could not read message count for {path}: {e}")
        return None
    if info.summary.statistics is None:
        return None
    return info.summary.statistics.message_count


def _outputs_dropped_all_messages(sources: list[str], outputs: list[Path]) -> bool:
    """True if every output has zero messages while some local source had messages.

    Guards against deleting sources when a transform silently produced empty output
    (e.g. the source was truncated before being read). Partial drops — dedup, time or
    channel filters — are intentionally not flagged; only total loss is.
    """
    out_total = 0
    for p in outputs:
        count = mcap_message_count(p)
        if count is None:
            return False  # unknown — validation already passed, don't second-guess it
        out_total += count
    if out_total > 0:
        return False
    for src in sources:
        if urlparse(src).scheme in ("http", "https"):
            continue  # URLs are never deleted, so their counts don't matter
        count = mcap_message_count(Path(src))
        if count:
            return True
    return False


def delete_source_files(sources: list[str], outputs: list[Path]) -> None:
    """Delete each local source file. Skip URLs and any source path that
    resolves to one of ``outputs`` (with a warning).
    """
    output_resolved = {p.resolve() for p in outputs}
    for src in sources:
        scheme = urlparse(src).scheme
        if scheme in ("http", "https"):
            logger.warning(f"Skipping delete: '{src}' is a remote URL")
            continue
        path = Path(src)
        try:
            resolved = path.resolve()
        except OSError as e:
            logger.warning(f"Skipping delete '{src}': {e}")
            continue
        if resolved in output_resolved:
            logger.warning(f"Skipping delete: source '{src}' is also an output")
            continue
        try:
            path.unlink()
            logger.info(f"Deleted source: {src}")
        except FileNotFoundError:
            logger.debug(f"Source already gone: {src}")
        except OSError:
            logger.exception(f"Failed to delete '{src}'")


def in_place_temp_path(source: Path) -> Path:
    """Temp output path next to ``source`` so the final rename stays on one filesystem."""
    return source.with_name(source.name + ".tmp")


def finalize_replace_source(*, source: Path, tmp_output: Path) -> int:
    """Validate ``tmp_output`` and atomically replace ``source`` with it.

    Returns 0 on success and 1 if the temp output failed validation or is empty
    while the source had messages. In those cases the source is preserved and the
    temp file is removed.
    """
    if not validate_mcap_output(tmp_output):
        logger.error(f"[red]Output failed validation: {tmp_output}[/red]")
        logger.error("Source file preserved — output not safe to replace source.")
        tmp_output.unlink(missing_ok=True)
        return 1
    if _outputs_dropped_all_messages([str(source)], [tmp_output]):
        logger.error("Output contains no messages but the source did — source file preserved.")
        tmp_output.unlink(missing_ok=True)
        return 1
    tmp_output.replace(source)
    logger.info(f"Replaced source: {source}")
    return 0


def finalize_delete_source(
    *,
    sources: list[str],
    outputs: list[Path],
) -> int:
    """Validate every output and, if all valid, delete the eligible sources.

    Returns 0 on success (sources deleted or skipped with warning) and 1 if there
    are no outputs, any output failed validation, or every output is empty while a
    source had messages. No sources are deleted in those cases.
    """
    if not outputs:
        logger.error("No output files were produced — source file(s) preserved.")
        return 1
    invalid = [p for p in outputs if not validate_mcap_output(p)]
    if invalid:
        for p in invalid:
            logger.error(f"[red]Output failed validation: {p}[/red]")
        logger.error("Source file(s) preserved — output not safe to replace source.")
        return 1
    if _outputs_dropped_all_messages(sources, outputs):
        logger.error("Output contains no messages but the source did — source file(s) preserved.")
        return 1
    delete_source_files(sources, outputs)
    return 0
