"""Shared processor pipeline for transform commands."""

import contextlib
import logging
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, cast
from urllib.parse import urlparse

from small_mcap import InvalidMagicError, McapError
from typing_extensions import Self

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
from pymcap_cli.core.rosbag2_layout import expand_bag_paths
from pymcap_cli.utils import confirm_output_overwrite, read_info

logger = logging.getLogger(__name__)


class _AsyncOutputStream:
    """Bounded, ordered output queue with a logical write position."""

    def __init__(self, output: BinaryIO, max_buffer_bytes: int) -> None:
        if max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be positive")
        self._output = output
        self._max_buffer_bytes = max_buffer_bytes
        self._condition = threading.Condition()
        self._pending: deque[bytes] = deque()
        self._queued_bytes = 0
        self._logical_position = output.tell()
        self._is_closing = False
        self._is_closed = False
        self._worker_error: Exception | None = None
        self._worker = threading.Thread(target=self._write_pending, name="mcap-output", daemon=True)
        self._worker.start()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self._is_closed

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def fileno(self) -> int:
        return self._output.fileno()

    def tell(self) -> int:
        with self._condition:
            self._raise_worker_error()
            return self._logical_position

    def write(self, data: bytes | bytearray | memoryview, /) -> int:
        if self._is_closed:
            raise ValueError("write to closed file")
        stable_data = data if type(data) is bytes else bytes(data)
        size = len(stable_data)
        with self._condition:
            while (
                self._queued_bytes > 0
                and self._queued_bytes + size > self._max_buffer_bytes
                and self._worker_error is None
            ):
                self._condition.wait()
            self._raise_worker_error()
            if self._is_closing:
                raise ValueError("write to closing file")
            self._pending.append(stable_data)
            self._queued_bytes += size
            self._logical_position += size
            self._condition.notify()
        return size

    def flush(self) -> None:
        if self._is_closed:
            raise ValueError("flush of closed file")
        with self._condition:
            while self._queued_bytes > 0 and self._worker_error is None:
                self._condition.wait()
            self._raise_worker_error()
        self._output.flush()

    def close(self) -> None:
        if self._is_closed:
            return
        error: Exception | None = None
        try:
            self.flush()
        except Exception as exc:  # noqa: BLE001 - preserve worker failures through cleanup
            error = exc
        with self._condition:
            self._is_closing = True
            self._condition.notify_all()
        self._worker.join()
        try:
            self._output.close()
        except Exception as exc:  # noqa: BLE001 - do not skip state cleanup on close failure
            if error is None:
                error = exc
        self._is_closed = True
        if error is not None:
            raise error

    def _write_pending(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._pending and not self._is_closing:
                        self._condition.wait()
                    if not self._pending:
                        return
                    data = self._pending.popleft()
                view = memoryview(data)
                written = 0
                while written < len(view):
                    count = self._output.write(view[written:])
                    if count is None or count <= 0:
                        raise OSError(  # noqa: TRY301 - relayed to the producer thread
                            "output stream made no write progress"
                        )
                    written += count
                with self._condition:
                    self._queued_bytes -= len(data)
                    self._condition.notify_all()
        except Exception as exc:  # noqa: BLE001 - relay arbitrary stream errors to producer
            with self._condition:
                self._worker_error = exc
                self._pending.clear()
                self._queued_bytes = 0
                self._is_closing = True
                self._condition.notify_all()

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise OSError("asynchronous output write failed") from self._worker_error


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


def _open_output_stream(
    output: Path,
    overwrite_policy: OverwriteCollisionPolicy,
    *,
    async_buffer_bytes: int = 0,
) -> BinaryIO:
    """Open a single-output destination with the configured overwrite policy."""
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output directory '{output.parent}' does not exist.")
    if overwrite_policy == OverwriteCollisionPolicy.ASK:
        confirm_output_overwrite(output, force=False)
    elif overwrite_policy == OverwriteCollisionPolicy.ERROR and output.exists():
        raise FileExistsError(f"Output file '{output}' already exists.")

    stream = output.open("wb", buffering=0 if async_buffer_bytes else -1)
    if async_buffer_bytes:
        return cast("BinaryIO", _AsyncOutputStream(stream, async_buffer_bytes))
    return stream


def run_processor(
    *,
    files: list[str],
    output: Path,
    input_options: InputOptions,
    output_options: OutputOptions,
    input_buffer_bytes: int = 8192,
) -> ProcessorResult:
    """Open files, build ProcessingOptions, run McapProcessor, return results.

    Raises any exception from McapProcessor.process() to the caller.
    """
    files = expand_bag_paths(files)
    with contextlib.ExitStack() as stack:
        input_files: list[InputFile] = []

        for f in files:
            stream, size = stack.enter_context(open_input(f, buffering=input_buffer_bytes))
            input_files.append(InputFile(stream=stream, size=size, options=input_options))

        output_stream = stack.enter_context(
            _open_output_stream(
                output,
                output_options.overwrite_policy,
                async_buffer_bytes=output_options.async_output_buffer_bytes,
            )
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


def _outputs_lost_messages(sources: list[str], outputs: list[Path]) -> bool:
    """True if the outputs hold fewer messages in total than the local sources.

    For lossless transforms (compress) every source message must appear in the
    output, so any shortfall means data was dropped. Counts that can't be read
    are treated as unknown and never trigger a block.
    """
    out_total = 0
    for p in outputs:
        count = mcap_message_count(p)
        if count is None:
            return False
        out_total += count
    src_total = 0
    for src in sources:
        if urlparse(src).scheme in ("http", "https"):
            continue
        count = mcap_message_count(Path(src))
        if count is None:
            return False
        src_total += count
    return out_total < src_total


def processing_had_errors(stats: ProcessingStats) -> bool:
    """True if the processor swallowed read/validation errors during the run.

    Such a run produces incomplete output even when it exits cleanly, so it is
    not safe to delete or replace the source from it.
    """
    return stats.errors_encountered > 0 or stats.validation_errors > 0


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
    if _outputs_lost_messages([str(source)], [tmp_output]):
        logger.error("Output has fewer messages than the source — source file preserved.")
        tmp_output.unlink(missing_ok=True)
        return 1
    tmp_output.replace(source)
    logger.info(f"Replaced source: {source}")
    return 0


def finalize_delete_source(
    *,
    sources: list[str],
    outputs: list[Path],
    require_lossless: bool = False,
) -> int:
    """Validate every output and, if all valid, delete the eligible sources.

    Returns 0 on success (sources deleted or skipped with warning) and 1 if there
    are no outputs, any output failed validation, or every output is empty while a
    source had messages. No sources are deleted in those cases.

    When ``require_lossless`` is set (transforms that must preserve every message,
    e.g. ``compress``), any shortfall in total output messages versus the sources
    also preserves them — not just total loss.
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
    if require_lossless:
        if _outputs_lost_messages(sources, outputs):
            logger.error("Output has fewer messages than the source(s) — source file(s) preserved.")
            return 1
    elif _outputs_dropped_all_messages(sources, outputs):
        logger.error("Output contains no messages but the source did — source file(s) preserved.")
        return 1
    delete_source_files(sources, outputs)
    return 0
