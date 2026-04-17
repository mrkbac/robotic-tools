"""Shared processor pipeline for transform commands."""

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

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
from pymcap_cli.utils import confirm_output_overwrite


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
