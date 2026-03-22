"""Shared processor pipeline for transform commands."""

import contextlib
from dataclasses import dataclass
from pathlib import Path

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.core.mcap_processor import (
    InputFile,
    InputOptions,
    McapProcessor,
    OutputOptions,
    ProcessingOptions,
    ProcessingStats,
)


@dataclass(slots=True)
class ProcessorResult:
    """Result of a processor run."""

    stats: ProcessingStats
    processor: McapProcessor


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

        output_stream = stack.enter_context(output.open("wb"))

        processing_options = ProcessingOptions(
            inputs=input_files,
            input_options=InputOptions.from_args(),
            output_options=output_options,
        )

        processor = McapProcessor(processing_options)
        stats = processor.process(output_stream)

    return ProcessorResult(stats=stats, processor=processor)
