"""Shared processor pipeline for multi-output transform commands (splitting)."""

import contextlib
from dataclasses import dataclass

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


def run_processor_multi(
    *,
    files: list[str],
    output_options: OutputOptions,
) -> ProcessorResult:
    """Open input files, build ProcessingOptions, run McapProcessor in multi-output mode.

    Unlike run_processor(), this does not open an output stream. The OutputManager
    creates and manages output files based on split routing.

    Raises any exception from McapProcessor.process() to the caller.
    """
    with contextlib.ExitStack() as stack:
        input_files: list[InputFile] = []

        for f in files:
            stream, size = stack.enter_context(open_input(f))
            input_files.append(
                InputFile(stream=stream, size=size, options=InputOptions.from_args())
            )

        processing_options = ProcessingOptions(
            inputs=input_files,
            input_options=InputOptions.from_args(),
            output_options=output_options,
        )

        processor = McapProcessor(processing_options)
        # Multi-output mode: pass None, OutputManager creates files
        stats = processor.process(output_stream=None)

    return ProcessorResult(stats=stats, processor=processor)
