"""Shared processor pipeline for multi-output transform commands (splitting)."""

import contextlib

from pymcap_cli.cmd._run_processor import ProcessorResult
from pymcap_cli.core.input_handler import open_input
from pymcap_cli.core.mcap_processor import (
    InputFile,
    InputOptions,
    McapProcessor,
    OutputOptions,
    ProcessingOptions,
)


def run_processor_multi(
    *,
    files: list[str],
    output_options: OutputOptions,
    input_options: InputOptions | None = None,
) -> ProcessorResult:
    """Open input files, build ProcessingOptions, run McapProcessor in multi-output mode.

    Unlike run_processor(), this does not open an output stream. The OutputManager
    creates and manages output files based on split routing.

    Raises any exception from McapProcessor.process() to the caller.
    """
    if input_options is None:
        input_options = InputOptions.from_args()
    with contextlib.ExitStack() as stack:
        input_files: list[InputFile] = []

        for f in files:
            stream, size = stack.enter_context(open_input(f))
            input_files.append(InputFile(stream=stream, size=size, options=input_options))

        processing_options = ProcessingOptions(
            inputs=input_files,
            input_options=input_options,
            output_options=output_options,
        )

        processor = McapProcessor(processing_options)
        # Multi-output mode: pass None, OutputManager creates files
        stats = processor.process(output_stream=None)

    return ProcessorResult(stats=stats, processor=processor)
