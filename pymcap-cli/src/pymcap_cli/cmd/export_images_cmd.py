"""Export image topics from an MCAP file to a folder of image files."""

import logging
from typing import Annotated

from cyclopts import Parameter

from pymcap_cli.cmd._cli_options import (
    EarlyBailOption,
    EndTimeOption,
    ExcludeTopicOption,
    ForceOverwriteOption,
    NumWorkersOption,
    OutputPathOption,
    StartTimeOption,
    TopicOption,
)
from pymcap_cli.cmd._message_filter_options import create_message_filter
from pymcap_cli.exporters import run_export
from pymcap_cli.exporters.image_exporter import ImageExporter

logger = logging.getLogger(__name__)


def export_images(
    file: str,
    output: OutputPathOption,
    *,
    force: ForceOverwriteOption = False,
    topic: TopicOption = None,
    exclude_topic: ExcludeTopicOption = None,
    start: StartTimeOption = "",
    end: EndTimeOption = "",
    early_bail: EarlyBailOption = False,
    raw_format: Annotated[str, Parameter(name=["--raw-format"])] = "png",
    output_format: Annotated[str, Parameter(name=["--format"])] = "native",
    num_workers: NumWorkersOption = 8,
) -> int:
    """Export image topics to per-topic folders of image files.

    ``CompressedImage`` payloads keep their original extension by default
    (``--format native``). Set ``--format`` to a Pillow format (such as
    ``jpeg``/``png``/``webp``) to force re-encoding compressed images.
    Raw ``Image`` messages are always encoded with ``--raw-format`` (default
    ``png``). Requires the ``image`` extra.
    """
    try:
        message_filter = create_message_filter(
            topic=topic,
            exclude_topic=exclude_topic,
            start=start,
            end=end,
            early_bail=early_bail,
        )
    except ValueError as exc:
        logger.error(str(exc))  # noqa: TRY400
        return 1

    return run_export(
        file=file,
        output=output,
        exporter=ImageExporter(raw_format=raw_format, output_format=output_format),
        message_filter=message_filter,
        force=force,
        num_workers=num_workers,
    )
