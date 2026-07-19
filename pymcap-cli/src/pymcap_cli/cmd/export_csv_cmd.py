"""Export an MCAP file to a directory of CSV files (one per topic)."""

import logging
from typing import Annotated

from cyclopts import Parameter

from pymcap_cli.cmd._cli_options import (
    MESSAGE_PATH_GROUP,
    VAR_REQUIRES_SELECT,
    EarlyBailOption,
    EndTimeOption,
    ExcludeTopicOption,
    ForceOverwriteOption,
    IncludeBlobsOption,
    MessagePathVariablesOption,
    NumWorkersOption,
    OutputPathOption,
    SelectColumnsOption,
    StartTimeOption,
    TopicOption,
)
from pymcap_cli.cmd._message_filter_options import create_message_filter
from pymcap_cli.cmd._message_path_options import create_message_path_variables
from pymcap_cli.exporters import run_export
from pymcap_cli.exporters.csv_exporter import CsvExporter

logger = logging.getLogger(__name__)


def export_csv(
    file: str,
    output: OutputPathOption,
    *,
    force: ForceOverwriteOption = False,
    topic: TopicOption = None,
    exclude_topic: ExcludeTopicOption = None,
    start: StartTimeOption = "",
    end: EndTimeOption = "",
    early_bail: EarlyBailOption = False,
    include_blobs: IncludeBlobsOption = False,
    num_workers: NumWorkersOption = 8,
    select: Annotated[SelectColumnsOption, Parameter(group=VAR_REQUIRES_SELECT)] = None,
    var: Annotated[
        MessagePathVariablesOption, Parameter(group=[MESSAGE_PATH_GROUP, VAR_REQUIRES_SELECT])
    ] = None,
) -> int:
    """Export an MCAP file to a directory of CSV files (one per topic).

    Nested message fields are flattened with dot notation
    (``pose.position.x``); arrays are kept as JSON strings to preserve row
    counts. Schemas with raw media payloads (``sensor_msgs/Image``,
    ``CompressedImage`` …) are skipped unless ``--include-blobs`` is passed.
    """
    try:
        message_filter = create_message_filter(
            topic=topic,
            exclude_topic=exclude_topic,
            start=start,
            end=end,
            early_bail=early_bail,
        )
        variables = create_message_path_variables(var) if select or var else {}
        exporter = CsvExporter(
            include_blobs=include_blobs,
            select=select,
            variables=variables,
        )
    except ValueError as exc:
        logger.error(str(exc))  # noqa: TRY400
        return 1

    return run_export(
        file=file,
        output=output,
        exporter=exporter,
        message_filter=message_filter,
        force=force,
        num_workers=num_workers,
    )
