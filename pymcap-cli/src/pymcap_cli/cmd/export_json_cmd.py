"""Export an MCAP file to a directory of NDJSON / JSON files (one per topic)."""

import logging
from typing import Annotated

from cyclopts import Parameter

from pymcap_cli.cmd._cli_options import (
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
from pymcap_cli.exporters.json_exporter import JsonExporter

logger = logging.getLogger(__name__)


def export_json(
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
    per_message: Annotated[bool, Parameter(name=["--per-message"])] = False,
    num_workers: NumWorkersOption = 8,
    select: SelectColumnsOption = None,
    var: MessagePathVariablesOption = None,
) -> int:
    """Export an MCAP file to NDJSON (one line per message) or per-message JSON files.

    Default is one ``<topic>.ndjson`` per topic. With ``--per-message``, each
    topic gets a directory containing one ``<log_time_ns>.json`` per message —
    handy when downstream tools expect one record per file.
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
        exporter = JsonExporter(
            include_blobs=include_blobs,
            per_message=per_message,
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
