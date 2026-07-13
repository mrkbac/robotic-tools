"""Export ``sensor_msgs/PointCloud2`` topics from an MCAP file to PCD files."""

import logging
from typing import Annotated

from cyclopts import Parameter

from pymcap_cli.cmd._message_filter_options import (
    EarlyBailOption,
    EndTimeOption,
    ExcludeTopicOption,
    StartTimeOption,
    TopicOption,
    create_message_filter,
)
from pymcap_cli.exporters import run_export
from pymcap_cli.exporters.pcd_exporter import PcdExporter
from pymcap_cli.types.types_manual import ForceOverwriteOption, OutputPathOption

logger = logging.getLogger(__name__)


def export_pcd(
    file: str,
    output: OutputPathOption,
    *,
    force: ForceOverwriteOption = False,
    topic: TopicOption = None,
    exclude_topic: ExcludeTopicOption = None,
    start: StartTimeOption = "",
    end: EndTimeOption = "",
    early_bail: EarlyBailOption = False,
    num_workers: Annotated[int, Parameter(name=["--num-workers"])] = 8,
) -> int:
    """Export ``sensor_msgs/PointCloud2`` topics to ASCII PCD files.

    Each message is written to ``<output>/<safe_topic>/<log_time_ns>.pcd`` in
    PCD v0.7 ASCII format — universally readable by ``pcl_viewer``, Open3D,
    and CloudCompare.
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
        exporter=PcdExporter(),
        message_filter=message_filter,
        force=force,
        num_workers=num_workers,
    )
