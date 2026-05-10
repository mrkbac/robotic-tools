"""Export ``sensor_msgs/PointCloud2`` topics from an MCAP file to PCD files."""

from typing import Annotated

from cyclopts import Parameter

from pymcap_cli.exporters import run_export
from pymcap_cli.exporters.pcd_exporter import PcdExporter
from pymcap_cli.types.types_manual import ForceOverwriteOption, OutputPathOption


def export_pcd(
    file: str,
    output: OutputPathOption,
    *,
    force: ForceOverwriteOption = False,
    topic: Annotated[list[str] | None, Parameter(name=["--topic", "-t"])] = None,
    num_workers: Annotated[int, Parameter(name=["--num-workers"])] = 8,
) -> int:
    """Export ``sensor_msgs/PointCloud2`` topics to ASCII PCD files.

    Each message is written to ``<output>/<safe_topic>/<log_time_ns>.pcd`` in
    PCD v0.7 ASCII format — universally readable by ``pcl_viewer``, Open3D,
    and CloudCompare.
    """
    return run_export(
        file=file,
        output=output,
        exporter=PcdExporter(),
        topics=topic,
        force=force,
        num_workers=num_workers,
    )
