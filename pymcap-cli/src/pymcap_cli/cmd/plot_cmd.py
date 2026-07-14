"""Plot command for pymcap-cli — visualize MCAP time-series data with plotly.

Wraps :class:`pymcap_cli.exporters.plot_exporter.PlotExporter`.
"""

import logging
from pathlib import Path
from typing import Annotated

from cyclopts import Group, Parameter
from ros_parser.message_path import ValidationError

from pymcap_cli.cmd._cli_options import (
    EarlyBailOption,
    EndTimeOption,
    ExcludeTopicOption,
    ForceOverwriteOption,
    MessagePathVariablesOption,
    OptionalOutputPathOption,
    StartTimeOption,
    TopicOption,
)
from pymcap_cli.cmd._message_filter_options import create_message_filter
from pymcap_cli.cmd._message_path_options import create_message_path_variables
from pymcap_cli.exporters import run_export
from pymcap_cli.exporters.plot_exporter import PlotExporter

logger = logging.getLogger(__name__)

OUTPUT_GROUP = Group("Output")


def plot(
    file: str,
    paths: list[str],
    *,
    topic: TopicOption = None,
    exclude_topic: ExcludeTopicOption = None,
    start: StartTimeOption = "",
    end: EndTimeOption = "",
    early_bail: EarlyBailOption = False,
    var: MessagePathVariablesOption = None,
    output: OptionalOutputPathOption = None,
    title: Annotated[
        str | None,
        Parameter(name=["--title"], group=OUTPUT_GROUP),
    ] = None,
    downsample: Annotated[
        int | None,
        Parameter(name=["-d", "--downsample"], group=OUTPUT_GROUP),
    ] = None,
    xy: Annotated[
        bool,
        Parameter(name=["--xy"], group=OUTPUT_GROUP),
    ] = False,
    force: ForceOverwriteOption = False,
) -> int:
    """Plot time-series data from an MCAP file using message paths.

    Extracts values along message paths and plots them over time using plotly.
    Supports numeric, boolean, and string values natively. Multiple paths can
    be overlaid. Array-valued paths expand into one trace per element index.

    Output format is chosen from the ``-o`` suffix: ``.html`` (interactive) or
    ``.png`` / ``.svg`` / ``.pdf`` / ``.jpg`` / ``.webp`` (static image, no
    browser needed — handy over SSH). With no ``-o`` the figure opens in a
    browser.

    Paths can be given a custom label: "Label=/topic.field"

    Examples:
      pymcap-cli plot recording.mcap /odom.pose.position.x
      pymcap-cli plot recording.mcap "Vel X=/odom.twist.twist.linear.x"
      pymcap-cli plot recording.mcap --xy /odom.pose.position.x /odom.pose.position.y
      pymcap-cli plot recording.mcap /odom.pose.position.x -d 1000
      pymcap-cli plot recording.mcap /odom.pose.position.x -S @10s -E @20s -o plot.html
      pymcap-cli plot recording.mcap "/joints.position[:].@degrees" -o joints.svg
    """
    if not paths:
        logger.error("At least one message path is required")
        return 1
    if xy and len(paths) != 2:
        logger.error("--xy mode requires exactly 2 paths")
        return 1

    output_path = Path(output) if output else None

    try:
        variables = create_message_path_variables(var)
        exporter = PlotExporter(
            output=output_path,
            paths=paths,
            title=title,
            downsample=downsample,
            xy=xy,
            force=force,
            source_name=Path(file).name,
            variables=variables,
        )
    except (ValueError, ValidationError) as exc:
        logger.error(str(exc))  # noqa: TRY400
        return 1
    except Exception:
        logger.exception(f"Invalid path syntax in {paths!r}")
        return 1

    try:
        message_filter = create_message_filter(
            topic=topic,
            exclude_topic=exclude_topic,
            start=start,
            end=end,
            early_bail=early_bail,
        )
    except ValueError as e:
        logger.error(str(e))  # noqa: TRY400
        return 1

    return run_export(
        file=file,
        output=output_path,
        exporter=exporter,
        message_filter=message_filter,
        force=force,
    )
