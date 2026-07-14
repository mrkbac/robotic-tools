"""Plot command for pymcap-cli — visualize MCAP values with Plotly.

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
from pymcap_cli.exporters.plot_exporter import (
    HistogramNormalization,
    PlotExporter,
    PlotKind,
)

logger = logging.getLogger(__name__)

CHART_GROUP = Group("Chart")


def plot(
    file: Annotated[str, Parameter(help="Input MCAP file or HTTP/HTTPS URL.")],
    paths: Annotated[
        list[str],
        Parameter(help="Message paths to plot; prefix with LABEL= to set a display name."),
    ],
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
        Parameter(
            name=["--title"],
            group=CHART_GROUP,
            help="Override the figure title, for example --title 'Wheel speed'.",
        ),
    ] = None,
    kind: Annotated[
        PlotKind,
        Parameter(
            name=["--kind"],
            group=CHART_GROUP,
            help=(
                "Chart type: time series, value histogram, or XY trajectory; "
                "for example --kind histogram."
            ),
        ),
    ] = "time",
    downsample: Annotated[
        int | None,
        Parameter(
            name=["-d", "--downsample"],
            group=CHART_GROUP,
            help=(
                "Reduce each time/XY series to at most N points using LTTB "
                "(minimum 3), for example -d 1000."
            ),
        ),
    ] = None,
    bins: Annotated[
        int | None,
        Parameter(
            name=["--bins"],
            group=CHART_GROUP,
            help=(
                "Maximum number of bins for numeric histograms; capped at the number "
                "of distinct values. Plotly chooses when omitted. Example: --bins 40."
            ),
        ),
    ] = None,
    normalize: Annotated[
        HistogramNormalization,
        Parameter(
            name=["--normalize"],
            group=CHART_GROUP,
            help=(
                "Histogram Y scale: raw count, probability, or probability density; "
                "for example --normalize probability."
            ),
        ),
    ] = "count",
    force: ForceOverwriteOption = False,
) -> int:
    """Plot MCAP message-path values as time series, histograms, or XY trajectories.

    Time plots support numeric, boolean, and string values. Histogram plots use
    numeric bins or categorical frequency bars, with one subplot per expanded
    series. Array-valued paths expand into one trace or subplot per element.

    Output format is chosen from the ``-o`` suffix: ``.html`` (interactive) or
    ``.png`` / ``.svg`` / ``.pdf`` / ``.jpg`` / ``.webp`` (static image, no
    browser needed — handy over SSH). With no ``-o`` the figure opens in a
    browser.

    Paths can be given a custom label: "Label=/topic.field"

    Examples:
      pymcap-cli plot recording.mcap /odom.pose.position.x
      pymcap-cli plot recording.mcap "Vel X=/odom.twist.twist.linear.x"
      pymcap-cli plot recording.mcap --kind xy /odom.pose.position.x /odom.pose.position.y
      pymcap-cli plot recording.mcap /imu.linear_acceleration.x --kind histogram --bins 40
      pymcap-cli plot recording.mcap /mode.name --kind histogram --normalize probability
      pymcap-cli plot recording.mcap /odom.pose.position.x -d 1000
      pymcap-cli plot recording.mcap /odom.pose.position.x -S @10s -E @20s -o plot.html
      pymcap-cli plot recording.mcap "/joints.position[:].@degrees" -o joints.svg
    """
    if not paths:
        logger.error("At least one message path is required")
        return 1
    if "--xy" in paths:
        logger.error("--xy was removed; use --kind xy")
        return 1

    output_path = Path(output) if output else None

    try:
        variables = create_message_path_variables(var)
        exporter = PlotExporter(
            output=output_path,
            paths=paths,
            title=title,
            downsample=downsample,
            kind=kind,
            bins=bins,
            normalize=normalize,
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
