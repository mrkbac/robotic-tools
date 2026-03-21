"""Plot command for pymcap-cli - visualize MCAP time-series data with plotly."""

from dataclasses import dataclass, field
from typing import Annotated

import plotly.graph_objects as go
from cyclopts import Group, Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import Schema
from rich.console import Console
from ros_parser import parse_schema_to_definitions
from ros_parser.message_path import (
    MessagePath,
    MessagePathError,
    ValidationError,
    parse_message_path,
)
from small_mcap import JSONDecoderFactory
from small_mcap.reader import read_message_decoded
from small_mcap.records import Channel

from pymcap_cli.input_handler import open_input
from pymcap_cli.utils import MAX_INT64, parse_timestamp_args

console_err = Console(stderr=True)

# Parameter groups
FILTERING_GROUP = Group("Filtering")
OUTPUT_GROUP = Group("Output")


@dataclass
class SeriesData:
    path_str: str
    parsed: MessagePath
    times: list[float] = field(default_factory=list)
    values: list[float | bool | str] = field(default_factory=list)


def _validate_series(
    series: SeriesData,
    schema_name: str,
    schema_data: bytes,
    query_str: str,
) -> bool:
    """Validate a series path against a message schema. Returns False on failure."""
    try:
        all_definitions = parse_schema_to_definitions(schema_name, schema_data)
    except Exception as e:  # noqa: BLE001
        console_err.print(f"[yellow]Warning: Could not parse schema '{schema_name}': {e}[/yellow]")
        return True  # allow continued processing

    root_msgdef = all_definitions.get(schema_name)
    if root_msgdef is None:
        parts = schema_name.split("/")
        short_name = f"{parts[0]}/{parts[-1]}"
        root_msgdef = all_definitions.get(short_name)

    if root_msgdef is None:
        console_err.print(
            f"[yellow]Warning: Could not find message definition "
            f"for schema '{schema_name}'[/yellow]"
        )
        return True

    try:
        series.parsed.validate(root_msgdef, all_definitions)
    except ValidationError as e:
        console_err.print(f"[red]Query validation error for path '{series.path_str}':[/red]")
        console_err.print(f"[red]{e}[/red]")
        console_err.print(
            f"\n[yellow]Path:[/yellow] {query_str}\n[yellow]Schema:[/yellow] {schema_name}"
        )
        return False

    return True


def plot(
    file: str,
    paths: list[str],
    *,
    start: Annotated[
        str,
        Parameter(name=["-S", "--start"], group=FILTERING_GROUP),
    ] = "",
    start_secs: Annotated[
        int,
        Parameter(name=["-s", "--start-secs"], group=FILTERING_GROUP),
    ] = 0,
    end: Annotated[
        str,
        Parameter(name=["-E", "--end"], group=FILTERING_GROUP),
    ] = "",
    end_secs: Annotated[
        int,
        Parameter(name=["-e", "--end-secs"], group=FILTERING_GROUP),
    ] = 0,
    output: Annotated[
        str | None,
        Parameter(name=["-o", "--output"], group=OUTPUT_GROUP),
    ] = None,
    title: Annotated[
        str | None,
        Parameter(name=["--title"], group=OUTPUT_GROUP),
    ] = None,
) -> int:
    """Plot time-series data from an MCAP file using message paths.

    Extracts values along message paths and plots them over time using plotly.
    Supports numeric, boolean, and string values natively. Multiple paths can be overlaid.

    Examples:
      # Plot a single field
      pymcap-cli plot recording.mcap /odom.pose.position.x

      # Overlay multiple fields
      pymcap-cli plot recording.mcap /odom.pose.position.x /odom.pose.position.y

      # With time range and save to HTML
      pymcap-cli plot recording.mcap /odom.pose.position.x -s 10 -e 20 -o plot.html

      # String field (plotly renders as categorical axis)
      pymcap-cli plot recording.mcap /diagnostics.status[0].level

      # With math modifiers
      pymcap-cli plot recording.mcap '/odom.pose.orientation.@rpy.yaw.@degrees'
    """
    if not paths:
        console_err.print("[red]Error: At least one message path is required[/red]")
        return 1

    # Phase 1: Parse all paths upfront — fail fast on syntax errors
    series_list: list[SeriesData] = []
    for path_str in paths:
        try:
            parsed = parse_message_path(path_str)
        except Exception as e:  # noqa: BLE001
            console_err.print(f"[red]Invalid path syntax '{path_str}': {e}[/red]")
            return 1
        series_list.append(SeriesData(path_str=path_str, parsed=parsed))

    # Collect all topics needed
    topics_needed: set[str] = {s.parsed.topic for s in series_list}

    # Parse time range
    start_time_ns = parse_timestamp_args(start, 0, start_secs) or 0
    end_time_ns = parse_timestamp_args(end, 0, end_secs)
    if end_time_ns is None:
        end_time_ns = MAX_INT64

    def should_include_message(channel: Channel, _schema: Schema | None) -> bool:
        return channel.topic in topics_needed

    # Phase 2: Iterate messages and collect data
    first_time_ns: int | None = None
    validated_topics: set[str] = set()

    try:
        with open_input(file) as (input_stream, _):
            for msg in read_message_decoded(
                input_stream,
                decoder_factories=[JSONDecoderFactory(), DecoderFactory()],
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                should_include=should_include_message,
            ):
                topic = msg.channel.topic

                # Validate on first message per topic
                if topic not in validated_topics:
                    validated_topics.add(topic)

                    if msg.schema is not None:
                        for series in series_list:
                            if series.parsed.topic == topic and not _validate_series(
                                series,
                                msg.schema.name,
                                msg.schema.data,
                                series.path_str,
                            ):
                                return 1

                if first_time_ns is None:
                    first_time_ns = msg.message.log_time

                t_sec = (msg.message.log_time - first_time_ns) / 1e9

                # Extract values for each matching series
                for series in series_list:
                    if series.parsed.topic != topic:
                        continue

                    try:
                        value = series.parsed.apply(msg.decoded_message)
                    except MessagePathError:
                        continue

                    if value is None or isinstance(value, (list, dict)):
                        continue

                    series.times.append(t_sec)
                    series.values.append(value)

    except KeyboardInterrupt:
        console_err.print("\n[yellow]Interrupted by user[/yellow]")
        return 0
    except Exception as e:  # noqa: BLE001
        console_err.print(f"[red]Error reading MCAP: {e}[/red]")
        return 1

    # Check for missing topics
    for missing in topics_needed - validated_topics:
        console_err.print(f"[red]Error: Topic '{missing}' not found in MCAP file[/red]")
        return 1

    # Warn about empty series
    for series in series_list:
        if not series.times:
            console_err.print(
                f"[yellow]Warning: No plottable data for path '{series.path_str}'[/yellow]"
            )

    if not any(s.times for s in series_list):
        console_err.print("[red]Error: No plottable data found for any path[/red]")
        return 1

    # Phase 4: Render plot
    fig = go.Figure()

    for series in series_list:
        if not series.times:
            continue

        fig.add_trace(
            go.Scatter(
                x=series.times,
                y=series.values,
                mode="lines",
                name=series.path_str,
            )
        )

    fig.update_layout(
        title=title or ", ".join(s.path_str for s in series_list),
        xaxis_title="Time (s)",
        yaxis_title="Value",
        hovermode="x unified",
    )

    if output:
        fig.write_html(output)
        console_err.print(f"[green]Plot saved to {output}[/green]")
    else:
        fig.show()

    return 0
