"""Plot command for pymcap-cli - visualize MCAP time-series data with plotly."""

from dataclasses import dataclass, field
from typing import IO, Annotated

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
from pymcap_cli.utils import MAX_INT64, ProgressTrackingIO, file_progress, parse_timestamp_args

console_err = Console(stderr=True)

# Parameter groups
FILTERING_GROUP = Group("Filtering")
OUTPUT_GROUP = Group("Output")


@dataclass
class SeriesData:
    label: str
    path_str: str
    parsed: MessagePath
    times: list[float] = field(default_factory=list)
    values: list[float | bool | str] = field(default_factory=list)


def _parse_path_arg(arg: str) -> tuple[str, str]:
    """Parse 'Label=/path' or just '/path'. Returns (label, path_str)."""
    if "=" in arg and not arg.startswith("/"):
        label, _, path_str = arg.partition("=")
        return label, path_str
    return arg, arg


def _downsample(
    times: list[float],
    values: list[float | bool | str],
    target: int,
) -> tuple[list[float], list[float | bool | str]]:
    """Downsample using LTTB (Largest Triangle Three Buckets).

    Preserves visually important points (peaks, valleys) unlike nth-point sampling.
    """
    n = len(times)
    if n <= target:
        return times, values

    out_t: list[float] = [times[0]]
    out_v: list[float | bool | str] = [values[0]]

    bucket_size = (n - 2) / (target - 2)
    prev_idx = 0

    for i in range(1, target - 1):
        bucket_start = int((i - 1) * bucket_size) + 1
        bucket_end = int(i * bucket_size) + 1
        next_start = int(i * bucket_size) + 1
        next_end = min(int((i + 1) * bucket_size) + 1, n)

        avg_t = sum(times[next_start:next_end]) / (next_end - next_start)
        avg_v = sum(float(v) for v in values[next_start:next_end]) / (next_end - next_start)

        prev_t = times[prev_idx]
        prev_v = float(values[prev_idx])

        best_idx = bucket_start
        best_area = -1.0
        for j in range(bucket_start, min(bucket_end, n)):
            area = abs(
                (prev_t - avg_t) * (float(values[j]) - prev_v)
                - (prev_t - times[j]) * (avg_v - prev_v)
            )
            if area > best_area:
                best_area = area
                best_idx = j

        out_t.append(times[best_idx])
        out_v.append(values[best_idx])
        prev_idx = best_idx

    out_t.append(times[-1])
    out_v.append(values[-1])
    return out_t, out_v


def _validate_series(
    series: SeriesData,
    schema_name: str,
    schema_data: bytes,
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
            f"\n[yellow]Path:[/yellow] {series.path_str}\n[yellow]Schema:[/yellow] {schema_name}"
        )
        return False

    return True


def _collect_data(
    input_stream: IO[bytes],
    file_size: int,
    series_list: list[SeriesData],
    topics_needed: set[str],
    start_time_ns: int,
    end_time_ns: int,
) -> set[str]:
    """Iterate MCAP messages and collect data into series. Returns validated topics."""
    first_time_ns: int | None = None
    validated_topics: set[str] = set()

    def should_include_message(channel: Channel, _schema: Schema | None) -> bool:
        return channel.topic in topics_needed

    with file_progress("[bold blue]Collecting plot data...", console=console_err) as progress:
        task = progress.add_task("Reading", total=file_size)
        wrapped = ProgressTrackingIO(input_stream, task, progress, 0)

        for msg in read_message_decoded(
            wrapped,
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
                        ):
                            raise ValidationError(f"Validation failed for '{series.path_str}'")

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

        progress.update(task, completed=file_size)

    return validated_topics


def _render_xy(
    series_list: list[SeriesData],
    title: str | None,
    output: str | None,
) -> int:
    """Render XY plot from exactly 2 series."""
    x_series, y_series = series_list[0], series_list[1]

    if not x_series.times or not y_series.times:
        console_err.print("[red]Error: No plottable data for XY plot[/red]")
        return 1

    # Match by timestamp (same-topic paths have identical timestamps)
    x_by_time = dict(zip(x_series.times, x_series.values, strict=True))
    matched_x: list[float | bool | str] = []
    matched_y: list[float | bool | str] = []
    for t, v in zip(y_series.times, y_series.values, strict=True):
        if t in x_by_time:
            matched_x.append(x_by_time[t])
            matched_y.append(v)

    if not matched_x:
        console_err.print("[red]Error: No matching timestamps between the two paths[/red]")
        return 1

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=matched_x,
            y=matched_y,
            mode="lines+markers",
            marker={"size": 3},
            name="trajectory",
        )
    )
    fig.update_layout(
        title=title or f"{x_series.label} vs {y_series.label}",
        xaxis_title=x_series.label,
        yaxis_title=y_series.label,
        hovermode="closest",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return _output_figure(fig, output)


def _output_figure(fig: go.Figure, output: str | None) -> int:
    """Save or show the figure."""
    if output:
        fig.write_html(output)
        console_err.print(f"[green]Plot saved to {output}[/green]")
    else:
        fig.show()
    return 0


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
    downsample: Annotated[
        int | None,
        Parameter(name=["-d", "--downsample"], group=OUTPUT_GROUP),
    ] = None,
    xy: Annotated[
        bool,
        Parameter(name=["--xy"], group=OUTPUT_GROUP),
    ] = False,
) -> int:
    """Plot time-series data from an MCAP file using message paths.

    Extracts values along message paths and plots them over time using plotly.
    Supports numeric, boolean, and string values natively. Multiple paths can
    be overlaid.

    Paths can be given a custom label: "Label=/topic.field"

    Examples:
      # Plot a single field
      pymcap-cli plot recording.mcap /odom.pose.position.x

      # Named series
      pymcap-cli plot recording.mcap "Vel X=/odom.twist.twist.linear.x"

      # XY trajectory plot
      pymcap-cli plot recording.mcap --xy /odom.pose.position.x /odom.pose.position.y

      # Downsample to 1000 points
      pymcap-cli plot recording.mcap /odom.pose.position.x -d 1000

      # With time range and save to HTML
      pymcap-cli plot recording.mcap /odom.pose.position.x -s 10 -e 20 -o plot.html
    """
    if not paths:
        console_err.print("[red]Error: At least one message path is required[/red]")
        return 1

    if xy and len(paths) != 2:
        console_err.print("[red]Error: --xy mode requires exactly 2 paths[/red]")
        return 1

    # Phase 1: Parse all paths upfront — fail fast on syntax errors
    series_list: list[SeriesData] = []
    for arg in paths:
        label, path_str = _parse_path_arg(arg)
        try:
            parsed = parse_message_path(path_str)
        except Exception as e:  # noqa: BLE001
            console_err.print(f"[red]Invalid path syntax '{path_str}': {e}[/red]")
            return 1
        series_list.append(SeriesData(label=label, path_str=path_str, parsed=parsed))

    topics_needed: set[str] = {s.parsed.topic for s in series_list}

    # Parse time range
    start_time_ns = parse_timestamp_args(start, 0, start_secs) or 0
    end_time_ns = parse_timestamp_args(end, 0, end_secs)
    if end_time_ns is None:
        end_time_ns = MAX_INT64

    # Phase 2: Collect data
    try:
        with open_input(file) as (input_stream, file_size):
            validated_topics = _collect_data(
                input_stream,
                file_size,
                series_list,
                topics_needed,
                start_time_ns,
                end_time_ns,
            )
    except ValidationError:
        return 1
    except KeyboardInterrupt:
        console_err.print("\n[yellow]Interrupted by user[/yellow]")
        return 0
    except Exception as e:  # noqa: BLE001
        console_err.print(f"[red]Error reading MCAP: {e}[/red]")
        return 1

    # Phase 3: Check for missing topics
    for missing in topics_needed - validated_topics:
        console_err.print(f"[red]Error: Topic '{missing}' not found in MCAP file[/red]")
        return 1

    # Warn about empty series
    for series in series_list:
        if not series.times:
            console_err.print(f"[yellow]Warning: No plottable data for '{series.label}'[/yellow]")

    if not any(s.times for s in series_list):
        console_err.print("[red]Error: No plottable data found for any path[/red]")
        return 1

    # Phase 3b: Downsample if requested
    if downsample:
        for series in series_list:
            if series.times and not any(isinstance(v, str) for v in series.values[:10]):
                before = len(series.times)
                series.times, series.values = _downsample(series.times, series.values, downsample)
                if before != len(series.times):
                    after = len(series.times)
                    console_err.print(
                        f"[dim]Downsampled '{series.label}': {before} → {after} points[/dim]"
                    )

    # Phase 4: Render
    if xy:
        return _render_xy(series_list, title, output)

    fig = go.Figure()

    for series in series_list:
        if not series.times:
            continue

        fig.add_trace(
            go.Scattergl(
                x=series.times,
                y=series.values,
                mode="lines",
                name=series.label,
            )
        )

    fig.update_layout(
        title=title or ", ".join(s.label for s in series_list),
        xaxis_title="Time (s)",
        yaxis_title="Value",
        hovermode="x unified",
    )

    return _output_figure(fig, output)
