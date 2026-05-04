"""Plot exporter — collect message-path series and render to a Plotly figure.

Lives in the Exporter pipeline so plot benefits from the shared progress UI,
topic filter, decoder-factory plumbing, and lifecycle hooks. The Plotly
figure is assembled in :meth:`PlotExporter.finish` once every per-topic
writer has flushed its buffered samples.

When ``output`` is ``None`` the figure is opened interactively
(``fig.show()``); otherwise it is written as standalone HTML.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import plotly.graph_objects as go
from ros_parser import parse_schema_to_definitions
from ros_parser.message_path import (
    MessagePath,
    MessagePathError,
    ValidationError,
    parse_message_path,
)

from pymcap_cli.exporters.base import JsonRos2Exporter, TopicWriter

if TYPE_CHECKING:
    from collections.abc import Mapping

    from small_mcap import DecodedMessage, Schema

    from pymcap_cli.exporters.base import TopicContext

logger = logging.getLogger(__name__)


@dataclass
class SeriesData:
    """One labelled time-series gathered from a single message-path query."""

    label: str
    path_str: str
    parsed: MessagePath
    times_ns: list[int] = field(default_factory=list)
    values: list[float | bool | str] = field(default_factory=list)


def parse_path_arg(arg: str) -> tuple[str, str]:
    """Parse 'Label=/path' or just '/path'. Returns (label, path_str)."""
    if "=" in arg and not arg.startswith("/"):
        label, _, path_str = arg.partition("=")
        return label, path_str
    return arg, arg


def downsample_lttb(
    times: list[float],
    values: list[float | bool | str],
    target: int,
) -> tuple[list[float], list[float | bool | str]]:
    """Largest-Triangle-Three-Buckets downsampling (preserves visual peaks)."""
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


def _validate_series_against_schema(
    series: SeriesData, schema_name: str, schema_data: bytes
) -> bool:
    """Validate one series against a ROS message schema.

    Returns True when the path is acceptable (or schema couldn't be parsed —
    we degrade gracefully); False when the path is structurally invalid for
    the schema and the run should bail.
    """
    try:
        all_definitions = parse_schema_to_definitions(schema_name, schema_data)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"could not parse schema {schema_name!r}: {exc}")
        return True

    root_msgdef = all_definitions.get(schema_name)
    if root_msgdef is None:
        parts = schema_name.split("/")
        short_name = f"{parts[0]}/{parts[-1]}"
        root_msgdef = all_definitions.get(short_name)
    if root_msgdef is None:
        logger.warning(f"no message definition for schema {schema_name!r}")
        return True

    try:
        series.parsed.validate(root_msgdef, all_definitions)
    except ValidationError:
        logger.exception(f"Query validation error for path {series.path_str!r}")
        return False
    return True


class _PlotTopicWriter(TopicWriter):
    """Buffer values for every series whose topic matches this writer."""

    def __init__(self, series: list[SeriesData]) -> None:
        self.series = series
        self._validated = False

    def write(self, msg: DecodedMessage) -> None:
        if not self._validated:
            self._validated = True
            if msg.schema is not None:
                for series in self.series:
                    ok = _validate_series_against_schema(
                        series,
                        msg.schema.name,
                        msg.schema.data,
                    )
                    if not ok:
                        raise ValidationError(f"Validation failed for {series.path_str!r}")

        log_time_ns = int(msg.message.log_time)
        for series in self.series:
            try:
                value = series.parsed.apply(msg.decoded_message)
            except MessagePathError:
                continue
            if value is None or isinstance(value, (list, dict)):
                continue
            series.times_ns.append(log_time_ns)
            series.values.append(value)

    def close(self) -> None:
        # All emission deferred to PlotExporter.finish.
        pass


class PlotExporter(JsonRos2Exporter):
    """Exporter that gathers message-path series into a single Plotly figure.

    A ``PlotExporter`` instance is single-use — the gathered series live on
    the exporter so :meth:`finish` can render them all together.
    """

    name: ClassVar[str] = "plot"

    def __init__(
        self,
        *,
        output: Path | None,
        paths: list[str],
        title: str | None = None,
        downsample: int | None = None,
        xy: bool = False,
        force: bool = False,
    ) -> None:
        if not paths:
            raise ValueError("PlotExporter requires at least one path")
        if xy and len(paths) != 2:
            raise ValueError("--xy mode requires exactly 2 paths")

        self._output = output
        self._title = title
        self._downsample = downsample
        self._xy = xy
        self._force = force

        # Pre-parse paths up-front; surface syntax errors immediately.
        self._series: list[SeriesData] = []
        for arg in paths:
            label, path_str = parse_path_arg(arg)
            parsed = parse_message_path(path_str)  # raises on syntax error
            self._series.append(SeriesData(label=label, path_str=path_str, parsed=parsed))

        self._series_by_topic: dict[str, list[SeriesData]] = {}
        for series in self._series:
            self._series_by_topic.setdefault(series.parsed.topic, []).append(series)

    @property
    def topics_needed(self) -> list[str]:
        return list(self._series_by_topic.keys())

    def accepts(self, schema: Schema | None) -> bool:  # noqa: ARG002
        # Topic filtering is handled upstream by run_export(topics=...).
        # Per-schema validation happens lazily on the first message inside
        # the writer.
        return True

    def validate_output(self, output: str | Path | None, *, force: bool) -> Path | None:
        if output is None:
            # Sentinel — the exporter writes (or doesn't) on its own from
            # finish(); the driver only uses the returned path for status
            # printing and as TopicContext.output_path. The writers don't
            # touch it.
            return Path()

        path = Path(output)
        if path.exists() and path.is_dir():
            logger.error(f"{path} is a directory; expected an HTML file path.")
            return None
        if path.exists() and not (force or self._force):
            logger.error(f"{path} exists. Use --force to overwrite.")
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def open_topic(self, ctx: TopicContext) -> _PlotTopicWriter:
        series = self._series_by_topic.get(ctx.topic, [])
        return _PlotTopicWriter(series)

    def finish(
        self,
        output_path: Path,  # noqa: ARG002 - plot writes to self._output.
        counts: Mapping[int, int],  # noqa: ARG002 - counts are irrelevant to rendering.
    ) -> None:
        # Convert absolute log_time_ns into "seconds since first sample".
        first_ns: int | None = None
        for series in self._series:
            if series.times_ns:
                first_ns = (
                    series.times_ns[0] if first_ns is None else min(first_ns, series.times_ns[0])
                )

        if first_ns is None:
            raise RuntimeError("No plottable data found for any path.")

        for series in self._series:
            if not series.times_ns:
                logger.warning(f"no plottable data for {series.label!r}")

        rendered_series = [
            (series, [(ts - first_ns) / 1e9 for ts in series.times_ns])
            for series in self._series
            if series.times_ns
        ]

        if self._downsample:
            new_rendered: list[tuple[SeriesData, list[float]]] = []
            for series, times_s in rendered_series:
                if any(isinstance(v, str) for v in series.values[:10]):
                    new_rendered.append((series, times_s))
                    continue
                before = len(times_s)
                t_out, v_out = downsample_lttb(times_s, series.values, self._downsample)
                if before != len(t_out):
                    logger.info(f"Downsampled {series.label!r}: {before} → {len(t_out)} points")
                # Stash the downsampled values back onto the series so XY
                # mode also benefits.
                series.values = v_out
                new_rendered.append((series, t_out))
            rendered_series = new_rendered

        if self._xy:
            self._render_xy(rendered_series)
            return

        fig = go.Figure()
        for series, times_s in rendered_series:
            fig.add_trace(
                go.Scattergl(
                    x=times_s,
                    y=series.values,
                    mode="lines",
                    name=series.label,
                )
            )
        fig.update_layout(
            title=self._title or ", ".join(s.label for s, _ in rendered_series),
            xaxis_title="Time (s)",
            yaxis_title="Value",
            hovermode="x unified",
        )
        self._emit(fig)

    def _render_xy(
        self,
        rendered_series: list[tuple[SeriesData, list[float]]],
    ) -> None:
        if len(rendered_series) != 2:
            raise RuntimeError("--xy mode needs both paths to have data")

        (x_series, x_times), (y_series, y_times) = rendered_series
        x_by_time = dict(zip(x_times, x_series.values, strict=True))
        matched_x: list[float | bool | str] = []
        matched_y: list[float | bool | str] = []
        for t, v in zip(y_times, y_series.values, strict=True):
            if t in x_by_time:
                matched_x.append(x_by_time[t])
                matched_y.append(v)

        if not matched_x:
            raise RuntimeError("No matching timestamps between the two paths")

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
            title=self._title or f"{x_series.label} vs {y_series.label}",
            xaxis_title=x_series.label,
            yaxis_title=y_series.label,
            hovermode="closest",
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        self._emit(fig)

    def _emit(self, fig: go.Figure) -> None:
        if self._output is None:
            fig.show()
            return
        fig.write_html(str(self._output))
        logger.info(f"Plot saved to {self._output}")
