"""Plot exporter — render message-path values as Plotly charts.

Lives in the Exporter pipeline so plot benefits from the shared progress UI,
topic filter, decoder-factory plumbing, and lifecycle hooks. The Plotly
figure is assembled in :meth:`PlotExporter.finish` once every per-topic
writer has flushed its buffered samples.

When ``output`` is ``None`` the figure is opened interactively
(``fig.show()``). Otherwise the output suffix picks the renderer: ``.html``
writes standalone HTML, while ``.png`` / ``.svg`` / ``.pdf`` / ``.jpg`` /
``.webp`` render a static image via kaleido (no browser needed — handy over
SSH).

Array-valued paths (e.g. ``/joints.position[:].@degrees``) expand into one
trace per element index, labelled ``<label>[i]``.
"""

from __future__ import annotations

import array
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ros_parser import parse_schema_to_definitions
from ros_parser.message_path import (
    NO_OUTPUT,
    MessagePath,
    MessagePathError,
    MessagePathEvaluator,
    MessagePathVariables,
    ValidationError,
    parse_message_path,
)

from pymcap_cli.core.named_message_path import parse_path_arg
from pymcap_cli.exporters.base import Exporter

if TYPE_CHECKING:
    from collections.abc import Mapping

    from small_mcap import Channel, DecodedMessage, Schema

    from pymcap_cli.exporters.base import TopicContext

logger = logging.getLogger(__name__)

PlotKind = Literal["time", "histogram", "xy"]
HistogramNormalization = Literal["count", "probability", "density"]

# Static-image suffixes rendered via kaleido; everything else falls back to HTML.
IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf"})

# Sequence types an array-valued path may yield. ``str``/``bytes`` are excluded:
# a string is a scalar sample, and a raw byte blob is not a plottable series.
_EXPANDABLE_TYPES = (list, tuple, memoryview, bytearray, array.array)

# Upper bound on traces produced by a single array-valued path. Guards against a
# huge unsliced array (e.g. a 10k-element field) exploding into 10k traces; the
# excess is dropped with a warning, never silently.
_MAX_ARRAY_COLUMNS = 64


@dataclass
class SeriesData:
    """One labelled time-series gathered from a single message-path query.

    A scalar path fills ``times_ns`` / ``values``. An array-valued path (a
    slice, index-free array field, or element-wise modifier that yields a
    sequence) fills ``array_times_ns`` / ``array_values``, keyed by element
    index; each index becomes its own trace in :meth:`PlotExporter.finish`.
    """

    label: str
    path_str: str
    parsed: MessagePath
    evaluator: MessagePathEvaluator | None = None
    times_ns: list[int] = field(default_factory=list)
    values: list[float | bool | str] = field(default_factory=list)
    array_times_ns: dict[int, list[int]] = field(default_factory=dict)
    array_values: dict[int, list[float | bool | str]] = field(default_factory=dict)
    warned_no_match: bool = False  # set when an early "won't resolve" warning was emitted

    @property
    def has_data(self) -> bool:
        return bool(self.times_ns) or bool(self.array_values)

    @property
    def point_count(self) -> int:
        """Total samples collected for this path across scalar and array columns."""
        return len(self.times_ns) + sum(len(col) for col in self.array_values.values())


@dataclass
class _PlotSeries:
    """A single flattened, ready-to-render trace (label + samples)."""

    label: str
    times_ns: list[int]
    values: list[float | bool | str]


def _expand_series(series: SeriesData) -> list[_PlotSeries]:
    """Flatten a gathered :class:`SeriesData` into one trace per plottable line."""
    out: list[_PlotSeries] = []
    if series.times_ns:
        out.append(_PlotSeries(series.label, series.times_ns, series.values))

    indices = sorted(series.array_values)
    if len(indices) > _MAX_ARRAY_COLUMNS:
        logger.warning(
            f"Path {series.path_str!r} produced {len(indices)} array columns; "
            f"plotting the first {_MAX_ARRAY_COLUMNS}. Narrow it with a slice "
            "like [0:8] to plot fewer."
        )
        indices = indices[:_MAX_ARRAY_COLUMNS]
    out.extend(
        _PlotSeries(f"{series.label}[{idx}]", series.array_times_ns[idx], series.array_values[idx])
        for idx in indices
    )
    return out


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


def downsample_xy_lttb(
    x_values: list[float],
    y_values: list[float],
    target: int,
) -> tuple[list[float], list[float]]:
    """LTTB downsampling over paired XY trajectory coordinates."""
    point_count = len(x_values)
    if point_count <= target:
        return x_values, y_values

    out_x = [x_values[0]]
    out_y = [y_values[0]]
    bucket_size = (point_count - 2) / (target - 2)
    previous_index = 0

    for bucket_index in range(1, target - 1):
        bucket_start = int((bucket_index - 1) * bucket_size) + 1
        bucket_end = min(int(bucket_index * bucket_size) + 1, point_count)
        next_start = int(bucket_index * bucket_size) + 1
        next_end = min(int((bucket_index + 1) * bucket_size) + 1, point_count)

        average_x = sum(x_values[next_start:next_end]) / (next_end - next_start)
        average_y = sum(y_values[next_start:next_end]) / (next_end - next_start)
        previous_x = x_values[previous_index]
        previous_y = y_values[previous_index]

        best_index = bucket_start
        best_area = -1.0
        best_distance = -1.0
        for candidate_index in range(bucket_start, bucket_end):
            candidate_x = x_values[candidate_index]
            candidate_y = y_values[candidate_index]
            area = abs(
                (previous_x - average_x) * (candidate_y - previous_y)
                - (previous_x - candidate_x) * (average_y - previous_y)
            )
            distance = (candidate_x - previous_x) ** 2 + (candidate_y - previous_y) ** 2
            if area > best_area or (area == best_area and distance > best_distance):
                best_area = area
                best_distance = distance
                best_index = candidate_index

        out_x.append(x_values[best_index])
        out_y.append(y_values[best_index])
        previous_index = best_index

    out_x.append(x_values[-1])
    out_y.append(y_values[-1])
    return out_x, out_y


def _validate_series_against_schema(
    series: SeriesData, schema_name: str, schema_data: bytes
) -> str | None:
    """Validate one series against a ROS message schema.

    Returns ``None`` when the path is acceptable (or the schema couldn't be
    parsed — we degrade gracefully); otherwise the validation error message,
    so the caller can surface it as an early, human-readable warning.
    """
    try:
        all_definitions = parse_schema_to_definitions(schema_name, schema_data)
    except Exception:  # noqa: BLE001 - non-ROS schema (e.g. JSON); can't validate.
        return None

    root_msgdef = all_definitions.get(schema_name)
    if root_msgdef is None:
        parts = schema_name.split("/")
        root_msgdef = all_definitions.get(f"{parts[0]}/{parts[-1]}")
    if root_msgdef is None:
        return None

    try:
        series.parsed.validate(root_msgdef, all_definitions)
    except ValidationError as exc:
        return str(exc)
    return None


class _PlotTopicWriter:
    """Buffer values for every series whose topic matches this writer."""

    def __init__(
        self,
        series: list[SeriesData],
        variables: MessagePathVariables | None = None,
    ) -> None:
        self.series = series
        self._variables = dict(variables or {})
        self._checked = False

    def write(self, msg: DecodedMessage) -> None:
        if not self._checked:
            self._checked = True
            self._check_first_message(msg)

        log_time_ns = int(msg.message.log_time)
        for series in self.series:
            try:
                if series.evaluator is not None:
                    value = series.evaluator.observe(
                        msg.decoded_message, log_time_ns, self._variables
                    )
                    if value is NO_OUTPUT:
                        continue
                else:
                    value = series.parsed.apply(msg.decoded_message, self._variables)
            except MessagePathError:
                continue
            if value is None or isinstance(value, dict):
                continue
            if isinstance(value, _EXPANDABLE_TYPES):
                for idx, element in enumerate(value):
                    if element is None or isinstance(element, (list, tuple, dict, memoryview)):
                        continue
                    series.array_times_ns.setdefault(idx, []).append(log_time_ns)
                    series.array_values.setdefault(idx, []).append(element)
                continue
            series.times_ns.append(log_time_ns)
            series.values.append(value)

    def _check_first_message(self, msg: DecodedMessage) -> None:
        """Warn early (on the first message of this topic) about paths that will
        never resolve — so a typo surfaces seconds in, not after the whole scan.
        """
        for series in self.series:
            topic = series.parsed.topic
            error = (
                _validate_series_against_schema(series, msg.schema.name, msg.schema.data)
                if msg.schema is not None
                else None
            )
            if error is not None:
                series.warned_no_match = True
                logger.warning(
                    f"Path {series.path_str!r} is not valid for topic {topic!r}: {error} "
                    "It will match nothing — check the field names (Ctrl-C to abort)."
                )
                continue
            if series.parsed.has_stream:
                # Stream paths need the stateful evaluator; a probe apply() always raises.
                continue
            # No schema to validate against (e.g. JSON): a runtime resolution
            # error on the first message means a bad field, not a filter miss.
            try:
                series.parsed.apply(msg.decoded_message, self._variables)
            except MessagePathError as exc:
                series.warned_no_match = True
                logger.warning(
                    f"Path {series.path_str!r} found no value in the first {topic!r} message "
                    f"({exc}). It will likely match nothing — check the field names "
                    "(Ctrl-C to abort)."
                )

    def close(self) -> None:
        # All emission deferred to PlotExporter.finish.
        pass


class PlotExporter(Exporter):
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
        kind: PlotKind = "time",
        bins: int | None = None,
        normalize: HistogramNormalization = "count",
        force: bool = False,
        source_name: str | None = None,
        variables: MessagePathVariables | None = None,
    ) -> None:
        if not paths:
            raise ValueError("PlotExporter requires at least one path")
        if kind == "xy" and len(paths) != 2:
            raise ValueError("XY plots require exactly 2 paths")
        if downsample is not None and downsample < 3:
            raise ValueError("--downsample must be at least 3")
        if kind == "histogram" and downsample is not None:
            raise ValueError("--downsample is only valid with --kind time or --kind xy")
        if bins is not None and bins <= 0:
            raise ValueError("--bins must be a positive integer")
        if kind != "histogram" and bins is not None:
            raise ValueError("--bins is only valid with --kind histogram")
        if kind != "histogram" and normalize != "count":
            raise ValueError("--normalize is only valid with --kind histogram")

        self._output = output
        self._title = title
        self._downsample = downsample
        self._kind = kind
        self._bins = bins
        self._normalize = normalize
        self._force = force
        self._source_name = source_name
        self._variables = dict(variables or {})

        # Pre-parse paths up-front; surface syntax errors immediately.
        self._series: list[SeriesData] = []
        for arg in paths:
            label, path_str = parse_path_arg(arg)
            parsed = parse_message_path(path_str)  # raises on syntax error
            if parsed.has_stream_reducer:
                raise ValueError(
                    f"Stream reducers (@@count, @@max, ...) produce a single value and "
                    f"cannot be plotted: {path_str!r}"
                )
            evaluator = MessagePathEvaluator(parsed) if parsed.has_stream else None
            self._series.append(
                SeriesData(label=label, path_str=path_str, parsed=parsed, evaluator=evaluator)
            )

        self._series_by_topic: dict[str, list[SeriesData]] = {}
        for series in self._series:
            self._series_by_topic.setdefault(series.parsed.topic, []).append(series)

    def decoder_factories(self) -> list[Any]:
        from mcap_ros2_support_fast.decoder import (  # noqa: PLC0415
            DecoderFactory as Ros2DecoderFactory,
        )
        from small_mcap import JSONDecoderFactory  # noqa: PLC0415

        return [JSONDecoderFactory(), Ros2DecoderFactory()]

    def accepts(self, channel: Channel, schema: Schema | None) -> bool:  # noqa: ARG002
        return channel.topic in self._series_by_topic

    def validate_output(self, output: str | Path | None, *, force: bool) -> Path | None:
        if output is None:
            # Sentinel — the exporter writes (or doesn't) on its own from
            # finish(); the driver only uses the returned path for status
            # printing and as TopicContext.output_path. The writers don't
            # touch it.
            return Path()

        path = Path(output)
        if path.exists() and path.is_dir():
            logger.error(f"{path} is a directory; expected an .html/.png/.svg/.pdf file path.")
            return None
        if path.exists() and not (force or self._force):
            logger.error(f"{path} exists. Use --force to overwrite.")
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def open_topic(self, ctx: TopicContext) -> _PlotTopicWriter:
        series = self._series_by_topic.get(ctx.topic, [])
        return _PlotTopicWriter(series, self._variables)

    def finish(
        self,
        output_path: Path,  # noqa: ARG002 - plot writes to self._output.
        counts: Mapping[int, int],  # noqa: ARG002 - counts are irrelevant to rendering.
    ) -> None:
        # Report per path what actually matched, so a mistyped topic/field is
        # obvious instead of silently producing an empty plot.
        self._report_matches()

        # Flatten each gathered query into one trace per plottable line
        # (array-valued paths fan out into per-index traces).
        plot_series = [ps for series in self._series for ps in _expand_series(series)]

        if not plot_series:
            raise RuntimeError(
                "No plottable data found for any path — every message path matched 0 messages."
            )

        if self._kind == "histogram":
            self._render_histogram(plot_series)
            return

        # Convert absolute log_time_ns into "seconds since first sample".
        first_ns: int | None = None
        for ps in plot_series:
            if ps.times_ns:
                first_ns = ps.times_ns[0] if first_ns is None else min(first_ns, ps.times_ns[0])

        if first_ns is None:
            raise RuntimeError(
                "No plottable data found for any path — every message path matched 0 messages."
            )

        rendered_series = [
            (ps, [(ts - first_ns) / 1e9 for ts in ps.times_ns]) for ps in plot_series if ps.times_ns
        ]

        if self._downsample and self._kind != "xy":
            new_rendered: list[tuple[_PlotSeries, list[float]]] = []
            for ps, times_s in rendered_series:
                if any(isinstance(v, str) for v in ps.values[:10]):
                    new_rendered.append((ps, times_s))
                    continue
                before = len(times_s)
                t_out, v_out = downsample_lttb(times_s, ps.values, self._downsample)
                if before != len(t_out):
                    logger.info(f"Downsampled {ps.label!r}: {before} → {len(t_out)} points")
                ps.values = v_out
                new_rendered.append((ps, t_out))
            rendered_series = new_rendered

        if self._kind == "xy":
            self._render_xy(rendered_series)
            return

        self._render_time(rendered_series)

    def _render_time(
        self,
        rendered_series: list[tuple[_PlotSeries, list[float]]],
    ) -> None:
        fig = go.Figure()
        for ps, times_s in rendered_series:
            fig.add_trace(
                go.Scattergl(
                    x=times_s,
                    y=ps.values,
                    mode="lines",
                    name=ps.label,
                )
            )
        fig.update_layout(
            title=self._compose_title(", ".join(ps.label for ps, _ in rendered_series)),
            xaxis_title="Time (s)",
            yaxis_title="Value",
            hovermode="x unified",
        )
        self._emit(fig)

    def _render_histogram(self, plot_series: list[_PlotSeries]) -> None:
        fig = make_subplots(
            rows=len(plot_series),
            cols=1,
            shared_xaxes=False,
            subplot_titles=[series.label for series in plot_series],
            vertical_spacing=min(0.12, 0.45 / len(plot_series)),
        )
        y_axis_title = {
            "count": "Count",
            "probability": "Probability",
            "density": "Density",
        }[self._normalize]

        for row, series in enumerate(plot_series, start=1):
            values = series.values
            row_y_axis_title = y_axis_title
            is_boolean = all(type(value) is bool for value in values)
            is_string = all(isinstance(value, str) for value in values)
            is_numeric = all(
                isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
            )

            if is_numeric:
                finite_values = [value for value in values if math.isfinite(float(value))]
                dropped = len(values) - len(finite_values)
                if dropped:
                    logger.warning(
                        f"Dropped {dropped} non-finite values from {series.label!r} histogram"
                    )
                if not finite_values:
                    raise RuntimeError(
                        f"Histogram series {series.label!r} has no finite numeric values"
                    )
                distinct_count = len(set(finite_values))
                if distinct_count == 1:
                    constant_value = float(finite_values[0])
                    value_label = f"{constant_value:g}"
                    half_range = max(abs(constant_value) * 0.01, 0.1)
                    frequency: int | float = (
                        len(finite_values) if self._normalize == "count" else 1.0
                    )
                    if self._normalize == "density":
                        row_y_axis_title = "Probability mass"
                        logger.info(
                            f"Density is undefined for constant series {series.label!r}; "
                            "showing probability mass 1"
                        )
                    fig.add_trace(
                        go.Bar(
                            x=[constant_value],
                            y=[frequency],
                            width=[half_range / 2],
                            name=series.label,
                            showlegend=False,
                        ),
                        row=row,
                        col=1,
                    )
                    fig.update_xaxes(
                        tickmode="array",
                        tickvals=[constant_value],
                        ticktext=[value_label],
                        range=[constant_value - half_range, constant_value + half_range],
                        row=row,
                        col=1,
                    )
                    fig.layout.annotations[
                        row - 1
                    ].text = f"{series.label} — constant {value_label}"
                else:
                    histnorm = {
                        "count": None,
                        "probability": "probability",
                        "density": "probability density",
                    }[self._normalize]
                    effective_bins: int | None = None
                    if self._bins is not None:
                        effective_bins = min(self._bins, distinct_count)
                        if effective_bins < self._bins:
                            logger.info(
                                f"Reduced histogram bins for {series.label!r}: "
                                f"{self._bins} → {effective_bins} for "
                                f"{distinct_count} distinct values"
                            )
                    fig.add_trace(
                        go.Histogram(
                            x=finite_values,
                            name=series.label,
                            nbinsx=effective_bins,
                            histnorm=histnorm,
                            showlegend=False,
                        ),
                        row=row,
                        col=1,
                    )
            elif is_boolean or is_string:
                if self._normalize == "density":
                    raise RuntimeError(
                        "Histogram density normalization is not defined for categorical "
                        f"series {series.label!r}"
                    )
                counts = Counter(values)
                categories = sorted(counts)
                frequencies: list[int | float] = [counts[value] for value in categories]
                if self._normalize == "probability":
                    frequencies = [frequency / len(values) for frequency in frequencies]
                fig.add_trace(
                    go.Bar(
                        x=categories,
                        y=frequencies,
                        name=series.label,
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )
            else:
                raise RuntimeError(
                    f"Histogram series {series.label!r} contains mixed numeric and categorical "
                    "values"
                )

            fig.update_xaxes(title_text="Value", row=row, col=1)
            fig.update_yaxes(title_text=row_y_axis_title, row=row, col=1)

        fig.update_layout(
            title=self._compose_title("Value distributions"),
            height=max(450, 300 * len(plot_series)),
            bargap=0.05,
            hovermode="closest",
        )
        self._emit(fig)

    def _render_xy(
        self,
        rendered_series: list[tuple[_PlotSeries, list[float]]],
    ) -> None:
        if len(rendered_series) != 2:
            raise RuntimeError(
                "XY plots need exactly two scalar series with data "
                "(array-valued paths expand into multiple series)."
            )

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

        if self._downsample and not any(
            isinstance(value, str) for value in (*matched_x, *matched_y)
        ):
            before = len(matched_x)
            sampled_x, sampled_y = downsample_xy_lttb(
                [float(value) for value in matched_x],
                [float(value) for value in matched_y],
                self._downsample,
            )
            matched_x[:] = sampled_x
            matched_y[:] = sampled_y
            if before != len(matched_x):
                logger.info(f"Downsampled XY trajectory: {before} → {len(matched_x)} points")

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
            title=self._compose_title(f"{x_series.label} vs {y_series.label}"),
            xaxis_title=x_series.label,
            yaxis_title=y_series.label,
            hovermode="closest",
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        self._emit(fig)

    def _report_matches(self) -> None:
        """Log, per requested path, how many samples matched — 0 means a typo."""
        for series in self._series:
            n = series.point_count
            if n == 0:
                if not series.warned_no_match:  # avoid repeating the early warning
                    logger.warning(
                        f"Path {series.path_str!r} matched 0 messages — check the topic and "
                        "field names (nothing will be plotted for it)."
                    )
                continue
            columns = len(series.array_values)
            suffix = f" across {columns} array columns" if columns else ""
            logger.info(f"Path {series.label!r}: {n:,} points{suffix}")

    def _compose_title(self, default: str) -> str:
        """Build the figure title, keeping the source filename for back-reference."""
        main = self._title or default
        if self._source_name:
            return f"{main}<br><sub>{self._source_name}</sub>"
        return main

    def _emit(self, fig: go.Figure) -> None:
        if self._output is None:
            fig.show()
            return
        if self._output.suffix.lower() in IMAGE_SUFFIXES:
            try:
                fig.write_image(str(self._output))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to render image {self._output}: {exc}. "
                    "Static image export needs kaleido — install it with "
                    "`uv add 'pymcap-cli[plot]'`."
                ) from exc
        else:
            fig.write_html(str(self._output))
        logger.info(f"Plot saved to {self._output}")
