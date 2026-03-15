from typing import ClassVar

from rich.text import Text
from ros_parser.message_path import (
    LarkError,
    MathModifier,
    MessagePath,
    MessagePathError,
    parse_message_path,
)
from textual import events
from textual.app import ComposeResult
from textual.binding import BindingType
from textual.reactive import reactive
from textual.validation import ValidationResult, Validator
from textual.widgets import Input, Static

from digitalis.reader.types import MessageEvent
from digitalis.transforms import TIMESERIES_OPS, TransformContext, apply_with_history
from digitalis.ui.panels.base import SCHEMA_ANY, BasePanel
from digitalis.ui.panels.plot_buffer import TimeSeriesBuffer
from digitalis.ui.panels.plot_renderer import SERIES_COLORS, PlotSeries, render_line_chart
from digitalis.utilities import NANOSECONDS_PER_SECOND

# Default visible time window in seconds
DEFAULT_WINDOW_S = 30.0
MIN_WINDOW_S = 0.5
MAX_WINDOW_S = 600.0
ZOOM_STEP = 1.5


def _to_full_path(field_path: str) -> str:
    """Prepend dummy topic, auto-inserting '.' if the path doesn't start with one."""
    if not field_path.startswith((".", "[", "{", "@")):
        field_path = "." + field_path
    return f"/dummy{field_path}"


class ChartDisplay(Static):
    """Widget that displays the braille chart."""


class PlotPathValidator(Validator):
    """Validates comma-separated message path expressions."""

    def validate(self, value: str) -> ValidationResult:
        if not value.strip():
            return self.success()

        errors: list[str] = []
        for raw_part in value.split(","):
            stripped = raw_part.strip()
            if not stripped:
                continue
            try:
                parse_message_path(_to_full_path(stripped))
            except (LarkError, MessagePathError) as e:
                errors.append(f"{stripped}: {e}")

        if errors:
            return self.failure("\n".join(errors))
        return self.success()


class Plot(BasePanel[MessageEvent]):
    SUPPORTED_SCHEMAS: ClassVar[set[str]] = {SCHEMA_ANY}
    PRIORITY: ClassVar[int] = 900

    DEFAULT_CSS = """
    Plot {
        ChartDisplay {
            width: 1fr;
            height: 1fr;
        }
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("plus", "zoom_in", "Zoom In"),
        ("minus", "zoom_out", "Zoom Out"),
        ("a", "auto_fit", "Auto Fit"),
        ("c", "clear_data", "Clear"),
    ]

    field_input: reactive[str] = reactive("")
    _parse_error: str = ""

    def __init__(self) -> None:
        super().__init__()
        self._buffers: list[TimeSeriesBuffer] = []
        self._contexts: list[TransformContext] = []
        self._paths: list[MessagePath] = []
        self._series_names: list[str] = []
        self._window_s: float = DEFAULT_WINDOW_S
        self._auto_fit: bool = False
        self._validator = PlotPathValidator()

    def compose(self) -> ComposeResult:
        yield Input(
            placeholder="Field paths (e.g. .angular_velocity.x, .angular_velocity.y)",
            compact=True,
            validators=[self._validator],
        )
        yield ChartDisplay()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle field path input changes."""
        self.field_input = event.value

        is_valid = not event.validation_result or event.validation_result.is_valid
        if is_valid:
            self._parse_paths(event.value)
            self._parse_error = ""
        else:
            self._parse_error = (
                event.validation_result.failure_descriptions[0]
                if event.validation_result and event.validation_result.failure_descriptions
                else "Invalid path"
            )
            self._buffers.clear()
            self._contexts.clear()
            self._paths.clear()
            self._series_names.clear()

        self._update_display()

    def _parse_paths(self, text: str) -> None:
        """Parse comma-separated field paths into validated paths."""
        self._buffers.clear()
        self._contexts.clear()
        self._paths.clear()
        self._series_names.clear()

        if not text.strip():
            return

        for raw_part in text.split(","):
            stripped = raw_part.strip()
            if not stripped:
                continue
            parsed = parse_message_path(_to_full_path(stripped))
            self._paths.append(parsed)
            self._buffers.append(TimeSeriesBuffer())
            self._contexts.append(TransformContext())
            self._series_names.append(stripped)

    def _has_timeseries_op(self, path: MessagePath) -> bool:
        """Check if path contains a time-series operation."""
        return any(
            isinstance(seg, MathModifier) and seg.operation in TIMESERIES_OPS
            for seg in path.segments
        )

    def watch_data(self, _data: MessageEvent | None) -> None:
        """Process new message data."""
        if not self.data or not self._paths:
            return

        for i, path in enumerate(self._paths):
            try:
                if self._has_timeseries_op(path):
                    value = apply_with_history(
                        path, self.data.message, self.data.timestamp_ns, self._contexts[i]
                    )
                else:
                    value = path.apply(self.data.message)

                if isinstance(value, (int, float)):
                    self._buffers[i].append(self.data.timestamp_ns, float(value))
            except (MessagePathError, ValueError, TypeError):
                pass

        self._update_display()

    def _update_display(self) -> None:
        """Update the chart display, showing errors or chart."""
        display = self.query_one(ChartDisplay)

        if self._parse_error:
            error_text = Text()
            error_text.append("Invalid path:\n", style="bold red")
            error_text.append(self._parse_error, style="red")
            display.update(error_text)
            return

        if not self._paths:
            display.update("")
            return

        self._render_chart()

    def _render_chart(self) -> None:
        """Render the chart to the display widget."""
        display = self.query_one(ChartDisplay)
        size = display.size

        if size.width < 12 or size.height < 4:
            return

        # Collect all raw series data
        raw_series: list[tuple[int, PlotSeries]] = []
        earliest_ns: float = float("inf")
        latest_ns: float = float("-inf")

        for i, buf in enumerate(self._buffers):
            if len(buf) == 0:
                continue
            timestamps, values = buf.get_data()
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            raw_series.append((i, PlotSeries(timestamps=timestamps, values=values, color=color)))
            earliest_ns = min(earliest_ns, float(timestamps[0]))
            latest_ns = max(latest_ns, float(timestamps[-1]))

        if not raw_series:
            display.update("")
            return

        # Compute visible time range
        if self._auto_fit:
            t_min_ns = earliest_ns
            t_max_ns = latest_ns
        else:
            # Rolling window anchored at latest data
            data_span_s = (latest_ns - earliest_ns) / NANOSECONDS_PER_SECOND
            # Shrink window to actual data span if buffer isn't full yet
            effective_window = min(self._window_s, data_span_s)
            window_ns = effective_window * NANOSECONDS_PER_SECOND
            t_max_ns = latest_ns
            t_min_ns = latest_ns - window_ns

        # Convert to relative seconds from window start
        t_origin = t_min_ns
        t_max_s = (t_max_ns - t_origin) / NANOSECONDS_PER_SECOND

        # Ensure minimum span
        if t_max_s < 0.001:
            t_max_s = 1.0

        # Clip series to visible window and convert to relative seconds
        visible_series: list[PlotSeries] = []
        all_v_min, all_v_max = float("inf"), float("-inf")

        for _, series in raw_series:
            mask = (series.timestamps >= t_min_ns) & (series.timestamps <= t_max_ns)
            if not mask.any():
                continue

            t_sec = (series.timestamps[mask] - t_origin) / NANOSECONDS_PER_SECOND
            vals = series.values[mask]
            visible_series.append(PlotSeries(timestamps=t_sec, values=vals, color=series.color))

            all_v_min = min(all_v_min, float(vals.min()))
            all_v_max = max(all_v_max, float(vals.max()))

        if not visible_series:
            display.update("")
            return

        # Add padding to value range
        v_span = all_v_max - all_v_min
        if v_span == 0:
            v_span = 1.0
        v_pad = v_span * 0.05
        y_range = (all_v_min - v_pad, all_v_max + v_pad)

        chart_height = max(size.height - 1, 3)  # Reserve one row for status
        chart = render_line_chart(
            visible_series,
            width=size.width,
            height=chart_height,
            x_range=(0.0, t_max_s),
            y_range=y_range,
        )

        # Convert to Rich Text
        text = Text()
        for row_idx, row in enumerate(chart):
            for char, color in row:
                text.append(char, style=color)
            if row_idx < len(chart) - 1:
                text.append("\n")

        # Status line: window info + legend
        text.append("\n")
        window_label = "all" if self._auto_fit else _format_duration(self._window_s)
        points = sum(len(b) for b in self._buffers)
        text.append(f" window:{window_label} pts:{points}", style="grey50")

        if len(self._series_names) > 1:
            for i, name in enumerate(self._series_names):
                color = SERIES_COLORS[i % len(SERIES_COLORS)]
                text.append(f" ■ {name}", style=color)

        display.update(text)

    # -- Zoom / pan actions --------------------------------------------------

    def _zoom(self, zoom_in: bool) -> None:
        """Zoom the time window."""
        self._auto_fit = False
        if zoom_in:
            self._window_s = max(MIN_WINDOW_S, self._window_s / ZOOM_STEP)
        else:
            self._window_s = min(MAX_WINDOW_S, self._window_s * ZOOM_STEP)
        self._update_display()

    def action_zoom_in(self) -> None:
        self._zoom(zoom_in=True)

    def action_zoom_out(self) -> None:
        self._zoom(zoom_in=False)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        """Zoom in on mouse scroll up."""
        event.stop()
        self._zoom(zoom_in=True)

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        """Zoom out on mouse scroll down."""
        event.stop()
        self._zoom(zoom_in=False)

    def action_auto_fit(self) -> None:
        self._auto_fit = True
        self._update_display()

    def action_clear_data(self) -> None:
        for buf in self._buffers:
            buf.clear()
        for ctx in self._contexts:
            ctx.prev_value = None
            ctx.prev_timestamp_ns = None
        self._update_display()


def _format_duration(seconds: float) -> str:
    """Format duration compactly for status line."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    return f"{minutes:.1f}m"
