"""Braille-character line chart renderer for terminal plots."""

import math
from dataclasses import dataclass

import numpy as np

# Braille dot encoding: each cell is 2 wide x 4 tall dots
# Dot positions map to bit offsets in Unicode Braille (U+2800-U+28FF)
DOT_MAP = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]]
BRAILLE_BASE = 0x2800

SERIES_COLORS = ["#1e90ff", "#ff6347", "#32cd32", "#ffd700", "#da70d6", "#00ffff"]


@dataclass
class PlotSeries:
    """A single series to render."""

    timestamps: np.ndarray
    values: np.ndarray
    color: str


def _nice_ticks(lo: float, hi: float, max_ticks: int = 5) -> list[float]:
    """Generate 'nice' tick values using 1/2/5 intervals."""
    if hi <= lo:
        return [lo]

    raw_step = (hi - lo) / max(max_ticks - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step)) if raw_step > 0 else 1
    residual = raw_step / magnitude

    if residual <= 1.5:
        nice_step = magnitude
    elif residual <= 3.5:
        nice_step = 2 * magnitude
    elif residual <= 7.5:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude

    start = math.floor(lo / nice_step) * nice_step
    ticks: list[float] = []
    tick = start
    while tick <= hi + nice_step * 0.01:
        if lo - nice_step * 0.01 <= tick:
            ticks.append(tick)
        tick += nice_step

    return ticks or [lo, hi]


def _format_tick(value: float) -> str:
    """Format a tick value compactly."""
    if value == 0:
        return "0"
    abs_val = abs(value)
    if abs_val >= 1000:
        return f"{value:.0f}"
    if abs_val >= 1:
        return f"{value:.1f}"
    if abs_val >= 0.01:
        return f"{value:.2f}"
    return f"{value:.1e}"


def _bresenham_dots(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    grid: list[list[int]],
    color_grid: list[list[str]],
    color: str,
) -> None:
    """Draw a line between two dot-space points using Bresenham's algorithm."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    while True:
        if 0 <= y0 < rows and 0 <= x0 < cols:
            grid[y0][x0] = 1
            color_grid[y0][x0] = color

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def render_line_chart(
    series_list: list[PlotSeries],
    width: int,
    height: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> list[list[tuple[str, str]]]:
    """Render line chart using braille characters.

    Args:
        series_list: List of PlotSeries to render.
        width: Terminal cell width for chart area.
        height: Terminal cell height for chart area.
        x_range: (min_x, max_x) data range for x-axis.
        y_range: (min_y, max_y) data range for y-axis.

    Returns:
        List of rows, each row is a list of (character, color) tuples.
    """
    x_lo, x_hi = x_range
    y_lo, y_hi = y_range

    chart_h = max(height - 1, 2)  # Reserve bottom row for X-axis

    # Pre-compute Y ticks and margin from actual label widths
    y_ticks = _nice_ticks(y_lo, y_hi, max_ticks=min(chart_h, 6))
    y_labels = {tick: _format_tick(tick) for tick in y_ticks}
    # Margin = longest label + 1 for the '|' separator
    y_margin = max((len(lbl) for lbl in y_labels.values()), default=3) + 1
    y_margin = max(y_margin, 4)  # minimum margin

    chart_w = max(width - y_margin, 4)

    # Dot-space dimensions (2x horizontal, 4x vertical per cell)
    dot_w = chart_w * 2
    dot_h = chart_h * 4

    # Initialize dot grid and color grid
    dot_grid: list[list[int]] = [[0] * dot_w for _ in range(dot_h)]
    color_grid: list[list[str]] = [[""] * dot_w for _ in range(dot_h)]

    # Avoid division by zero
    x_span = x_hi - x_lo if x_hi > x_lo else 1.0
    y_span = y_hi - y_lo if y_hi > y_lo else 1.0

    # Plot each series
    for series in series_list:
        if len(series.timestamps) < 2:
            # Single point: just mark it
            if len(series.timestamps) == 1:
                dx = int((series.timestamps[0] - x_lo) / x_span * (dot_w - 1))
                dy = int((1.0 - (series.values[0] - y_lo) / y_span) * (dot_h - 1))
                dx = max(0, min(dot_w - 1, dx))
                dy = max(0, min(dot_h - 1, dy))
                dot_grid[dy][dx] = 1
                color_grid[dy][dx] = series.color
            continue

        # Map data to dot coordinates
        xs = ((series.timestamps - x_lo) / x_span * (dot_w - 1)).astype(int)
        ys = ((1.0 - (series.values - y_lo) / y_span) * (dot_h - 1)).astype(int)
        xs = np.clip(xs, 0, dot_w - 1)
        ys = np.clip(ys, 0, dot_h - 1)

        # Draw lines between consecutive points
        for i in range(len(xs) - 1):
            _bresenham_dots(
                int(xs[i]),
                int(ys[i]),
                int(xs[i + 1]),
                int(ys[i + 1]),
                dot_grid,
                color_grid,
                series.color,
            )

    # Convert dot grid to braille characters
    chart_rows: list[list[tuple[str, str]]] = []

    for row in range(chart_h):
        line: list[tuple[str, str]] = []

        # Y-axis label
        label = ""
        for tick in y_ticks:
            tick_row = int((1.0 - (tick - y_lo) / y_span) * (chart_h - 1))
            if tick_row == row:
                label = y_labels[tick]
                break
        line.append((f"{label:>{y_margin - 1}}|", "grey50"))

        # Chart cells
        for col in range(chart_w):
            bitmask = 0
            cell_color = ""
            for dr in range(4):
                for dc in range(2):
                    dy = row * 4 + dr
                    dx = col * 2 + dc
                    if dy < dot_h and dx < dot_w and dot_grid[dy][dx]:
                        bitmask |= DOT_MAP[dr][dc]
                        if color_grid[dy][dx]:
                            cell_color = color_grid[dy][dx]

            char = chr(BRAILLE_BASE + bitmask)
            line.append((char, cell_color or "grey50"))

        chart_rows.append(line)

    # X-axis labels
    x_ticks = _nice_ticks(x_lo, x_hi, max_ticks=min(chart_w // 10, 6))
    x_axis: list[tuple[str, str]] = []
    x_axis.append((" " * y_margin, "grey50"))

    # Build X-axis string
    x_line = [" "] * chart_w
    for tick in x_ticks:
        pos = int((tick - x_lo) / x_span * (chart_w - 1)) if x_span > 0 else 0
        pos = max(0, min(chart_w - 1, pos))
        label = _format_tick(tick)
        start = max(0, pos - len(label) // 2)
        for i, ch in enumerate(label):
            if start + i < chart_w:
                x_line[start + i] = ch

    x_axis.append(("".join(x_line), "grey50"))
    chart_rows.append(x_axis)

    return chart_rows
