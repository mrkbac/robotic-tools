import itertools
from collections.abc import Callable, Iterable
from functools import partial
from typing import TypeVar

import numpy as np
from rich.color import Color
from rich.segment import Segment
from rich.style import Style

from digitalis.turbo import interpolate_turbo_color

# Type definitions
HalfCellSize = tuple[int, int]  # (width_chars, height_chars)
CenterPoint = tuple[float, float]  # (center_x, center_y)

# Half-block character constants
CH_SPACE = " "
CH_UPPER = "▀"
CH_LOWER = "▄"


def turbo_color_from_value(
    value: float | np.ndarray, value_range: tuple[float, float], *, clip: bool = True
) -> Color | None:
    """Map value to turbo colormap hex string.

    Args:
        value: The value to map
        value_range: Optional (min, max) range. If None, assumes (1, 100) for legacy compatibility
    """

    v_min, v_max = value_range
    if not np.isfinite(value) or value == -np.inf or v_max <= v_min:
        return None

    normalized = (value - v_min) / (v_max - v_min)
    if clip:
        normalized = np.clip(normalized, 0.0, 1.0)
    elif normalized < 0 or normalized > 1:
        return None

    r, g, b = interpolate_turbo_color(normalized)
    return Color.from_rgb(int(r * 255), int(g * 255), int(b * 255))


_T = TypeVar("_T")


def _batched(iterable: Iterable[_T], n: int) -> Iterable[tuple[_T, ...]]:
    """itertools.batched for python <3.12"""
    it = iter(iterable)
    while chunk := tuple(itertools.islice(it, n)):
        yield chunk


def render_halfblock_grid(
    occupancy_grid: np.ndarray,
    value_grid: np.ndarray,
    size: HalfCellSize,
    color_mapper: Callable[[float | np.ndarray], Color | None],
    background_style: Style | None = None,
) -> Iterable[Segment]:
    """Render a grid using half-block characters.

    Args:
        occupancy_grid: Grid where truthy values indicate occupied cells
        value_grid: Grid with values for color mapping (only used for occupied cells)
        size: Terminal size in characters (width, height)
        color_mapper: Function to map values to color strings
        background_style: Style for empty cells

    Yields:
        Segment: Rich segments for terminal rendering
    """
    width_chars, height_chars = size

    for top_idx, bot_idx in _batched(range(2 * height_chars - 1, -1, -1), 2):
        line = []
        for x in range(width_chars):
            top_occ = bool(occupancy_grid[top_idx, x])
            bot_occ = bool(occupancy_grid[bot_idx, x])

            if not top_occ and not bot_occ:  # Neither occupied
                ch = CH_SPACE
                style = background_style
            elif not top_occ and bot_occ:  # Only bottom occupied
                ch = CH_LOWER
                bot_value = value_grid[bot_idx, x]
                color = color_mapper(bot_value)
                style = Style(color=color)
            elif top_occ and not bot_occ:  # Only top occupied
                ch = CH_UPPER
                top_value = value_grid[top_idx, x]
                color = color_mapper(top_value)
                style = Style(color=color)
            else:  # Both occupied
                ch = CH_LOWER
                top_value = value_grid[top_idx, x]
                top_color = color_mapper(top_value)
                bot_value = value_grid[bot_idx, x]
                bot_color = color_mapper(bot_value)
                style = Style(color=bot_color, bgcolor=top_color)

            line.append(Segment(ch, style))
        # Simplify the line by removing consecutive segments with the same style
        yield from Segment.simplify(line)

        yield Segment.line()


def _validate_inputs(
    points: np.ndarray, size: HalfCellSize, resolution: float, center_point: CenterPoint
) -> None:
    """Validate input parameters."""
    if not isinstance(points, np.ndarray) or points.shape[-1:] != (3,):
        raise ValueError("points must be a numpy array with shape (..., 3)")

    w, h = size
    if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
        raise ValueError(f"size must contain positive integers, got ({w}, {h})")

    if not (np.isfinite([resolution, *center_point]).all() and resolution > 0):
        raise ValueError("resolution and center_point must be finite with resolution > 0")


def _create_grids(
    points: np.ndarray, size: HalfCellSize, resolution: float, center_point: CenterPoint
) -> tuple[np.ndarray, np.ndarray]:
    """Create occupancy and z-value grids from points."""
    width_chars, height_chars = size
    grid_h, grid_w = height_chars * 2, width_chars

    # Initialize grids
    grid_occ = np.zeros((grid_h, grid_w), dtype=bool)
    grid_z = np.full((grid_h, grid_w), -np.inf)

    if points.size == 0:
        return grid_occ, grid_z

    # Filter finite points
    mask = np.isfinite(points).all(axis=1)
    if not mask.any():
        return grid_occ, grid_z

    pts = points[mask]

    # Calculate grid bounds
    center = np.array(center_point)
    world_size = np.array([grid_w * resolution, grid_h * resolution])
    x_min, y_min = center - world_size / 2
    x_max, y_max = center + world_size / 2

    # Filter points within viewport bounds
    in_bounds = (
        (pts[:, 0] >= x_min) & (pts[:, 0] < x_max) & (pts[:, 1] >= y_min) & (pts[:, 1] < y_max)
    )

    if not in_bounds.any():
        return grid_occ, grid_z

    visible_pts = pts[in_bounds]

    # Convert world coordinates to grid indices
    x_indices = ((visible_pts[:, 0] - x_min) / resolution).astype(int)
    y_indices = ((visible_pts[:, 1] - y_min) / resolution).astype(int)

    # Update grids using advanced indexing
    grid_occ[y_indices, x_indices] = True
    np.maximum.at(grid_z, (y_indices, x_indices), visible_pts[:, 2])

    return grid_occ, grid_z


def render_pointcloud(
    points: np.ndarray,
    size: HalfCellSize,
    resolution: float,
    center_point: CenterPoint,
    background_style: Style | None = None,
) -> list[Segment]:
    """
    Render a 2D point cloud using half-cell characters.

    Each terminal character encodes two vertical 'pixel rows':
    - ' '  : no occupancy
    - '▀'  : top half occupied
    - '▄'  : bottom half occupied

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3) with columns [x, y, z].
    size :
        Target render size in terminal characters (width, height).
    resolution :
        Meters per character cell (e.g., 0.1 = 10cm per cell).
    center_point :
        World coordinates (x, y) at the center of the display.
    background_style :
        Rich style for background (spaces). Default None (inherit).

    Returns
    -------
        Rich segments for rendering the point cloud.
    """
    _validate_inputs(points, size, resolution, center_point)

    # Calculate global z-range from full point cloud to ensure consistent colors
    if points.size == 0:
        z_min, z_max = 0.0, 0.0
    else:
        finite_mask = np.isfinite(points).all(axis=1)
        if finite_mask.any():
            finite_z = points[finite_mask, 2]
            z_min, z_max = float(np.min(finite_z)), float(np.max(finite_z))
        else:
            z_min, z_max = 0.0, 0.0

    grid_occ, grid_z = _create_grids(points, size, resolution, center_point)

    return list(
        render_halfblock_grid(
            occupancy_grid=grid_occ,
            value_grid=grid_z,
            size=size,
            color_mapper=partial(turbo_color_from_value, value_range=(z_min, z_max)),
            background_style=background_style,
        )
    )
