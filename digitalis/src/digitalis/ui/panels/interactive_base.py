from abc import abstractmethod
from typing import ClassVar

import numpy as np
from textual import events, work
from textual.app import ComposeResult, RenderResult
from textual.binding import Binding, BindingType
from textual.reactive import reactive
from textual.widgets import Static

from digitalis.reader.types import MessageEvent
from digitalis.ui.panels.base import BasePanel
from digitalis.utilities import RichRender, nanoseconds_to_iso


class InteractiveRenderPanel(BasePanel[MessageEvent]):
    """Base class for panels with interactive zoom/pan capabilities and background rendering.

    Provides shared functionality for:
    - Zoom in/out with configurable resolution ranges
    - Pan by dragging with mouse
    - Fit view to data
    - Background rendering with @work decorator
    - Status display with timestamp
    - Consistent key bindings and mouse interactions
    """

    # Interactive state - shared by all panels
    center: reactive[tuple[float, float]] = reactive((0, 0))
    resolution: reactive[float] = reactive(0.1)  # Default, overridden by subclasses
    _rendered: reactive[RichRender | None] = reactive(None)

    # Zoom constraints - configurable by subclasses
    MIN_RESOLUTION: float = 0.001
    MAX_RESOLUTION: float = 10.0
    ZOOM_FACTOR: float = 1.1

    # Fit action text - customizable by subclasses
    FIT_ACTION_TEXT: str = "Fit View"

    def __init__(self, *, default_resolution: float = 0.1) -> None:
        super().__init__()
        self.resolution = default_resolution
        self._first_data: bool = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("+", "zoom_in", "Zoom In"),
        Binding("-", "zoom_out", "Zoom Out"),
        Binding("a", "fit_view", "Fit View"),
    ]

    def compose(self) -> ComposeResult:
        """Common composition: top status bar."""
        yield Static(id="top")

    # Abstract methods for subclass implementation
    @abstractmethod
    @work(exclusive=True)
    async def _process_background_render(self) -> None:
        """Background worker for panel-specific rendering.

        Should:
        1. Check if data is available and widget is properly sized
        2. Process data into renderable format
        3. Set self._rendered to RichRender result
        4. Call self._update_status()
        """

    @abstractmethod
    def action_fit_view(self) -> None:
        """Fit view to panel-specific data bounds."""

    @abstractmethod
    def _update_status(self) -> None:
        """Update the top status display with panel-specific information."""

    @abstractmethod
    def _has_renderable_data(self) -> bool:
        """Check if panel has data worth auto-fitting to.

        Returns:
            True if panel has sufficient data to auto-fit view
        """

    @abstractmethod
    def _get_hover_info(
        self, world_pos: tuple[float, float], _screen_x: int, _screen_y: int
    ) -> str | None:
        """Get hover information for the given position.

        Args:
            world_pos: World coordinates (x, y) corresponding to mouse position
            _screen_x: Screen X coordinate of mouse
            _screen_y: Screen Y coordinate of mouse

        Returns:
            Tooltip text to display, or None for no tooltip
        """

    # Shared interactive behavior
    def watch_data(self, data: MessageEvent | None) -> None:
        """Watch for data changes to reset first_data flag when topic changes."""
        if data is None:
            # Topic changed or cleared - reset auto-fit flag
            self._first_data = True

    def _trigger_background_render(self) -> None:
        """Trigger background rendering if conditions are met."""
        if self.data is not None:
            self._process_background_render()

    def _handle_first_data_auto_fit(self) -> None:
        """Handle auto-fit behavior on first data reception.

        This method should be called by subclasses in their watch_data method
        after processing new data.
        """
        if not self._has_renderable_data():
            return

        # Auto-fit on first data (only if widget is mounted)
        if self._first_data:
            self._first_data = False
            if self.size.width > 0 and self.size.height > 0:
                self.action_fit_view()
            else:
                # Defer fit until widget is mounted
                self.call_after_refresh(self.action_fit_view)
        else:
            self._trigger_background_render()

    def watch_center(self, _center: tuple[float, float]) -> None:
        """Re-render when center changes."""
        self._trigger_background_render()

    def watch_resolution(self, _resolution: float) -> None:
        """Re-render when resolution changes."""
        self._trigger_background_render()

    def on_resize(self, _event: events.Resize) -> None:
        """Re-render when widget size changes."""
        self._trigger_background_render()

    def render(self) -> RenderResult:
        """Common render logic with loading states."""
        if self.data is None:
            return self._get_no_data_message()

        if self._rendered is None:
            return self._get_loading_message()

        return self._rendered

    def _get_no_data_message(self) -> str:
        """Override in subclasses for custom 'no data' message."""
        return "No data available"

    def _get_loading_message(self) -> str:
        """Override in subclasses for custom 'loading' message."""
        return "Processing data..."

    # Mouse interaction methods
    def on_mouse_move(self, event: events.MouseMove) -> None:
        """Handle mouse dragging for panning and hovering for tooltips."""
        if event.button == 1:
            # Handle dragging for panning
            self.center = (
                self.center[0] - event.delta_x * self.resolution,
                self.center[1]
                + (event.delta_y * self.resolution) * 2,  # 2x for half-block rendering
            )
        else:
            # Handle hovering for tooltips
            self._update_hover_tooltip(event.x, event.y)

    def _update_hover_tooltip(self, x: int, y: int) -> None:
        """Update tooltip based on mouse position."""
        world_pos = self._screen_to_world(x, y)
        hover_info = self._get_hover_info(world_pos, x, y)
        self.tooltip = hover_info or ""

    def _screen_to_world(self, x: int, y: int) -> tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        center_x, center_y = self.center
        widget_center = (self.size.width / 2, self.size.height / 2)
        world_x = center_x + (x - widget_center[0]) * self.resolution
        world_y = center_y - (y - widget_center[1]) * self.resolution * 2  # 2x for half-block
        return world_x, world_y

    def action_zoom(self, zoom_in: bool, mouse_pos: tuple[int, int] | None = None) -> None:
        """Zoom in/out with optional cursor-based zooming."""
        old_resolution = self.resolution

        if zoom_in:
            new_resolution = self.resolution / self.ZOOM_FACTOR
        else:
            new_resolution = self.resolution * self.ZOOM_FACTOR

        # Apply constraints
        new_resolution = max(self.MIN_RESOLUTION, min(self.MAX_RESOLUTION, new_resolution))

        if mouse_pos and new_resolution != old_resolution:
            # Zoom towards cursor position
            widget_center = (self.size.width / 2, self.size.height / 2)

            # Convert mouse position to world coordinates
            world_x = self.center[0] + (mouse_pos[0] - widget_center[0]) * old_resolution
            world_y = self.center[1] - (mouse_pos[1] - widget_center[1]) * old_resolution

            # Calculate new center to keep the point under cursor stable
            new_center_x = world_x - (mouse_pos[0] - widget_center[0]) * new_resolution
            new_center_y = world_y + (mouse_pos[1] - widget_center[1]) * new_resolution

            self.center = (new_center_x, new_center_y)

        self.resolution = new_resolution

    def action_zoom_in(self) -> None:
        """Zoom in from center (keyboard shortcut)."""
        self.action_zoom(zoom_in=True)

    def action_zoom_out(self) -> None:
        """Zoom out from center (keyboard shortcut)."""
        self.action_zoom(zoom_in=False)

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        """Zoom out with cursor-based zooming."""
        mouse_pos = (event.x, event.y)
        self.action_zoom(zoom_in=False, mouse_pos=mouse_pos)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        """Zoom in with cursor-based zooming."""
        mouse_pos = (event.x, event.y)
        self.action_zoom(zoom_in=True, mouse_pos=mouse_pos)

    # Utility methods for common patterns
    def _get_common_status_prefix(self) -> str:
        """Get common status prefix with timestamp."""
        if not self.data:
            return ""
        return nanoseconds_to_iso(self.data.timestamp_ns)

    def _get_view_bounds_info(self) -> str:
        """Get view bounds information for status display."""
        range_x = self.size.width * self.resolution
        range_y = (self.size.height * 2) * self.resolution  # 2x for half-block rendering
        bounds_x = (self.center[0] - range_x / 2, self.center[0] + range_x / 2)
        bounds_y = (self.center[1] - range_y / 2, self.center[1] + range_y / 2)

        return (
            f"X: {bounds_x[0]:.2f} to {bounds_x[1]:.2f} | "
            f"Y: {bounds_y[0]:.2f} to {bounds_y[1]:.2f} | "
            f"Res: {self.resolution:.2f}m/cell"
        )

    def _calculate_fit_resolution(
        self, data_bounds: tuple[tuple[float, float], tuple[float, float]], padding: float = 1.1
    ) -> float:
        """Calculate optimal resolution to fit data bounds in view.

        Args:
            data_bounds: ((min_x, max_x), (min_y, max_y))
            padding: Padding factor (1.1 = 10% padding)

        Returns:
            Optimal resolution clamped to min/max constraints
        """
        (min_x, max_x), (min_y, max_y) = data_bounds
        ranges = (max_x - min_x, max_y - min_y)

        if ranges[0] == 0 and ranges[1] == 0:
            return self.MIN_RESOLUTION

        # Check if widget is properly sized (mounted)
        if self.size.width <= 0 or self.size.height <= 0:
            return self.MIN_RESOLUTION

        req_res = max(ranges[0] / self.size.width, ranges[1] / (self.size.height * 2)) * padding
        return float(np.clip(req_res, self.MIN_RESOLUTION, self.MAX_RESOLUTION))

    def _center_on_bounds(
        self, data_bounds: tuple[tuple[float, float], tuple[float, float]]
    ) -> None:
        """Center view on data bounds.

        Args:
            data_bounds: ((min_x, max_x), (min_y, max_y))
        """
        (min_x, max_x), (min_y, max_y) = data_bounds
        self.center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
