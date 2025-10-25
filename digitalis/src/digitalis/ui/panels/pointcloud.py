import logging
from typing import ClassVar

import numpy as np
import pointcloud2
from textual import work
from textual.binding import Binding, BindingType
from textual.widgets import Static

from digitalis.reader.types import MessageEvent
from digitalis.ui.panels.interactive_base import InteractiveRenderPanel
from digitalis.ui.panels.pointcloud_renderer import render_pointcloud
from digitalis.utilities import RichRender


class PointCloud(InteractiveRenderPanel):
    SUPPORTED_SCHEMAS: ClassVar[set[str]] = {
        "sensor_msgs/msg/PointCloud2",  # ROS2
        "sensor_msgs/PointCloud2",  # ROS1
    }

    # Panel-specific configuration
    MIN_RESOLUTION = 0.001
    MAX_RESOLUTION = 10.0
    FIT_ACTION_TEXT = "Fit Points"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("+", "zoom_in", "Zoom In"),
        Binding("-", "zoom_out", "Zoom Out"),
        Binding("a", "fit_view", "Fit Points"),
    ]

    def __init__(self) -> None:
        super().__init__(default_resolution=0.1)
        self.points3d: np.ndarray | None = None

    def watch_data(self, data: MessageEvent | None) -> None:
        """Update the point cloud data."""
        super().watch_data(data)  # Reset first_data flag on topic change
        if not data:
            return

        cls = pointcloud2.read_points(data.message)
        names = cls.dtype.names
        if names is None or "x" not in names or "y" not in names or "z" not in names:
            logging.error("Invalid point cloud data: missing x, y, or z coordinates")
            self.data = None
            return

        self.points3d = np.column_stack((cls["x"], cls["y"], cls["z"]))

        # Use centralized auto-fit logic
        self._handle_first_data_auto_fit()

    @work(exclusive=True)
    async def _process_background_render(self) -> None:
        """Background worker to process point cloud rendering."""
        if self.points3d is None:
            return

        size = self.size
        if size.width <= 0 or size.height <= 0:
            return

        self._rendered = RichRender(
            render_pointcloud(
                points=self.points3d,
                size=(size.width, size.height),
                resolution=self.resolution,
                center_point=self.center,
            )
        )

        # Update status display
        self._update_status()

    def _update_status(self) -> None:
        """Update the status display."""
        if not self.data:
            return

        top = self.query_one("#top", Static)
        ts = self._get_common_status_prefix()
        center_str = f"Center: {self.center[0]:.2f}, {self.center[1]:.2f}"
        bounds_str = self._get_view_bounds_info()

        top.update(f"{ts} | {center_str} | {bounds_str}")

    def action_fit_view(self) -> None:
        """Fit all points in view by calculating optimal center and resolution."""
        if self.points3d is None or self.points3d.size == 0:
            return

        # Get finite points bounds
        finite_mask = np.isfinite(self.points3d).all(axis=1)
        if not finite_mask.any():
            return

        finite_points = self.points3d[finite_mask]
        min_coords = np.min(finite_points, axis=0)
        max_coords = np.max(finite_points, axis=0)

        # Use base class utilities for fit calculation
        data_bounds = ((min_coords[0], max_coords[0]), (min_coords[1], max_coords[1]))
        self._center_on_bounds(data_bounds)
        self.resolution = self._calculate_fit_resolution(data_bounds, padding=1.1)

    def _get_no_data_message(self) -> str:
        """Custom no data message."""
        return "No point cloud data available"

    def _get_loading_message(self) -> str:
        """Custom loading message."""
        return "Processing point cloud..."

    def _has_renderable_data(self) -> bool:
        """Check if we have point cloud data worth auto-fitting to."""
        return self.points3d is not None and self.points3d.size > 0

    def _get_hover_info(
        self, world_pos: tuple[float, float], _screen_x: int, _screen_y: int
    ) -> str | None:
        """Get hover information for the given position."""
        if self.points3d is None:
            return None

        wx, wy = world_pos
        return f"World: ({wx:.2f}, {wy:.2f})"
