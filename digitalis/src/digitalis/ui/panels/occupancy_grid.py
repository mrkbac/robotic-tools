from typing import Any, ClassVar, Protocol

import numpy as np
from textual import work
from textual.binding import Binding, BindingType
from textual.widgets import Static

from digitalis.reader.types import MessageEvent
from digitalis.ui.panels.interactive_base import InteractiveRenderPanel
from digitalis.ui.panels.pointcloud_renderer import render_pointcloud
from digitalis.utilities import RichRender


class MapInfo(Protocol):
    resolution: float
    """The map resolution [m/cell]"""

    width: int
    """map width [cells]"""

    height: int
    """map height [cells]"""


class OccupancyGridMessage(Protocol):
    header: Any
    info: MapInfo
    data: list[int]  # Grid data in row-major order


class OccupancyGrid(InteractiveRenderPanel):
    SUPPORTED_SCHEMAS: ClassVar[set[str]] = {
        "nav_msgs/msg/OccupancyGrid",  # ROS2
        "nav_msgs/OccupancyGrid",  # ROS1
    }

    # Panel-specific configuration - suitable for typical occupancy grids
    MIN_RESOLUTION = 0.01  # 1cm per cell
    MAX_RESOLUTION = 10.0  # 10m per cell
    FIT_ACTION_TEXT = "Fit Grid"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("+", "zoom_in", "Zoom In"),
        Binding("-", "zoom_out", "Zoom Out"),
        Binding("a", "fit_view", "Fit Grid"),
    ]

    def __init__(self) -> None:
        super().__init__(default_resolution=0.1)  # 10cm default resolution
        self.grid_data: np.ndarray | None = None
        self.grid_width: int = 0
        self.grid_height: int = 0
        self.grid_resolution: float = 0.0  # The actual resolution from the occupancy grid
        self.grid_origin: tuple[float, float] = (0.0, 0.0)  # Grid origin in world coordinates

    def watch_data(self, data: MessageEvent | None) -> None:
        """Update the occupancy grid data."""
        super().watch_data(data)  # Reset first_data flag on topic change
        if not data:
            return

        message = data.message
        if not hasattr(message, "data") or not hasattr(message, "info"):
            return

        info = message.info
        grid_data = message.data

        if not info or not grid_data:
            return

        # Extract grid properties
        self.grid_width = getattr(info, "width", 0)
        self.grid_height = getattr(info, "height", 0)
        self.grid_resolution = getattr(info, "resolution", 0.0)

        if (
            self.grid_width <= 0
            or self.grid_height <= 0
            or len(grid_data) != self.grid_width * self.grid_height
        ):
            return

        # Convert to numpy array and reshape
        self.grid_data = np.array(grid_data, dtype=np.int8).reshape(
            self.grid_height, self.grid_width
        )

        # Set grid origin - assuming grid center is at (0,0) in world coordinates
        # This can be enhanced later to use actual pose information from the message
        world_width = self.grid_width * self.grid_resolution
        world_height = self.grid_height * self.grid_resolution
        self.grid_origin = (-world_width / 2, -world_height / 2)

        # Use centralized auto-fit logic
        self._handle_first_data_auto_fit()

    def _grid_to_points(self) -> np.ndarray:
        """Convert occupancy grid to point cloud format.

        Returns:
            Array of shape (N, 3) with columns [world_x, world_y, occupancy_value]
        """
        if self.grid_data is None:
            return np.empty((0, 3))

        # Find occupied cells (non-zero values)
        occupied_indices = np.where(self.grid_data != 0)

        if len(occupied_indices[0]) == 0:
            return np.empty((0, 3))

        # Convert grid indices to world coordinates
        world_x = self.grid_origin[0] + occupied_indices[1] * self.grid_resolution
        world_y = self.grid_origin[1] + occupied_indices[0] * self.grid_resolution

        # Use occupancy values as Z coordinate for coloring
        occupancy_values = self.grid_data[occupied_indices].astype(float)

        return np.column_stack((world_x, world_y, occupancy_values))

    @work(exclusive=True)
    async def _process_background_render(self) -> None:
        """Background worker to render occupancy grid using pointcloud renderer."""
        if self.grid_data is None:
            return

        size = self.size
        if size.width <= 0 or size.height <= 0:
            return

        # Convert occupancy grid to points
        points = self._grid_to_points()

        # Use existing pointcloud renderer
        self._rendered = RichRender(
            render_pointcloud(
                points=points,
                size=(size.width, size.height),
                resolution=self.resolution,
                center_point=self.center,
            )
        )

        # Update status display
        self._update_status()

    def _update_status(self) -> None:
        """Update the status display with grid info and view bounds."""
        if not self.data:
            return

        top = self.query_one("#top", Static)
        ts = self._get_common_status_prefix()

        # Grid information
        grid_info = (
            f"{self.grid_width}x{self.grid_height} cells | "
            f"Grid res: {self.grid_resolution:.3f} m/cell"
        )

        # View information
        view_info = self._get_view_bounds_info()

        top.update(f"{ts} | {grid_info} | {view_info}")

    def action_fit_view(self) -> None:
        """Fit entire occupancy grid in view."""
        if self.grid_data is None:
            return

        # Calculate world bounds of entire grid
        world_width = self.grid_width * self.grid_resolution
        world_height = self.grid_height * self.grid_resolution

        # Center on grid
        grid_center_x = self.grid_origin[0] + world_width / 2
        grid_center_y = self.grid_origin[1] + world_height / 2
        self.center = (grid_center_x, grid_center_y)

        # Calculate resolution to fit entire grid with padding
        data_bounds = (
            (self.grid_origin[0], self.grid_origin[0] + world_width),
            (self.grid_origin[1], self.grid_origin[1] + world_height),
        )
        self.resolution = self._calculate_fit_resolution(data_bounds, padding=1.1)

    def _get_no_data_message(self) -> str:
        """Custom no data message."""
        return "No occupancy grid data available"

    def _get_loading_message(self) -> str:
        """Custom loading message."""
        return "Processing occupancy grid..."

    def _has_renderable_data(self) -> bool:
        """Check if we have occupancy grid data worth auto-fitting to."""
        return self.grid_data is not None

    def _get_hover_info(
        self, world_pos: tuple[float, float], _screen_x: int, _screen_y: int
    ) -> str | None:
        """Get hover information for the given position."""
        if self.grid_data is None:
            return None

        wx, wy = world_pos
        # Convert to grid coordinates
        grid_x = int((wx - self.grid_origin[0]) / self.grid_resolution)
        grid_y = int((wy - self.grid_origin[1]) / self.grid_resolution)

        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            occupancy = self.grid_data[grid_y, grid_x]
            return (
                f"Grid: ({grid_x}, {grid_y}) | Occupancy: {occupancy} | "
                f"World: ({wx:.2f}, {wy:.2f})m"
            )
        return f"World: ({wx:.2f}, {wy:.2f})m | Outside grid bounds"
