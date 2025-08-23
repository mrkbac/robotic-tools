import logging
import math
from typing import Any, ClassVar, Protocol

import numpy as np
from textual import work
from textual.binding import Binding, BindingType
from textual.widgets import Static

from digitalis.reader.types import MessageEvent
from digitalis.ui.panels.interactive_base import InteractiveRenderPanel
from digitalis.ui.panels.pointcloud_renderer import render_pointcloud
from digitalis.utilities import RichRender


class NavSatStatus(Protocol):
    status: int
    service: int


class NavSatFixMessage(Protocol):
    header: Any
    status: NavSatStatus
    latitude: float
    longitude: float
    altitude: float
    position_covariance: list[float]


def _convert_gps_to_local(
    latitude: float, longitude: float, origin_lat: float, origin_lon: float
) -> tuple[float, float]:
    """Convert GPS coordinates to local X/Y coordinates using simple approximation.

    Args:
        latitude: GPS latitude in degrees
        longitude: GPS longitude in degrees
        origin_lat: Origin latitude for local coordinate system
        origin_lon: Origin longitude for local coordinate system

    Returns:
        (x, y) coordinates in meters relative to origin
    """
    # Simple approximation for small areas
    # Convert degrees to meters using approximate conversion factors
    lat_to_meters = 111320.0  # meters per degree latitude
    lon_to_meters = 111320.0 * math.cos(math.radians(origin_lat))  # adjust for latitude

    x = (longitude - origin_lon) * lon_to_meters
    y = (latitude - origin_lat) * lat_to_meters

    return x, y


def _status_to_string(status: int) -> str:
    """Convert NavSat status code to human-readable string."""
    status_map = {
        -2: "Unknown",
        -1: "No Fix",
        0: "Fix",
        1: "SBAS Fix",
        2: "GBAS Fix",
    }
    return status_map.get(status, f"Status {status}")


def _service_to_string(service: int) -> str:
    """Convert NavSat service bitfield to human-readable string."""
    services = []
    if service & 1:  # SERVICE_GPS
        services.append("GPS")
    if service & 2:  # SERVICE_GLONASS
        services.append("GLONASS")
    if service & 4:  # SERVICE_COMPASS (BeiDou)
        services.append("BeiDou")

    if not services:
        return "None"
    return "+".join(services)


class NavSatFix(InteractiveRenderPanel):
    SUPPORTED_SCHEMAS: ClassVar[set[str]] = {
        "sensor_msgs/msg/NavSatFix",  # ROS2
        "sensor_msgs/NavSatFix",  # ROS1
    }

    # Panel-specific configuration
    MIN_RESOLUTION = 0.1  # 10cm
    MAX_RESOLUTION = 1000.0  # 1km
    FIT_ACTION_TEXT = "Fit Trajectory"

    # Track limits for trajectory buffer
    MAX_POINTS = 1000

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("+", "zoom_in", "Zoom In"),
        Binding("-", "zoom_out", "Zoom Out"),
        Binding("a", "fit_view", "Fit Trajectory"),
        Binding("c", "clear_trajectory", "Clear Trajectory"),
    ]

    def __init__(self) -> None:
        super().__init__(default_resolution=10.0)
        self.gps_points: list[tuple[float, float, float]] = []  # (lat, lon, timestamp)
        self.origin_lat: float | None = None
        self.origin_lon: float | None = None

    def watch_data(self, data: MessageEvent | None) -> None:
        """Update the GPS data and add to trajectory."""
        if not data:
            return

        message = data.message
        if not hasattr(message, "latitude") or not hasattr(message, "longitude"):
            logging.error("Invalid NavSatFix data: missing latitude or longitude")
            return

        # Check for valid coordinates
        lat = message.latitude
        lon = message.longitude

        if not (np.isfinite(lat) and np.isfinite(lon)):
            logging.warning("Invalid GPS coordinates: lat=%s, lon=%s", lat, lon)
            return

        # Only add points with valid fix
        if hasattr(message, "status") and message.status.status < 0:
            logging.debug("Skipping GPS point with no fix (status=%d)", message.status.status)
            return

        # Set origin on first valid point
        if self.origin_lat is None or self.origin_lon is None:
            self.origin_lat = lat
            self.origin_lon = lon

        # Add point to trajectory
        timestamp = data.timestamp_ns / 1e9  # Convert to seconds
        self.gps_points.append((lat, lon, timestamp))

        # Limit buffer size
        if len(self.gps_points) > self.MAX_POINTS:
            self.gps_points = self.gps_points[-self.MAX_POINTS :]

        # Use centralized auto-fit logic
        self._handle_first_data_auto_fit()

    @work(exclusive=True)
    async def _process_background_render(self) -> None:
        """Background worker to render GPS trajectory."""
        if not self.gps_points or self.origin_lat is None or self.origin_lon is None:
            return

        size = self.size
        if size.width <= 0 or size.height <= 0:
            return

        # Convert GPS points to local coordinates
        local_points = []
        for lat, lon, timestamp in self.gps_points:
            x, y = _convert_gps_to_local(lat, lon, self.origin_lat, self.origin_lon)
            # Use timestamp as Z value for color coding
            local_points.append([x, y, timestamp])

        points_array = np.array(local_points, dtype=np.float64)

        self._rendered = RichRender(
            render_pointcloud(
                points=points_array,
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

        message = self.data.message
        status_str = "No Status"
        service_str = "No Service"
        current_pos = "No Position"

        if hasattr(message, "status"):
            status_str = _status_to_string(message.status.status)
            service_str = _service_to_string(message.status.service)

        if hasattr(message, "latitude") and hasattr(message, "longitude"):
            lat = message.latitude
            lon = message.longitude
            if np.isfinite(lat) and np.isfinite(lon):
                current_pos = f"{lat:.6f}°, {lon:.6f}°"

        status_line = (
            f"{ts} | {status_str} ({service_str}) | "
            f"Pos: {current_pos} | "
            f"Points: {len(self.gps_points)} | "
            f"Res: {self.resolution:.1f}m/cell"
        )

        top.update(status_line)

    def action_fit_view(self) -> None:
        """Fit all GPS points in view."""
        if not self.gps_points or self.origin_lat is None or self.origin_lon is None:
            return

        # Convert all points to local coordinates
        local_coords = []
        for lat, lon, _ in self.gps_points:
            x, y = _convert_gps_to_local(lat, lon, self.origin_lat, self.origin_lon)
            local_coords.append((x, y))

        if not local_coords:
            return

        coords_array = np.array(local_coords)
        min_coords = np.min(coords_array, axis=0)
        max_coords = np.max(coords_array, axis=0)

        # Use base class utilities for fit calculation
        data_bounds = ((min_coords[0], max_coords[0]), (min_coords[1], max_coords[1]))
        self._center_on_bounds(data_bounds)
        self.resolution = self._calculate_fit_resolution(data_bounds, padding=1.2)

    def action_clear_trajectory(self) -> None:
        """Clear the GPS trajectory."""
        self.gps_points.clear()
        self.origin_lat = None
        self.origin_lon = None
        self._first_data = True
        self._rendered = None

    def _get_no_data_message(self) -> str:
        """Custom no data message."""
        return "No GPS data available"

    def _get_loading_message(self) -> str:
        """Custom loading message."""
        if not self.gps_points:
            return "No valid GPS points received"
        return "Processing GPS trajectory..."

    def _has_renderable_data(self) -> bool:
        """Check if we have GPS data worth auto-fitting to."""
        return len(self.gps_points) > 0

    def _get_hover_info(
        self, world_pos: tuple[float, float], _screen_x: int, _screen_y: int
    ) -> str | None:
        """Get hover information for the given position."""
        if not self.gps_points or self.origin_lat is None or self.origin_lon is None:
            return None

        wx, wy = world_pos
        # Convert back to GPS coordinates
        lat = self.origin_lat + wy / 111320.0
        lon = self.origin_lon + wx / (111320.0 * math.cos(math.radians(self.origin_lat)))

        return f"GPS: ({lat:.6f}°, {lon:.6f}°) | Local: ({wx:.1f}, {wy:.1f})m"
