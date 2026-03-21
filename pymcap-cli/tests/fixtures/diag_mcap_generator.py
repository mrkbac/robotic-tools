"""Generate test MCAP files with diagnostic messages."""

import io
from pathlib import Path

from mcap_ros2_support_fast import ROS2EncoderFactory
from small_mcap import CompressionType, McapWriter

# ROS2 message schema for diagnostic_msgs/msg/DiagnosticArray
DIAGNOSTIC_ARRAY_SCHEMA = """std_msgs/Header header
diagnostic_msgs/DiagnosticStatus[] status

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec

================================================================================
MSG: diagnostic_msgs/DiagnosticStatus
byte level
string name
string message
string hardware_id
diagnostic_msgs/KeyValue[] values

================================================================================
MSG: diagnostic_msgs/KeyValue
string key
string value"""


def create_diagnostics_mcap(
    compression: CompressionType = CompressionType.ZSTD,
) -> bytes:
    """Create an MCAP file with diagnostic messages.

    Contains a mix of OK, WARN, ERROR components with level transitions.
    """
    output = io.BytesIO()
    writer = McapWriter(
        output,
        chunk_size=1024 * 1024,
        compression=compression,
        encoder_factory=ROS2EncoderFactory(),
    )
    writer.start()

    schema_id = 1
    channel_id = 1
    writer.add_schema(
        schema_id,
        "diagnostic_msgs/msg/DiagnosticArray",
        "ros2msg",
        DIAGNOSTIC_ARRAY_SCHEMA.encode(),
    )
    writer.add_channel(channel_id, "/diagnostics", "cdr", schema_id)

    base_time_ns = 1_700_000_000_000_000_000  # ~2023-11-14

    # Message 1: radar OK, encoder OK
    _add_diag(
        writer,
        channel_id,
        base_time_ns,
        [
            _status(
                0,
                "radar_front: receiver",
                "",
                "radar_front",
                [
                    ("Timeout", "False"),
                    ("HW temperature", "40"),
                ],
            ),
            _status(
                0,
                "encoder_top: Encoder Status",
                "",
                "encoder_top",
                [
                    ("encoder_reset_count", "0"),
                ],
            ),
            _status(0, "camera_front: image_raw", "OK", "cam_front", []),
        ],
    )

    # Message 2: encoder goes to WARN
    _add_diag(
        writer,
        channel_id,
        base_time_ns + 5_000_000_000,
        [
            _status(
                0,
                "radar_front: receiver",
                "",
                "radar_front",
                [
                    ("Timeout", "False"),
                    ("HW temperature", "41"),
                ],
            ),
            _status(
                1,
                "encoder_top: Encoder Status",
                "Large position diff",
                "encoder_top",
                [
                    ("encoder_reset_count", "1"),
                    ("encoder_has_large_pos_diff", "True"),
                ],
            ),
            _status(0, "camera_front: image_raw", "OK", "cam_front", []),
        ],
    )

    # Message 3: encoder goes to ERROR
    _add_diag(
        writer,
        channel_id,
        base_time_ns + 10_000_000_000,
        [
            _status(
                0,
                "radar_front: receiver",
                "",
                "radar_front",
                [
                    ("Timeout", "False"),
                    ("HW temperature", "42"),
                ],
            ),
            _status(
                2,
                "encoder_top: Encoder Status",
                "Encoder have errors!",
                "encoder_top",
                [
                    ("encoder_reset_count", "2"),
                    ("encoder_has_large_pos_diff", "True"),
                    ("encoder_has_large_vel_diff", "True"),
                ],
            ),
            _status(0, "camera_front: image_raw", "OK", "cam_front", []),
        ],
    )

    # Message 4: encoder back to OK, camera goes ERROR
    _add_diag(
        writer,
        channel_id,
        base_time_ns + 15_000_000_000,
        [
            _status(
                0,
                "radar_front: receiver",
                "",
                "radar_front",
                [
                    ("Timeout", "False"),
                    ("HW temperature", "43"),
                ],
            ),
            _status(
                0,
                "encoder_top: Encoder Status",
                "OK",
                "encoder_top",
                [
                    ("encoder_reset_count", "3"),
                ],
            ),
            _status(2, "camera_front: image_raw", "Frame drop", "cam_front", []),
        ],
    )

    # Message 5: everything OK again
    _add_diag(
        writer,
        channel_id,
        base_time_ns + 20_000_000_000,
        [
            _status(
                0,
                "radar_front: receiver",
                "",
                "radar_front",
                [
                    ("Timeout", "False"),
                    ("HW temperature", "40"),
                ],
            ),
            _status(
                0,
                "encoder_top: Encoder Status",
                "OK",
                "encoder_top",
                [
                    ("encoder_reset_count", "3"),
                ],
            ),
            _status(0, "camera_front: image_raw", "OK", "cam_front", []),
        ],
    )

    writer.finish()
    return output.getvalue()


def _status(
    level: int,
    name: str,
    message: str,
    hardware_id: str,
    values: list[tuple[str, str]],
) -> dict:
    return {
        "level": level,
        "name": name,
        "message": message,
        "hardware_id": hardware_id,
        "values": [{"key": k, "value": v} for k, v in values],
    }


def _add_diag(writer: McapWriter, channel_id: int, log_time: int, statuses: list[dict]) -> None:
    sec = log_time // 1_000_000_000
    nanosec = log_time % 1_000_000_000
    msg = {
        "header": {
            "stamp": {"sec": sec, "nanosec": nanosec},
            "frame_id": "",
        },
        "status": statuses,
    }
    writer.add_message_encode(
        channel_id=channel_id,
        log_time=log_time,
        data=msg,
        publish_time=log_time,
    )


def ensure_diag_fixtures(base_dir: Path) -> dict[str, Path]:
    """Generate diagnostic MCAP fixtures, returning paths."""
    fixtures: dict[str, Path] = {}

    path = base_dir / "diagnostics.mcap"
    if not path.exists():
        path.write_bytes(create_diagnostics_mcap())
    fixtures["diagnostics"] = path

    return fixtures
