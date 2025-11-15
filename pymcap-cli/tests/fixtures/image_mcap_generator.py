"""Generate test MCAP files with image messages for video encoding tests."""

import io
from pathlib import Path

from mcap_ros2_support_fast import ROS2EncoderFactory
from PIL import Image
from small_mcap import CompressionType, McapWriter

# ROS2 message schema for sensor_msgs/msg/Image
SENSOR_MSGS_IMAGE_SCHEMA = """std_msgs/Header header
uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step
uint8[] data

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""


# ROS2 message schema for sensor_msgs/msg/CompressedImage
SENSOR_MSGS_COMPRESSED_IMAGE_SCHEMA = """std_msgs/Header header
string format
uint8[] data

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""


def create_simple_rgb_frame(width: int, height: int, frame_idx: int) -> bytes:
    """Create a simple RGB test pattern.

    Args:
        width: Image width
        height: Image height
        frame_idx: Frame index (affects color)

    Returns:
        RGB24 image data
    """
    data = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            idx = (y * width + x) * 3
            # Create a gradient pattern
            data[idx] = (x * 255 // width + frame_idx * 10) % 256  # R
            data[idx + 1] = (y * 255 // height) % 256  # G
            data[idx + 2] = (frame_idx * 20) % 256  # B
    return bytes(data)


def create_jpeg_frame(width: int, height: int, frame_idx: int) -> bytes:
    """Create a simple JPEG test frame.

    Args:
        width: Image width
        height: Image height
        frame_idx: Frame index

    Returns:
        JPEG image data
    """
    # For testing, create a minimal valid JPEG
    # This is a placeholder - in real tests we'd use PIL or similar
    # For now, create a simple solid color JPEG
    try:
        rgb_data = create_simple_rgb_frame(width, height, frame_idx)
        img = Image.frombytes("RGB", (width, height), rgb_data)
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=85)
        return output.getvalue()
    except ImportError:
        # Fallback: return minimal JPEG header (won't be valid but useful for structure tests)
        return b"\xff\xd8\xff\xe0" + b"\x00" * 10  # JPEG SOI + JFIF marker


def create_image_mcap(
    num_frames: int = 30,
    width: int = 320,
    height: int = 240,
    message_type: str = "Image",
    encoding: str = "rgb8",
    compression: CompressionType = CompressionType.NONE,
) -> bytes:
    """Create an MCAP file with image messages.

    Args:
        num_frames: Number of frames to generate
        width: Image width
        height: Image height
        message_type: "Image" or "CompressedImage"
        encoding: Image encoding for raw images
        compression: MCAP compression type

    Returns:
        MCAP file bytes
    """
    output = io.BytesIO()
    writer = McapWriter(
        output,
        chunk_size=1024 * 1024,
        compression=compression,
        encoder_factory=ROS2EncoderFactory(),
    )
    writer.start()

    if message_type == "Image":
        schema_name = "sensor_msgs/msg/Image"
        schema_data = SENSOR_MSGS_IMAGE_SCHEMA.encode()
        topic = "/camera/image_raw"
    else:  # CompressedImage
        schema_name = "sensor_msgs/msg/CompressedImage"
        schema_data = SENSOR_MSGS_COMPRESSED_IMAGE_SCHEMA.encode()
        topic = "/camera/image_compressed"

    # Register schema and channel
    schema_id = 1
    channel_id = 1
    writer.add_schema(schema_id, schema_name, "ros2msg", schema_data)
    writer.add_channel(channel_id, topic, "cdr", schema_id)

    # Generate frames at 30 FPS
    fps = 30
    time_step_ns = int(1e9 / fps)

    for i in range(num_frames):
        log_time = i * time_step_ns
        sec = log_time // int(1e9)
        nanosec = log_time % int(1e9)

        if message_type == "Image":
            # Generate raw RGB frame
            rgb_data = create_simple_rgb_frame(width, height, i)
            step = width * 3 if encoding == "rgb8" else width
            message_obj = {
                "header": {
                    "stamp": {"sec": sec, "nanosec": nanosec},
                    "frame_id": "camera",
                },
                "height": height,
                "width": width,
                "encoding": encoding,
                "is_bigendian": False,
                "step": step,
                "data": rgb_data,
            }
        else:  # CompressedImage
            # Generate JPEG frame
            jpeg_data = create_jpeg_frame(width, height, i)
            message_obj = {
                "header": {
                    "stamp": {"sec": sec, "nanosec": nanosec},
                    "frame_id": "camera",
                },
                "format": "jpeg",
                "data": jpeg_data,
            }

        writer.add_message_encode(
            channel_id=channel_id,
            log_time=log_time,
            data=message_obj,
            publish_time=log_time,
            sequence=i,
        )

    writer.finish()
    return output.getvalue()


def save_image_fixture(name: str, data: bytes, fixtures_dir: Path | None = None) -> Path:
    """Save MCAP image data to fixtures directory."""
    if fixtures_dir is None:
        fixtures_dir = Path(__file__).parent
    filepath = fixtures_dir / f"{name}.mcap"
    filepath.write_bytes(data)
    return filepath


def ensure_image_fixtures(fixtures_dir: Path | None = None) -> dict[str, Path]:
    """Ensure all image test fixtures exist and return their paths."""
    if fixtures_dir is None:
        fixtures_dir = Path(__file__).parent

    fixtures = {}

    # RGB Image (raw)
    fixtures["image_rgb"] = save_image_fixture(
        "image_rgb",
        create_image_mcap(num_frames=30, message_type="Image", encoding="rgb8"),
        fixtures_dir,
    )

    # CompressedImage (JPEG)
    fixtures["image_compressed"] = save_image_fixture(
        "image_compressed",
        create_image_mcap(num_frames=30, message_type="CompressedImage"),
        fixtures_dir,
    )

    # Small test (just a few frames for quick tests)
    fixtures["image_small"] = save_image_fixture(
        "image_small",
        create_image_mcap(num_frames=5, width=160, height=120, message_type="CompressedImage"),
        fixtures_dir,
    )

    return fixtures


if __name__ == "__main__":
    # Generate all fixtures when run directly
    fixtures_dir = Path(__file__).parent
    fixtures = ensure_image_fixtures(fixtures_dir)
    print("Generated image test fixtures:")
    for name, path in fixtures.items():
        size_kb = path.stat().st_size / 1024
        print(f"  {name:20s} -> {path.name:30s} ({size_kb:.1f} KB)")
