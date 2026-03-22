# ruff: noqa: PLC0415
"""Tests for the decompressors and DecompressDecoderFactory."""

import io

from mcap_ros2_support_fast import ROS2EncoderFactory
from small_mcap import McapWriter, read_message_decoded

FOXGLOVE_COMPRESSED_VIDEO = """\
builtin_interfaces/Time timestamp
string frame_id
uint8[] data
string format

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""

COMPRESSED_POINTCLOUD2 = """\
std_msgs/Header header
uint32 height
uint32 width
sensor_msgs/PointField[] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] compressed_data
bool is_dense
string format

================================================================================
MSG: sensor_msgs/PointField
uint8 INT8    = 1
uint8 UINT8   = 2
uint8 INT16   = 3
uint8 UINT16  = 4
uint8 INT32   = 5
uint8 UINT32  = 6
uint8 FLOAT32 = 7
uint8 FLOAT64 = 8
string name
uint32 offset
uint8  datatype
uint32 count

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""


def _create_compressed_video_mcap(
    frames: list[tuple[int, bytes, str]],
    *,
    topic: str = "/camera/compressed_video",
) -> io.BytesIO:
    buf = io.BytesIO()
    writer = McapWriter(buf, encoder_factory=ROS2EncoderFactory())
    writer.start()

    writer.add_schema(
        1, "foxglove_msgs/msg/CompressedVideo", "ros2msg", FOXGLOVE_COMPRESSED_VIDEO.encode()
    )
    writer.add_channel(1, topic, "cdr", 1)

    for log_time, video_data, fmt in frames:
        msg = {
            "timestamp": {"sec": log_time // 1_000_000_000, "nanosec": log_time % 1_000_000_000},
            "frame_id": "camera_link",
            "data": video_data,
            "format": fmt,
        }
        writer.add_message_encode(channel_id=1, log_time=log_time, publish_time=log_time, data=msg)

    writer.finish()
    buf.seek(0)
    return buf


def _create_compressed_pointcloud_mcap(
    clouds: list[tuple[int, dict]],
    *,
    topic: str = "/lidar/compressed",
) -> io.BytesIO:
    buf = io.BytesIO()
    writer = McapWriter(buf, encoder_factory=ROS2EncoderFactory())
    writer.start()

    writer.add_schema(
        1,
        "point_cloud_interfaces/msg/CompressedPointCloud2",
        "ros2msg",
        COMPRESSED_POINTCLOUD2.encode(),
    )
    writer.add_channel(1, topic, "cdr", 1)

    for log_time, cloud_msg in clouds:
        writer.add_message_encode(
            channel_id=1, log_time=log_time, publish_time=log_time, data=cloud_msg
        )

    writer.finish()
    buf.seek(0)
    return buf


def _make_h264_keyframe(width: int = 64, height: int = 32) -> bytes:
    from fractions import Fraction

    import av

    encoder = av.CodecContext.create("libx264", "w")
    encoder.width = width
    encoder.height = height
    encoder.pix_fmt = "yuv420p"
    encoder.time_base = Fraction(1, 30)
    encoder.gop_size = 1
    encoder.max_b_frames = 0
    encoder.options = {"preset": "ultrafast", "tune": "zerolatency"}
    encoder.open()

    frame = av.VideoFrame(width, height, "yuv420p")
    for plane in frame.planes:
        plane.update(bytes(plane.buffer_size))
    frame.pts = 0

    packets = encoder.encode(frame)
    packets += encoder.encode(None)
    return b"".join(bytes(p) for p in packets)


def _make_compressed_pointcloud(n_points: int = 10) -> tuple[bytes, bytes]:
    import numpy as np
    from pureini import (
        CompressionOption,
        EncodingInfo,
        EncodingOptions,
        FieldType,
        PointcloudEncoder,
        PointField,
    )

    point_step = 12
    raw_data = np.random.default_rng(42).random((n_points, 3), dtype=np.float32)
    raw_bytes = raw_data.tobytes()

    info = EncodingInfo(
        fields=[
            PointField(name="x", offset=0, type=FieldType.FLOAT32),
            PointField(name="y", offset=4, type=FieldType.FLOAT32),
            PointField(name="z", offset=8, type=FieldType.FLOAT32),
        ],
        width=n_points,
        height=1,
        point_step=point_step,
        encoding_opt=EncodingOptions.NONE,
        compression_opt=CompressionOption.ZSTD,
    )

    encoder = PointcloudEncoder(info)
    compressed = encoder.encode(raw_bytes)
    return compressed, raw_bytes


# ---------------------------------------------------------------------------
# Video decompressor tests
# ---------------------------------------------------------------------------


class TestVideoDecompressor:
    def test_decompress_h264_to_jpeg(self):
        from pymcap_cli.encoding.video_pyav import PyAVVideoDecompressor

        dec = PyAVVideoDecompressor("compressed", 80)
        h264_data = _make_h264_keyframe()

        result = dec.decompress(h264_data, "h264")
        assert result is not None
        assert result.is_jpeg is True
        assert result.data[:2] == b"\xff\xd8"

    def test_decompress_h264_to_raw(self):
        from pymcap_cli.encoding.video_pyav import PyAVVideoDecompressor

        dec = PyAVVideoDecompressor("raw", 90)
        h264_data = _make_h264_keyframe(width=64, height=32)

        result = dec.decompress(h264_data, "h264")
        assert result is not None
        assert result.is_jpeg is False
        assert result.height == 32
        assert result.width == 64
        assert len(result.data) == 64 * 32 * 3


class TestVideoDecompressFactoryTimestamp:
    """Test that VideoDecompressFactory preserves timestamps in the output message."""

    def test_timestamp_preserved_in_compressed_output(self):
        from pymcap_cli.encoding.decompress import VideoDecompressFactory
        from pymcap_cli.encoding.encoder_common import EncoderMode

        h264_data = _make_h264_keyframe()
        log_time = 999 * 1_000_000_000 + 12345

        buf = _create_compressed_video_mcap([(log_time, h264_data, "h264")])
        factory = VideoDecompressFactory(
            video_format="compressed", jpeg_quality=90, backend=EncoderMode.PYAV
        )

        for msg in read_message_decoded(buf, decoder_factories=[factory]):
            assert msg.decoded_message is not None
            header = msg.decoded_message["header"]
            assert header["stamp"]["sec"] == 999
            assert header["stamp"]["nanosec"] == 12345
            assert header["frame_id"] == "camera_link"
            assert msg.decoded_message["format"] == "jpeg"
            assert msg.decoded_message["data"][:2] == b"\xff\xd8"
            break
        else:
            raise AssertionError("No messages decoded")


# ---------------------------------------------------------------------------
# Pointcloud decompressor tests
# ---------------------------------------------------------------------------


class TestPointCloudDecompressor:
    def test_roundtrip_decompress(self):
        from pymcap_cli.encoding.pointcloud import PointCloudDecompressFactory

        compressed, original = _make_compressed_pointcloud(n_points=100)
        factory = PointCloudDecompressFactory()

        msg = {
            "header": {"stamp": {"sec": 10, "nanosec": 0}, "frame_id": "lidar"},
            "height": 1,
            "width": 100,
            "fields": [
                {"name": "x", "offset": 0, "datatype": 7, "count": 1},
                {"name": "y", "offset": 4, "datatype": 7, "count": 1},
                {"name": "z", "offset": 8, "datatype": 7, "count": 1},
            ],
            "is_bigendian": False,
            "point_step": 12,
            "row_step": 1200,
            "compressed_data": compressed,
            "is_dense": True,
            "format": "cloudini",
        }

        result = factory._decompress(msg)
        assert result["header"] == msg["header"]
        assert result["height"] == 1
        assert result["width"] == 100
        assert result["point_step"] == 12
        assert result["is_dense"] is True
        assert result["data"] == original
        assert "compressed_data" not in result

    def test_preserves_metadata(self):
        from pymcap_cli.encoding.pointcloud import PointCloudDecompressFactory

        compressed, _ = _make_compressed_pointcloud(n_points=5)
        factory = PointCloudDecompressFactory()

        fields = [
            {"name": "x", "offset": 0, "datatype": 7, "count": 1},
            {"name": "y", "offset": 4, "datatype": 7, "count": 1},
            {"name": "z", "offset": 8, "datatype": 7, "count": 1},
        ]

        msg = {
            "header": {"stamp": {"sec": 42, "nanosec": 123}, "frame_id": "velodyne"},
            "height": 1,
            "width": 5,
            "fields": fields,
            "is_bigendian": False,
            "point_step": 12,
            "row_step": 60,
            "compressed_data": compressed,
            "is_dense": False,
            "format": "cloudini",
        }

        result = factory._decompress(msg)
        assert result["header"]["frame_id"] == "velodyne"
        assert result["fields"] == fields
        assert result["is_bigendian"] is False
        assert result["row_step"] == 60


# ---------------------------------------------------------------------------
# Factory integration tests
# ---------------------------------------------------------------------------


class TestVideoDecompressFactory:
    def test_decode_compressed_video_from_mcap(self):
        from pymcap_cli.encoding.decompress import VideoDecompressFactory

        h264_data = _make_h264_keyframe()
        buf = _create_compressed_video_mcap([(1_000_000_000, h264_data, "h264")])

        factory = VideoDecompressFactory()
        results = list(read_message_decoded(buf, decoder_factories=[factory]))

        assert len(results) == 1
        decoded = results[0].decoded_message
        assert decoded is not None
        assert decoded["format"] == "jpeg"
        assert decoded["data"][:2] == b"\xff\xd8"

    def test_multi_topic_gets_separate_decoders(self):
        from pymcap_cli.encoding.decompress import VideoDecompressFactory

        h264_data = _make_h264_keyframe()

        buf = io.BytesIO()
        writer = McapWriter(buf, encoder_factory=ROS2EncoderFactory())
        writer.start()

        writer.add_schema(
            1, "foxglove_msgs/msg/CompressedVideo", "ros2msg", FOXGLOVE_COMPRESSED_VIDEO.encode()
        )
        writer.add_channel(1, "/cam_left", "cdr", 1)
        writer.add_channel(2, "/cam_right", "cdr", 1)

        msg = {
            "timestamp": {"sec": 1, "nanosec": 0},
            "frame_id": "cam",
            "data": h264_data,
            "format": "h264",
        }
        writer.add_message_encode(channel_id=1, log_time=1, publish_time=1, data=msg)
        writer.add_message_encode(channel_id=2, log_time=2, publish_time=2, data=msg)

        writer.finish()
        buf.seek(0)

        factory = VideoDecompressFactory()
        results = list(read_message_decoded(buf, decoder_factories=[factory]))

        assert len(results) == 2
        for r in results:
            decoded = r.decoded_message
            assert decoded is not None
            assert decoded["format"] == "jpeg"

        # Factory should have created 2 separate decompressors
        assert len(factory._decompressors) == 2


class TestVideoDecompressFactoryFlush:
    def test_flush_all_empty(self):
        from pymcap_cli.encoding.decompress import VideoDecompressFactory

        factory = VideoDecompressFactory()
        frames = factory.flush_all()
        assert frames == []

    def test_flush_all_after_decoding(self):
        from pymcap_cli.encoding.decompress import VideoDecompressFactory

        # Feed 1 keyframe through the factory and verify flush works
        h264_data = _make_h264_keyframe()
        buf = _create_compressed_video_mcap([(1_000_000_000, h264_data, "h264")])

        factory = VideoDecompressFactory()
        decoded = [
            msg
            for msg in read_message_decoded(buf, decoder_factories=[factory])
            if msg.decoded_message is not None
        ]
        flushed = factory.flush_all()

        # The single keyframe should decode immediately (no buffering)
        assert len(decoded) == 1
        assert decoded[0].decoded_message["format"] == "jpeg"
        assert flushed == []
        # Verify decompressor was created for the channel
        assert len(factory._decompressors) == 1


class TestPointCloudDecompressFactory:
    def test_decode_compressed_pointcloud_from_mcap(self):
        from pymcap_cli.encoding.pointcloud import PointCloudDecompressFactory

        compressed, original = _make_compressed_pointcloud(n_points=50)

        cloud_msg = {
            "header": {"stamp": {"sec": 1, "nanosec": 0}, "frame_id": "lidar"},
            "height": 1,
            "width": 50,
            "fields": [
                {"name": "x", "offset": 0, "datatype": 7, "count": 1},
                {"name": "y", "offset": 4, "datatype": 7, "count": 1},
                {"name": "z", "offset": 8, "datatype": 7, "count": 1},
            ],
            "is_bigendian": False,
            "point_step": 12,
            "row_step": 600,
            "compressed_data": compressed,
            "is_dense": True,
            "format": "cloudini",
        }

        buf = _create_compressed_pointcloud_mcap([(1_000_000_000, cloud_msg)])

        factory = PointCloudDecompressFactory()
        results = list(read_message_decoded(buf, decoder_factories=[factory]))

        assert len(results) == 1
        decoded = results[0].decoded_message
        assert decoded["data"] == original
        assert decoded["width"] == 50


class TestBothFactoriesTogether:
    def test_non_compressed_topics_pass_through(self):
        """With both factories, non-matching schemas should raise (no CDR fallback)."""
        from mcap_ros2_support_fast.decoder import DecoderFactory
        from pymcap_cli.encoding.decompress import VideoDecompressFactory
        from pymcap_cli.encoding.pointcloud import PointCloudDecompressFactory

        buf = io.BytesIO()
        writer = McapWriter(buf, encoder_factory=ROS2EncoderFactory())
        writer.start()

        schema_text = "int32 data"
        writer.add_schema(1, "std_msgs/msg/Int32", "ros2msg", schema_text.encode())
        writer.add_channel(1, "/counter", "cdr", 1)

        writer.add_message_encode(channel_id=1, log_time=1, publish_time=1, data={"data": 42})
        writer.finish()
        buf.seek(0)

        # The decompress factories don't handle Int32 — need a CDR factory too
        factories = [VideoDecompressFactory(), PointCloudDecompressFactory(), DecoderFactory()]
        results = list(read_message_decoded(buf, decoder_factories=factories))

        assert len(results) == 1
        assert results[0].decoded_message.data == 42
