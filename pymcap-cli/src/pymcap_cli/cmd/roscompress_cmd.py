"""Command to compress image and point cloud topics in MCAP files."""

import logging
import platform
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, cast

import av
import av.error
from av.video.frame import VideoFrame
from cyclopts import Group, Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from small_mcap import McapWriter, get_summary
from small_mcap.reader import read_message_decoded

from pymcap_cli.image_utils import (
    COMPRESSED_SCHEMAS,
    IMAGE_SCHEMAS,
    VideoEncoderError,
    decode_compressed_frame,
    raw_image_to_array,
    test_encoder,
)
from pymcap_cli.input_handler import open_input
from pymcap_cli.osc_utils import OSCProgressColumn
from pymcap_cli.types_manual import (
    ForceOverwriteOption,
    OutputPathOption,
)
from pymcap_cli.utils import confirm_output_overwrite

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from av.video.codeccontext import VideoCodecContext
    from small_mcap.reader import DecodedMessage


FOXGLOVE_COMPRESSED_VIDEO = """builtin_interfaces/Time timestamp
string frame_id
uint8[] data
string format

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""

POINTCLOUD2_SCHEMAS = {"sensor_msgs/msg/PointCloud2", "sensor_msgs/PointCloud2"}

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

console = Console()
logger = logging.getLogger(__name__)

# Parameter groups
ENCODING_GROUP = Group("Encoding")
POINTCLOUD_GROUP = Group("Point Cloud")


def _detect_encoder() -> str:
    """Detect the best available H.264 encoder for this system."""
    # Try hardware encoders first
    system = platform.system()
    if system == "Darwin":
        if test_encoder("h264_videotoolbox"):
            return "h264_videotoolbox"
    elif system == "Linux":
        for encoder in ["h264_nvenc", "h264_vaapi"]:
            if test_encoder(encoder):
                return encoder

    # Fallback to software encoder
    return "libx264"


def _decode_compressed_image(compressed_data: bytes) -> VideoFrame:
    """Decode a compressed image (JPEG/PNG) to a VideoFrame in native format."""
    return decode_compressed_frame(compressed_data)


def _prefetch_image_decodes(
    messages: "Iterable[DecodedMessage]",
    pool: ThreadPoolExecutor,
    prefetch: int = 8,
) -> "Iterator[tuple[DecodedMessage, Future[VideoFrame] | None]]":
    """Wrap message iterator to decode JPEGs in background threads.

    Buffers up to `prefetch` messages ahead, submitting JPEG decode jobs
    to the thread pool eagerly. By the time we process a message, its
    decode is likely already complete.
    """
    buffer: deque[tuple[DecodedMessage, Future[VideoFrame] | None]] = deque()

    for msg in messages:
        schema_name = msg.schema.name if msg.schema else ""
        if schema_name in COMPRESSED_SCHEMAS:
            data = bytes(msg.decoded_message.data)
            future: Future[VideoFrame] | None = pool.submit(_decode_compressed_image, data)
        else:
            future = None
        buffer.append((msg, future))

        if len(buffer) > prefetch:
            yield buffer.popleft()

    while buffer:
        yield buffer.popleft()


@dataclass(frozen=True, slots=True)
class _EncoderConfig:
    """Configuration for a video encoder."""

    width: int
    height: int
    codec_name: str


def _build_encoder_options(
    codec_name: str, quality: int, width: int, height: int
) -> tuple[dict[str, str], int | None]:
    """Build codec-specific encoder options from user-facing quality (CRF).

    Returns (options_dict, bit_rate_or_none).
    """
    if codec_name == "libx264":
        return {
            "crf": str(quality),
            "preset": "superfast",
            "tune": "zerolatency",
        }, None
    if codec_name == "libx265":
        return {
            "crf": str(quality),
            "preset": "superfast",
        }, None
    if codec_name in {"h264_videotoolbox", "hevc_videotoolbox"}:
        # CRF → bitrate: 5 Mbps baseline scaled by quality and resolution
        pixel_scale = (width * height) / (1920 * 1080)
        bit_rate = int(5_000_000 * (2 ** ((28 - quality) / 6)) * pixel_scale)
        return {}, bit_rate
    if codec_name == "h264_nvenc":
        return {"rc": "vbr", "cq": str(quality)}, None
    if codec_name == "h264_vaapi":
        return {"qp": str(quality)}, None
    return {}, None


class _VideoEncoder:
    """PyAV-based video encoder for converting images to compressed video."""

    def __init__(
        self,
        width: int,
        height: int,
        codec_name: str,
        quality: int = 28,
        target_fps: float = 30.0,
        gop_size: int = 30,
    ) -> None:
        self.config = _EncoderConfig(width=width, height=height, codec_name=codec_name)
        self._target_fps = max(target_fps, 1.0)
        self._frame_index = 0
        self._quality = quality
        self._gop_size = gop_size

        try:
            self._context: VideoCodecContext = cast(
                "VideoCodecContext", av.CodecContext.create(codec_name, "w")
            )
        except av.error.FFmpegError as exc:
            raise VideoEncoderError(f"Failed to create encoder {codec_name}: {exc}") from exc

        # Configure encoder
        fps_int = max(round(self._target_fps), 1)
        self._context.width = width
        self._context.height = height
        self._context.pix_fmt = "yuv420p"
        self._context.time_base = Fraction(1, fps_int)
        self._context.framerate = Fraction(fps_int, 1)
        self._context.gop_size = gop_size
        self._context.max_b_frames = 0  # Ensure every frame produces immediate output

        # Set codec-specific options
        options, bit_rate = _build_encoder_options(codec_name, quality, width, height)
        if bit_rate is not None:
            self._context.bit_rate = bit_rate
        if options:
            self._context.options = options

        try:
            self._context.open()
        except av.error.FFmpegError as exc:
            raise VideoEncoderError(f"Failed to open encoder {codec_name}: {exc}") from exc

    # Pixel formats that are plane-compatible with yuv420p (only color range differs)
    # and can be passed directly to the encoder without an expensive sws_scale reformat.
    _YUV420P_COMPAT = frozenset({"yuv420p", "yuvj420p"})

    def encode(self, frame: VideoFrame) -> bytes | None:
        """Encode a single frame and return compressed video bytes, or None if buffered."""
        # Only reformat when dimensions differ or the format is truly incompatible.
        # yuvj420p (full-range) is plane-compatible with yuv420p; FFmpeg handles the
        # color-range flag internally so we can skip the costly sws_scale conversion.
        needs_resize = frame.width != self.config.width or frame.height != self.config.height
        needs_fmt = frame.format.name not in self._YUV420P_COMPAT
        if needs_resize or needs_fmt:
            frame = frame.reformat(
                width=self.config.width, height=self.config.height, format=self._context.pix_fmt
            )
        frame.pts = self._frame_index
        self._frame_index += 1

        try:
            packets = list(self._context.encode(frame))
        except av.error.FFmpegError as exc:
            raise VideoEncoderError(f"Encoding error: {exc}") from exc

        if not packets:
            return None

        return b"".join(bytes(packet) for packet in packets)

    def flush(self) -> bytes | None:
        """Flush remaining buffered frames from the encoder."""
        try:
            packets = list(self._context.encode(None))
        except av.error.FFmpegError:
            return None
        if not packets:
            return None
        return b"".join(bytes(packet) for packet in packets)

    def close(self) -> None:
        """Close the encoder (no-op, context cleaned up by garbage collector)."""


def _build_encoding_info(
    msg: object,
    encoding_opt: "EncodingOptions",  # noqa: F821  # ty: ignore[unresolved-reference]
    compression_opt: "CompressionOption",  # noqa: F821  # ty: ignore[unresolved-reference]
    resolution: float,
) -> "EncodingInfo":  # noqa: F821  # ty: ignore[unresolved-reference]
    """Build pureini EncodingInfo from a decoded ROS2 PointCloud2 message."""
    from pureini import EncodingInfo, FieldType, PointField  # noqa: PLC0415

    info = EncodingInfo()
    info.width = msg.width  # type: ignore[attr-defined]
    info.height = msg.height  # type: ignore[attr-defined]
    info.point_step = msg.point_step  # type: ignore[attr-defined]
    info.encoding_opt = encoding_opt
    info.compression_opt = compression_opt

    info.fields = []
    for ros_field in msg.fields:  # type: ignore[attr-defined]
        field = PointField(
            name=ros_field.name,
            offset=ros_field.offset,
            type=FieldType(ros_field.datatype),
            resolution=resolution if ros_field.datatype == 7 else None,
        )
        info.fields.append(field)

    return info


def roscompress(
    file: str,
    output: OutputPathOption,
    *,
    force: ForceOverwriteOption = False,
    quality: Annotated[
        int,
        Parameter(
            name=["--quality", "-q"],
            group=ENCODING_GROUP,
        ),
    ] = 28,
    codec: Annotated[
        str,
        Parameter(
            name=["--codec"],
            group=ENCODING_GROUP,
        ),
    ] = "h264",
    encoder: Annotated[
        str | None,
        Parameter(
            name=["--encoder"],
            group=ENCODING_GROUP,
        ),
    ] = None,
    resolution: Annotated[
        float,
        Parameter(
            name=["--resolution"],
            group=POINTCLOUD_GROUP,
        ),
    ] = 0.01,
    pc_encoding: Annotated[
        Literal["lossy", "lossless", "none"],
        Parameter(
            name=["--pc-encoding"],
            group=POINTCLOUD_GROUP,
        ),
    ] = "lossy",
    pc_compression: Annotated[
        Literal["zstd", "lz4", "none"],
        Parameter(
            name=["--pc-compression"],
            group=POINTCLOUD_GROUP,
        ),
    ] = "zstd",
    video: Annotated[
        bool,
        Parameter(
            name=["--video/--no-video"],
            group=ENCODING_GROUP,
        ),
    ] = True,
    pointcloud: Annotated[
        bool,
        Parameter(
            name=["--pointcloud/--no-pointcloud"],
            group=POINTCLOUD_GROUP,
        ),
    ] = True,
) -> int:
    """Compress ROS MCAP by converting image and point cloud topics.

    Converts CompressedImage/Image topics to CompressedVideo and
    PointCloud2 topics to CompressedPointCloud2 using pureini.

    Parameters
    ----------
    file
        Input MCAP file (local file or HTTP/HTTPS URL).
    output
        Output filename.
    force
        Force overwrite of output file without confirmation.
    quality
        Video quality (CRF: lower = better, 0-51). Default: 28.
    codec
        Video codec (h264, h265). Default: h264.
    encoder
        Force specific encoder (libx264, h264_videotoolbox, etc.). If None, auto-detect.
    resolution
        Resolution for lossy float compression of point clouds. Default: 0.01.
    pc_encoding
        Point cloud encoding mode (lossy, lossless, none). Default: lossy.
    pc_compression
        Point cloud second-stage compression (zstd, lz4, none). Default: zstd.
    video
        Enable video compression of image topics. Default: True.
    pointcloud
        Enable point cloud compression. Default: True.
    """
    confirm_output_overwrite(output, force)

    # Detect encoder
    encoder_name = ""
    if video:
        if encoder:
            if not test_encoder(encoder):
                console.print(f"[red]Error:[/red] Encoder '{encoder}' not available on this system")
                return 1
            encoder_name = encoder
        else:
            encoder_name = _detect_encoder()

    # Lazy-import pureini for point cloud compression
    pureini_available = True
    if pointcloud:
        try:
            from pureini import (  # noqa: PLC0415
                CompressionOption,
                EncodingOptions,
                PointcloudEncoder,
            )
        except ImportError:
            pureini_available = False

    # Map string options to pureini enums
    if pointcloud and pureini_available:
        encoding_map = {
            "lossy": EncodingOptions.LOSSY,
            "lossless": EncodingOptions.LOSSLESS,
            "none": EncodingOptions.NONE,
        }
        compression_map = {
            "zstd": CompressionOption.ZSTD,
            "lz4": CompressionOption.LZ4,
            "none": CompressionOption.NONE,
        }
        pc_encoding_opt = encoding_map[pc_encoding]
        pc_compression_opt = compression_map[pc_compression]

    console.print(f"[cyan]Input:[/cyan] {file}")
    console.print(f"[cyan]Output:[/cyan] {output}")
    if video:
        console.print(f"[cyan]Encoder:[/cyan] {encoder_name}")
        console.print(f"[cyan]Quality (CRF):[/cyan] {quality}")
    else:
        console.print("[cyan]Video compression:[/cyan] disabled")
    if pointcloud and pureini_available:
        console.print(f"[cyan]Point cloud encoding:[/cyan] {pc_encoding}")
        console.print(f"[cyan]Point cloud compression:[/cyan] {pc_compression}")
        if pc_encoding == "lossy":
            console.print(f"[cyan]Point cloud resolution:[/cyan] {resolution}")
    elif not pointcloud:
        console.print("[cyan]Point cloud compression:[/cyan] disabled")

    # Get message count from summary for progress bar
    total_message_count: int | None = None
    with open_input(file) as (f, _file_size):
        if (
            (summary := get_summary(f))
            and summary.statistics
            and summary.statistics.channel_message_counts
        ):
            total_message_count = sum(summary.statistics.channel_message_counts.values())

    # Track encoders per topic (lazy initialization)
    encoders: dict[str, _VideoEncoder] = {}
    # (EncodingInfo, PointcloudEncoder) — typed as object since pureini is lazily imported
    pc_encoders: dict[str, tuple[object, object]] = {}
    decoder_factory = DecoderFactory()
    encoder_factory = ROS2EncoderFactory()

    # Statistics
    messages_converted = 0
    messages_copied = 0
    topics_converted: set[str] = set()
    pointcloud_messages_converted = 0
    pointcloud_topics_converted: set[str] = set()

    try:
        with (
            open_input(file) as (input_stream, _),
            output.open("wb") as output_stream,
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                OSCProgressColumn(title="Compressing images"),
                console=console,
            ) as progress,
        ):
            task_id = progress.add_task("Processing messages", total=total_message_count)

            writer = McapWriter(
                output_stream,
                encoder_factory=encoder_factory,
                num_workers=4,
            )
            writer.start()

            # Track schema/channel IDs
            schema_ids: dict[str, int] = {}  # schema_name -> schema_id
            channel_ids: dict[str, int] = {}  # topic -> channel_id
            next_schema_id = 1
            next_channel_id = 1

            decode_pool = ThreadPoolExecutor(max_workers=4)
            messages = read_message_decoded(input_stream, decoder_factories=[decoder_factory])
            prefetched = _prefetch_image_decodes(messages, decode_pool)

            for msg, decode_future in prefetched:
                schema_name = msg.schema.name if msg.schema else ""

                if video and schema_name in IMAGE_SCHEMAS:
                    # Convert to CompressedVideo
                    topic = msg.channel.topic

                    # Lazy initialization of encoder for this topic
                    frame: VideoFrame | None = None
                    if topic not in encoders:
                        # Decode first frame to get dimensions
                        if schema_name in COMPRESSED_SCHEMAS:
                            first_frame = decode_future.result()  # type: ignore[union-attr]
                            width, height = first_frame.width, first_frame.height
                            frame = first_frame  # Reuse decoded frame
                        else:  # RAW_SCHEMAS
                            rgb_array = raw_image_to_array(msg.decoded_message)
                            height, width = rgb_array.shape[:2]
                            frame = av.VideoFrame.from_ndarray(rgb_array, format="rgb24")

                        # Ensure even dimensions (required for yuv420p)
                        width -= width % 2
                        height -= height % 2

                        # Create encoder for this topic
                        try:
                            encoders[topic] = _VideoEncoder(
                                width=width,
                                height=height,
                                codec_name=encoder_name,
                                quality=quality,
                                target_fps=30.0,
                                gop_size=30,
                            )
                            topics_converted.add(topic)
                            console.print(
                                f"[green]✓[/green] Converting {topic}: {width}x{height} "
                                f"({schema_name} → CompressedVideo)"
                            )
                        except VideoEncoderError as exc:
                            console.print(
                                f"[red]Error:[/red] Failed to create encoder for {topic}: {exc}"
                            )
                            return 1

                    # Decode image (skip if already decoded for encoder init)
                    if frame is None:
                        if decode_future is not None:
                            frame = decode_future.result()
                        elif schema_name in COMPRESSED_SCHEMAS:
                            compressed_data = bytes(msg.decoded_message.data)
                            frame = _decode_compressed_image(compressed_data)
                        else:  # RAW_SCHEMAS
                            rgb_array = raw_image_to_array(msg.decoded_message)
                            frame = av.VideoFrame.from_ndarray(rgb_array, format="rgb24")

                    # Encode to video
                    try:
                        video_data = encoders[topic].encode(frame)
                    except VideoEncoderError as exc:
                        # Try fallback to software encoder if hardware encoder fails
                        if encoders[topic].config.codec_name != "libx264":
                            console.print(
                                f"[yellow]Warning:[/yellow] Hardware encoder failed for {topic}, "
                                f"falling back to libx264"
                            )
                            encoder_name = "libx264"
                            # Recreate encoder with software encoder
                            width = encoders[topic].config.width
                            height = encoders[topic].config.height
                            encoders[topic].close()
                            try:
                                encoders[topic] = _VideoEncoder(
                                    width=width,
                                    height=height,
                                    codec_name="libx264",
                                    quality=quality,
                                    target_fps=30.0,
                                    gop_size=30,
                                )
                                video_data = encoders[topic].encode(frame)
                            except VideoEncoderError as fallback_exc:
                                console.print(
                                    f"[red]Error:[/red] Software encoder also failed for {topic}: "
                                    f"{fallback_exc}"
                                )
                                return 1
                        else:
                            console.print(
                                f"[red]Error:[/red] Failed to encode frame for {topic}: {exc}"
                            )
                            return 1

                    # Skip writing if encoder buffered the frame (no output yet)
                    if video_data is None:
                        progress.update(task_id, advance=1)
                        continue

                    # Create CompressedVideo message
                    compressed_video_msg = {
                        "timestamp": {
                            "sec": msg.decoded_message.header.stamp.sec,
                            "nanosec": msg.decoded_message.header.stamp.nanosec,
                        },
                        "frame_id": msg.decoded_message.header.frame_id,
                        "data": video_data,
                        "format": codec,
                    }

                    # Register schema if needed
                    compressed_video_schema = "foxglove_msgs/msg/CompressedVideo"
                    if compressed_video_schema not in schema_ids:
                        schema_id = next_schema_id
                        next_schema_id += 1
                        writer.add_schema(
                            schema_id,
                            compressed_video_schema,
                            "ros2msg",
                            FOXGLOVE_COMPRESSED_VIDEO.encode(),
                        )
                        schema_ids[compressed_video_schema] = schema_id
                    else:
                        schema_id = schema_ids[compressed_video_schema]

                    # Register channel if needed
                    if topic not in channel_ids:
                        channel_id = next_channel_id
                        next_channel_id += 1
                        writer.add_channel(channel_id, topic, "cdr", schema_id)
                        channel_ids[topic] = channel_id
                    else:
                        channel_id = channel_ids[topic]

                    # Write as CompressedVideo
                    writer.add_message_encode(
                        channel_id=channel_id,
                        log_time=msg.message.log_time,
                        data=compressed_video_msg,
                        publish_time=msg.message.publish_time,
                    )
                    messages_converted += 1

                elif pointcloud and schema_name in POINTCLOUD2_SCHEMAS:
                    if not pureini_available:
                        console.print(
                            "[red]Error:[/red] pureini is required for PointCloud2 compression. "
                            "Install with: uv add 'pymcap-cli[pointcloud]'"
                        )
                        return 1

                    topic = msg.channel.topic

                    # Build encoding info and compress
                    info = _build_encoding_info(
                        msg.decoded_message, pc_encoding_opt, pc_compression_opt, resolution
                    )
                    # Cache PointcloudEncoder per topic, recreate if info changes
                    cached = pc_encoders.get(topic)
                    if cached is None or cached[0] != info:
                        pc_encoders[topic] = (info, PointcloudEncoder(info))
                    pc_encoder = pc_encoders[topic][1]
                    compressed = pc_encoder.encode(bytes(msg.decoded_message.data))  # ty: ignore[unresolved-attribute]

                    if topic not in pointcloud_topics_converted:
                        pointcloud_topics_converted.add(topic)
                        console.print(
                            f"[green]✓[/green] Converting {topic} "
                            f"({schema_name} → CompressedPointCloud2)"
                        )

                    # Create CompressedPointCloud2 message
                    decoded = msg.decoded_message
                    compressed_pc_msg = {
                        "header": {
                            "stamp": {
                                "sec": decoded.header.stamp.sec,
                                "nanosec": decoded.header.stamp.nanosec,
                            },
                            "frame_id": decoded.header.frame_id,
                        },
                        "height": decoded.height,
                        "width": decoded.width,
                        "fields": [
                            {
                                "name": f.name,
                                "offset": f.offset,
                                "datatype": f.datatype,
                                "count": f.count,
                            }
                            for f in decoded.fields
                        ],
                        "is_bigendian": decoded.is_bigendian,
                        "point_step": decoded.point_step,
                        "row_step": decoded.row_step,
                        "compressed_data": compressed,
                        "is_dense": decoded.is_dense,
                        "format": "cloudini",
                    }

                    # Register schema if needed
                    compressed_pc_schema = "point_cloud_interfaces/msg/CompressedPointCloud2"
                    if compressed_pc_schema not in schema_ids:
                        schema_id = next_schema_id
                        next_schema_id += 1
                        writer.add_schema(
                            schema_id,
                            compressed_pc_schema,
                            "ros2msg",
                            COMPRESSED_POINTCLOUD2.encode(),
                        )
                        schema_ids[compressed_pc_schema] = schema_id
                    else:
                        schema_id = schema_ids[compressed_pc_schema]

                    # Register channel if needed
                    if topic not in channel_ids:
                        channel_id = next_channel_id
                        next_channel_id += 1
                        writer.add_channel(channel_id, topic, "cdr", schema_id)
                        channel_ids[topic] = channel_id
                    else:
                        channel_id = channel_ids[topic]

                    # Write as CompressedPointCloud2
                    writer.add_message_encode(
                        channel_id=channel_id,
                        log_time=msg.message.log_time,
                        data=compressed_pc_msg,
                        publish_time=msg.message.publish_time,
                    )
                    pointcloud_messages_converted += 1

                else:
                    # Copy unchanged
                    topic = msg.channel.topic

                    if topic not in channel_ids:
                        # Register schema if needed
                        if msg.schema:
                            if msg.schema.name not in schema_ids:
                                schema_id = next_schema_id
                                next_schema_id += 1
                                writer.add_schema(
                                    schema_id, msg.schema.name, msg.schema.encoding, msg.schema.data
                                )
                                schema_ids[msg.schema.name] = schema_id
                            else:
                                schema_id = schema_ids[msg.schema.name]
                        else:
                            schema_id = 0

                        channel_id = next_channel_id
                        next_channel_id += 1
                        writer.add_channel(
                            channel_id=channel_id,
                            topic=topic,
                            message_encoding=msg.channel.message_encoding,
                            schema_id=schema_id,
                            metadata=msg.channel.metadata,
                        )
                        channel_ids[topic] = channel_id
                    else:
                        channel_id = channel_ids[topic]

                    # Write message with original data
                    writer.add_message(
                        channel_id=channel_id,
                        log_time=msg.message.log_time,
                        data=msg.message.data,
                        publish_time=msg.message.publish_time,
                    )
                    messages_copied += 1

                progress.update(task_id, advance=1)

            decode_pool.shutdown(wait=False)
            writer.finish()

    finally:
        # Flush and clean up encoders
        for video_encoder in encoders.values():
            video_encoder.flush()
            video_encoder.close()

    # Report statistics
    total_converted = messages_converted + pointcloud_messages_converted
    console.print("\n[green bold]✓ Compression complete![/green bold]")
    if topics_converted:
        console.print(f"[cyan]Video topics converted:[/cyan] {len(topics_converted)}")
        for topic in sorted(topics_converted):
            console.print(f"  - {topic}")
        console.print(f"[cyan]Video messages converted:[/cyan] {messages_converted:,}")
    if pointcloud_topics_converted:
        console.print(
            f"[cyan]Point cloud topics converted:[/cyan] {len(pointcloud_topics_converted)}"
        )
        for topic in sorted(pointcloud_topics_converted):
            console.print(f"  - {topic}")
        console.print(
            f"[cyan]Point cloud messages converted:[/cyan] {pointcloud_messages_converted:,}"
        )
    console.print(f"[cyan]Messages copied:[/cyan] {messages_copied:,}")
    console.print(f"[cyan]Total messages:[/cyan] {total_converted + messages_copied:,}")

    # Show file size comparison
    input_size = Path(file).stat().st_size
    output_size = output.stat().st_size
    reduction_pct = ((input_size - output_size) / input_size) * 100 if input_size > 0 else 0

    console.print(f"\n[cyan]Input size:[/cyan] {input_size / 1024 / 1024:.2f} MB")
    console.print(f"[cyan]Output size:[/cyan] {output_size / 1024 / 1024:.2f} MB")
    if reduction_pct > 0:
        console.print(f"[green]Reduction:[/green] {reduction_pct:.1f}%")
    else:
        console.print(f"[yellow]Size change:[/yellow] {-reduction_pct:.1f}% increase")

    return 0
