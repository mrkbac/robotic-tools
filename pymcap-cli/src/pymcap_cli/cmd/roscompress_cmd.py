"""Command to compress image and point cloud topics in MCAP files."""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Annotated, Literal

import av
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
    COMPRESSED_POINTCLOUD2,
    COMPRESSED_SCHEMAS,
    FOXGLOVE_COMPRESSED_VIDEO,
    IMAGE_SCHEMAS,
    POINTCLOUD2_SCHEMAS,
    PointCloudCompressor,
    VideoEncoder,
    VideoEncoderError,
    calculate_downscale_dimensions,
    decode_compressed_frame,
    get_software_encoder,
    raw_image_to_array,
    resolve_encoder,
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

    from small_mcap.reader import DecodedMessage


console = Console()

# Parameter groups
ENCODING_GROUP = Group("Encoding")
POINTCLOUD_GROUP = Group("Point Cloud")


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
        Literal["h264", "h265"],
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
    scale: Annotated[
        int | None,
        Parameter(
            name=["--scale", "-s"],
            group=ENCODING_GROUP,
        ),
    ] = None,
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
    scale
        Cap the maximum image dimension (width or height) while preserving aspect ratio.
        When None, use original resolution.
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
            encoder_name = resolve_encoder(codec)

    # Create point cloud compressor
    pc_compressor: PointCloudCompressor | None = None
    if pointcloud:
        try:
            pc_compressor = PointCloudCompressor(
                encoding=pc_encoding, compression=pc_compression, resolution=resolution
            )
        except ImportError:
            console.print(
                "[red]Error:[/red] pureini is required for PointCloud2 compression. "
                "Install with: uv add 'pymcap-cli[pointcloud]'"
            )
            return 1

    console.print(f"[cyan]Input:[/cyan] {file}")
    console.print(f"[cyan]Output:[/cyan] {output}")
    if video:
        console.print(f"[cyan]Encoder:[/cyan] {encoder_name}")
        console.print(f"[cyan]Quality (CRF):[/cyan] {quality}")
        if scale is not None:
            console.print(f"[cyan]Scale (max dim):[/cyan] {scale}px")
    else:
        console.print("[cyan]Video compression:[/cyan] disabled")
    if pointcloud:
        console.print(f"[cyan]Point cloud encoding:[/cyan] {pc_encoding}")
        console.print(f"[cyan]Point cloud compression:[/cyan] {pc_compression}")
        if pc_encoding == "lossy":
            console.print(f"[cyan]Point cloud resolution:[/cyan] {resolution}")
    else:
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
    encoders: dict[str, VideoEncoder] = {}
    decoder_factory = DecoderFactory()
    encoder_factory = ROS2EncoderFactory()

    # Statistics
    messages_converted = 0
    messages_copied = 0
    topics_converted: set[str] = set()
    pointcloud_messages_converted = 0
    pointcloud_topics_converted: set[str] = set()
    last_video_times: dict[str, tuple[int, int]] = {}  # topic -> (log_time, publish_time)

    with (
        open_input(file) as (input_stream, input_size),
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
        messages = read_message_decoded(
            input_stream, decoder_factories=[decoder_factory], num_workers=4
        )
        prefetched = _prefetch_image_decodes(messages, decode_pool, prefetch=16)

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

                    # Downscale if --scale is set, and ensure even dimensions
                    if scale is not None:
                        width, height = calculate_downscale_dimensions(width, height, scale)
                    else:
                        width -= width % 2
                        height -= height % 2

                    # Create encoder for this topic
                    try:
                        encoders[topic] = VideoEncoder(
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
                    sw_encoder = get_software_encoder(codec)
                    if encoders[topic].config.codec_name != sw_encoder:
                        console.print(
                            f"[yellow]Warning:[/yellow] Hardware encoder failed for {topic}, "
                            f"falling back to {sw_encoder}"
                        )
                        # Recreate encoder with software encoder
                        width = encoders[topic].config.width
                        height = encoders[topic].config.height
                        try:
                            encoders[topic] = VideoEncoder(
                                width=width,
                                height=height,
                                codec_name=sw_encoder,
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
                last_video_times[topic] = (msg.message.log_time, msg.message.publish_time)

            elif pointcloud and schema_name in POINTCLOUD2_SCHEMAS:
                topic = msg.channel.topic

                compressed = pc_compressor.compress(msg.decoded_message)  # type: ignore[union-attr]

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

        # Flush remaining frames from video encoders
        for topic_name, video_enc in encoders.items():
            flushed = video_enc.flush()
            if flushed and topic_name in channel_ids and topic_name in last_video_times:
                log_time, publish_time = last_video_times[topic_name]
                writer.add_message_encode(
                    channel_id=channel_ids[topic_name],
                    log_time=log_time,
                    data={
                        "timestamp": {"sec": 0, "nanosec": 0},
                        "frame_id": "",
                        "data": flushed,
                        "format": codec,
                    },
                    publish_time=publish_time,
                )

        decode_pool.shutdown(wait=True)
        writer.finish()

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
    output_size = output.stat().st_size
    if input_size > 0:
        reduction_pct = ((input_size - output_size) / input_size) * 100

        console.print(f"\n[cyan]Input size:[/cyan] {input_size / 1024 / 1024:.2f} MB")
        console.print(f"[cyan]Output size:[/cyan] {output_size / 1024 / 1024:.2f} MB")
        if reduction_pct > 0:
            console.print(f"[green]Reduction:[/green] {reduction_pct:.1f}%")
        else:
            console.print(f"[yellow]Size change:[/yellow] {-reduction_pct:.1f}% increase")
    else:
        console.print(f"\n[cyan]Output size:[/cyan] {output_size / 1024 / 1024:.2f} MB")

    return 0
