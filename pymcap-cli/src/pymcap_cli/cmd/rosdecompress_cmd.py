"""Command to decompress CompressedVideo and CompressedPointCloud2 topics in MCAP files."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Annotated, Any, Literal

if TYPE_CHECKING:
    from small_mcap.reader import DecodedMessage

from cyclopts import Group, Parameter
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from rich.console import Console
from small_mcap import McapWriter
from small_mcap.nop_decoder import NOPDecoderFactory
from small_mcap.reader import read_message_decoded

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.core.mcap_transform import (
    copy_message,
    create_progress,
    ensure_channel,
    ensure_schema,
    get_total_message_count,
    print_size_comparison,
)
from pymcap_cli.encoding.decompress import (
    _COMPRESSED_VIDEO_SCHEMA,
    COMPRESSED_IMAGE,
    IMAGE,
    POINTCLOUD2,
    VideoDecompressFactory,
)
from pymcap_cli.encoding.encoder_common import EncoderMode
from pymcap_cli.encoding.pointcloud import (
    _COMPRESSED_POINTCLOUD2_SCHEMA,
    PointCloudDecompressFactory,
)
from pymcap_cli.types.types_manual import (  # noqa: TC001 — runtime for cyclopts
    ForceOverwriteOption,
    OutputPathOption,
)
from pymcap_cli.utils import confirm_output_overwrite

console = Console()

VIDEO_GROUP = Group("Video")
POINTCLOUD_GROUP = Group("Point Cloud")


def _build_header(msg: DecodedMessage) -> dict[str, Any]:
    """Build a ROS header dict from a DecodedMessage's original CompressedVideo fields."""
    decoded = msg.decoded_message
    if decoded is not None and isinstance(decoded, dict):
        return decoded.get("header", {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""})
    # Fallback: use message timestamps.
    sec = msg.message.log_time // 1_000_000_000
    nanosec = msg.message.log_time % 1_000_000_000
    return {"stamp": {"sec": sec, "nanosec": nanosec}, "frame_id": ""}


def rosdecompress(
    file: str,
    output: OutputPathOption,
    *,
    force: ForceOverwriteOption = False,
    video: Annotated[
        bool,
        Parameter(
            name=["--video/--no-video"],
            group=VIDEO_GROUP,
        ),
    ] = True,
    video_format: Annotated[
        Literal["compressed", "raw"],
        Parameter(
            name=["--video-format"],
            group=VIDEO_GROUP,
        ),
    ] = "compressed",
    jpeg_quality: Annotated[
        int,
        Parameter(
            name=["--jpeg-quality"],
            group=VIDEO_GROUP,
        ),
    ] = 90,
    backend: Annotated[
        EncoderMode,
        Parameter(
            name=["--backend"],
            group=VIDEO_GROUP,
        ),
    ] = EncoderMode.AUTO,
    pointcloud: Annotated[
        bool,
        Parameter(
            name=["--pointcloud/--no-pointcloud"],
            group=POINTCLOUD_GROUP,
        ),
    ] = True,
) -> int:
    """Decompress ROS MCAP by converting compressed topics back to standard formats.

    Converts CompressedVideo topics back to CompressedImage (JPEG) or raw Image,
    and CompressedPointCloud2 topics back to PointCloud2.

    Parameters
    ----------
    file
        Input MCAP file (local file or HTTP/HTTPS URL).
    output
        Output filename.
    force
        Force overwrite of output file without confirmation.
    video
        Enable video decompression. Default: True.
    video_format
        Output format for video topics: "compressed" (JPEG) or "raw" (uncompressed Image).
        Default: compressed.
    jpeg_quality
        JPEG quality (1-100) when video_format=compressed. Default: 90.
    backend
        Video decoder backend: auto, pyav, or ffmpeg-cli. Default: auto.
    pointcloud
        Enable point cloud decompression. Default: True.
    """
    confirm_output_overwrite(output, force)

    # Print config.
    console.print(f"[cyan]Input:[/cyan] {file}")
    console.print(f"[cyan]Output:[/cyan] {output}")
    if video:
        console.print(f"[cyan]Video format:[/cyan] {video_format}")
        if video_format == "compressed":
            console.print(f"[cyan]JPEG quality:[/cyan] {jpeg_quality}")
        console.print(f"[cyan]Backend:[/cyan] {backend.value}")
    else:
        console.print("[cyan]Video decompression:[/cyan] disabled")
    if pointcloud:
        console.print("[cyan]Point cloud decompression:[/cyan] enabled")
    else:
        console.print("[cyan]Point cloud decompression:[/cyan] disabled")

    total_message_count = get_total_message_count(file)

    # Create decoder factories.
    # Video/pointcloud factories are channel-aware and handle compressed topics.
    # CDR factory handles all other ROS2 schemas (pass-through messages).
    factories: list[VideoDecompressFactory | PointCloudDecompressFactory | NOPDecoderFactory] = []
    if video:
        factories.append(
            VideoDecompressFactory(
                video_format=video_format,
                jpeg_quality=jpeg_quality,
                backend=backend,
            )
        )
    if pointcloud:
        factories.append(PointCloudDecompressFactory())
    factories.append(NOPDecoderFactory())
    encoder_factory = ROS2EncoderFactory()

    # Statistics.
    video_messages = 0
    pointcloud_messages = 0
    messages_copied = 0
    video_topics: set[str] = set()
    pointcloud_topics: set[str] = set()

    schema_ids: dict[str, int] = {}
    channel_ids: dict[str, int] = {}

    # Pending video messages whose decoded data hasn't arrived yet (decoder buffering).
    pending_video: dict[str, deque[DecodedMessage]] = {}

    # Track which video factory is used for flushing.
    video_factory: VideoDecompressFactory | None = None
    for f in factories:
        if isinstance(f, VideoDecompressFactory):
            video_factory = f
            break

    with (
        open_input(file) as (input_stream, input_size),
        output.open("wb") as output_stream,
        create_progress(console, title="Decompressing topics") as progress,
    ):
        task_id = progress.add_task("Processing messages", total=total_message_count)

        writer = McapWriter(output_stream, encoder_factory=encoder_factory, num_workers=4)
        writer.start()

        messages = read_message_decoded(input_stream, decoder_factories=factories, num_workers=4)

        for msg in messages:
            schema_name = msg.schema.name if msg.schema else ""
            topic = msg.channel.topic

            if schema_name == _COMPRESSED_VIDEO_SCHEMA and video:
                decoded = msg.decoded_message  # triggers CDR decode via VideoDecompressFactory
                if video_format == "compressed":
                    out_schema_name = "sensor_msgs/msg/CompressedImage"
                    out_schema_data = COMPRESSED_IMAGE
                else:
                    out_schema_name = "sensor_msgs/msg/Image"
                    out_schema_data = IMAGE

                # Register schema/channel on first encounter.
                schema_id = ensure_schema(
                    writer, out_schema_name, "ros2msg", out_schema_data.encode(), schema_ids
                )
                ensure_channel(writer, topic, "cdr", schema_id, channel_ids)

                # Buffer message metadata.
                if topic not in pending_video:
                    pending_video[topic] = deque()
                pending_video[topic].append(msg)

                if decoded is None:
                    progress.advance(task_id)
                    continue

                # Write using oldest pending message's metadata.
                pending_msg = pending_video[topic].popleft()
                writer.add_message_encode(
                    channel_id=channel_ids[topic],
                    log_time=pending_msg.message.log_time,
                    publish_time=pending_msg.message.publish_time,
                    data=decoded,
                )
                video_messages += 1
                video_topics.add(topic)

            elif schema_name == _COMPRESSED_POINTCLOUD2_SCHEMA and pointcloud:
                decoded = msg.decoded_message
                schema_id = ensure_schema(
                    writer, "sensor_msgs/msg/PointCloud2", "ros2msg", POINTCLOUD2.encode(), schema_ids
                )
                channel_id = ensure_channel(writer, topic, "cdr", schema_id, channel_ids)
                writer.add_message_encode(
                    channel_id=channel_id,
                    log_time=msg.message.log_time,
                    publish_time=msg.message.publish_time,
                    data=decoded,
                )
                pointcloud_messages += 1
                pointcloud_topics.add(topic)

            else:
                # Pass-through: copy raw bytes — never touches decoded_message.
                copy_message(msg, writer, schema_ids, channel_ids)
                messages_copied += 1
                progress.advance(task_id)
                continue

            progress.advance(task_id)

        # Flush buffered frames from video decompressors.
        if video_factory is not None:
            flushed_frames = video_factory.flush_all()
            for frame in flushed_frames:
                # Find a topic with pending messages.
                topic_name = next(
                    (t for t, p in pending_video.items() if p and t in channel_ids),
                    None,
                )
                if topic_name is None:
                    break
                pending_msg = pending_video[topic_name].popleft()
                header = _build_header(pending_msg)
                if frame.is_jpeg:
                    msg_data: dict[str, Any] = {
                        "header": header,
                        "format": "jpeg",
                        "data": frame.data,
                    }
                else:
                    msg_data = {
                        "header": header,
                        "height": frame.height,
                        "width": frame.width,
                        "encoding": "rgb8",
                        "is_bigendian": 0,
                        "step": frame.width * 3,
                        "data": frame.data,
                    }
                writer.add_message_encode(
                    channel_id=channel_ids[topic_name],
                    log_time=pending_msg.message.log_time,
                    publish_time=pending_msg.message.publish_time,
                    data=msg_data,
                )
                video_messages += 1

        writer.finish()
        output_size = output_stream.tell()

    # Print statistics.
    console.print()
    if video_messages:
        target = "CompressedImage (JPEG)" if video_format == "compressed" else "Image (raw)"
        console.print(
            f"[green]Video:[/green] {video_messages:,} messages -> {target} "
            f"({len(video_topics)} topic{'s' if len(video_topics) != 1 else ''})"
        )
    if pointcloud_messages:
        console.print(
            f"[green]Point cloud:[/green] {pointcloud_messages:,} messages -> PointCloud2 "
            f"({len(pointcloud_topics)} topic{'s' if len(pointcloud_topics) != 1 else ''})"
        )
    if messages_copied:
        console.print(f"[dim]Copied:[/dim] {messages_copied:,} messages unchanged")

    print_size_comparison(console, input_size, output_size)

    return 0
