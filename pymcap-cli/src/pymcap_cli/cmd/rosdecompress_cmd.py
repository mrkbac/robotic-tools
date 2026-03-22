"""Command to decompress CompressedVideo and CompressedPointCloud2 topics in MCAP files."""

from __future__ import annotations

from typing import Annotated, Literal

from cyclopts import Group, Parameter
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from rich.console import Console
from small_mcap import McapWriter
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
    factories: list[VideoDecompressFactory | PointCloudDecompressFactory] = []
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
    encoder_factory = ROS2EncoderFactory()

    # Statistics.
    video_messages = 0
    pointcloud_messages = 0
    messages_copied = 0
    video_topics: set[str] = set()
    pointcloud_topics: set[str] = set()

    schema_ids: dict[str, int] = {}
    channel_ids: dict[str, int] = {}

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
            decoded = msg.decoded_message

            # Determine if this was a compressed topic and what output schema to use.
            if schema_name == _COMPRESSED_VIDEO_SCHEMA and video:
                if decoded is None:
                    # Decoder needs more data (waiting for keyframe)
                    progress.advance(task_id)
                    continue
                if video_format == "compressed":
                    out_schema_name = "sensor_msgs/msg/CompressedImage"
                    out_schema_data = COMPRESSED_IMAGE
                else:
                    out_schema_name = "sensor_msgs/msg/Image"
                    out_schema_data = IMAGE
                video_messages += 1
                video_topics.add(topic)

            elif schema_name == _COMPRESSED_POINTCLOUD2_SCHEMA and pointcloud:
                out_schema_name = "sensor_msgs/msg/PointCloud2"
                out_schema_data = POINTCLOUD2
                pointcloud_messages += 1
                pointcloud_topics.add(topic)

            else:
                # Pass-through: copy message unchanged.
                copy_message(msg, writer, schema_ids, channel_ids)
                messages_copied += 1
                progress.advance(task_id)
                continue

            # Transformed message — register schema/channel and write.
            schema_id = ensure_schema(
                writer, out_schema_name, "ros2msg", out_schema_data.encode(), schema_ids
            )
            channel_id = ensure_channel(writer, topic, "cdr", schema_id, channel_ids)
            writer.add_message_encode(
                channel_id=channel_id,
                log_time=msg.message.log_time,
                publish_time=msg.message.publish_time,
                data=decoded,
            )

            progress.advance(task_id)

        writer.finish()

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

    print_size_comparison(console, input_size, output_stream.tell())

    return 0
