"""Video encoding command for pymcap-cli using av with grid view."""

from __future__ import annotations

import io
import math
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import av
import av.error
import numpy as np
import typer
from av import VideoFrame
from mcap_ros2_support_fast.decoder import DecoderFactory
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
from small_mcap import get_summary, include_topics, read_message_decoded

from pymcap_cli.autocompletion import complete_topic_by_schema
from pymcap_cli.input_handler import open_input
from pymcap_cli.mcap_processor import confirm_output_overwrite

if TYPE_CHECKING:
    from av.container import InputContainer, OutputContainer


console = Console()
app = typer.Typer()


COMPRESSED_SCHEMAS = {"sensor_msgs/msg/CompressedImage", "sensor_msgs/CompressedImage"}
RAW_SCHEMAS = {"sensor_msgs/msg/Image", "sensor_msgs/Image"}
IMAGE_SCHEMAS = COMPRESSED_SCHEMAS | RAW_SCHEMAS


def complete_image_topics(ctx: typer.Context, incomplete: str) -> list[str]:
    return complete_topic_by_schema(ctx, incomplete, schemas=IMAGE_SCHEMAS)


def _validate_topics(topics: list[str]) -> list[str]:
    if not topics:
        raise typer.BadParameter(
            "At least one --topic is required. Repeat the flag for multiple topics."
        )
    return topics


class VideoCodec(str, Enum):
    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    AV1 = "av1"


class QualityPreset(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EncoderBackend(str, Enum):
    AUTO = "auto"
    SOFTWARE = "software"
    VIDEOTOOLBOX = "videotoolbox"
    NVENC = "nvenc"
    VAAPI = "vaapi"


class ImageType(Enum):
    COMPRESSED = auto()
    RAW = auto()


class VideoEncoderError(Exception):
    """Raised when encoding fails."""


@dataclass
class TopicInfo:
    """Discovered metadata for a topic."""

    topic: str
    schema_name: str
    message_type: ImageType
    width: int
    height: int


SOFTWARE_ENCODERS = {
    VideoCodec.H264: "libx264",
    VideoCodec.H265: "libx265",
    VideoCodec.VP9: "libvpx-vp9",
    VideoCodec.AV1: "libsvtav1",
}

HARDWARE_ENCODERS = {
    VideoCodec.H264: {
        "videotoolbox": "h264_videotoolbox",
        "nvenc": "h264_nvenc",
        "vaapi": "h264_vaapi",
    },
    VideoCodec.H265: {
        "videotoolbox": "hevc_videotoolbox",
        "nvenc": "hevc_nvenc",
        "vaapi": "hevc_vaapi",
    },
}

QUALITY_PRESETS = {
    VideoCodec.H264: {QualityPreset.HIGH: 32, QualityPreset.MEDIUM: 35, QualityPreset.LOW: 40},
    VideoCodec.H265: {QualityPreset.HIGH: 32, QualityPreset.MEDIUM: 35, QualityPreset.LOW: 40},
    VideoCodec.VP9: {QualityPreset.HIGH: 42, QualityPreset.MEDIUM: 45, QualityPreset.LOW: 50},
    VideoCodec.AV1: {QualityPreset.HIGH: 37, QualityPreset.MEDIUM: 40, QualityPreset.LOW: 45},
}


def _decode_and_resize_image(
    message: Any, message_type: ImageType, target_width: int, target_height: int
) -> np.ndarray:
    """Decode a ROS image message and resize to target dimensions."""
    if message_type is ImageType.COMPRESSED:
        compressed = _process_compressed_image(message)
        frame = _decode_compressed_frame(compressed)
        frame = frame.reformat(width=target_width, height=target_height, format="rgb24")
        return frame.to_ndarray(format="rgb24")

    rgb_array = _raw_image_to_array(message)
    frame = av.VideoFrame.from_ndarray(rgb_array, format="rgb24")
    frame = frame.reformat(width=target_width, height=target_height, format="rgb24")
    return frame.to_ndarray(format="rgb24")


def _process_compressed_image(message: Any) -> bytes:
    if not hasattr(message, "data") or not message.data:
        raise VideoEncoderError("CompressedImage has no data")
    return bytes(message.data)


def _decode_compressed_frame(compressed_data: bytes) -> VideoFrame:
    try:
        container: InputContainer = av.open(  # type: ignore[assignment]
            io.BytesIO(compressed_data), format="image2"
        )
        for frame in container.decode(video=0):
            container.close()
            return frame
    except Exception as exc:  # pragma: no cover - depends on data
        raise VideoEncoderError(f"Failed to decode compressed image: {exc}") from exc

    raise VideoEncoderError("Decoder produced no frames")


def _raw_image_to_array(message: Any) -> np.ndarray:
    if not hasattr(message, "data") or not message.data:
        raise VideoEncoderError("Image has no data")
    if not hasattr(message, "width") or not hasattr(message, "height"):
        raise VideoEncoderError("Image missing width/height")
    if not hasattr(message, "encoding"):
        raise VideoEncoderError("Image missing encoding")

    width = message.width
    height = message.height
    encoding = str(message.encoding).lower()
    data = bytes(message.data)
    if encoding in {"rgb", "rgb8"}:
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        return array.copy()  # type: ignore[no-any-return]
    if encoding in {"bgr", "bgr8"}:
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        return array[..., ::-1].copy()  # type: ignore[no-any-return]
    if encoding in {"mono", "mono8", "8uc1"}:
        mono_array = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        return np.repeat(mono_array[:, :, None], 3, axis=2)
    raise VideoEncoderError(f"Unsupported image encoding: {message.encoding}")


def _discover_topic_info(message: Any, topic: str) -> TopicInfo:
    """Discover topic metadata from first message."""
    schema_name = message.schema.name if message.schema else ""

    if schema_name in COMPRESSED_SCHEMAS:
        message_type = ImageType.COMPRESSED
        frame = _decode_compressed_frame(_process_compressed_image(message.decoded_message))
        width = frame.width
        height = frame.height
    elif schema_name in RAW_SCHEMAS:
        message_type = ImageType.RAW
        rgb = _raw_image_to_array(message.decoded_message)
        height, width = rgb.shape[:2]
    else:
        raise VideoEncoderError(f"Topic '{topic}' is not an image topic (schema: {schema_name})")

    return TopicInfo(
        topic=topic,
        schema_name=schema_name,
        message_type=message_type,
        width=width,
        height=height,
    )


def _detect_encoder(codec: VideoCodec, encoder_backend: EncoderBackend) -> str:
    if encoder_backend == EncoderBackend.SOFTWARE:
        encoder = SOFTWARE_ENCODERS.get(codec)
        if not encoder:
            raise VideoEncoderError(f"No software encoder available for codec: {codec.value}")
        return encoder
    if encoder_backend in (
        EncoderBackend.VIDEOTOOLBOX,
        EncoderBackend.NVENC,
        EncoderBackend.VAAPI,
    ):
        hw = HARDWARE_ENCODERS.get(codec, {})
        encoder = hw.get(encoder_backend.value)
        if not encoder:
            raise VideoEncoderError(
                f"Hardware encoder '{encoder_backend.value}' not available for codec: {codec.value}"
            )
        if not _test_encoder(encoder):
            raise VideoEncoderError(
                f"Hardware encoder '{encoder}' not available on this system. "
                "Try --encoder software."
            )
        return encoder
    if encoder_backend == EncoderBackend.AUTO:
        hw = HARDWARE_ENCODERS.get(codec, {})
        system = platform.system()
        if system == "Darwin" and "videotoolbox" in hw:
            encoder = hw["videotoolbox"]
            if _test_encoder(encoder):
                return encoder
        if system == "Linux":
            for backend in ("nvenc", "vaapi"):
                if backend in hw and _test_encoder(hw[backend]):
                    return hw[backend]
        encoder = SOFTWARE_ENCODERS.get(codec)
        if not encoder:
            raise VideoEncoderError(f"No encoder available for codec: {codec.value}")
        return encoder
    raise VideoEncoderError(f"Unknown encoder backend: {encoder_backend}")


def _test_encoder(encoder_name: str) -> bool:
    try:
        av.CodecContext.create(encoder_name, "w")
    except (av.error.FFmpegError, ValueError):
        return False
    else:
        return True


def _get_encoder_options(codec: VideoCodec, encoder_name: str) -> dict[str, str]:
    options: dict[str, str] = {}
    # All encoders now use bitrate mode, so don't set CRF/CQ/QP
    # Just set encoder-specific presets if needed
    if "nvenc" in encoder_name:
        options["preset"] = "p4"
    elif (
        codec in (VideoCodec.H264, VideoCodec.H265) and "libx264" in encoder_name
    ) or "libx265" in encoder_name:
        options["preset"] = "medium"
    return options


def _compose_grid(
    tile_frames: Sequence[np.ndarray], layout: tuple[int, int], tile_size: tuple[int, int]
) -> np.ndarray:
    rows, cols = layout
    tile_height, tile_width = tile_size
    grid = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=np.uint8)
    for idx, tile in enumerate(tile_frames):
        row = idx // cols
        col = idx % cols
        y0 = row * tile_height
        x0 = col * tile_width
        grid[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile
    return grid


def encode_video(
    mcap_path: str,
    topics: list[str],
    output_path: Path,
    codec: VideoCodec,
    encoder_backend: EncoderBackend,
    quality: int,
) -> None:
    """Encode video from MCAP file using single-pass streaming."""
    # Check schema and channels, validate topics, get statistics
    available_image_topics: list[str] = []
    total_message_count: int | None = None

    with open_input(mcap_path) as (f, _file_size):
        if summary := get_summary(f):
            # Find all schemas that are image types
            image_schema_ids = {
                schema.id for schema in summary.schemas.values() if schema.name in IMAGE_SCHEMAS
            }

            # Find all channels with image schemas
            requested_topic_channels = {
                channel.topic: channel.id
                for channel in summary.channels.values()
                if channel.schema_id in image_schema_ids and channel.topic in topics
            }
            available_image_topics = sorted(
                [
                    channel.topic
                    for channel in summary.channels.values()
                    if channel.schema_id in image_schema_ids
                ]
            )

            # Validate all requested topics exist
            missing_topics = set(topics) - set(requested_topic_channels.keys())
            if missing_topics:
                available_str = "\n".join(f"  - {t}" for t in available_image_topics)
                raise VideoEncoderError(
                    "Topic(s) not found or not image topics:"
                    + ", ".join(sorted(missing_topics))
                    + "\n\n"
                    f"Available image topics:\n{available_str}"
                )

            # Calculate total message count from statistics if available
            if summary.statistics and summary.statistics.channel_message_counts:
                total_message_count = sum(
                    count
                    for channel_id, count in summary.statistics.channel_message_counts.items()
                    if channel_id in requested_topic_channels.values()
                )

    console.print(f"[cyan]Reading MCAP file:[/cyan] {mcap_path}")
    console.print(f"[cyan]Topics:[/cyan] {', '.join(topics)}")
    if total_message_count is not None:
        console.print(f"[cyan]Total messages:[/cyan] {total_message_count:,}")
    console.print(f"[cyan]Output:[/cyan] {output_path}")
    console.print(f"[cyan]Codec:[/cyan] {codec.value}")

    encoder_name = _detect_encoder(codec, encoder_backend)
    console.print(f"[cyan]Encoder:[/cyan] {encoder_name}")

    # Compute grid layout
    rows, cols = (1, 1)
    if len(topics) == 2:
        rows, cols = (1, 2)
    elif len(topics) > 2:
        rows = math.ceil(math.sqrt(len(topics)))
        cols = math.ceil(len(topics) / rows)
    console.print(f"[green]✓[/green] Layout: {rows} row(s) x {cols} column(s)")

    decoder_factory = DecoderFactory()

    # State for streaming
    topic_infos: dict[str, TopicInfo] = {}  # Discovered metadata per topic
    last_frames: dict[str, np.ndarray] = {}  # Cache of last decoded frame per topic
    container: OutputContainer | None = None
    stream: Any = None
    frame_idx = 0
    first_timestamp_ns: int | None = None  # First message timestamp for PTS calculation
    last_pts: int = -1  # Track last PTS to ensure monotonic increase
    target_tile_width: int | None = None
    target_tile_height: int | None = None
    grid_width: int | None = None
    grid_height: int | None = None

    console.print("\n[yellow]Processing messages...[/yellow]")

    with (
        open_input(mcap_path) as (handle, _),
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress,
    ):
        task_id = progress.add_task("Encoding frames", total=total_message_count)

        for msg in read_message_decoded(
            handle,
            should_include=include_topics(topics),
            decoder_factories=[decoder_factory],
        ):
            topic = msg.channel.topic

            # Discover topic info on first message
            if topic not in topic_infos:
                topic_infos[topic] = _discover_topic_info(msg, topic)
                console.print(
                    f"[green]✓[/green] Discovered {topic}: "
                    f"{topic_infos[topic].width}x{topic_infos[topic].height} "
                    f"({topic_infos[topic].message_type.name})"
                )

                # Initialize encoder once we have all topics
                if len(topic_infos) == len(topics) and container is None:
                    # Compute tile dimensions
                    target_tile_width = max(2, min(info.width for info in topic_infos.values()))
                    target_tile_height = max(2, min(info.height for info in topic_infos.values()))
                    target_tile_width -= target_tile_width % 2
                    target_tile_height -= target_tile_height % 2

                    grid_width = target_tile_width * cols
                    grid_height = target_tile_height * rows
                    grid_width -= grid_width % 2
                    grid_height -= grid_height % 2

                    console.print(
                        f"[green]✓[/green] Grid: {grid_width}x{grid_height} "
                        f"(tiles: {target_tile_width}x{target_tile_height})"
                    )

                    # Initialize black frames for all topics
                    for t in topics:
                        last_frames[t] = np.zeros(
                            (target_tile_height, target_tile_width, 3), dtype=np.uint8
                        )

                    # Create output container and stream
                    container = av.open(
                        str(output_path),
                        "w",
                        format=None,
                        options={"movflags": "faststart"},
                    )

                    try:
                        stream = container.add_stream(codec_name=encoder_name)
                    except (av.error.FFmpegError, ValueError) as exc:
                        container.close()
                        raise VideoEncoderError(
                            f"Failed to create video stream with encoder '{encoder_name}':"
                            "This encoder may not be available on your system."
                            "Try --encoder software."
                        ) from exc

                    stream.width = grid_width
                    stream.height = grid_height
                    stream.pix_fmt = "yuv420p"
                    # Use microsecond time_base (standard for video, more compatible)
                    stream.time_base = Fraction(1, 1_000_000)
                    stream.codec_context.time_base = Fraction(1, 1_000_000)
                    # Set a nominal framerate (actual timing from PTS)
                    stream.codec_context.framerate = Fraction(30, 1)
                    stream.codec_context.gop_size = 60  # 2 seconds at 30fps

                    # Set encoder options
                    options = _get_encoder_options(codec, encoder_name)

                    # Set bitrate (simplified, no input bitrate calculation)
                    if quality <= 20:  # high quality
                        target_bitrate = 10_000_000  # 10 Mbps
                    elif quality <= 25:  # medium quality
                        target_bitrate = 5_000_000  # 5 Mbps
                    else:  # low quality
                        target_bitrate = 2_000_000  # 2 Mbps

                    stream.codec_context.bit_rate = target_bitrate

                    if (
                        "libx264" in encoder_name
                        or "libx265" in encoder_name
                        or "videotoolbox" in encoder_name
                    ):
                        options["bf"] = "0"

                    stream.codec_context.options = options
                    console.print(f"[cyan]Encoder options:[/cyan] {options}")

            # Get message timestamp
            message_timestamp_ns = msg.message.log_time

            # Skip messages until encoder is ready
            if container is None or stream is None:
                continue

            # Set first timestamp on first encoded frame
            if first_timestamp_ns is None:
                first_timestamp_ns = message_timestamp_ns

            # Decode and cache this frame
            assert target_tile_width is not None
            assert target_tile_height is not None
            last_frames[topic] = _decode_and_resize_image(
                msg.decoded_message,
                topic_infos[topic].message_type,
                target_tile_width,
                target_tile_height,
            )

            # Compose grid from current cached frames
            tiles = [last_frames[t] for t in topics]
            grid_frame = _compose_grid(tiles, (rows, cols), (target_tile_height, target_tile_width))

            # Encode frame with actual timestamp
            try:
                frame = av.VideoFrame.from_ndarray(grid_frame, format="rgb24")
                frame = frame.reformat(format="yuv420p")
                # Set PTS in microseconds (matching stream.time_base = 1/1_000_000)
                current_pts = (message_timestamp_ns - first_timestamp_ns) // 1000
                # Ensure monotonic increase (required by MP4/H.264)
                if current_pts <= last_pts:
                    current_pts = last_pts + 1
                frame.pts = current_pts
                last_pts = current_pts
                packets = stream.encode(frame)
                for packet in packets:
                    container.mux(packet)
                frame_idx += 1
                progress.update(task_id, advance=1)
            except (av.error.FFmpegError, ValueError) as exc:
                if container:
                    container.close()
                raise VideoEncoderError(
                    f"Encoding failed at frame {frame_idx}: {exc}\n"
                    f"Encoder: {encoder_name}, Resolution: {grid_width}x{grid_height}, "
                    f"Codec: {codec.value}\n"
                    f"Try using --encoder software or a different codec."
                ) from exc

        # Check if we encoded any frames
        if container is None or frame_idx == 0:
            if container:
                container.close()
            missing = set(topics) - set(topic_infos.keys())
            if missing:
                # Show available topics if we can
                available_str = ""
                if available_image_topics:
                    available_str = "\n\nAvailable image topics:\n" + "\n".join(
                        f"  - {t}" for t in available_image_topics
                    )
                raise VideoEncoderError(
                    f"No messages found for topics: {', '.join(sorted(missing))}\n"
                    f"These topics exist in the file but had no messages.{available_str}"
                )
            raise VideoEncoderError(
                "No frames were encoded. Check that topics contain valid image messages."
            )

        # Flush encoder
        try:
            for packet in stream.encode(None):
                container.mux(packet)
        except (av.error.FFmpegError, ValueError) as exc:
            container.close()
            raise VideoEncoderError(
                f"Failed to flush encoder: {exc}\nThe video file may be incomplete or corrupted."
            ) from exc

        container.close()

    console.print(f"\n[green bold]✓ Video created:[/green bold] {output_path}")
    console.print(f"[cyan]Total frames:[/cyan] {frame_idx:,}")


@app.command(
    epilog="""
Examples:
  pymcap-cli video data.mcap -t /camera/front -o output.mp4
  pymcap-cli video data.mcap -t /cam/left -t /cam/right -o grid.mp4
"""
)
def video(
    file: Annotated[
        str,
        typer.Argument(help="Path to the MCAP file (local file or HTTP/HTTPS URL)"),
    ],
    topics: Annotated[
        list[str],
        typer.Option(
            "--topic",
            "-t",
            callback=_validate_topics,
            help="Image topic to convert (repeat for multiple topics)",
            autocompletion=complete_image_topics,
            rich_help_panel="Input Options",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            ...,
            "--output",
            "-o",
            help="Output video file path (e.g., output.mp4)",
            rich_help_panel="Output Options",
        ),
    ],
    codec: Annotated[
        VideoCodec,
        typer.Option(
            "--codec",
            case_sensitive=False,
            help="Video codec",
            rich_help_panel="Encoding Options",
            show_default=True,
        ),
    ] = VideoCodec.H264,
    quality: Annotated[
        QualityPreset,
        typer.Option(
            "--quality",
            help="Quality preset (ignored if --crf provided)",
            rich_help_panel="Encoding Options",
            show_default=True,
        ),
    ] = QualityPreset.MEDIUM,
    crf: Annotated[
        int | None,
        typer.Option(
            "--crf",
            min=0,
            max=51,
            help="Manual CRF value (lower = better) overrides --quality",
            rich_help_panel="Encoding Options",
        ),
    ] = None,
    encoder: Annotated[
        EncoderBackend,
        typer.Option(
            "--encoder",
            help="Encoder backend (auto/software/videotoolbox/nvenc/vaapi)",
            rich_help_panel="Encoding Options",
            show_default=True,
        ),
    ] = EncoderBackend.AUTO,
    force: Annotated[
        bool,
        typer.Option(
            "-f",
            "--force",
            help="Force overwrite of output file",
            rich_help_panel="Output Options",
            show_default=True,
        ),
    ] = False,
) -> None:
    """Encode video from image topics in an MCAP file."""
    if not output.parent.exists():
        console.print(f"[red]Error:[/red] Output directory not found: {output.parent}")
        raise typer.Exit(1)
    confirm_output_overwrite(output, force)
    quality_value = crf if crf is not None else QUALITY_PRESETS[codec][quality]

    try:
        encode_video(
            mcap_path=file,
            topics=topics,
            output_path=output,
            codec=codec,
            encoder_backend=encoder,
            quality=quality_value,
        )
    except VideoEncoderError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
