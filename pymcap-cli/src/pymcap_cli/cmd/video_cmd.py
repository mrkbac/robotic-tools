"""Video encoding command for pymcap-cli."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import platform
import shutil
import subprocess
import threading
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, cast

import typer
from mcap_ros2_support_fast.decoder import DecoderFactory
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from small_mcap import get_summary, include_topics, read_message_decoded

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer()

# Supported image message schemas
COMPRESSED_SCHEMAS = {"sensor_msgs/msg/CompressedImage", "sensor_msgs/CompressedImage"}
RAW_SCHEMAS = {"sensor_msgs/msg/Image", "sensor_msgs/Image"}
IMAGE_SCHEMAS = COMPRESSED_SCHEMAS | RAW_SCHEMAS


@dataclass(slots=True)
class TopicStreamSpec:
    topic: str
    message_type: ImageType
    width: int
    height: int
    fps: float
    message_count: int


class ImageType(Enum):
    COMPRESSED = auto()
    RAW = auto()


class VideoCodec(Enum):
    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    AV1 = "av1"


class QualityPreset(str, Enum):
    """Quality preset for video encoding."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EncoderBackend(str, Enum):
    """Encoder backend selection."""

    AUTO = "auto"
    SOFTWARE = "software"
    VIDEOTOOLBOX = "videotoolbox"
    NVENC = "nvenc"
    VAAPI = "vaapi"


class HardwareBackend(Enum):
    VIDEOTOOLBOX = "videotoolbox"
    NVENC = "nvenc"
    VAAPI = "vaapi"


class VideoEncoderError(Exception):
    """Exception raised when video encoding fails."""


# Codec to software encoder mapping
CODEC_SOFTWARE_ENCODERS = {
    VideoCodec.H264: "libx264",
    VideoCodec.H265: "libx265",
    VideoCodec.VP9: "libvpx-vp9",
    VideoCodec.AV1: "libsvtav1",  # Faster than libaom-av1
}

# Hardware encoder mapping by codec and platform/backend
HARDWARE_ENCODERS = {
    VideoCodec.H264: {
        HardwareBackend.VIDEOTOOLBOX: "h264_videotoolbox",  # macOS
        HardwareBackend.NVENC: "h264_nvenc",  # NVIDIA
        HardwareBackend.VAAPI: "h264_vaapi",  # Intel/AMD on Linux
    },
    VideoCodec.H265: {
        HardwareBackend.VIDEOTOOLBOX: "hevc_videotoolbox",
        HardwareBackend.NVENC: "hevc_nvenc",
        HardwareBackend.VAAPI: "hevc_vaapi",
    },
}

# Quality presets (CRF values)
QUALITY_PRESETS = {
    VideoCodec.H264: {"high": 18, "medium": 23, "low": 28},
    VideoCodec.H265: {"high": 20, "medium": 25, "low": 30},
    VideoCodec.VP9: {"high": 30, "medium": 35, "low": 40},
    VideoCodec.AV1: {"high": 25, "medium": 30, "low": 35},
}


def _determine_layout(topic_count: int) -> tuple[int, int]:
    if topic_count <= 1:
        return (1, 1)
    if topic_count == 2:
        return (1, 2)
    rows = math.ceil(math.sqrt(topic_count))
    cols = math.ceil(topic_count / rows)
    return (rows, cols)


def _build_filter_complex(
    layout: tuple[int, int],
    stream_count: int,
    target_width: int,
    target_height: int,
    topic_names: list[str] | None = None,
) -> str:
    cols = layout[1]

    # Single topic case
    if stream_count == 1:
        filter_str = f"[0:v]scale={target_width}:{target_height}"
        if topic_names:
            drawtext = _build_drawtext_filter(topic_names[0], target_height)
            filter_str += f",{drawtext}"
        filter_str += "[out]"
        return filter_str

    # Multi-topic case: scale and add watermark to each stream
    filters = []
    for idx in range(stream_count):
        filter_chain = f"[{idx}:v]scale={target_width}:{target_height}"
        if topic_names and idx < len(topic_names):
            drawtext = _build_drawtext_filter(topic_names[idx], target_height)
            filter_chain += f",{drawtext}"
        filter_chain += f"[s{idx}]"
        filters.append(filter_chain)

    layout_entries = [
        f"{(idx % cols) * target_width}_{(idx // cols) * target_height}"
        for idx in range(stream_count)
    ]

    stacked_inputs = "".join(f"[s{idx}]" for idx in range(stream_count))
    layout_str = "|".join(layout_entries)
    filters.append(
        f"{stacked_inputs}xstack=inputs={stream_count}:layout={layout_str}:fill=black[out]"
    )
    return ";".join(filters)


def _collect_topic_stream_spec(mcap_path: Path, topic: str) -> TopicStreamSpec:
    message_type, message_count = _scan_mcap(mcap_path, topic)

    if message_count == 0:
        raise VideoEncoderError(f"No messages found for topic: {topic}")

    decoder_factory = DecoderFactory()
    width: int | None = None
    height: int | None = None
    first_timestamp: int | None = None
    last_timestamp: int | None = None
    fps: float | None = None
    frames_sampled = 0
    max_samples = min(10, message_count)

    with mcap_path.open("rb") as handle:
        for msg in read_message_decoded(
            handle,
            should_include=include_topics(topic),
            decoder_factories=[decoder_factory],
        ):
            # Track timestamps for FPS calculation
            if first_timestamp is None:
                first_timestamp = msg.message.log_time
            last_timestamp = msg.message.log_time
            frames_sampled += 1

            # Get dimensions from first frame
            if width is None or height is None:
                if message_type == ImageType.COMPRESSED:
                    frame_data = _process_compressed_image(msg.decoded_message)
                    width, height = _get_compressed_dimensions(frame_data)
                else:
                    frame_data, img_width, img_height = _process_raw_image(msg.decoded_message)
                    width = img_width
                    height = img_height

            # Stop after sampling enough frames and getting dimensions
            if width is not None and height is not None and frames_sampled >= max_samples:
                break

    if width is None or height is None:
        raise VideoEncoderError(f"Failed to determine dimensions for topic: {topic}")

    # Calculate FPS from sampled frames
    if (
        first_timestamp is not None
        and last_timestamp is not None
        and frames_sampled > 1
        and last_timestamp > first_timestamp
    ):
        duration_s = (last_timestamp - first_timestamp) / 1e9
        fps = (frames_sampled - 1) / duration_s
    else:
        fps = 1.0

    return TopicStreamSpec(
        topic=topic,
        message_type=message_type,
        width=width,
        height=height,
        fps=fps,
        message_count=message_count,
    )


def _start_multi_topic_ffmpeg(
    topic_specs: Sequence[TopicStreamSpec],
    layout: tuple[int, int],
    output_path: Path,
    codec: VideoCodec,
    encoder: str,
    quality: int,
    enable_watermarks: bool,
    target_width: int,
    target_height: int,
    target_fps: float,
) -> tuple[subprocess.Popen[bytes], list[BinaryIO]]:
    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    read_fds: list[int] = []
    write_fds: list[int] = []

    for spec in topic_specs:
        read_fd, write_fd = os.pipe()
        read_fds.append(read_fd)
        write_fds.append(write_fd)

        if spec.message_type == ImageType.COMPRESSED:
            cmd.extend(
                [
                    "-f",
                    "image2pipe",
                    "-framerate",
                    f"{spec.fps:.6f}",
                    "-i",
                    f"pipe:{read_fd}",
                ]
            )
        else:
            cmd.extend(
                [
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{spec.width}x{spec.height}",
                    "-framerate",
                    f"{spec.fps:.6f}",
                    "-i",
                    f"pipe:{read_fd}",
                ]
            )

    # Build filter complex with per-topic watermarks if enabled
    topic_names = [spec.topic for spec in topic_specs] if enable_watermarks else None
    filter_str = _build_filter_complex(
        layout, len(topic_specs), target_width, target_height, topic_names
    )

    cmd.extend(["-filter_complex", filter_str, "-map", "[out]"])
    cmd.extend(["-c:v", encoder])
    cmd.extend(_get_encoder_quality_params(codec, encoder, quality))
    if target_fps > 0:
        cmd.extend(["-r", f"{target_fps:.6f}"])
    cmd.extend(["-pix_fmt", "yuv420p", str(output_path)])

    try:
        process = subprocess.Popen(  # noqa: S603
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            pass_fds=tuple(read_fds),
        )
    except FileNotFoundError as exc:
        for fd in read_fds + write_fds:
            os.close(fd)
        raise VideoEncoderError("ffmpeg not found") from exc
    except Exception:
        for fd in read_fds + write_fds:
            os.close(fd)
        raise

    for fd in read_fds:
        os.close(fd)

    writers = [cast("BinaryIO", os.fdopen(fd, "wb", buffering=0)) for fd in write_fds]
    return process, writers


def _stream_topic_worker(
    spec: TopicStreamSpec,
    mcap_path: Path,
    writer: BinaryIO,
    error_bucket: list[tuple[str, Exception]],
    error_lock: threading.Lock,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> None:
    decoder_factory = DecoderFactory()
    try:
        with mcap_path.open("rb") as handle:
            for msg in read_message_decoded(
                handle,
                should_include=include_topics(spec.topic),
                decoder_factories=[decoder_factory],
            ):
                if spec.message_type == ImageType.COMPRESSED:
                    frame_data = _process_compressed_image(msg.decoded_message)
                else:
                    frame_data, _, _ = _process_raw_image(msg.decoded_message)
                writer.write(frame_data)

                # Update progress if available
                if progress is not None and task_id is not None:
                    progress.update(task_id, advance=1)
    except Exception as exc:  # noqa: BLE001
        with error_lock:
            error_bucket.append((spec.topic, exc))
    finally:
        with contextlib.suppress(Exception):
            writer.close()


# VideoToolbox bitrate calculation constants
# Formula: bitrate = BASE_BITRATE * 2^(-(crf - CRF_REFERENCE) / CRF_SCALE)
VIDEOTOOLBOX_BASE_BITRATE = 20  # Mbps at CRF 18
VIDEOTOOLBOX_CRF_REFERENCE = 10
VIDEOTOOLBOX_CRF_SCALE = 5


class FPSColumn(ProgressColumn):
    """Custom column to display encoding frames per second."""

    def render(self, task: Task) -> Text:
        """Render the FPS."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("-- fps/s", style="progress.data.speed")
        return Text(f"{speed:.0f} fps/s", style="progress.data.speed")


def _test_encoder(encoder_name: str) -> bool:
    """Test if an ffmpeg encoder is available.

    Args:
        encoder_name: Name of the encoder to test (e.g., "h264_videotoolbox")

    Returns:
        True if encoder is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    else:
        return encoder_name in result.stdout


def _detect_encoder(
    codec: VideoCodec, encoder_preference: Literal["auto", "software"] | HardwareBackend
) -> str:
    """Detect the best available encoder for the given codec.

    Args:
        codec: Video codec
        encoder_preference: User preference (auto, software, or HardwareBackend enum)

    Returns:
        Name of the encoder to use

    Raises:
        VideoEncoderError: If no suitable encoder is found
    """
    # If user explicitly requested software, use it
    if encoder_preference == "software":
        encoder = CODEC_SOFTWARE_ENCODERS.get(codec)
        if not encoder:
            raise VideoEncoderError(f"No software encoder available for codec: {codec.value}")
        return encoder

    # If user requested specific hardware encoder
    if isinstance(encoder_preference, HardwareBackend):
        hw_encoders = HARDWARE_ENCODERS.get(codec, {})
        encoder = hw_encoders.get(encoder_preference)
        if not encoder:
            msg = (
                f"Hardware encoder '{encoder_preference.value}' "
                f"not available for codec: {codec.value}"
            )
            raise VideoEncoderError(msg)
        if not _test_encoder(encoder):
            raise VideoEncoderError(
                f"Hardware encoder '{encoder}' not available on this system. Try --encoder software"
            )
        return encoder

    # Auto mode: try hardware encoders based on platform, fall back to software
    if encoder_preference == "auto":
        system = platform.system()
        hw_encoders = HARDWARE_ENCODERS.get(codec, {})

        # Try platform-specific hardware encoders
        if system == "Darwin" and HardwareBackend.VIDEOTOOLBOX in hw_encoders:
            encoder = hw_encoders[HardwareBackend.VIDEOTOOLBOX]
            if _test_encoder(encoder):
                logger.info("Using hardware encoder: %s", encoder)
                return encoder

        elif system == "Linux":
            # Try NVENC first (NVIDIA), then VAAPI (Intel/AMD)
            for hw_backend in (HardwareBackend.NVENC, HardwareBackend.VAAPI):
                if hw_backend in hw_encoders:
                    encoder = hw_encoders[hw_backend]
                    if _test_encoder(encoder):
                        logger.info("Using hardware encoder: %s", encoder)
                        return encoder

        # Fall back to software encoder
        encoder = CODEC_SOFTWARE_ENCODERS.get(codec)
        if not encoder:
            raise VideoEncoderError(f"No encoder available for codec: {codec.value}")
        logger.info("Using software encoder: %s", encoder)
        return encoder

    raise VideoEncoderError(f"Unknown encoder preference: {encoder_preference}")


def _get_encoder_quality_params(
    codec: VideoCodec,
    encoder: str,
    quality: int,
) -> list[str]:
    """Get encoder-specific quality parameters.

    Args:
        codec: Video codec
        encoder: Encoder name (e.g., "h264_videotoolbox", "libx264")
        quality: Quality value (CRF value: lower = better quality)

    Returns:
        List of ffmpeg arguments for quality settings
    """
    params: list[str] = []

    # Detect encoder type by name
    if "videotoolbox" in encoder:
        # VideoToolbox (macOS hardware encoder)
        # VideoToolbox uses bitrate control, not quality. Use -b:v instead.
        # Map CRF-like values to bitrate (Mbps):
        # CRF 18 (high) -> ~50 Mbps, CRF 23 (medium) -> ~18 Mbps, CRF 28 (low) -> ~6 Mbps
        bitrate_mbps = VIDEOTOOLBOX_BASE_BITRATE * math.pow(
            2, -(quality - VIDEOTOOLBOX_CRF_REFERENCE) / VIDEOTOOLBOX_CRF_SCALE
        )
        bitrate_str = f"{bitrate_mbps:.1f}M"
        params.extend(["-b:v", bitrate_str])
    elif "nvenc" in encoder:
        # NVENC (NVIDIA hardware encoder)
        params.extend(["-preset", "p4", "-cq", str(quality)])  # p4 = medium preset
    elif "vaapi" in encoder:
        # VAAPI (Intel/AMD hardware encoder)
        params.extend(["-qp", str(quality)])
    # Software encoders
    elif codec in (VideoCodec.H264, VideoCodec.H265):
        params.extend(["-preset", "medium", "-crf", str(quality)])
    elif codec == VideoCodec.VP9:
        params.extend(["-crf", str(quality), "-b:v", "0"])  # Constant quality mode
    elif codec == VideoCodec.AV1:
        params.extend(["-crf", str(quality)])

    return params


def _scan_mcap(mcap_path: Path, topic: str) -> tuple[ImageType, int]:
    with mcap_path.open("rb") as f:
        summary = get_summary(f)

    if summary is None:
        raise VideoEncoderError("MCAP file has no summary, try rebuilding it")

    # Find all image schemas and their channels
    image_schema_ids: set[int] = {
        schema.id for schema in summary.schemas.values() if schema.name in IMAGE_SCHEMAS
    }
    compressed_schema_ids: set[int] = {
        schema.id for schema in summary.schemas.values() if schema.name in COMPRESSED_SCHEMAS
    }

    image_channels = [
        channel for channel in summary.channels.values() if channel.schema_id in image_schema_ids
    ]
    image_topics = {channel.topic for channel in image_channels}

    if topic not in image_topics:
        available = "\n".join(f"  - {t}" for t in sorted(image_topics))
        msg = (
            f"Topic '{topic}' not found or is not an image topic.\n\n"
            f"Available image topics:\n{available}"
        )
        raise VideoEncoderError(msg)

    channel = next(channel for channel in image_channels if channel.topic == topic)
    message_type = (
        ImageType.COMPRESSED if channel.schema_id in compressed_schema_ids else ImageType.RAW
    )

    if not summary.statistics:
        raise VideoEncoderError("MCAP file has no statistics, try rebuilding it")

    message_count = summary.statistics.channel_message_counts.get(channel.id, 0)

    return message_type, message_count


def _process_compressed_image(message: Any) -> bytes:
    """Extract compressed image data from CompressedImage message.

    Args:
        message: Decoded CompressedImage message

    Returns:
        Compressed image bytes (JPEG, PNG, etc.)

    Raises:
        VideoEncoderError: If data is invalid
    """
    if not hasattr(message, "data") or not message.data:
        raise VideoEncoderError("CompressedImage has no data")

    return bytes(message.data)


def _process_raw_image(message: Any) -> tuple[bytes, int, int]:
    """Convert raw Image message to RGB24 bytes.

    Args:
        message: Decoded Image message

    Returns:
        Tuple of (rgb24_bytes, width, height)

    Raises:
        VideoEncoderError: If conversion fails
    """
    if not hasattr(message, "data") or not message.data:
        raise VideoEncoderError("Image has no data")
    if not hasattr(message, "width") or not hasattr(message, "height"):
        raise VideoEncoderError("Image missing width/height")
    if not hasattr(message, "encoding"):
        raise VideoEncoderError("Image missing encoding")

    width = message.width
    height = message.height
    encoding = message.encoding

    # Convert data to bytes
    data = bytes(message.data) if isinstance(message.data, list) else message.data

    # Convert to RGB24 based on encoding
    if encoding in ("rgb8", "rgb"):
        return data, width, height
    if encoding in ("bgr8", "bgr"):
        # Convert BGR to RGB
        rgb_data = bytearray(len(data))
        for i in range(0, len(data), 3):
            rgb_data[i] = data[i + 2]  # R
            rgb_data[i + 1] = data[i + 1]  # G
            rgb_data[i + 2] = data[i]  # B
        return bytes(rgb_data), width, height
    if encoding in ("mono8", "8UC1", "mono"):
        # Convert grayscale to RGB by duplicating channels
        rgb_data = bytearray(len(data) * 3)
        for i, pixel in enumerate(data):
            rgb_data[i * 3] = pixel
            rgb_data[i * 3 + 1] = pixel
            rgb_data[i * 3 + 2] = pixel
        return bytes(rgb_data), width, height
    raise VideoEncoderError(f"Unsupported image encoding: {encoding}")


def _get_compressed_dimensions(compressed_data: bytes) -> tuple[int, int]:
    """Get image dimensions from compressed image data using ffprobe.

    Args:
        compressed_data: Compressed image bytes

    Returns:
        Tuple of (width, height)

    Raises:
        VideoEncoderError: If dimensions cannot be determined
    """
    try:
        result = subprocess.run(
            [  # noqa: S607
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                "-",
            ],
            input=compressed_data,
            capture_output=True,
            timeout=5,
            check=True,
        )
        width_str, height_str = result.stdout.decode().strip().split("x")
        return int(width_str), int(height_str)
    except Exception as e:
        raise VideoEncoderError(f"Failed to get image dimensions: {e}") from e


def _build_drawtext_filter(watermark_text: str, frame_height: int) -> str:
    escaped_text = (
        watermark_text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\\\\\''")
    )
    font_size = max(12, int(frame_height * 0.025))
    return (
        f"drawtext=text='{escaped_text}':"
        f"x=10:y=10:"
        f"fontsize={font_size}:"
        f"fontcolor=white:"
        f"box=1:"
        f"boxcolor=black@0.5:"
        f"boxborderw=5"
    )


def encode_video(
    mcap_path: Path,
    topics: list[str],
    output_path: Path,
    codec: VideoCodec,
    encoder_preference: Literal["auto", "software"] | HardwareBackend,
    quality: int,
    watermark: bool,
) -> None:
    """Encode image topics from MCAP to video.

    Args:
        mcap_path: Path to input MCAP file
        topics: List of topic names to extract images from (1+ topics)
        output_path: Path to output video file
        codec: Video codec
        encoder_preference: Encoder backend preference ("auto", "software", or HardwareBackend)
        quality: Quality value (CRF or equivalent)
        watermark: Whether to add per-topic watermarks showing topic names

    Raises:
        VideoEncoderError: If encoding fails
    """
    if len(topics) < 1:
        raise VideoEncoderError("At least one topic is required")

    console.print(f"[cyan]Reading MCAP file:[/cyan] {mcap_path}")
    console.print(f"[cyan]Topics:[/cyan] {', '.join(topics)}")
    console.print(f"[cyan]Output:[/cyan] {output_path}")
    console.print(f"[cyan]Codec:[/cyan] {codec.value}")

    encoder = _detect_encoder(codec, encoder_preference)
    console.print(f"[cyan]Encoder:[/cyan] {encoder}")

    layout = _determine_layout(len(topics))
    console.print(f"[green]✓[/green] Layout: {layout[0]} row(s) x {layout[1]} column(s)")

    console.print("\n[yellow]Collecting topic metadata...[/yellow]")
    topic_specs: list[TopicStreamSpec] = []
    for topic in topics:
        spec = _collect_topic_stream_spec(mcap_path, topic)
        topic_specs.append(spec)
        console.print(
            f"[green]✓[/green] {topic}: {spec.width}x{spec.height} @ {spec.fps:.2f} fps"
            f" ({spec.message_count:,} frames)"
        )

    target_width = max(2, min(spec.width for spec in topic_specs))
    target_height = max(2, min(spec.height for spec in topic_specs))
    target_width -= target_width % 2
    target_height -= target_height % 2
    target_fps = max((spec.fps for spec in topic_specs), default=1.0)

    console.print(
        f"\n[yellow]Encoding video at {target_width}x{target_height}"
        f" @ {target_fps:.2f} fps...[/yellow]"
    )

    ffmpeg_process, writers = _start_multi_topic_ffmpeg(
        topic_specs=topic_specs,
        layout=layout,
        output_path=output_path,
        codec=codec,
        encoder=encoder,
        quality=quality,
        enable_watermarks=watermark,
        target_width=target_width,
        target_height=target_height,
        target_fps=target_fps,
    )

    error_lock = threading.Lock()
    errors: list[tuple[str, Exception]] = []
    threads: list[threading.Thread] = []

    # Create progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        FPSColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        # Create progress tasks for each topic
        task_ids = []
        for spec in topic_specs:
            task_id = progress.add_task(
                f"Encoding {spec.topic}",
                total=spec.message_count,
            )
            task_ids.append(task_id)

        # Start worker threads with progress tracking
        for spec, writer, task_id in zip(topic_specs, writers, task_ids, strict=True):
            thread = threading.Thread(
                target=_stream_topic_worker,
                name=f"pymcap-{spec.topic}",
                args=(spec, mcap_path, writer, errors, error_lock, progress, task_id),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    if errors:
        ffmpeg_process.terminate()
        stderr = ""
        if ffmpeg_process.stderr:
            stderr = ffmpeg_process.stderr.read().decode("utf-8", errors="ignore")
        ffmpeg_process.wait()
        failed_topic, exc = errors[0]
        raise VideoEncoderError(
            f"Failed to stream topic '{failed_topic}': {exc}\n{stderr}"
        ) from exc

    return_code = ffmpeg_process.wait()
    if return_code != 0:
        stderr = ""
        if ffmpeg_process.stderr:
            stderr = ffmpeg_process.stderr.read().decode("utf-8", errors="ignore")
        raise VideoEncoderError(f"ffmpeg encoding failed:\n{stderr}")

    console.print(f"\n[green bold]✓ Video created:[/green bold] {output_path}")


@app.command()
def video(
    file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Path to the MCAP file",
    ),
    topics: list[str] = typer.Option(
        ...,
        "--topic",
        "-t",
        help="Image topic to convert (repeat for multiple topics)",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output video file path (e.g., output.mp4)",
    ),
    codec: VideoCodec = typer.Option(
        VideoCodec.H264,
        "--codec",
        case_sensitive=False,
        help="Video codec to use",
    ),
    quality: QualityPreset = typer.Option(
        QualityPreset.MEDIUM,
        "--quality",
        help="Quality preset",
    ),
    crf: int | None = typer.Option(
        None,
        "--crf",
        min=0,
        max=51,
        help="Manual CRF/quality value (lower = better quality, overrides --quality)",
    ),
    encoder: EncoderBackend = typer.Option(
        EncoderBackend.AUTO,
        "--encoder",
        help="Encoder backend",
    ),
    watermark: bool = typer.Option(
        False,
        "--watermark",
        help="Enable per-topic watermarks showing topic names",
    ),
) -> None:
    """Generate video from image topics in MCAP files.

    Generate MP4 video from topics using ffmpeg.
    """
    # Validate input file exists
    if not file.exists():
        console.print(f"[red]Error:[/red] Input file not found: {file}")
        raise typer.Exit(1)

    # Validate output directory exists
    if not output.parent.exists():
        console.print(f"[red]Error:[/red] Output directory not found: {output.parent}")
        raise typer.Exit(1)

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "ffmpeg or ffprobe not found. Please install ffmpeg:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: Download from https://ffmpeg.org/"
        )

    # Convert encoder preference to the appropriate type
    encoder_pref: Literal["auto", "software"] | HardwareBackend
    if encoder.value in ("auto", "software"):
        encoder_pref = encoder.value  # type: ignore[assignment]
    else:
        # Map to HardwareBackend enum
        encoder_pref = HardwareBackend(encoder.value)

    # Determine quality value
    quality_value = crf if crf is not None else QUALITY_PRESETS[codec][quality.value]

    try:
        encode_video(
            mcap_path=file,
            topics=topics,
            output_path=output,
            codec=codec,
            encoder_preference=encoder_pref,
            quality=quality_value,
            watermark=watermark,
        )
    except VideoEncoderError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e
