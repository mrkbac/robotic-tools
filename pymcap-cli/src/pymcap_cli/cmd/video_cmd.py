from __future__ import annotations

import logging
import math
import platform
import shutil
import subprocess
import sys
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)
console = Console()

# Supported image message schemas
COMPRESSED_SCHEMAS = {"sensor_msgs/msg/CompressedImage", "sensor_msgs/CompressedImage"}
RAW_SCHEMAS = {"sensor_msgs/msg/Image", "sensor_msgs/Image"}
IMAGE_SCHEMAS = COMPRESSED_SCHEMAS | RAW_SCHEMAS


class ImageType(Enum):
    COMPRESSED = auto()
    RAW = auto()


class VideoCodec(Enum):
    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    AV1 = "av1"


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

# VideoToolbox bitrate calculation constants
# Formula: bitrate = BASE_BITRATE * 2^(-(crf - CRF_REFERENCE) / CRF_SCALE)
VIDEOTOOLBOX_BASE_BITRATE = 20  # Mbps at CRF 18
VIDEOTOOLBOX_CRF_REFERENCE = 10
VIDEOTOOLBOX_CRF_SCALE = 5


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


def _scan_mcap(mcap_path: Path, topic: str) -> tuple[ImageType, int, float]:
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
    # TODO: improve FPS calculation
    duration_ns = summary.statistics.message_end_time - summary.statistics.message_start_time
    fps = message_count / (duration_ns / 1e9) if duration_ns > 0 else 0.0

    return message_type, message_count, fps


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


def _start_ffmpeg(
    output_path: Path,
    width: int,
    height: int,
    message_type: ImageType,
    codec: VideoCodec,
    encoder: str,
    quality: int,
    fps: float,
) -> subprocess.Popen[bytes]:
    cmd: list[str] = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    # Input format differs based on message type
    if message_type == ImageType.COMPRESSED:
        # Read compressed images (JPEG/PNG) from stdin
        # Let ffmpeg auto-detect the image format
        cmd.extend(
            [
                "-f",
                "image2pipe",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
            ]
        )
    else:  # Raw Image
        # Read raw RGB24 frames from stdin
        cmd.extend(
            [
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
            ]
        )

    # Output encoder
    cmd.extend(["-c:v", encoder])

    # Add encoder-specific quality parameters
    quality_params = _get_encoder_quality_params(codec, encoder, quality)
    cmd.extend(quality_params)

    # Pixel format for compatibility
    cmd.extend(["-pix_fmt", "yuv420p"])

    # Output file
    cmd.append(str(output_path))

    try:
        return subprocess.Popen(  # noqa: S603
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise VideoEncoderError("ffmpeg not found") from e
    except Exception as e:
        raise VideoEncoderError(f"Failed to start ffmpeg: {e}") from e


def encode_video(
    mcap_path: Path,
    topic: str,
    output_path: Path,
    codec: VideoCodec = VideoCodec.H264,
    encoder_preference: Literal["auto", "software"] | HardwareBackend = "auto",
    quality: int = 23,
) -> None:
    """Encode an image topic from MCAP to video.

    Args:
        mcap_path: Path to input MCAP file
        topic: Topic name to extract images from
        output_path: Path to output video file
        codec: Video codec
        encoder_preference: Encoder backend preference ("auto", "software", or HardwareBackend)
        quality: Quality value (CRF or equivalent)

    Raises:
        VideoEncoderError: If encoding fails
    """
    console.print(f"[cyan]Reading MCAP file:[/cyan] {mcap_path}")
    console.print(f"[cyan]Topic:[/cyan] {topic}")
    console.print(f"[cyan]Output:[/cyan] {output_path}")
    console.print(f"[cyan]Codec:[/cyan] {codec.value}")

    # Detect encoder
    encoder = _detect_encoder(codec, encoder_preference)
    console.print(f"[cyan]Encoder:[/cyan] {encoder}")

    message_type, message_count, fps = _scan_mcap(mcap_path, topic)

    if message_count == 0:
        raise VideoEncoderError(f"No messages found for topic: {topic}")

    console.print(f"[green]✓[/green] Found {message_count:,} messages")
    console.print(f"[green]✓[/green] Message type: {message_type}")

    # Encode video
    console.print("\n[yellow]Encoding video...[/yellow]")
    _encode_frames(
        mcap_path=mcap_path,
        topic=topic,
        output_path=output_path,
        message_type=message_type,
        message_count=message_count,
        codec=codec,
        encoder=encoder,
        quality=quality,
        fps=fps,
    )

    console.print(f"\n[green bold]✓ Video created:[/green bold] {output_path}")


def _encode_frames(
    mcap_path: Path,
    topic: str,
    output_path: Path,
    message_type: ImageType,
    message_count: int,
    codec: VideoCodec,
    encoder: str,
    quality: int,
    fps: float,
) -> None:
    """Extract frames from MCAP and encode to video with accurate timing.

    Args:
        mcap_path: Path to MCAP file
        topic: Topic to extract
        output_path: Output video path
        message_type: Type of image message (compressed or raw)
        message_count: Expected number of messages for progress tracking
        codec: Video codec
        encoder: Encoder name
        quality: Quality value
        fps: Frames per second

    Raises:
        VideoEncoderError: If encoding fails
    """
    decoder_factory = DecoderFactory()

    # Start ffmpeg process
    ffmpeg_process = None
    frames_written = 0
    width: int | None = None
    height: int | None = None
    fps_pts: list[int] = []  # List of presentation timestamps for FPS calculation

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Encoding video...", total=message_count)

        with mcap_path.open("rb") as f:
            for msg in read_message_decoded(
                f,
                should_include=include_topics(topic),
                decoder_factories=[decoder_factory],
            ):
                # Extract frame data based on message type
                if message_type == ImageType.COMPRESSED:
                    frame_data = _process_compressed_image(msg.decoded_message)
                    # For compressed images, we need to get dimensions from first frame
                    if width is None or height is None:
                        width, height = _get_compressed_dimensions(frame_data)
                else:  # Image
                    frame_data, img_width, img_height = _process_raw_image(msg.decoded_message)
                    if width is None or height is None:
                        width = img_width
                        height = img_height

                # Start ffmpeg on first frame
                if ffmpeg_process is None:
                    if width is None or height is None:
                        raise VideoEncoderError("Failed to determine image dimensions")
                    ffmpeg_process = _start_ffmpeg(
                        output_path=output_path,
                        width=width,
                        height=height,
                        message_type=message_type,
                        codec=codec,
                        encoder=encoder,
                        quality=quality,
                        fps=fps,
                    )

                # Write frame
                assert ffmpeg_process.stdin is not None
                ffmpeg_process.stdin.write(frame_data)
                frames_written += 1

                # Track timestamp for FPS calculation
                fps_pts.append(msg.message.log_time)

                progress.update(task, advance=1)

    # Close ffmpeg
    if ffmpeg_process:
        if ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()
        return_code = ffmpeg_process.wait()

        if return_code != 0 and ffmpeg_process.stderr:
            stderr = ffmpeg_process.stderr.read().decode("utf-8", errors="ignore")
            raise VideoEncoderError(f"ffmpeg encoding failed:\n{stderr}")

    # Calculate and display statistics
    if frames_written > 0 and len(fps_pts) >= 2:
        duration_ns = fps_pts[-1] - fps_pts[0]
        duration_s = duration_ns / 1e9
        avg_fps = (frames_written - 1) / duration_s if duration_s > 0 else 0
        console.print(f"[dim]Frames written: {frames_written:,}")
        console.print(f"[dim]Duration: {duration_s:.2f}s")
        console.print(f"[dim]Average FPS: {avg_fps:.2f}[/dim]")


def add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Add the video command parser to the subparsers."""
    parser = subparsers.add_parser(
        "video",
        help="Generate video from image topics in MCAP files",
        description="Generate MP4 video from topics using ffmpeg",
    )

    parser.add_argument(
        "file",
        help="Path to the MCAP file",
        type=str,
    )

    parser.add_argument(
        "--topic",
        required=True,
        help="Image topic to convert (e.g., /camera/front)",
        type=str,
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output video file path (e.g., output.mp4)",
        type=str,
    )

    parser.add_argument(
        "--codec",
        choices=[c.value for c in VideoCodec],
        default=VideoCodec.H264.value,
        help="Video codec to use (default: h264)",
    )

    # Quality options (mutually exclusive)
    quality_group = parser.add_mutually_exclusive_group()
    quality_group.add_argument(
        "--quality",
        choices=["high", "medium", "low"],
        default="medium",
        help="Quality preset (default: medium)",
    )
    quality_group.add_argument(
        "--crf",
        type=int,
        metavar="<0-51>",
        help="Manual CRF/quality value (lower = better quality, overrides --quality)",
    )

    parser.add_argument(
        "--encoder",
        choices=["auto", "software"] + [b.value for b in HardwareBackend],
        default="auto",
        help="Encoder backend (default: auto - detect hardware, fallback to software)",
    )

    return parser


def handle_command(args: argparse.Namespace) -> None:
    """Handle the video command execution."""
    input_file = Path(args.file)
    output_file = Path(args.output)

    # Validate input file exists
    if not input_file.exists():
        console.print(f"[red]Error:[/red] Input file not found: {input_file}")
        sys.exit(1)

    # Validate output directory exists
    if not output_file.parent.exists():
        console.print(f"[red]Error:[/red] Output directory not found: {output_file.parent}")
        sys.exit(1)

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "ffmpeg or ffprobe not found. Please install ffmpeg:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: Download from https://ffmpeg.org/"
        )

    # Convert codec string to enum
    codec = VideoCodec(args.codec)

    # Convert encoder preference to enum if it's a hardware backend
    encoder_pref: Literal["auto", "software"] | HardwareBackend
    if args.encoder in ("auto", "software"):
        encoder_pref = args.encoder
    else:
        encoder_pref = HardwareBackend(args.encoder)

    # Determine quality value
    quality_value = args.crf if args.crf is not None else QUALITY_PRESETS[codec][args.quality]

    try:
        encode_video(
            mcap_path=input_file,
            topic=args.topic,
            output_path=output_file,
            codec=codec,
            encoder_preference=encoder_pref,
            quality=quality_value,
        )
    except VideoEncoderError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
