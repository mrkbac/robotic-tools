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
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, BinaryIO, Literal, cast

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
from small_mcap import include_topics, read_message_decoded
from small_mcap.rebuild import RebuildInfo, rebuild_summary

from pymcap_cli.autocompletion import complete_topic_by_schema
from pymcap_cli.mcap_processor import confirm_output_overwrite

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer()

# Supported image message schemas
COMPRESSED_SCHEMAS = {"sensor_msgs/msg/CompressedImage", "sensor_msgs/CompressedImage"}
RAW_SCHEMAS = {"sensor_msgs/msg/Image", "sensor_msgs/Image"}
IMAGE_SCHEMAS = COMPRESSED_SCHEMAS | RAW_SCHEMAS


def complete_image_topics(ctx: typer.Context, incomplete: str) -> list[str]:
    """Autocomplete function for image topics."""
    return complete_topic_by_schema(ctx, incomplete, schemas=IMAGE_SCHEMAS)


@dataclass(slots=True)
class TopicStreamSpec:
    topic: str
    message_type: ImageType
    width: int
    height: int
    fps: float
    message_count: int
    first_timestamp: int  # Nanoseconds since epoch
    last_timestamp: int  # Nanoseconds since epoch


class ImageType(Enum):
    COMPRESSED = auto()
    RAW = auto()


class VideoCodec(Enum):
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


class HardwareBackend(Enum):
    VIDEOTOOLBOX = "videotoolbox"
    NVENC = "nvenc"
    VAAPI = "vaapi"


class VideoEncoderError(Exception):
    """Exception raised when video encoding fails."""


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

QUALITY_PRESETS = {
    VideoCodec.H264: {"high": 18, "medium": 23, "low": 28},
    VideoCodec.H265: {"high": 20, "medium": 25, "low": 30},
    VideoCodec.VP9: {"high": 30, "medium": 35, "low": 40},
    VideoCodec.AV1: {"high": 25, "medium": 30, "low": 35},
}


def _build_filter_complex(
    layout: tuple[int, int],
    stream_count: int,
    target_width: int,
    target_height: int,
    topic_names: list[str] | None = None,
) -> str:
    cols = layout[1]

    def build_drawtext(text: str, height: int) -> str:
        escaped = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\\\\\''")
        font_size = max(12, int(height * 0.025))
        return (
            f"drawtext=text='{escaped}':x=10:y=10:fontsize={font_size}:"
            f"fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5"
        )

    # Single topic case
    if stream_count == 1:
        filter_str = f"[0:v]scale={target_width}:{target_height}"
        if topic_names:
            filter_str += f",{build_drawtext(topic_names[0], target_height)}"
        filter_str += "[out]"
        return filter_str

    # Multi-topic case: scale and add watermark to each stream
    filters = []
    for idx in range(stream_count):
        filter_chain = f"[{idx}:v]scale={target_width}:{target_height}"
        if topic_names and idx < len(topic_names):
            filter_chain += f",{build_drawtext(topic_names[idx], target_height)}"
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


def _extract_topic_metadata_from_rebuild(
    rebuild_info: RebuildInfo, topic: str
) -> tuple[int, ImageType, int, int, int]:
    """Extract topic metadata from rebuild info without decompressing chunks."""
    summary = rebuild_info.summary

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
        raise VideoEncoderError(
            f"Topic '{topic}' not found or is not an image topic.\n\n"
            f"Available image topics:\n{available}"
        )

    channel = next(channel for channel in image_channels if channel.topic == topic)
    channel_id = channel.id
    message_type = (
        ImageType.COMPRESSED if channel.schema_id in compressed_schema_ids else ImageType.RAW
    )

    # Extract timestamps and message count from MessageIndex records
    first_timestamp: int | None = None
    last_timestamp: int | None = None
    message_count = 0

    if rebuild_info.chunk_information:
        for message_indexes in rebuild_info.chunk_information.values():
            for msg_idx in message_indexes:
                if msg_idx.channel_id == channel_id:
                    for timestamp, _offset in msg_idx.records:
                        message_count += 1
                        if first_timestamp is None or timestamp < first_timestamp:
                            first_timestamp = timestamp
                        if last_timestamp is None or timestamp > last_timestamp:
                            last_timestamp = timestamp

    if message_count == 0:
        raise VideoEncoderError(f"No messages found for topic: {topic}")

    if first_timestamp is None or last_timestamp is None:
        raise VideoEncoderError(f"Failed to get timestamps for topic: {topic}")

    return channel_id, message_type, message_count, first_timestamp, last_timestamp


def _collect_topic_stream_spec(
    mcap_path: Path, topic: str, rebuild_info: RebuildInfo
) -> TopicStreamSpec:
    """Collect topic stream specifications with minimal file reads."""
    # Extract metadata from rebuild info (no file I/O, no decompression)
    _channel_id, message_type, message_count, first_timestamp, last_timestamp = (
        _extract_topic_metadata_from_rebuild(rebuild_info, topic)
    )

    # Sample only first ~10 messages to get dimensions
    decoder_factory = DecoderFactory()
    width: int | None = None
    height: int | None = None
    frames_sampled = 0
    max_samples = min(10, message_count)

    with mcap_path.open("rb") as handle:
        for msg in read_message_decoded(
            handle,
            should_include=include_topics(topic),
            decoder_factories=[decoder_factory],
        ):
            # Get dimensions from first frame
            if width is None or height is None:
                if message_type == ImageType.COMPRESSED:
                    frame_data = _process_compressed_image(msg.decoded_message)
                    width, height = _get_compressed_dimensions(frame_data)
                else:
                    frame_data, img_width, img_height = _process_raw_image(msg.decoded_message)
                    width = img_width
                    height = img_height

            frames_sampled += 1
            # Stop early once we have dimensions
            if width is not None and height is not None and frames_sampled >= max_samples:
                break

    if width is None or height is None:
        raise VideoEncoderError(f"Failed to determine dimensions for topic: {topic}")

    # Calculate FPS from full timestamp range and message count (more accurate than sampling)
    if message_count > 1 and last_timestamp > first_timestamp:
        duration_s = (last_timestamp - first_timestamp) / 1e9
        fps = (message_count - 1) / duration_s
    else:
        fps = 1.0

    return TopicStreamSpec(
        topic=topic,
        message_type=message_type,
        width=width,
        height=height,
        fps=fps,
        message_count=message_count,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
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
    enable_subtitles: bool,
    mcap_path: Path,
    topics: list[str],
    global_start_time: int,
    global_end_time: int,
) -> tuple[subprocess.Popen[bytes], list[BinaryIO], BinaryIO | None]:
    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    read_fds: list[int] = []
    write_fds: list[int] = []

    # All worker threads emit frames at the target FPS (duplicating or padding as needed),
    # so declare the same rate for every ffmpeg input to keep timestamps aligned.
    effective_input_fps = target_fps if target_fps > 0 else None

    for spec in topic_specs:
        read_fd, write_fd = os.pipe()
        read_fds.append(read_fd)
        write_fds.append(write_fd)
        input_fps = effective_input_fps or spec.fps

        if spec.message_type == ImageType.COMPRESSED:
            cmd.extend(
                [
                    "-f",
                    "image2pipe",
                    "-framerate",
                    f"{input_fps:.6f}",
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
                    f"{input_fps:.6f}",
                    "-i",
                    f"pipe:{read_fd}",
                ]
            )

    # Create subtitle pipe if enabled
    subtitle_read_fd: int | None = None
    subtitle_write_fd: int | None = None
    if enable_subtitles:
        subtitle_read_fd, subtitle_write_fd = os.pipe()
        read_fds.append(subtitle_read_fd)
        write_fds.append(subtitle_write_fd)

        # Add subtitle input
        cmd.extend(["-f", "srt", "-i", f"pipe:{subtitle_read_fd}"])

    # Build filter complex with per-topic watermarks if enabled
    topic_names = [spec.topic for spec in topic_specs] if enable_watermarks else None
    filter_str = _build_filter_complex(
        layout, len(topic_specs), target_width, target_height, topic_names
    )

    cmd.extend(["-filter_complex", filter_str, "-map", "[out]"])

    # Add subtitle mapping if enabled
    if enable_subtitles:
        # Map the subtitle input stream (it's the last input after all video inputs)
        subtitle_input_index = len(topic_specs)
        cmd.extend(["-map", f"{subtitle_input_index}:s"])

    cmd.extend(["-c:v", encoder])
    cmd.extend(_get_encoder_quality_params(codec, encoder, quality))
    if target_fps > 0:
        cmd.extend(["-r", f"{target_fps:.6f}"])
    cmd.extend(["-pix_fmt", "yuv420p"])

    # Add subtitle codec and metadata if enabled
    if enable_subtitles:
        cmd.extend(["-c:s", "mov_text"])
        cmd.extend(["-metadata:s:s:0", "language=eng"])
        cmd.extend(["-metadata:s:s:0", "title=Recording Timestamps"])

    # Add MCAP metadata
    cmd.extend(["-metadata", f"title=MCAP Recording: {mcap_path.name}"])
    cmd.extend(["-metadata", f"comment=Topics: {', '.join(topics)}"])
    start_iso = _ns_to_iso8601(global_start_time)
    end_iso = _ns_to_iso8601(global_end_time)
    cmd.extend(["-metadata", f"description=Recording: {start_iso} to {end_iso}"])

    cmd.append(str(output_path))

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

    # Separate subtitle writer from video writers
    subtitle_writer: BinaryIO | None = None
    video_write_fds = write_fds.copy()

    if enable_subtitles and subtitle_write_fd is not None:
        video_write_fds.remove(subtitle_write_fd)
        subtitle_writer = cast("BinaryIO", os.fdopen(subtitle_write_fd, "wb", buffering=0))

    writers = [cast("BinaryIO", os.fdopen(fd, "wb", buffering=0)) for fd in video_write_fds]
    return process, writers, subtitle_writer


def _stream_topic_worker(
    spec: TopicStreamSpec,
    mcap_path: Path,
    writer: BinaryIO,
    error_bucket: list[tuple[str, Exception]],
    error_lock: threading.Lock,
    global_start_time: int,
    expected_frame_count: int,
    global_duration_ns: int,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> None:
    """Stream frames for a topic with timestamp-aligned frame generation."""
    decoder_factory = DecoderFactory()
    try:
        # Generate black frame (cached for reuse)
        if spec.message_type == ImageType.COMPRESSED:
            black_frame = _create_black_compressed_frame(spec.width, spec.height)
        else:
            black_frame = _create_black_raw_frame(spec.width, spec.height)

        # Open MCAP file and create streaming iterator
        handle = mcap_path.open("rb")
        message_iterator = iter(
            read_message_decoded(
                handle,
                should_include=include_topics(spec.topic),
                decoder_factories=[decoder_factory],
            )
        )

        # Helper function to process a message into frame data
        def process_message(msg: Any) -> tuple[int, bytes]:
            timestamp = msg.message.log_time
            if spec.message_type == ImageType.COMPRESSED:
                frame_data = _process_compressed_image(msg.decoded_message)
            else:
                frame_data, _, _ = _process_raw_image(msg.decoded_message)
            return (timestamp, frame_data)

        # Prime the sliding window with first two messages
        current_msg: tuple[int, bytes] | None = None
        next_msg: tuple[int, bytes] | None = None

        try:
            msg = next(message_iterator)
            current_msg = process_message(msg)
            msg = next(message_iterator)
            next_msg = process_message(msg)
        except StopIteration:
            # 0 or 1 messages - current_msg and/or next_msg remain None
            pass

        last_valid_frame: bytes | None = None

        # Pre-calculate duration as timedelta for cleaner arithmetic
        duration_td = _ns_to_timedelta(global_duration_ns)

        # Generate exactly expected_frame_count frames with precise timing
        for frame_idx in range(expected_frame_count):
            # Calculate precise frame time for this frame
            # Distribute frames evenly across the duration
            if expected_frame_count == 1:
                current_time = global_start_time
            else:
                # Use timedelta for precise fractional time calculation
                frame_offset_td = (duration_td * frame_idx) / (expected_frame_count - 1)
                current_time = global_start_time + _timedelta_to_ns(frame_offset_td)

            # Advance iterator while next message should be consumed
            while next_msg is not None and next_msg[0] <= current_time:
                current_msg = next_msg
                try:
                    msg = next(message_iterator)
                    next_msg = process_message(msg)
                except StopIteration:
                    next_msg = None

            # Determine which frame to write
            if current_time < spec.first_timestamp:
                # Before first message: black frame
                frame_to_write = black_frame
            elif current_time > spec.last_timestamp:
                # After last message: black frame
                frame_to_write = black_frame
            else:
                # Use current message if it's at or before current_time
                if current_msg is not None and current_msg[0] <= current_time:
                    last_valid_frame = current_msg[1]

                # Use the last valid frame (handles gaps by duplicating)
                frame_to_write = last_valid_frame if last_valid_frame is not None else black_frame

            # Write the frame
            writer.write(frame_to_write)

            # Update progress
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)

        # Close MCAP file handle
        handle.close()

    except Exception as exc:  # noqa: BLE001
        with error_lock:
            error_bucket.append((spec.topic, exc))
    finally:
        with contextlib.suppress(Exception):
            writer.close()


def _subtitle_writer_worker(
    writer: BinaryIO,
    global_start_time: int,
    global_duration_ns: int,
    subtitle_interval: float,
    error_bucket: list[tuple[str, Exception]],
    error_lock: threading.Lock,
) -> None:
    """Generate and stream SRT subtitles during video encoding."""
    try:
        # Calculate subtitle cues based on interval
        cue_index = 1
        duration_td = _ns_to_timedelta(global_duration_ns)
        interval_td = timedelta(seconds=subtitle_interval)
        current_td = timedelta(0)

        while current_td < duration_td:
            # Calculate end time for this cue
            end_td = min(current_td + interval_td, duration_td)

            # Calculate actual timestamp for this subtitle
            timestamp_ns = global_start_time + _timedelta_to_ns(current_td)

            # Convert to seconds for SRT formatting
            current_time_s = current_td.total_seconds()
            end_time_s = end_td.total_seconds()

            # Format and write SRT cue
            srt_cue = _format_srt_cue(cue_index, current_time_s, end_time_s, timestamp_ns)
            writer.write(srt_cue)

            # Move to next cue
            cue_index += 1
            current_td = end_td

    except Exception as exc:  # noqa: BLE001
        with error_lock:
            error_bucket.append(("subtitles", exc))
    finally:
        with contextlib.suppress(Exception):
            writer.close()


VIDEOTOOLBOX_BASE_BITRATE = 20
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
    """Test if an ffmpeg encoder is available."""
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
    """Detect the best available encoder for the given codec."""
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
            raise VideoEncoderError(
                f"Hardware encoder '{encoder_preference.value}' not available "
                f"for codec: {codec.value}"
            )
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
    """Get encoder-specific quality parameters."""
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


def _ns_to_iso8601(timestamp_ns: int) -> str:
    """Convert nanosecond timestamp to ISO 8601 datetime string."""
    dt = datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _ns_to_timedelta(timestamp_ns: int) -> timedelta:
    """Convert nanoseconds to timedelta."""
    return timedelta(microseconds=timestamp_ns / 1000)


def _timedelta_to_ns(td: timedelta) -> int:
    return int(td.total_seconds() * 1e9)


def _format_srt_cue(index: int, start_s: float, end_s: float, timestamp_ns: int) -> bytes:
    """Format a single SRT subtitle cue with ISO 8601 timestamp."""
    # Convert start_s to SRT time format
    td_start = timedelta(seconds=start_s)
    ts_start = int(td_start.total_seconds())
    start_str = (
        f"{ts_start // 3600:02d}:{(ts_start % 3600) // 60:02d}:"
        f"{ts_start % 60:02d},{td_start.microseconds // 1000:03d}"
    )

    # Convert end_s to SRT time format
    td_end = timedelta(seconds=end_s)
    ts_end = int(td_end.total_seconds())
    end_str = (
        f"{ts_end // 3600:02d}:{(ts_end % 3600) // 60:02d}:"
        f"{ts_end % 60:02d},{td_end.microseconds // 1000:03d}"
    )

    timestamp_str = _ns_to_iso8601(timestamp_ns)
    return f"{index}\n{start_str} --> {end_str}\n{timestamp_str}\n\n".encode()


def _create_black_compressed_frame(width: int, height: int) -> bytes:
    """Create a black JPEG frame using ffmpeg."""
    try:
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s={width}x{height}:d=1",
                "-frames:v",
                "1",
                "-f",
                "image2",
                "-",
            ],
            capture_output=True,
            timeout=5,
            check=True,
        )
    except Exception as e:
        raise VideoEncoderError(f"Failed to create black frame: {e}") from e
    else:
        return result.stdout


def _create_black_raw_frame(width: int, height: int) -> bytes:
    """Create a black RGB24 frame (all zeros)."""
    return bytes(width * height * 3)


def _process_compressed_image(message: Any) -> bytes:
    """Extract compressed image data from CompressedImage message."""
    if not hasattr(message, "data") or not message.data:
        raise VideoEncoderError("CompressedImage has no data")

    return bytes(message.data)


def _process_raw_image(message: Any) -> tuple[bytes, int, int]:
    """Convert raw Image message to RGB24 bytes."""
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
    """Get image dimensions from compressed image data using ffprobe."""
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


def encode_video(
    mcap_path: Path,
    topics: list[str],
    output_path: Path,
    codec: VideoCodec,
    encoder_preference: Literal["auto", "software"] | HardwareBackend,
    quality: int,
    watermark: bool,
    enable_subtitles: bool = True,
    subtitle_interval: float = 1.0,
) -> None:
    """Encode image topics from MCAP to video."""
    console.print(f"[cyan]Reading MCAP file:[/cyan] {mcap_path}")
    console.print(f"[cyan]Topics:[/cyan] {', '.join(topics)}")
    console.print(f"[cyan]Output:[/cyan] {output_path}")
    console.print(f"[cyan]Codec:[/cyan] {codec.value}")

    encoder = _detect_encoder(codec, encoder_preference)
    console.print(f"[cyan]Encoder:[/cyan] {encoder}")

    # Determine grid layout
    topic_count = len(topics)
    if topic_count <= 1:
        layout = (1, 1)
    elif topic_count == 2:
        layout = (1, 2)
    else:
        rows = math.ceil(math.sqrt(topic_count))
        cols = math.ceil(topic_count / rows)
        layout = (rows, cols)
    console.print(f"[green]✓[/green] Layout: {layout[0]} row(s) x {layout[1]} column(s)")

    # Read MCAP summary once to get all metadata
    console.print("\n[yellow]Reading MCAP summary...[/yellow]")
    with mcap_path.open("rb") as handle:
        rebuild_info = rebuild_summary(
            handle,
            validate_crc=False,
            calculate_channel_sizes=False,
            exact_sizes=False,
        )
    console.print("[green]✓[/green] Summary loaded")

    console.print("\n[yellow]Collecting topic metadata...[/yellow]")
    topic_specs: list[TopicStreamSpec] = []
    for topic in topics:
        spec = _collect_topic_stream_spec(mcap_path, topic, rebuild_info)
        topic_specs.append(spec)
        console.print(
            f"[green]✓[/green] {topic}: {spec.width}x{spec.height} @ {spec.fps:.2f} fps"
            f" ({spec.message_count:,} messages)"
        )

    # Calculate global time range across all topics
    global_start_time = min(spec.first_timestamp for spec in topic_specs)
    global_end_time = max(spec.last_timestamp for spec in topic_specs)
    global_duration_ns = global_end_time - global_start_time
    global_duration_s = global_duration_ns / 1e9

    console.print(
        f"[green]✓[/green] Recording duration: {global_duration_s:.3f}s "
        f"({global_start_time} to {global_end_time})"
    )

    target_width = max(2, min(spec.width for spec in topic_specs))
    target_height = max(2, min(spec.height for spec in topic_specs))
    target_width -= target_width % 2
    target_height -= target_height % 2
    target_fps = max((spec.fps for spec in topic_specs), default=1.0)

    # Calculate exact frame count for ±1 frame precision
    # Use round() to get nearest integer number of frames
    expected_frame_count = round(global_duration_s * target_fps)

    # Ensure at least 1 frame
    expected_frame_count = max(expected_frame_count, 1)

    console.print(
        f"\n[yellow]Encoding video at {target_width}x{target_height}"
        f" @ {target_fps:.2f} fps...[/yellow]"
    )
    if enable_subtitles:
        console.print("[yellow]Embedding timestamp subtitles...[/yellow]")

    ffmpeg_process, writers, subtitle_writer = _start_multi_topic_ffmpeg(
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
        enable_subtitles=enable_subtitles,
        mcap_path=mcap_path,
        topics=topics,
        global_start_time=global_start_time,
        global_end_time=global_end_time,
    )

    error_lock = threading.Lock()
    errors: list[tuple[str, Exception]] = []
    threads: list[threading.Thread] = []

    # Start subtitle writer thread if enabled
    if enable_subtitles and subtitle_writer is not None:
        subtitle_thread = threading.Thread(
            target=_subtitle_writer_worker,
            name="pymcap-subtitles",
            args=(
                subtitle_writer,
                global_start_time,
                global_duration_ns,
                subtitle_interval,
                errors,
                error_lock,
            ),
            daemon=True,
        )
        subtitle_thread.start()
        threads.append(subtitle_thread)

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
                total=expected_frame_count,
            )
            task_ids.append(task_id)

        # Start worker threads with progress tracking
        for spec, writer, task_id in zip(topic_specs, writers, task_ids, strict=True):
            thread = threading.Thread(
                target=_stream_topic_worker,
                name=f"pymcap-{spec.topic}",
                args=(
                    spec,
                    mcap_path,
                    writer,
                    errors,
                    error_lock,
                    global_start_time,
                    expected_frame_count,
                    global_duration_ns,
                    progress,
                    task_id,
                ),
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


def _validate_topics(topics: list[str]) -> list[str]:
    """Validate that at least one topic is provided."""
    if not topics:
        raise typer.BadParameter(
            "At least one topic is required. Use --topic TOPIC_NAME or -t TOPIC_NAME"
        )
    return topics


@app.command(
    epilog="""
Examples:
  pymcap-cli video data.mcap -t /camera/front -o output.mp4
  pymcap-cli video data.mcap -t /cam/front -t /cam/back --watermark -o grid.mp4
  pymcap-cli video data.mcap -t /camera --codec h265 --quality high -o hq.mp4
"""
)
def video(
    file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the MCAP file",
        ),
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
            help="Video codec to use",
            rich_help_panel="Encoding Options",
            show_default=True,
        ),
    ] = VideoCodec.H264,
    quality: Annotated[
        QualityPreset,
        typer.Option(
            "--quality",
            help="Quality preset (ignored if --crf specified)",
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
            help="Manual CRF/quality value (lower = better quality, OVERRIDES --quality)",
            rich_help_panel="Encoding Options",
        ),
    ] = None,
    encoder: Annotated[
        EncoderBackend,
        typer.Option(
            "--encoder",
            help="Encoder backend (auto, software, or hardware acceleration)",
            rich_help_panel="Encoding Options",
            show_default=True,
        ),
    ] = EncoderBackend.AUTO,
    watermark: Annotated[
        bool,
        typer.Option(
            "--watermark",
            help="Add topic name overlay in top-left corner of each video stream",
            rich_help_panel="Display Options",
            show_default=True,
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "-f",
            "--force",
            help="Force overwrite of output file without confirmation",
            rich_help_panel="Output Options",
            show_default=True,
        ),
    ] = False,
    subtitles: Annotated[
        bool,
        typer.Option(
            "--subtitles/--no-subtitles",
            help="Embed timestamp subtitles (ISO 8601 format)",
            rich_help_panel="Metadata Options",
            show_default=True,
        ),
    ] = True,
    subtitle_interval: Annotated[
        float,
        typer.Option(
            "--subtitle-interval",
            help="Subtitle update interval in seconds",
            min=0.1,
            rich_help_panel="Metadata Options",
            show_default=True,
        ),
    ] = 1.0,
) -> None:
    """Generate video from image topics in MCAP files.

    Generate MP4 video from topics using ffmpeg.
    """
    # Validate output directory exists (input file is validated by typer with exists=True)
    if not output.parent.exists():
        console.print(f"[red]Error:[/red] Output directory not found: {output.parent}")
        raise typer.Exit(1)

    # Confirm overwrite if needed
    confirm_output_overwrite(output, force)

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "ffmpeg or ffprobe not found. Install: brew install ffmpeg (macOS) "
            "or sudo apt install ffmpeg (Ubuntu)"
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
            enable_subtitles=subtitles,
            subtitle_interval=subtitle_interval,
        )
    except VideoEncoderError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e
