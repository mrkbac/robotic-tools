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
from small_mcap import include_topics, read_message_decoded
from small_mcap.rebuild import rebuild_summary

from pymcap_cli.autocompletion import complete_topic_by_schema
from pymcap_cli.mcap_processor import confirm_output_overwrite

if TYPE_CHECKING:
    from av.container import InputContainer

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


@dataclass(slots=True)
class TopicStreamSpec:
    topic: str
    message_type: ImageType
    width: int
    height: int
    fps: float
    message_count: int
    first_timestamp: int
    last_timestamp: int


@dataclass(frozen=True)
class EncoderConfig:
    codec_name: str
    width: int
    height: int
    fps: float
    quality: int
    is_hardware: bool


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
    VideoCodec.H264: {"high": 18, "medium": 23, "low": 28},
    VideoCodec.H265: {"high": 20, "medium": 25, "low": 30},
    VideoCodec.VP9: {"high": 30, "medium": 35, "low": 40},
    VideoCodec.AV1: {"high": 25, "medium": 30, "low": 35},
}


class TopicFrameStream:
    """Stream frames for a topic, duplicating last frame to fill gaps."""

    def __init__(
        self, spec: TopicStreamSpec, mcap_path: Path, tile_width: int, tile_height: int
    ) -> None:
        self.spec = spec
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.decoder_factory = DecoderFactory()
        self.handle = mcap_path.open("rb")
        self.iterator = iter(
            read_message_decoded(
                self.handle,
                should_include=include_topics(spec.topic),
                decoder_factories=[self.decoder_factory],
            )
        )
        self.black_frame = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
        self.current_frame: tuple[int, np.ndarray] | None = None
        self.closed = False
        self.next_frame: tuple[int, np.ndarray] | None = self._read_next()
        self.last_valid_frame: np.ndarray | None = None

    def _read_next(self) -> tuple[int, np.ndarray] | None:
        if self.closed:
            return None
        for msg in self.iterator:
            timestamp = msg.message.log_time
            frame_array = _message_to_array(
                msg.decoded_message,
                self.spec.message_type,
                self.tile_width,
                self.tile_height,
            )
            return (timestamp, frame_array)
        self.close()
        return None

    def frame_at(self, timestamp_ns: int) -> np.ndarray:
        if timestamp_ns < self.spec.first_timestamp or timestamp_ns > self.spec.last_timestamp:
            return self.black_frame

        while self.next_frame is not None and self.next_frame[0] <= timestamp_ns:
            self.current_frame = self.next_frame
            self.next_frame = self._read_next()

        if self.current_frame is not None and self.current_frame[0] <= timestamp_ns:
            self.last_valid_frame = self.current_frame[1]

        return self.last_valid_frame if self.last_valid_frame is not None else self.black_frame

    def close(self) -> None:
        if not self.closed:
            self.handle.close()
            self.closed = True


def _message_to_array(
    message: Any, message_type: ImageType, target_width: int, target_height: int
) -> np.ndarray:
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


def _extract_topic_metadata_from_rebuild(
    rebuild_info: Any, topic: str
) -> tuple[int, ImageType, int, int, int]:
    summary = rebuild_info.summary
    image_schema_ids = {
        schema.id for schema in summary.schemas.values() if schema.name in IMAGE_SCHEMAS
    }
    compressed_schema_ids = {
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
    if message_count == 0 or first_timestamp is None or last_timestamp is None:
        raise VideoEncoderError(f"No messages found for topic: {topic}")
    return channel_id, message_type, message_count, first_timestamp, last_timestamp


def _collect_topic_stream_spec(mcap_path: Path, topic: str, rebuild_info: Any) -> TopicStreamSpec:
    _channel_id, message_type, message_count, first_timestamp, last_timestamp = (
        _extract_topic_metadata_from_rebuild(rebuild_info, topic)
    )
    decoder_factory = DecoderFactory()
    width: int | None = None
    height: int | None = None
    samples = 0
    max_samples = min(10, message_count)
    with mcap_path.open("rb") as handle:
        for msg in read_message_decoded(
            handle,
            should_include=include_topics(topic),
            decoder_factories=[decoder_factory],
        ):
            if message_type is ImageType.COMPRESSED:
                frame = _decode_compressed_frame(_process_compressed_image(msg.decoded_message))
                width = frame.width
                height = frame.height
            else:
                rgb = _raw_image_to_array(msg.decoded_message)
                height, width = rgb.shape[:2]
            samples += 1
            if width is not None and height is not None and samples >= max_samples:
                break
    if width is None or height is None:
        raise VideoEncoderError(f"Failed to determine dimensions for topic: {topic}")
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


def _get_encoder_options(codec: VideoCodec, encoder_name: str, quality: int) -> dict[str, str]:
    options: dict[str, str] = {}
    if "videotoolbox" in encoder_name:
        options["q:v"] = str(max(0, min(100, 100 - quality * 2)))
    elif "nvenc" in encoder_name:
        options["preset"] = "p4"
        options["cq"] = str(quality)
    elif "vaapi" in encoder_name:
        options["qp"] = str(quality)
    elif codec in (VideoCodec.H264, VideoCodec.H265):
        options["preset"] = "medium"
        options["crf"] = str(quality)
    elif codec == VideoCodec.VP9:
        options["crf"] = str(quality)
        options["b:v"] = "0"
    elif codec == VideoCodec.AV1:
        options["crf"] = str(quality)
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
    mcap_path: Path,
    topics: list[str],
    output_path: Path,
    codec: VideoCodec,
    encoder_backend: EncoderBackend,
    quality: int,
) -> None:
    console.print(f"[cyan]Reading MCAP file:[/cyan] {mcap_path}")
    console.print(f"[cyan]Topics:[/cyan] {', '.join(topics)}")
    console.print(f"[cyan]Output:[/cyan] {output_path}")
    console.print(f"[cyan]Codec:[/cyan] {codec.value}")

    encoder_name = _detect_encoder(codec, encoder_backend)
    console.print(f"[cyan]Encoder:[/cyan] {encoder_name}")

    with mcap_path.open("rb") as handle:
        rebuild_info = rebuild_summary(
            handle,
            validate_crc=False,
            calculate_channel_sizes=False,
            exact_sizes=False,
        )

    topic_specs = [_collect_topic_stream_spec(mcap_path, topic, rebuild_info) for topic in topics]

    rows, cols = (1, 1)
    if len(topics) == 2:
        rows, cols = (1, 2)
    elif len(topics) > 2:
        rows = math.ceil(math.sqrt(len(topics)))
        cols = math.ceil(len(topics) / rows)
    console.print(f"[green]✓[/green] Layout: {rows} row(s) x {cols} column(s)")

    target_tile_width = max(2, min(spec.width for spec in topic_specs))
    target_tile_height = max(2, min(spec.height for spec in topic_specs))
    target_tile_width -= target_tile_width % 2
    target_tile_height -= target_tile_height % 2

    global_start = min(spec.first_timestamp for spec in topic_specs)
    global_end = max(spec.last_timestamp for spec in topic_specs)
    global_duration_ns = max(global_end - global_start, 1)
    target_fps = max((spec.fps for spec in topic_specs), default=0.0, key=lambda x: x)
    target_fps = max(target_fps, 1.0)
    expected_frames = max(1, round((global_duration_ns / 1e9) * target_fps))

    grid_width = target_tile_width * cols
    grid_height = target_tile_height * rows
    grid_width -= grid_width % 2
    grid_height -= grid_height % 2

    # MP4-specific configuration
    try:
        container = av.open(str(output_path), "w", format=None, options={"movflags": "faststart"})
    except (av.error.FFmpegError, ValueError) as exc:
        raise VideoEncoderError(
            f"Failed to open output file '{output_path}': {exc}\n"
            f"Ensure the file path is writable and the format is supported."
        ) from exc

    fps_int = max(1, round(target_fps))

    try:
        stream_raw = container.add_stream(codec_name=encoder_name, rate=fps_int)
    except (av.error.FFmpegError, ValueError) as exc:
        container.close()
        raise VideoEncoderError(
            f"Failed to create video stream with encoder '{encoder_name}': {exc}\n"
            f"This encoder may not be available on your system. Try --encoder software."
        ) from exc

    # Type-narrow the stream to VideoStream for mypy
    stream = stream_raw

    stream.width = grid_width  # type: ignore[union-attr]
    stream.height = grid_height  # type: ignore[union-attr]
    stream.pix_fmt = "yuv420p"  # type: ignore[union-attr]
    stream.time_base = Fraction(1, fps_int)
    stream.codec_context.framerate = Fraction(fps_int, 1)  # type: ignore[attr-defined]
    stream.codec_context.gop_size = max(1, fps_int * 2)  # type: ignore[attr-defined]

    # Get encoder options and ensure B-frames are disabled for MP4 compatibility
    options = _get_encoder_options(codec, encoder_name, quality)
    if "libx264" in encoder_name or "libx265" in encoder_name:
        options["bf"] = "0"
        options["g"] = str(max(1, fps_int * 2))
    elif "videotoolbox" in encoder_name or "nvenc" in encoder_name or "vaapi" in encoder_name:
        # Hardware encoders: ensure B-frames are disabled
        if "videotoolbox" in encoder_name:
            options["bf"] = "0"

    stream.codec_context.options = options
    console.print(f"[cyan]Encoder options:[/cyan] {options}")

    frame_streams = [
        TopicFrameStream(spec, mcap_path, target_tile_width, target_tile_height)
        for spec in topic_specs
    ]

    console.print(
        f"\n[yellow]Encoding {expected_frames:,} frames at {grid_width}x{grid_height} "
        f"@ {target_fps:.2f} fps...[/yellow]"
    )

    if expected_frames == 1:
        timestamps = [global_start]
    else:
        step = Fraction(global_duration_ns, expected_frames - 1)
        timestamps = [global_start + int(step * idx) for idx in range(expected_frames)]

    try:
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
            task_id = progress.add_task("Encoding", total=expected_frames)
            for frame_idx, timestamp_ns in enumerate(timestamps):
                try:
                    tiles = [stream.frame_at(timestamp_ns) for stream in frame_streams]
                    grid_frame = _compose_grid(
                        tiles, (rows, cols), (target_tile_height, target_tile_width)
                    )
                    frame = av.VideoFrame.from_ndarray(grid_frame, format="rgb24")
                    frame = frame.reformat(format="yuv420p")
                    frame.pts = frame_idx
                    packets = stream.encode(frame)  # type: ignore[union-attr,arg-type]
                    for packet in packets:
                        container.mux(packet)
                    progress.update(task_id, advance=1)
                except (av.error.FFmpegError, ValueError) as exc:
                    container.close()
                    for fs in frame_streams:
                        fs.close()
                    raise VideoEncoderError(
                        f"Encoding failed at frame {frame_idx + 1}/{expected_frames}: {exc}\n"
                        f"Encoder: {encoder_name}, Resolution: {grid_width}x{grid_height}, "
                        f"Codec: {codec.value}\n"
                        f"This may indicate an incompatibility between the encoder and "
                        f"MP4 format.\n"
                        f"Try using --encoder software or a different codec."
                    ) from exc

        # Flush encoder
        try:
            for packet in stream.encode(None):  # type: ignore[union-attr]
                container.mux(packet)
        except (av.error.FFmpegError, ValueError) as exc:
            container.close()
            for fs in frame_streams:
                fs.close()
            raise VideoEncoderError(
                f"Failed to flush encoder: {exc}\nThe video file may be incomplete or corrupted."
            ) from exc

        container.close()
    except Exception:
        # Cleanup on any error
        for fs in frame_streams:
            fs.close()
        raise

    for fs in frame_streams:
        fs.close()

    console.print(f"\n[green bold]✓ Video created:[/green bold] {output_path}")


@app.command(
    epilog="""
Examples:
  pymcap-cli video data.mcap -t /camera/front -o output.mp4
  pymcap-cli video data.mcap -t /cam/left -t /cam/right -o grid.mp4
"""
)
def video(
    file: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=True, dir_okay=False, readable=True, help="Path to the MCAP file"
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
    if not output.parent.exists():
        console.print(f"[red]Error:[/red] Output directory not found: {output.parent}")
        raise typer.Exit(1)
    confirm_output_overwrite(output, force)
    quality_value = crf if crf is not None else QUALITY_PRESETS[codec][quality.value]
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
