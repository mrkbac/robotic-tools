"""Command to compress CompressedImage topics to CompressedVideo in MCAP files."""

import io
import logging
import platform
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import av
import av.error
import numpy as np
from av.video.frame import VideoFrame
from cyclopts import Group, Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from numpy.typing import NDArray
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

from pymcap_cli.input_handler import open_input
from pymcap_cli.osc_utils import OSCProgressColumn
from pymcap_cli.types_manual import (
    ForceOverwriteOption,
    OutputPathOption,
)
from pymcap_cli.utils import confirm_output_overwrite

if TYPE_CHECKING:
    from av.container import InputContainer
    from av.video.codeccontext import VideoCodecContext


FOXGLOVE_COMPRESSED_VIDEO = """builtin_interfaces/Time timestamp
string frame_id
uint8[] data
string format

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec"""

COMPRESSED_SCHEMAS = {"sensor_msgs/msg/CompressedImage", "sensor_msgs/CompressedImage"}
RAW_SCHEMAS = {"sensor_msgs/msg/Image", "sensor_msgs/Image"}
IMAGE_SCHEMAS = COMPRESSED_SCHEMAS | RAW_SCHEMAS

console = Console()
logger = logging.getLogger(__name__)

# Parameter groups
ENCODING_GROUP = Group("Encoding")


class VideoEncoderError(Exception):
    """Raised when video encoding fails."""


def _test_encoder(encoder_name: str) -> bool:
    """Test if an encoder is available on this system."""
    try:
        av.CodecContext.create(encoder_name, "w")
    except (av.error.FFmpegError, ValueError):
        return False
    else:
        return True


def _detect_encoder() -> str:
    """Detect the best available H.264 encoder for this system."""
    # Try hardware encoders first
    system = platform.system()
    if system == "Darwin":
        if _test_encoder("h264_videotoolbox"):
            return "h264_videotoolbox"
    elif system == "Linux":
        for encoder in ["h264_nvenc", "h264_vaapi"]:
            if _test_encoder(encoder):
                return encoder

    # Fallback to software encoder
    return "libx264"


def _decode_compressed_image(compressed_data: bytes) -> VideoFrame:
    """Decode a compressed image (JPEG/PNG) to a VideoFrame."""
    try:
        container = cast("InputContainer", av.open(io.BytesIO(compressed_data), format="image2"))
        for frame in container.decode(video=0):
            container.close()
            return frame.reformat(format="rgb24")
    except Exception as exc:
        raise VideoEncoderError(f"Failed to decode compressed image: {exc}") from exc

    raise VideoEncoderError("Decoder produced no frames")


def _raw_image_to_array(message: Any) -> NDArray[np.uint8]:
    """Convert a ROS Image message to an RGB numpy array."""
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
        return array.copy()  # type: ignore[no-any-return,unused-ignore]
    if encoding in {"bgr", "bgr8"}:
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        return array[..., ::-1].copy()  # type: ignore[no-any-return,unused-ignore]
    if encoding in {"mono", "mono8", "8uc1"}:
        mono_array = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        return np.repeat(mono_array[:, :, None], 3, axis=2)

    raise VideoEncoderError(f"Unsupported image encoding: {message.encoding}")


@dataclass(frozen=True, slots=True)
class _EncoderConfig:
    """Configuration for a video encoder."""

    width: int
    height: int
    codec_name: str


class _VideoEncoder:
    """PyAV-based video encoder for converting images to compressed video."""

    def __init__(
        self,
        width: int,
        height: int,
        codec_name: str,
        quality: int = 28,
        preset: str = "medium",
        target_fps: float = 30.0,
        gop_size: int = 30,
    ) -> None:
        self.config = _EncoderConfig(width=width, height=height, codec_name=codec_name)
        self._target_fps = max(target_fps, 1.0)
        self._frame_index = 0
        self._quality = quality
        self._preset = preset
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

        # Set encoder-specific options
        options: dict[str, str] = {}
        if codec_name in {"libx264", "h264_videotoolbox"}:
            options["preset"] = preset
            options["crf"] = str(quality)
            if codec_name == "libx264":
                options["tune"] = "zerolatency"
        elif codec_name in {"libx265", "hevc_videotoolbox"}:
            options["preset"] = preset
            options["crf"] = str(quality)

        if options:
            self._context.options = options

        try:
            self._context.open()
        except av.error.FFmpegError as exc:
            raise VideoEncoderError(f"Failed to open encoder {codec_name}: {exc}") from exc

    def encode(self, frame: VideoFrame) -> bytes:
        """Encode a single frame and return compressed video bytes."""
        # Ensure frame is in the correct format and dimensions
        if (
            frame.width != self.config.width
            or frame.height != self.config.height
            or frame.format.name != "rgb24"
        ):
            frame = frame.reformat(
                width=self.config.width, height=self.config.height, format="rgb24"
            )

        # Convert to encoder pixel format
        frame = frame.reformat(format=self._context.pix_fmt)
        frame.pts = self._frame_index
        self._frame_index += 1

        try:
            packets = list(self._context.encode(frame))
        except av.error.FFmpegError as exc:
            raise VideoEncoderError(f"Encoding error: {exc}") from exc

        # If no packets, try flushing (some codecs buffer internally)
        if not packets:
            try:
                packets = list(self._context.encode(None))
            except av.error.FFmpegError as exc:
                raise VideoEncoderError(f"Encoder flush error: {exc}") from exc

        data = b"".join(bytes(packet) for packet in packets)
        if not data:
            raise VideoEncoderError("Encoder produced empty packet")
        return data

    def close(self) -> None:
        """Close the encoder (no-op, context cleaned up by garbage collector)."""


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
) -> int:
    """Compress ROS MCAP by converting CompressedImage/Image topics to CompressedVideo.

    This reduces file size while maintaining visual quality by using video compression
    instead of individual image compression.

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
    """
    confirm_output_overwrite(output, force)

    # Detect encoder
    if encoder:
        if not _test_encoder(encoder):
            console.print(f"[red]Error:[/red] Encoder '{encoder}' not available on this system")
            return 1
        encoder_name = encoder
    else:
        encoder_name = _detect_encoder()

    console.print(f"[cyan]Input:[/cyan] {file}")
    console.print(f"[cyan]Output:[/cyan] {output}")
    console.print(f"[cyan]Encoder:[/cyan] {encoder_name}")
    console.print(f"[cyan]Quality (CRF):[/cyan] {quality}")

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
    decoder_factory = DecoderFactory()
    encoder_factory = ROS2EncoderFactory()

    # Statistics
    messages_converted = 0
    messages_copied = 0
    topics_converted: set[str] = set()

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
            )
            writer.start()

            # Track schema/channel IDs
            schema_ids: dict[str, int] = {}  # schema_name -> schema_id
            channel_ids: dict[str, int] = {}  # topic -> channel_id
            next_schema_id = 1
            next_channel_id = 1
            registered_channels: set[int] = set()

            for msg in read_message_decoded(input_stream, decoder_factories=[decoder_factory]):
                schema_name = msg.schema.name if msg.schema else ""

                if schema_name in IMAGE_SCHEMAS:
                    # Convert to CompressedVideo
                    topic = msg.channel.topic

                    # Lazy initialization of encoder for this topic
                    if topic not in encoders:
                        # Decode first frame to get dimensions
                        if schema_name in COMPRESSED_SCHEMAS:
                            compressed_data = bytes(msg.decoded_message.data)
                            first_frame = _decode_compressed_image(compressed_data)
                            width, height = first_frame.width, first_frame.height
                        else:  # RAW_SCHEMAS
                            rgb_array = _raw_image_to_array(msg.decoded_message)
                            height, width = rgb_array.shape[:2]

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
                                preset="medium",
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

                    # Decode image
                    if schema_name in COMPRESSED_SCHEMAS:
                        compressed_data = bytes(msg.decoded_message.data)
                        frame = _decode_compressed_image(compressed_data)
                    else:  # RAW_SCHEMAS
                        rgb_array = _raw_image_to_array(msg.decoded_message)
                        frame = av.VideoFrame.from_ndarray(rgb_array, format="rgb24")

                    # Encode to video
                    try:
                        video_data = encoders[topic].encode(frame)
                    except VideoEncoderError as exc:
                        # Try fallback to software encoder if hardware encoder fails
                        if encoder_name != "libx264":
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
                                    codec_name=encoder_name,
                                    quality=quality,
                                    preset="medium",
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

                    # Create CompressedVideo message
                    compressed_video_msg = {
                        "timestamp": {
                            "sec": msg.decoded_message.header.stamp.sec,
                            "nanosec": msg.decoded_message.header.stamp.nanosec,
                        },
                        "frame_id": msg.decoded_message.header.frame_id,
                        "data": list(video_data),
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

                else:
                    # Copy unchanged - register schema/channel if not already registered
                    channel_id = msg.channel.id

                    if channel_id not in registered_channels:
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

                        # Register channel with same ID as input
                        writer.add_channel(
                            channel_id=channel_id,
                            topic=msg.channel.topic,
                            message_encoding=msg.channel.message_encoding,
                            schema_id=schema_id,
                            metadata=msg.channel.metadata,
                        )
                        registered_channels.add(channel_id)

                    # Write message with original data
                    writer.add_message(
                        channel_id=channel_id,
                        log_time=msg.message.log_time,
                        data=msg.message.data,
                        publish_time=msg.message.publish_time,
                    )
                    messages_copied += 1

                progress.update(task_id, advance=1)

            writer.finish()

    finally:
        # Clean up encoders
        for video_encoder in encoders.values():
            video_encoder.close()

    # Report statistics
    console.print("\n[green bold]✓ Compression complete![/green bold]")
    console.print(f"[cyan]Topics converted:[/cyan] {len(topics_converted)}")
    if topics_converted:
        for topic in sorted(topics_converted):
            console.print(f"  - {topic}")
    console.print(f"[cyan]Messages converted:[/cyan] {messages_converted:,}")
    console.print(f"[cyan]Messages copied:[/cyan] {messages_copied:,}")
    console.print(f"[cyan]Total messages:[/cyan] {messages_converted + messages_copied:,}")

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
