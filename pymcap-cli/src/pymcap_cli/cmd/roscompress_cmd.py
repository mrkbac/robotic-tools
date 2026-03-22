"""Command to compress image and point cloud topics in MCAP files."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol

from cyclopts import Group, Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory
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
from pymcap_cli.encoding.encoder_common import (
    COMPRESSED_SCHEMAS,
    DEFAULT_FPS,
    DEFAULT_GOP_SIZE,
    FOXGLOVE_COMPRESSED_VIDEO,
    IMAGE_SCHEMAS,
    EncoderConfig,
    EncoderMode,
    VideoEncoderError,
    calculate_downscale_dimensions,
    get_software_encoder,
)
from pymcap_cli.encoding.pointcloud import COMPRESSED_POINTCLOUD2, POINTCLOUD2_SCHEMAS
from pymcap_cli.types.types_manual import (  # noqa: TC001 — runtime for cyclopts
    ForceOverwriteOption,
    OutputPathOption,
)
from pymcap_cli.utils import confirm_output_overwrite

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from rich.progress import Progress
    from small_mcap.reader import DecodedMessage


console = Console()

# Parameter groups
ENCODING_GROUP = Group("Encoding")
POINTCLOUD_GROUP = Group("Point Cloud")


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


class CompressBackend(Protocol):
    """Protocol for video encoding backends (PyAV or ffmpeg-cli)."""

    def test_encoder(self, encoder_name: str) -> bool: ...

    def resolve_encoder(self, codec: str) -> str: ...

    def decode_image(
        self, msg: DecodedMessage, schema_name: str, *, scale: int | None = None
    ) -> tuple[Any, int, int]:
        """Decode an image message to an intermediate format.

        Returns (intermediate_frame, width, height).
        """
        ...

    def create_encoder(
        self,
        width: int,
        height: int,
        codec_name: str,
        quality: int,
        *,
        input_pix_fmt: str | None = None,
        scale: tuple[int, int] | None = None,
    ) -> Any: ...


class PyAVBackend:
    """Video encoding backend using PyAV (in-process)."""

    def test_encoder(self, encoder_name: str) -> bool:
        import av  # noqa: PLC0415
        import av.error  # noqa: PLC0415

        try:
            av.CodecContext.create(encoder_name, "w")
        except (av.error.FFmpegError, ValueError):
            return False
        return True

    def resolve_encoder(self, codec: str) -> str:
        from pymcap_cli.encoding.video_pyav import resolve_encoder  # noqa: PLC0415

        return resolve_encoder(codec)

    def decode_image(
        self,
        msg: DecodedMessage,
        schema_name: str,
        *,
        scale: int | None = None,  # noqa: ARG002
    ) -> tuple[Any, int, int]:
        # PyAV decodes at native resolution; scaling is done via calculate_downscale.
        if schema_name in COMPRESSED_SCHEMAS:
            return self.decode_compressed(bytes(msg.decoded_message.data))
        return self._decode_raw(msg.decoded_message)

    def decode_compressed(self, data: bytes) -> tuple[Any, int, int]:
        from pymcap_cli.encoding.video_pyav import decode_compressed_frame  # noqa: PLC0415

        frame = decode_compressed_frame(data)
        return frame, frame.width, frame.height

    def _decode_raw(self, decoded_message: Any) -> tuple[Any, int, int]:
        import av  # noqa: PLC0415

        from pymcap_cli.encoding.video_pyav import raw_image_to_array  # noqa: PLC0415

        rgb_array = raw_image_to_array(decoded_message)
        frame = av.VideoFrame.from_ndarray(rgb_array, format="rgb24")
        return frame, frame.width, frame.height

    def create_encoder(
        self,
        width: int,
        height: int,
        codec_name: str,
        quality: int,
        *,
        input_pix_fmt: str | None = None,  # noqa: ARG002
        scale: tuple[int, int] | None = None,  # noqa: ARG002
    ) -> Any:
        from pymcap_cli.encoding.video_pyav import VideoEncoder  # noqa: PLC0415

        return VideoEncoder(
            width=width,
            height=height,
            codec_name=codec_name,
            quality=quality,
            target_fps=DEFAULT_FPS,
            gop_size=DEFAULT_GOP_SIZE,
        )


class FfmpegCliBackend:
    """Video encoding backend using ffmpeg subprocess."""

    def __init__(self) -> None:
        self._topic_pix_fmt: dict[str, str | None] = {}

    def get_pix_fmt(self, topic: str) -> str | None:
        """Return the input pixel format discovered for *topic*, or ``None`` (image2pipe)."""
        return self._topic_pix_fmt.get(topic)

    def test_encoder(self, encoder_name: str) -> bool:
        from pymcap_cli.encoding.video_ffmpeg import check_encoder_cli  # noqa: PLC0415

        return check_encoder_cli(encoder_name)

    def resolve_encoder(self, codec: str) -> str:
        from pymcap_cli.encoding.video_ffmpeg import resolve_encoder  # noqa: PLC0415

        return resolve_encoder(codec)

    def decode_image(
        self,
        msg: DecodedMessage,
        schema_name: str,
        *,
        scale: int | None = None,  # noqa: ARG002
    ) -> tuple[Any, int, int]:
        data = bytes(msg.decoded_message.data)
        topic = msg.channel.topic

        if schema_name in COMPRESSED_SCHEMAS:
            from pymcap_cli.encoding.video_ffmpeg import probe_image_dimensions  # noqa: PLC0415

            self._topic_pix_fmt[topic] = None
            width, height = probe_image_dimensions(data)
            return data, width, height

        from pymcap_cli.encoding.video_ffmpeg import ROS_ENCODING_TO_PIX_FMT  # noqa: PLC0415

        encoding = str(msg.decoded_message.encoding).lower()
        pix_fmt = ROS_ENCODING_TO_PIX_FMT.get(encoding)
        if not pix_fmt:
            raise VideoEncoderError(f"Unsupported image encoding: {msg.decoded_message.encoding}")
        self._topic_pix_fmt[topic] = pix_fmt
        return data, msg.decoded_message.width, msg.decoded_message.height

    def create_encoder(
        self,
        width: int,
        height: int,
        codec_name: str,
        quality: int,
        *,
        input_pix_fmt: str | None = None,
        scale: tuple[int, int] | None = None,
    ) -> Any:
        from pymcap_cli.encoding.video_ffmpeg import FFmpegVideoEncoder  # noqa: PLC0415

        return FFmpegVideoEncoder(
            width=width,
            height=height,
            codec_name=codec_name,
            quality=quality,
            target_fps=DEFAULT_FPS,
            gop_size=DEFAULT_GOP_SIZE,
            input_pix_fmt=input_pix_fmt,
            scale=scale,
        )


# ---------------------------------------------------------------------------
# Prefetch (PyAV backend only)
# ---------------------------------------------------------------------------


def _prefetch_image_decodes(
    messages: Iterable[DecodedMessage],
    backend: PyAVBackend,
    pool: ThreadPoolExecutor,
    prefetch: int = 8,
) -> Iterator[tuple[DecodedMessage, Future[Any] | None]]:
    """Wrap message iterator to decode JPEGs in background threads (PyAV)."""
    buffer: deque[tuple[DecodedMessage, Future[Any] | None]] = deque()

    for msg in messages:
        schema_name = msg.schema.name if msg.schema else ""
        if schema_name in COMPRESSED_SCHEMAS:
            data = bytes(msg.decoded_message.data)
            future: Future[Any] | None = pool.submit(backend.decode_compressed, data)
        else:
            future = None
        buffer.append((msg, future))

        if len(buffer) > prefetch:
            yield buffer.popleft()

    while buffer:
        yield buffer.popleft()


# ---------------------------------------------------------------------------
# PointCloud compression helper
# ---------------------------------------------------------------------------


def _create_pointcloud_compressor(
    pc_encoding: str, pc_compression: str, resolution: float
) -> Any | None:
    try:
        from pymcap_cli.encoding.pointcloud import PointCloudCompressor  # noqa: PLC0415

        return PointCloudCompressor(
            encoding=pc_encoding, compression=pc_compression, resolution=resolution
        )
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


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
    backend: Annotated[
        EncoderMode,
        Parameter(
            name=["--backend"],
            group=ENCODING_GROUP,
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

    # Resolve backend.
    use_cli = backend == EncoderMode.FFMPEG_CLI
    compress_backend: CompressBackend
    if use_cli:
        compress_backend = FfmpegCliBackend()
    else:
        pyav_backend = PyAVBackend()
        if backend == EncoderMode.AUTO:
            try:
                pyav_backend.resolve_encoder(codec)
            except Exception:  # noqa: BLE001
                use_cli = True
                compress_backend = FfmpegCliBackend()
            else:
                compress_backend = pyav_backend
        else:
            compress_backend = pyav_backend

    # Detect encoder.
    encoder_name = ""
    if video:
        if encoder:
            if not compress_backend.test_encoder(encoder):
                console.print(f"[red]Error:[/red] Encoder '{encoder}' not available on this system")
                return 1
            encoder_name = encoder
        else:
            encoder_name = compress_backend.resolve_encoder(codec)

    # Create point cloud compressor.
    pc_compressor: Any | None = None
    if pointcloud:
        pc_compressor = _create_pointcloud_compressor(pc_encoding, pc_compression, resolution)
        if pc_compressor is None:
            console.print(
                "[red]Error:[/red] pureini is required for PointCloud2 compression. "
                "Install with: uv add 'pymcap-cli[pointcloud]'"
            )
            return 1

    backend_label = "ffmpeg-cli" if use_cli else "pyav"
    console.print(f"[cyan]Input:[/cyan] {file}")
    console.print(f"[cyan]Output:[/cyan] {output}")
    if video:
        console.print(f"[cyan]Encoder:[/cyan] {encoder_name}")
        console.print(f"[cyan]Backend:[/cyan] {backend_label}")
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

    # Get message count from summary for progress bar.
    total_message_count = get_total_message_count(file)

    # Track encoders per topic (lazy initialization).
    encoders: dict[str, Any] = {}
    decoder_factory = DecoderFactory()
    encoder_factory = ROS2EncoderFactory()

    # Statistics.
    counters = {"converted": 0, "copied": 0, "pc_converted": 0}
    topics_converted: set[str] = set()
    pointcloud_topics_converted: set[str] = set()
    last_video_times: dict[str, tuple[int, int]] = {}

    with (
        open_input(file) as (input_stream, input_size),
        output.open("wb") as output_stream,
        create_progress(console, title="Compressing images") as progress,
    ):
        task_id = progress.add_task("Processing messages", total=total_message_count)

        writer = McapWriter(
            output_stream,
            encoder_factory=encoder_factory,
            num_workers=4,
        )
        writer.start()

        # Track schema/channel IDs.
        schema_ids: dict[str, int] = {}
        channel_ids: dict[str, int] = {}

        messages = read_message_decoded(
            input_stream, decoder_factories=[decoder_factory], num_workers=4
        )

        # Wrap with prefetching for PyAV backend.
        decode_pool: ThreadPoolExecutor | None = None
        msg_iter: Iterator[tuple[DecodedMessage, Future[Any] | None]]
        if not use_cli and isinstance(compress_backend, PyAVBackend):
            decode_pool = ThreadPoolExecutor(max_workers=4)
            msg_iter = _prefetch_image_decodes(messages, compress_backend, decode_pool, prefetch=16)
        else:
            msg_iter = _iter_no_futures(messages)

        _run_compress_loop(
            msg_iter,
            compress_backend,
            video,
            pointcloud,
            encoders,
            encoder_name,
            codec,
            quality,
            scale,
            pc_compressor,
            writer,
            schema_ids,
            channel_ids,
            topics_converted,
            pointcloud_topics_converted,
            last_video_times,
            progress,
            task_id,
            counters,
        )
        if decode_pool is not None:
            decode_pool.shutdown(wait=True)

        # Flush remaining frames from video encoders.
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

        writer.finish()

    # Report statistics.
    messages_converted = counters["converted"]
    messages_copied = counters["copied"]
    pointcloud_messages_converted = counters["pc_converted"]
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

    # Show file size comparison.
    print_size_comparison(console, input_size, output.stat().st_size)

    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_no_futures(
    messages: Iterable[DecodedMessage],
) -> Iterator[tuple[DecodedMessage, None]]:
    """Wrap plain message iterator to match the prefetched signature."""
    for msg in messages:
        yield msg, None


def _write_compressed_video(
    writer: McapWriter,
    channel_id: int,
    msg: DecodedMessage,
    video_data: bytes,
    codec: str,
) -> None:
    compressed_video_msg = {
        "timestamp": {
            "sec": msg.decoded_message.header.stamp.sec,
            "nanosec": msg.decoded_message.header.stamp.nanosec,
        },
        "frame_id": msg.decoded_message.header.frame_id,
        "data": video_data,
        "format": codec,
    }
    writer.add_message_encode(
        channel_id=channel_id,
        log_time=msg.message.log_time,
        data=compressed_video_msg,
        publish_time=msg.message.publish_time,
    )


def _handle_pointcloud(
    msg: DecodedMessage,
    pc_compressor: Any,
    writer: McapWriter,
    schema_ids: dict[str, int],
    channel_ids: dict[str, int],
    pointcloud_topics_converted: set[str],
) -> None:
    topic = msg.channel.topic
    compressed = pc_compressor.compress(msg.decoded_message)

    if topic not in pointcloud_topics_converted:
        pointcloud_topics_converted.add(topic)
        schema_name = msg.schema.name if msg.schema else ""
        console.print(
            f"[green]✓[/green] Converting {topic} ({schema_name} → CompressedPointCloud2)"
        )

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

    schema_id = ensure_schema(
        writer,
        "point_cloud_interfaces/msg/CompressedPointCloud2",
        "ros2msg",
        COMPRESSED_POINTCLOUD2.encode(),
        schema_ids,
    )
    channel_id = ensure_channel(writer, topic, "cdr", schema_id, channel_ids)
    writer.add_message_encode(
        channel_id=channel_id,
        log_time=msg.message.log_time,
        data=compressed_pc_msg,
        publish_time=msg.message.publish_time,
    )


# ---------------------------------------------------------------------------
# Unified processing loop
# ---------------------------------------------------------------------------


def _get_topic_pix_fmt(backend: CompressBackend, topic: str) -> str | None:
    """Return the input pixel format for *topic*, or ``None`` for image2pipe mode."""
    if isinstance(backend, FfmpegCliBackend):
        return backend.get_pix_fmt(topic)
    return None


def _run_compress_loop(
    messages: Iterator[tuple[DecodedMessage, Future[Any] | None]],
    backend: CompressBackend,
    video: bool,
    pointcloud: bool,
    encoders: dict[str, Any],
    encoder_name: str,
    codec: str,
    quality: int,
    scale: int | None,
    pc_compressor: Any | None,
    writer: McapWriter,
    schema_ids: dict[str, int],
    channel_ids: dict[str, int],
    topics_converted: set[str],
    pointcloud_topics_converted: set[str],
    last_video_times: dict[str, tuple[int, int]],
    progress: Progress,
    task_id: int,
    counters: dict[str, int],
) -> None:
    for msg, decode_future in messages:
        schema_name = msg.schema.name if msg.schema else ""

        if video and schema_name in IMAGE_SCHEMAS:
            topic = msg.channel.topic

            frame: Any = None
            pix_fmt = _get_topic_pix_fmt(backend, topic)
            if topic not in encoders:
                # First message for this topic — discover dimensions and create encoder.
                if decode_future is not None:
                    frame, width, height = decode_future.result()
                else:
                    frame, width, height = backend.decode_image(msg, schema_name, scale=scale)
                pix_fmt = _get_topic_pix_fmt(backend, topic)

                if scale is not None:
                    width, height = calculate_downscale_dimensions(width, height, scale)
                else:
                    width -= width % 2
                    height -= height % 2
                try:
                    encoders[topic] = backend.create_encoder(
                        width,
                        height,
                        encoder_name,
                        quality,
                        input_pix_fmt=pix_fmt,
                        scale=(width, height) if pix_fmt is None and scale is not None else None,
                    )
                    topics_converted.add(topic)
                    console.print(
                        f"[green]✓[/green] Converting {topic}: {width}x{height} "
                        f"({schema_name} → CompressedVideo)"
                    )
                except VideoEncoderError as exc:
                    console.print(f"[red]Error:[/red] Failed to create encoder for {topic}: {exc}")
                    return

            if frame is None:
                if decode_future is not None:
                    frame, _, _ = decode_future.result()
                else:
                    frame, _, _ = backend.decode_image(msg, schema_name, scale=scale)

            try:
                video_data = encoders[topic].encode(frame)
            except VideoEncoderError:
                sw = get_software_encoder(codec)
                if encoders[topic].config.codec_name != sw:
                    console.print(
                        f"[yellow]Warning:[/yellow] Encoder failed for {topic}, "
                        f"falling back to {sw}"
                    )
                    cfg: EncoderConfig = encoders[topic].config
                    encoders[topic] = backend.create_encoder(
                        cfg.width,
                        cfg.height,
                        sw,
                        quality,
                        input_pix_fmt=pix_fmt,
                        scale=(cfg.width, cfg.height) if pix_fmt is None and scale else None,
                    )
                    video_data = encoders[topic].encode(frame)
                else:
                    raise

            if video_data is None:
                progress.update(task_id, advance=1)
                continue

            schema_id = ensure_schema(
                writer,
                "foxglove_msgs/msg/CompressedVideo",
                "ros2msg",
                FOXGLOVE_COMPRESSED_VIDEO.encode(),
                schema_ids,
            )
            channel_id = ensure_channel(writer, topic, "cdr", schema_id, channel_ids)
            _write_compressed_video(writer, channel_id, msg, video_data, codec)
            counters["converted"] += 1
            last_video_times[topic] = (msg.message.log_time, msg.message.publish_time)

        elif pointcloud and schema_name in POINTCLOUD2_SCHEMAS:
            _handle_pointcloud(
                msg, pc_compressor, writer, schema_ids, channel_ids, pointcloud_topics_converted
            )
            counters["pc_converted"] += 1

        else:
            copy_message(msg, writer, schema_ids, channel_ids)
            counters["copied"] += 1

        progress.update(task_id, advance=1)
