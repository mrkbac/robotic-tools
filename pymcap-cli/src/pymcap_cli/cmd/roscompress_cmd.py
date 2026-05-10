"""Command to compress image and point cloud topics in MCAP files."""

import logging
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Annotated, Any, Literal

from cyclopts import Group, Parameter
from mcap_codec_support.pointcloud import (
    COMPRESSED_POINTCLOUD2,
    COMPRESSED_POINTCLOUD2_SCHEMA,
    FOXGLOVE_COMPRESSED_POINTCLOUD,
    FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA,
    POINTCLOUD2_SCHEMAS,
    PointCloudCompressionError,
    PointCloudCompressorProtocol,
    build_compressed_pointcloud2_message,
    build_foxglove_compressed_pointcloud_message,
)
from mcap_codec_support.video import (
    COMPRESSED_IMAGE,
    FOXGLOVE_COMPRESSED_VIDEO,
    IMAGE_SCHEMAS,
    RAW_SCHEMAS,
    EncoderConfig,
    EncoderMode,
    VideoCompressionBackend,
    VideoEncoderError,
    calculate_downscale_dimensions,
    create_video_compression_backend,
    encode_raw_image_to_jpeg,
    get_software_encoder,
    prefetch_image_decodes,
)
from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from rich.console import Console
from rich.progress import Progress
from small_mcap import DecodedMessage, McapWriter, read_message_decoded

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.core.mcap_transform import (
    copy_message,
    create_progress,
    ensure_channel,
    ensure_schema,
    get_total_message_count,
    print_size_comparison,
)
from pymcap_cli.exporters._common import normalize_schema_name
from pymcap_cli.types.types_manual import ForceOverwriteOption, OutputPathOption
from pymcap_cli.utils import confirm_output_overwrite

logger = logging.getLogger(__name__)
console = Console()

# Parameter groups
ENCODING_GROUP = Group("Encoding")
POINTCLOUD_GROUP = Group("Point Cloud")


# PointCloud compression helper
# ---------------------------------------------------------------------------


def _create_pointcloud_compressor(
    pc_format: str,
    pc_encoding: str,
    pc_compression: str,
    resolution: float,
    draco_compression_level: int,
) -> PointCloudCompressorProtocol | None:
    try:
        if pc_format == "draco":
            from mcap_codec_support.pointcloud import DracoPointCloudCompressor  # noqa: PLC0415

            return DracoPointCloudCompressor(
                resolution=resolution,
                compression_level=draco_compression_level,
            )

        from mcap_codec_support.pointcloud import CloudiniPointCloudCompressor  # noqa: PLC0415

        return CloudiniPointCloudCompressor(
            encoding=pc_encoding,
            compression=pc_compression,
            resolution=resolution,
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
    pc_format: Annotated[
        Literal["cloudini", "draco"],
        Parameter(
            name=["--pc-format"],
            group=POINTCLOUD_GROUP,
        ),
    ] = "cloudini",
    pc_schema: Annotated[
        Literal["auto", "pointcloud2", "foxglove"],
        Parameter(
            name=["--pc-schema"],
            group=POINTCLOUD_GROUP,
        ),
    ] = "auto",
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
    draco_compression_level: Annotated[
        int,
        Parameter(
            name=["--draco-compression-level"],
            group=POINTCLOUD_GROUP,
        ),
    ] = 7,
    scale: Annotated[
        int | None,
        Parameter(
            name=["--scale", "-s"],
            group=ENCODING_GROUP,
        ),
    ] = None,
    image_format: Annotated[
        Literal["video", "jpeg", "none"],
        Parameter(
            name=["--image-format"],
            group=ENCODING_GROUP,
        ),
    ] = "video",
    jpeg_quality: Annotated[
        int,
        Parameter(
            name=["--jpeg-quality"],
            group=ENCODING_GROUP,
        ),
    ] = 90,
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

    Converts image topics to CompressedVideo or JPEG CompressedImage and
    PointCloud2 topics to compressed point cloud messages using Cloudini or Draco.

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
        Resolution for lossy point cloud compression. Default: 0.01.
    pc_format
        Point cloud output format (cloudini or draco). Default: cloudini.
    pc_schema
        Point cloud output schema (auto, pointcloud2, foxglove). ``auto`` uses
        CompressedPointCloud2 for Cloudini and Foxglove CompressedPointCloud for Draco.
    pc_encoding
        Cloudini point cloud encoding mode (lossy, lossless, none). Default: lossy.
    pc_compression
        Cloudini point cloud second-stage compression (zstd, lz4, none). Default: zstd.
    draco_compression_level
        Draco compression level (0-10). Default: 7.
    image_format
        How to encode image topics:
        ``video`` (default) — convert raw and compressed images to CompressedVideo
        (H.264/H.265). ``jpeg`` — encode raw Image topics as JPEG CompressedImage;
        already-compressed images are copied unchanged. ``none`` — copy all image
        topics unchanged.
    jpeg_quality
        JPEG quality (1-100, higher = better) when ``image_format=jpeg``. Default: 90.
    pointcloud
        Enable point cloud compression. Default: True.
    """
    confirm_output_overwrite(output, force)

    if not 1 <= jpeg_quality <= 100:
        logger.error(f"--jpeg-quality must be in [1, 100], got {jpeg_quality}")
        return 1
    if not 0 <= draco_compression_level <= 10:
        logger.error(f"--draco-compression-level must be in [0, 10], got {draco_compression_level}")
        return 1

    do_video = image_format == "video"
    do_jpeg = image_format == "jpeg"

    # Resolve backend.
    compress_backend = create_video_compression_backend(backend, codec, do_video=do_video)

    # Detect encoder.
    encoder_name = ""
    if do_video:
        if encoder:
            if not compress_backend.test_encoder(encoder):
                logger.error(f"Encoder '{encoder}' not available on this system")
                return 1
            encoder_name = encoder
        else:
            encoder_name = compress_backend.resolve_encoder(codec)

    # Create point cloud compressor.
    pc_compressor: PointCloudCompressorProtocol | None = None
    if pointcloud:
        pc_compressor = _create_pointcloud_compressor(
            pc_format,
            pc_encoding,
            pc_compression,
            resolution,
            draco_compression_level,
        )
        if pc_compressor is None:
            extra = "draco" if pc_format == "draco" else "pointcloud"
            package = f"pymcap-cli[{extra}]"
            logger.error(
                f"{pc_format} dependencies are required for PointCloud2 compression. "
                f"Install with: uv add '{package}'"
            )
            return 1

    logger.info(f"Input: {file}")
    logger.info(f"Output: {output}")
    if do_video:
        logger.info(f"Image mode: video ({encoder_name}, {compress_backend.label})")
        logger.info(f"Quality (CRF): {quality}")
        if scale is not None:
            logger.info(f"Scale (max dim): {scale}px")
    elif do_jpeg:
        logger.info(f"Image mode: jpeg (raw → CompressedImage, q={jpeg_quality})")
        if scale is not None:
            logger.info(f"Scale (max dim): {scale}px")
    else:
        logger.info("Image mode: none (copy unchanged)")
    if pointcloud:
        logger.info(f"Point cloud format: {pc_format}")
        logger.info(f"Point cloud schema: {pc_schema}")
        if pc_format == "cloudini":
            logger.info(f"Point cloud encoding: {pc_encoding}")
            logger.info(f"Point cloud compression: {pc_compression}")
        else:
            logger.info(f"Draco compression level: {draco_compression_level}")
        if pc_format == "draco" or pc_encoding == "lossy":
            logger.info(f"Point cloud resolution: {resolution}")
    else:
        logger.info("Point cloud compression: disabled")

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
    pending_messages: dict[str, deque[DecodedMessage]] = {}

    with (
        open_input(file) as (input_stream, input_size),
        output.open("wb") as output_stream,
        create_progress(title="Compressing images") as progress,
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

        # Video mode benefits from prefetching compressed image decodes. JPEG mode only
        # transcodes raw Image topics and copies CompressedImage topics unchanged.
        decode_pool: ThreadPoolExecutor | None = None
        msg_iter: Iterator[tuple[DecodedMessage, Future[Any] | None]]
        if do_video and compress_backend.prefetch_supported:
            decode_pool = ThreadPoolExecutor(max_workers=4)
            msg_iter = prefetch_image_decodes(messages, compress_backend, decode_pool, prefetch=16)
        else:
            msg_iter = _iter_no_futures(messages)

        compress_ok = _run_compress_loop(
            msg_iter,
            compress_backend,
            do_video,
            do_jpeg,
            jpeg_quality,
            pointcloud,
            pc_format,
            pc_schema,
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
            pending_messages,
            progress,
            task_id,
            counters,
        )
        if decode_pool is not None:
            decode_pool.shutdown(wait=True)

        # Flush remaining frames from video encoders. JPEG encoders are intra-only
        # and write inline, so pending_messages stays empty and we skip them here.
        if compress_ok:
            for topic_name, video_enc in encoders.items():
                if topic_name not in channel_ids:
                    continue
                pending = pending_messages.get(topic_name, deque())
                if not pending:
                    continue
                for packet in video_enc.flush_packets():
                    if pending:
                        pending_msg = pending.popleft()
                        _write_compressed_video(
                            writer, channel_ids[topic_name], pending_msg, packet, codec
                        )
                        counters["converted"] += 1

        writer.finish()

    if not compress_ok:
        return 1

    # Report statistics.
    messages_converted = counters["converted"]
    messages_copied = counters["copied"]
    pointcloud_messages_converted = counters["pc_converted"]
    total_converted = messages_converted + pointcloud_messages_converted
    logger.info("[green bold]✓ Compression complete![/green bold]")
    if topics_converted:
        target_label = "JPEG" if image_format == "jpeg" else "Video"
        console.print(f"[cyan]{target_label} topics converted:[/cyan] {len(topics_converted)}")
        for topic in sorted(topics_converted):
            console.print(f"  - {topic}")
        console.print(f"[cyan]{target_label} messages converted:[/cyan] {messages_converted:,}")
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
    print_size_comparison(input_size, output.stat().st_size)

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
    pc_compressor: PointCloudCompressorProtocol,
    pc_format: str,
    pc_schema: str,
    writer: McapWriter,
    schema_ids: dict[str, int],
    channel_ids: dict[str, int],
    pointcloud_topics_converted: set[str],
) -> None:
    topic = msg.channel.topic
    decoded = msg.decoded_message
    resolved_schema = pc_schema
    if resolved_schema == "auto":
        resolved_schema = "foxglove" if pc_format == "draco" else "pointcloud2"

    compressed = pc_compressor.compress(decoded)
    if resolved_schema == "foxglove":
        compressed_pc_msg = build_foxglove_compressed_pointcloud_message(
            decoded,
            compressed,
            fmt=pc_format,
        )
        target = f"CompressedPointCloud/{pc_format}"
        schema_name_out = FOXGLOVE_COMPRESSED_POINTCLOUD_SCHEMA
        schema_data = FOXGLOVE_COMPRESSED_POINTCLOUD.encode()
    else:
        compressed_pc_msg = build_compressed_pointcloud2_message(
            decoded,
            compressed,
            fmt=pc_format,
        )
        target = f"CompressedPointCloud2/{pc_format}"
        schema_name_out = COMPRESSED_POINTCLOUD2_SCHEMA
        schema_data = COMPRESSED_POINTCLOUD2.encode()

    if topic not in pointcloud_topics_converted:
        pointcloud_topics_converted.add(topic)
        schema_name = normalize_schema_name(msg.schema.name) if msg.schema else ""
        logger.info(f"[green]✓[/green] Converting {topic} ({schema_name} → {target})")

    schema_id = ensure_schema(writer, schema_name_out, "ros2msg", schema_data, schema_ids)
    channel_id = ensure_channel(writer, topic, "cdr", schema_id, channel_ids)
    writer.add_message_encode(
        channel_id=channel_id,
        log_time=msg.message.log_time,
        data=compressed_pc_msg,
        publish_time=msg.message.publish_time,
    )


# ---------------------------------------------------------------------------
# Raw → JPEG path (CompressedImage output)
# ---------------------------------------------------------------------------


def _handle_raw_to_jpeg(
    msg: DecodedMessage,
    backend: VideoCompressionBackend,
    decode_future: Future[Any] | None,
    encoders: dict[str, Any],
    jpeg_quality: int,
    scale: int | None,
    writer: McapWriter,
    schema_ids: dict[str, int],
    channel_ids: dict[str, int],
    topics_converted: set[str],
    counters: dict[str, int],
) -> bool:
    """Encode a raw Image message as JPEG and write it as a CompressedImage."""
    del backend, decode_future

    topic = msg.channel.topic
    schema_name = normalize_schema_name(msg.schema.name) if msg.schema else ""

    try:
        jpeg_bytes, target_w, target_h = encode_raw_image_to_jpeg(
            msg.decoded_message, jpeg_quality=jpeg_quality, scale=scale
        )
    except VideoEncoderError:
        logger.exception(f"Failed to encode JPEG for {topic}")
        return False

    if topic not in encoders:
        encoders[topic] = True
        topics_converted.add(topic)
        schema_id = ensure_schema(
            writer,
            "sensor_msgs/msg/CompressedImage",
            "ros2msg",
            COMPRESSED_IMAGE.encode(),
            schema_ids,
        )
        ensure_channel(writer, topic, "cdr", schema_id, channel_ids)
        logger.info(
            f"[green]✓[/green] Converting {topic}: {target_w}x{target_h} "
            f"({schema_name} → CompressedImage/jpeg)"
        )

    decoded = msg.decoded_message
    compressed_msg = {
        "header": {
            "stamp": {
                "sec": decoded.header.stamp.sec,
                "nanosec": decoded.header.stamp.nanosec,
            },
            "frame_id": decoded.header.frame_id,
        },
        "format": "jpeg",
        "data": jpeg_bytes,
    }
    writer.add_message_encode(
        channel_id=channel_ids[topic],
        log_time=msg.message.log_time,
        data=compressed_msg,
        publish_time=msg.message.publish_time,
    )
    counters["converted"] += 1
    return True


# ---------------------------------------------------------------------------
# Unified processing loop
# ---------------------------------------------------------------------------


def _run_compress_loop(
    messages: Iterator[tuple[DecodedMessage, Future[Any] | None]],
    backend: VideoCompressionBackend,
    do_video: bool,
    do_jpeg: bool,
    jpeg_quality: int,
    pointcloud: bool,
    pc_format: str,
    pc_schema: str,
    encoders: dict[str, Any],
    encoder_name: str,
    codec: str,
    quality: int,
    scale: int | None,
    pc_compressor: PointCloudCompressorProtocol | None,
    writer: McapWriter,
    schema_ids: dict[str, int],
    channel_ids: dict[str, int],
    topics_converted: set[str],
    pointcloud_topics_converted: set[str],
    last_video_times: dict[str, tuple[int, int]],
    pending_messages: dict[str, deque[DecodedMessage]],
    progress: Progress,
    task_id: int,
    counters: dict[str, int],
) -> bool:
    for msg, decode_future in messages:
        schema_name = normalize_schema_name(msg.schema.name) if msg.schema else ""

        if do_jpeg and schema_name in RAW_SCHEMAS:
            if not _handle_raw_to_jpeg(
                msg,
                backend,
                decode_future,
                encoders,
                jpeg_quality,
                scale,
                writer,
                schema_ids,
                channel_ids,
                topics_converted,
                counters,
            ):
                return False

        elif do_video and schema_name in IMAGE_SCHEMAS:
            topic = msg.channel.topic

            frame: Any = None
            if topic not in encoders:
                # First message for this topic — discover dimensions and create encoder.
                if decode_future is not None:
                    frame, width, height = decode_future.result()
                else:
                    frame, width, height = backend.decode_image(msg, schema_name)
                pix_fmt = backend.get_pix_fmt(topic)

                if scale is not None:
                    width, height = calculate_downscale_dimensions(width, height, scale)
                # Always ensure even dimensions (required for yuv420p).
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
                    # Register schema/channel immediately so flush can write messages.
                    vid_schema_id = ensure_schema(
                        writer,
                        "foxglove_msgs/msg/CompressedVideo",
                        "ros2msg",
                        FOXGLOVE_COMPRESSED_VIDEO.encode(),
                        schema_ids,
                    )
                    ensure_channel(writer, topic, "cdr", vid_schema_id, channel_ids)
                    logger.info(
                        f"[green]✓[/green] Converting {topic}: {width}x{height} "
                        f"({schema_name} → CompressedVideo)"
                    )
                except VideoEncoderError:
                    logger.exception(f"Failed to create encoder for {topic}")
                    return False

            if frame is None:
                if decode_future is not None:
                    frame, _, _ = decode_future.result()
                else:
                    frame, _, _ = backend.decode_image(msg, schema_name)

            try:
                video_data = encoders[topic].encode(frame)
            except VideoEncoderError:
                sw = get_software_encoder(codec)
                if encoders[topic].config.codec_name != sw:
                    logger.warning(f"Encoder failed for {topic}, falling back to {sw}")
                    cfg: EncoderConfig = encoders[topic].config
                    pix_fmt = backend.get_pix_fmt(topic)
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

            # Buffer this message's metadata for when encoder output arrives.
            if topic not in pending_messages:
                pending_messages[topic] = deque()
            pending_messages[topic].append(msg)

            if video_data is None:
                progress.update(task_id, advance=1)
                continue

            # Write output using the oldest pending message's metadata.
            pending_msg = pending_messages[topic].popleft()
            _write_compressed_video(writer, channel_ids[topic], pending_msg, video_data, codec)
            counters["converted"] += 1
            last_video_times[topic] = (msg.message.log_time, msg.message.publish_time)

        elif pointcloud and schema_name in POINTCLOUD2_SCHEMAS:
            try:
                _handle_pointcloud(
                    msg,
                    pc_compressor,
                    pc_format,
                    pc_schema,
                    writer,
                    schema_ids,
                    channel_ids,
                    pointcloud_topics_converted,
                )
            except PointCloudCompressionError as exc:
                logger.warning(f"Skipping point cloud compression for {msg.channel.topic}: {exc}")
                copy_message(msg, writer, schema_ids, channel_ids)
                counters["copied"] += 1
            else:
                counters["pc_converted"] += 1

        else:
            copy_message(msg, writer, schema_ids, channel_ids)
            counters["copied"] += 1

        progress.update(task_id, advance=1)

    return True
