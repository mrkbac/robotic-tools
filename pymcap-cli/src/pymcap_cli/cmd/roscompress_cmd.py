"""Command to compress image and point cloud topics in MCAP files.

Thin preset over the processing pipeline: builds the transcode processors
(video / still-image / point cloud) and runs them through ``run_processor``, so the
command shares the pipeline's machinery (fast-copy, chunk grouping, ordering)
and composes with everything else. The heavy lifting lives in the processors
(``core/processors/video_compress.py``, ``pointcloud_compress.py``,
``image_compress.py``).
"""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from mcap_codec_support.video import EncoderMode, VideoEncoderError
from rich.console import Console

from pymcap_cli.cmd._cli_options import (
    BackendOption,
    CodecOption,
    DracoCompressionLevelOption,
    EncoderOption,
    EndTimeOption,
    ExcludeTopicOption,
    ForceOverwriteOption,
    ImageFormatOption,
    JpegQualityOption,
    OutputPathOption,
    PointCloudCompressionOption,
    PointCloudDropInvalidOption,
    PointCloudEncodingOption,
    PointCloudFormatOption,
    PointCloudOption,
    PointCloudSchemaOption,
    PointCloudSortFieldOption,
    QualityOption,
    ResolutionOption,
    ScaleOption,
    StartTimeOption,
    TopicOption,
)
from pymcap_cli.cmd._message_filter_options import create_message_filter
from pymcap_cli.cmd._pointcloud_cleanup import (
    pointcloud_worker_count,
    resolve_pointcloud_cleanup,
)
from pymcap_cli.cmd._run_processor import resolve_overwrite_policy, run_processor
from pymcap_cli.constants import DEFAULT_ROSCOMPRESS_CHUNK_SPAN_NS
from pymcap_cli.core.mcap_processor import InputOptions, OutputOptions
from pymcap_cli.core.mcap_transform import print_size_comparison
from pymcap_cli.core.processors.chunk_groupers import SchemaCompressionGrouper
from pymcap_cli.utils import output_overwrites_input

if TYPE_CHECKING:
    from pymcap_cli.core.processors.base import InputProcessor, OutputProcessor

logger = logging.getLogger(__name__)
console = Console()

# roscompress emits already-compressed payloads (CompressedVideo / CompressedImage
# / CompressedPointCloud); route them to a *per-topic* uncompressed chunk group so
# the container zstd pass isn't wasted on data that won't shrink (and never touches
# them on future reads). Per-topic (not one shared) groups keep each topic's
# monotonic stream time-ordered: the video transcode emits topics with a per-topic
# frame-count lag, so a shared group would interleave them into wide, heavily
# overlapping chunks.
_COMPRESSED_OUTPUT_PATTERN = re.compile(r"Compressed(Image|Video|PointCloud)")
_INPUT_BUFFER_BYTES = 8 * 1024 * 1024
_ASYNC_OUTPUT_BUFFER_BYTES = 16 * 1024 * 1024


def roscompress(
    file: str,
    output: OutputPathOption,
    *,
    force: ForceOverwriteOption = False,
    quality: QualityOption = 28,
    codec: CodecOption = "h264",
    encoder: EncoderOption = None,
    resolution: ResolutionOption = 0.01,
    pc_format: PointCloudFormatOption = "cloudini",
    pc_schema: PointCloudSchemaOption = "auto",
    pc_encoding: PointCloudEncodingOption = "lossy",
    pc_compression: PointCloudCompressionOption = "zstd",
    draco_compression_level: DracoCompressionLevelOption = 7,
    scale: ScaleOption = None,
    image_format: ImageFormatOption = "video",
    jpeg_quality: JpegQualityOption = 90,
    backend: BackendOption = EncoderMode.AUTO,
    pointcloud: PointCloudOption = True,
    pointcloud_drop_invalid: PointCloudDropInvalidOption = None,
    pointcloud_sort_field: PointCloudSortFieldOption = None,
    topic: TopicOption = None,
    exclude_topic: ExcludeTopicOption = None,
    start: StartTimeOption = "",
    end: EndTimeOption = "",
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
        Video codec (h264, h265, vp9, av1). Default: h264.
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
        ``png`` — encode raw Image topics as PNG CompressedImage;
        already-compressed images are copied unchanged. ``none`` — copy all image
        topics unchanged.
    jpeg_quality
        JPEG quality (1-100, higher = better) when ``image_format=jpeg``. Default: 90.
    pointcloud
        Enable point cloud compression. Default: True.
    pointcloud_drop_invalid
        Drop invalid ``(0,0,0)``/NaN points from PointCloud2 messages. Defaults
        to enabled when point cloud compression is enabled, and disabled when
        compression is disabled unless a point-cloud cleanup flag is supplied.
    pointcloud_sort_field
        Stable-sort cleaned PointCloud2 points by this field. Defaults to no
        sorting. Use ``line`` to group lidar rings.
    exclude_topic
        Drop topics matching a full-match regex (repeatable). Excluded topics
        are skipped before decoding, e.g. ``-x '/debug/.*'``.
    """
    if output_overwrites_input(file, output):
        logger.error("Output path is the same file as the input; choose a different output file.")
        return 1

    if not 1 <= jpeg_quality <= 100:
        logger.error(f"--jpeg-quality must be in [1, 100], got {jpeg_quality}")
        return 1
    if not 0 <= draco_compression_level <= 10:
        logger.error(f"--draco-compression-level must be in [0, 10], got {draco_compression_level}")
        return 1

    try:
        cleanup = resolve_pointcloud_cleanup(
            pointcloud_compression_enabled=pointcloud,
            pointcloud_drop_invalid=pointcloud_drop_invalid,
            pointcloud_sort_field=pointcloud_sort_field,
        )
    except ValueError as exc:
        logger.error(str(exc))  # noqa: TRY400
        return 1

    overwrite_policy = resolve_overwrite_policy(force=force, no_clobber=False)
    assert overwrite_policy is not None  # no_clobber is fixed False here

    # Build the transcode processor chain. Constructors do the real work
    # (encoder probing, codec dependency import), so a missing optional
    # dependency or unavailable encoder surfaces here as a clean CLI error.
    extras: list[InputProcessor] = []
    try:
        if image_format == "video":
            from pymcap_cli.core.processors.video_compress import (  # noqa: PLC0415
                VideoCompressProcessor,
            )

            extras.append(
                VideoCompressProcessor(
                    codec=codec,
                    quality=quality,
                    encoder=encoder,
                    scale=scale,
                    backend=backend,
                )
            )
        elif image_format in {"jpeg", "png"}:
            from pymcap_cli.core.processors.image_compress import (  # noqa: PLC0415
                ImageCompressProcessor,
            )

            extras.append(
                ImageCompressProcessor(
                    image_format=image_format,
                    jpeg_quality=jpeg_quality,
                    scale=scale,
                )
            )

        if cleanup.enabled:
            from pymcap_cli.core.processors.pointcloud_clean import (  # noqa: PLC0415
                PointcloudCleanProcessor,
            )

            extras.append(
                PointcloudCleanProcessor(
                    drop_invalid=cleanup.drop_invalid,
                    sort_field=cleanup.sort_field,
                )
            )

        if pointcloud:
            from pymcap_cli.core.processors.pointcloud_compress import (  # noqa: PLC0415
                PointcloudCompressProcessor,
            )

            extras.append(
                PointcloudCompressProcessor(
                    pc_format=pc_format,
                    pc_schema=pc_schema,
                    pc_encoding=pc_encoding,
                    pc_compression=pc_compression,
                    resolution=resolution,
                    draco_compression_level=draco_compression_level,
                    # Always parallelize point-cloud compression on its own pool.
                    # Profiling the video path showed the main thread is NOT idle
                    # (it drives reading + video dispatch/drain + writing), so
                    # compressing point clouds inline on it serializes behind that
                    # work rather than "riding for free". A dedicated pool lets
                    # Cloudini/Draco work overlap the main loop.
                    workers=pointcloud_worker_count(),
                )
            )
    except ImportError:
        extra = "draco" if pc_format == "draco" else "pointcloud"
        logger.error(  # noqa: TRY400
            f"Optional dependencies are required for this mode. "
            f"Install with: uv add 'pymcap-cli[{extra}]'"
        )
        return 1
    except VideoEncoderError as exc:
        logger.error(str(exc))  # noqa: TRY400
        return 1

    logger.info(f"Input: {file}")
    logger.info(f"Output: {output}")
    if exclude_topic:
        logger.info(f"Excluding topics matching: {', '.join(exclude_topic)}")
    if image_format == "video":
        logger.info(f"Image mode: video ({encoder or 'auto'}, {codec}, backend={backend.value})")
        logger.info(f"Quality (CRF): {quality}")
    elif image_format == "jpeg":
        logger.info(f"Image mode: jpeg (raw → CompressedImage, q={jpeg_quality})")
    elif image_format == "png":
        logger.info("Image mode: png (raw → CompressedImage)")
    else:
        logger.info("Image mode: none (copy unchanged)")
    if scale is not None and image_format != "none":
        logger.info(f"Scale (max dim): {scale}px")
    if pointcloud:
        logger.info(f"Point cloud: {pc_format} (schema={pc_schema})")
    else:
        logger.info("Point cloud compression: disabled")
    if cleanup.enabled:
        parts: list[str] = []
        if cleanup.drop_invalid:
            parts.append("drop (0,0,0)/NaN points")
        if cleanup.sort_field is not None:
            parts.append(f"group by {cleanup.sort_field}")
        logger.info(f"Point cloud cleanup: {', '.join(parts)}")
    else:
        logger.info("Point cloud cleanup: disabled")

    try:
        message_filter = create_message_filter(
            topic=topic,
            exclude_topic=exclude_topic,
            start=start,
            end=end,
            early_bail=False,
        )
    except ValueError as exc:
        logger.error(str(exc))  # noqa: TRY400
        return 1

    input_options = InputOptions.from_message_filter(
        message_filter,
        extra_processors=extras or None,
    )
    # Route the compressed output into a per-topic uncompressed chunk group — the
    # payloads are already compressed, so a container zstd pass only burns CPU, and
    # per-topic grouping keeps each topic time-ordered and non-overlapping. Cap the
    # chunk span so a low-byte-rate topic doesn't accumulate one very wide chunk.
    output_processors: list[OutputProcessor] = []
    max_chunk_span_ns: int | None = None
    if image_format != "none" or pointcloud:
        output_processors.append(
            SchemaCompressionGrouper([_COMPRESSED_OUTPUT_PATTERN], per_channel=True)
        )
        max_chunk_span_ns = DEFAULT_ROSCOMPRESS_CHUNK_SPAN_NS
    output_options = OutputOptions(
        output_processors=output_processors,
        overwrite_policy=overwrite_policy,
        max_chunk_span_ns=max_chunk_span_ns,
        async_output_buffer_bytes=_ASYNC_OUTPUT_BUFFER_BYTES,
    )

    try:
        result = run_processor(
            files=[file],
            output=output,
            input_options=input_options,
            output_options=output_options,
            input_buffer_bytes=_INPUT_BUFFER_BYTES,
        )
    except Exception:
        logger.exception("Error during compression")
        # A failed run may have truncated/partially written the output; don't
        # leave a corrupt file behind (the output was opened with an
        # overwrite/truncate policy, so it is ours to remove).
        output.unlink(missing_ok=True)
        return 1

    logger.info("[green bold]✓ Compression complete![/green bold]")
    stats = result.stats.writer_statistics
    console.print(f"[cyan]Messages written:[/cyan] {stats.message_count:,}")

    input_size = _local_size(file)
    if input_size and output.exists():
        print_size_comparison(input_size, output.stat().st_size)

    return 0


def _local_size(file: str) -> int:
    """Best-effort byte size of a local input file (0 for URLs / missing)."""
    try:
        return Path(file).stat().st_size
    except OSError:
        return 0
