"""Command to compress image and point cloud topics in MCAP files.

Thin preset over the processing pipeline: builds the transcode processors
(video / still-image / point cloud) and runs them through ``run_processor``, so the
command shares the pipeline's machinery (fast-copy, chunk grouping, ordering)
and composes with everything else. The heavy lifting lives in the processors
(``core/processors/video_compress.py``, ``pointcloud_compress.py``,
``image_compress.py``).
"""

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from cyclopts import Group, Parameter
from mcap_codec_support.video import EncoderMode, VideoEncoderError
from rich.console import Console

from pymcap_cli.cmd._run_processor import resolve_overwrite_policy, run_processor
from pymcap_cli.core.mcap_processor import InputOptions, OutputOptions
from pymcap_cli.core.mcap_transform import print_size_comparison
from pymcap_cli.core.processors.chunk_groupers import SchemaCompressionGrouper
from pymcap_cli.types.types_manual import ForceOverwriteOption, OutputPathOption
from pymcap_cli.utils import output_overwrites_input

if TYPE_CHECKING:
    from pymcap_cli.core.processors.base import InputProcessor, OutputProcessor

logger = logging.getLogger(__name__)
console = Console()

# roscompress emits already-compressed payloads (CompressedVideo / CompressedImage
# / CompressedPointCloud); route them to their own uncompressed chunk group so the
# container zstd pass isn't wasted on data that won't shrink (and never touches
# them on future reads).
_COMPRESSED_OUTPUT_PATTERN = re.compile(r"Compressed(Image|Video|PointCloud)")

# Parameter groups
ENCODING_GROUP = Group("Encoding")
POINTCLOUD_GROUP = Group("Point Cloud")


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
        Literal["video", "jpeg", "png", "none"],
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
            name=["--pointcloud"],
            group=POINTCLOUD_GROUP,
        ),
    ] = True,
    clean_pointcloud: Annotated[
        bool,
        Parameter(
            name=["--clean-pointcloud"],
            group=POINTCLOUD_GROUP,
        ),
    ] = True,
    exclude_topic_glob: Annotated[
        list[str] | None,
        Parameter(
            name=["--exclude-topic-glob", "-x"],
            help="Drop topics matching this shell-style glob (repeatable), e.g. '/debug/*'.",
        ),
    ] = None,
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
        ``png`` — encode raw Image topics as PNG CompressedImage;
        already-compressed images are copied unchanged. ``none`` — copy all image
        topics unchanged.
    jpeg_quality
        JPEG quality (1-100, higher = better) when ``image_format=jpeg``. Default: 90.
    pointcloud
        Enable point cloud compression. Default: True.
    clean_pointcloud
        Before compressing, drop invalid ``(0,0,0)``/NaN points and group the
        remaining points by laser ring (``line``). Shrinks such clouds ~30%%
        at no extra time cost. Default: True.
    exclude_topic_glob
        Drop topics whose name matches any of these shell-style globs
        (repeatable). Excluded topics are skipped before decoding, e.g.
        ``--exclude-topic-glob '/debug/*'``.
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
                    clean=clean_pointcloud,
                    # Parallelize point-cloud compression only when video isn't
                    # also being transcoded: with video, point clouds already
                    # ride for free in the main thread's idle time (hidden behind
                    # the video worker threads), and a second pool just adds CPU
                    # contention. Without video, the main thread would otherwise
                    # compress them serially.
                    workers=0 if image_format == "video" else _pointcloud_workers(),
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
    if exclude_topic_glob:
        logger.info(f"Excluding topics matching: {', '.join(exclude_topic_glob)}")
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
        if clean_pointcloud:
            logger.info("Point cloud cleanup: drop (0,0,0)/NaN points, group by line")
    else:
        logger.info("Point cloud compression: disabled")

    input_options = InputOptions.from_args(
        exclude_topic_glob=exclude_topic_glob or None,
        extra_processors=extras or None,
    )
    # Route the compressed output into its own uncompressed chunk group — the
    # payloads are already compressed, so a container zstd pass only burns CPU.
    output_processors: list[OutputProcessor] = []
    if image_format != "none" or pointcloud:
        output_processors.append(SchemaCompressionGrouper([_COMPRESSED_OUTPUT_PATTERN]))
    output_options = OutputOptions(
        output_processors=output_processors,
        overwrite_policy=overwrite_policy,
    )

    try:
        result = run_processor(
            files=[file],
            output=output,
            input_options=input_options,
            output_options=output_options,
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


def _pointcloud_workers() -> int:
    """Worker count for parallel point-cloud compression.

    Point-cloud encode overlaps the read/re-chunk floor, so wall time stops
    improving past ~4 workers (measured knee); more just occupies cores.
    """
    return min(4, max(2, (os.cpu_count() or 4) - 2))
