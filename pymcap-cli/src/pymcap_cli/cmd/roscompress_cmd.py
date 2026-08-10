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
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from cyclopts import Parameter
from mcap_codec_support.video import VideoEncoderError
from rich.console import Console

from pymcap_cli.cmd._cli_options import (
    ENCODING_GROUP,
    IMAGE_POINTCLOUD_MODE_CONSTRAINT,
    POINTCLOUD_GROUP,
    BackendOption,
    CodecOption,
    DracoCompressionLevelOption,
    EncoderOption,
    EndTimeOption,
    ExcludeTopicOption,
    FfmpegArgsOption,
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
    PointCloudTopicOptionsOption,
    QualityOption,
    ResolutionOption,
    ScaleOption,
    StartTimeOption,
    TopicOption,
    VideoTopicFfmpegArgsOption,
    VideoTopicOptionsOption,
)
from pymcap_cli.cmd._message_filter_options import create_message_filter
from pymcap_cli.cmd._pointcloud_cleanup import pointcloud_worker_count
from pymcap_cli.cmd._roscompress import (
    RoscompressConfig,
    create_image_processor,
    create_pointcloud_cleanup_processor,
    create_pointcloud_compress_processor,
    create_video_compress_processor,
    fuses_cloudini_cleanup,
    resolve_cleanup,
)
from pymcap_cli.cmd._run_processor import resolve_overwrite_policy, run_processor
from pymcap_cli.constants import DEFAULT_ROSCOMPRESS_CHUNK_SPAN_NS
from pymcap_cli.core.mcap_processor import InputOptions, OutputOptions
from pymcap_cli.core.mcap_transform import print_size_comparison
from pymcap_cli.core.message_filter import TopicSelection
from pymcap_cli.core.processors.base import TopicMatchingProcessor
from pymcap_cli.core.processors.chunk_groupers import SchemaCompressionGrouper
from pymcap_cli.utils import compile_topic_patterns, output_overwrites_input

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


@dataclass(frozen=True, slots=True)
class _TopicOptionSpec:
    pattern: str
    values: dict[str, str]


def _parse_topic_pattern(specification: str, *, option_name: str, syntax: str) -> tuple[str, str]:
    """Split ``PATTERN:REST`` and validate the pattern as a topic regex."""
    invalid_pattern: ValueError | None = None
    # Regex groups can contain colons, as in ``(?:front|back)``, so the
    # delimiter is the first colon that ends a valid regex prefix.
    for index, character in enumerate(specification):
        if character != ":":
            continue
        pattern = specification[:index].strip()
        rest = specification[index + 1 :]
        if not pattern or not rest.strip():
            continue
        try:
            compile_topic_patterns([pattern])
        except ValueError as exc:
            invalid_pattern = exc
            continue
        return pattern, rest
    if invalid_pattern is not None:
        raise invalid_pattern
    raise ValueError(f"{option_name} must use {syntax} syntax")


def _parse_topic_option_spec(
    specification: str,
    *,
    option_name: str,
    allowed_keys: frozenset[str],
) -> _TopicOptionSpec:
    syntax = "PATTERN:key=value[,key=value...]"
    pattern, raw_values = _parse_topic_pattern(
        specification, option_name=option_name, syntax=syntax
    )

    values: dict[str, str] = {}
    for assignment in raw_values.split(","):
        key, equals, value = assignment.partition("=")
        key = key.strip().replace("_", "-")
        value = value.strip()
        if not equals or not key or not value:
            raise ValueError(f"{option_name} must use {syntax} syntax")
        if key not in allowed_keys:
            kind = "point-cloud" if option_name == "--pointcloud-topic-options" else "video"
            raise ValueError(f"unknown {kind} topic option '{key}'")
        if key in values:
            raise ValueError(f"duplicate {option_name} key '{key}' for '{pattern}'")
        values[key] = value
    return _TopicOptionSpec(pattern=pattern, values=values)


def _choice(value: str, *, key: str, choices: frozenset[str]) -> str:
    if value not in choices:
        raise ValueError(f"{key} must be one of: {', '.join(sorted(choices))}")
    return value


def _integer(value: str, *, key: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {value!r}") from exc


def _resolve_pointcloud_topic_options(
    specifications: list[str] | None,
    defaults: RoscompressConfig,
) -> dict[str, RoscompressConfig]:
    resolved: dict[str, RoscompressConfig] = {}
    allowed_keys = frozenset(
        {
            "resolution",
            "pc-format",
            "pc-schema",
            "pc-encoding",
            "pc-compression",
            "draco-compression-level",
        }
    )
    for specification in specifications or []:
        parsed = _parse_topic_option_spec(
            specification,
            option_name="--pointcloud-topic-options",
            allowed_keys=allowed_keys,
        )
        settings = resolved.get(parsed.pattern, defaults)
        for key, value in parsed.values.items():
            if key == "resolution":
                try:
                    topic_resolution = float(value)
                except ValueError as exc:
                    raise ValueError(f"resolution must be a number, got {value!r}") from exc
                if topic_resolution <= 0:
                    raise ValueError("resolution must be positive")
                settings = replace(settings, resolution=topic_resolution)
            elif key == "pc-format":
                settings = replace(
                    settings,
                    pc_format=_choice(value, key=key, choices=frozenset({"cloudini", "draco"})),
                )
            elif key == "pc-schema":
                settings = replace(
                    settings,
                    pc_schema=_choice(
                        value, key=key, choices=frozenset({"auto", "pointcloud2", "foxglove"})
                    ),
                )
            elif key == "pc-encoding":
                settings = replace(
                    settings,
                    pc_encoding=_choice(
                        value, key=key, choices=frozenset({"lossy", "lossless", "none"})
                    ),
                )
            elif key == "pc-compression":
                settings = replace(
                    settings,
                    pc_compression=_choice(
                        value, key=key, choices=frozenset({"zstd", "lz4", "none"})
                    ),
                )
            else:
                level = _integer(value, key=key)
                if not 0 <= level <= 10:
                    raise ValueError("draco-compression-level must be between 0 and 10")
                settings = replace(settings, draco_compression_level=level)
        resolved[parsed.pattern] = settings
    return resolved


def _resolve_video_topic_options(
    specifications: list[str] | None,
    defaults: RoscompressConfig,
    ffmpeg_specifications: list[str] | None = None,
) -> dict[str, RoscompressConfig]:
    resolved: dict[str, RoscompressConfig] = {}
    allowed_keys = frozenset({"quality", "codec", "encoder", "scale", "backend"})
    for specification in specifications or []:
        parsed = _parse_topic_option_spec(
            specification,
            option_name="--video-topic-options",
            allowed_keys=allowed_keys,
        )
        settings = resolved.get(parsed.pattern, defaults)
        for key, value in parsed.values.items():
            if key == "quality":
                topic_quality = _integer(value, key=key)
                if not 0 <= topic_quality <= 51:
                    raise ValueError("quality must be between 0 and 51")
                settings = replace(settings, quality=topic_quality)
            elif key == "codec":
                settings = replace(
                    settings,
                    codec=_choice(
                        value, key=key, choices=frozenset({"h264", "h265", "vp9", "av1"})
                    ),
                )
            elif key == "encoder":
                settings = replace(settings, encoder=None if value in {"auto", "none"} else value)
            elif key == "scale":
                if value in {"none", "original"}:
                    settings = replace(settings, scale=None)
                else:
                    topic_scale = _integer(value, key=key)
                    if topic_scale <= 0:
                        raise ValueError("scale must be positive")
                    settings = replace(settings, scale=topic_scale)
            else:
                settings = replace(
                    settings,
                    backend=_choice(
                        value,
                        key=key,
                        choices=frozenset({"auto", "ffmpeg-cli", "gstreamer", "pyav"}),
                    ),
                )
        resolved[parsed.pattern] = settings
    for specification in ffmpeg_specifications or []:
        pattern, raw_args = _parse_topic_pattern(
            specification,
            option_name="--video-topic-ffmpeg-args",
            syntax="PATTERN:ARGS",
        )
        settings = resolved.get(pattern, defaults)
        topic_args = (
            ()
            if raw_args.strip() == "none"
            else settings.ffmpeg_args + _parse_ffmpeg_args(raw_args)
        )
        resolved[pattern] = replace(
            settings,
            ffmpeg_args=topic_args,
        )
    return resolved


@dataclass(frozen=True, slots=True)
class _ProfileEntry:
    """One compressor's settings and the topics it owns.

    ``pattern`` is ``None`` for the trailing catch-all entry that compresses
    every topic no profile claimed.
    """

    pattern: str | None
    settings: RoscompressConfig
    topics: TopicSelection


@dataclass(frozen=True, slots=True)
class _ProfiledProcessor:
    """A processor kept to report unused profiles after channels are read."""

    kind: str
    pattern: str | None
    processor: TopicMatchingProcessor


def _profile_entries(
    profiles: dict[str, RoscompressConfig], defaults: RoscompressConfig
) -> list[_ProfileEntry]:
    """Profiles in declaration order (first match wins), then the catch-all.

    Overlapping patterns are resolved by excluding every earlier pattern from a
    profile's own selection, so each topic is compressed exactly once.
    """
    patterns = list(profiles)
    entries = [
        _ProfileEntry(
            pattern=pattern,
            settings=profiles[pattern],
            topics=TopicSelection.from_patterns(include=[pattern], exclude=patterns[:index]),
        )
        for index, pattern in enumerate(patterns)
    ]
    entries.append(
        _ProfileEntry(
            pattern=None,
            settings=defaults,
            topics=TopicSelection.from_patterns(exclude=patterns),
        )
    )
    return entries


def _parse_ffmpeg_args(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    try:
        arguments = tuple(shlex.split(value))
    except ValueError as exc:
        raise ValueError(f"invalid FFmpeg arguments: {exc}") from exc
    if not arguments:
        raise ValueError("FFmpeg arguments must not be empty")
    return arguments


def roscompress(
    file: str,
    output: OutputPathOption,
    *,
    force: ForceOverwriteOption = False,
    quality: Annotated[
        QualityOption, Parameter(group=[ENCODING_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT])
    ] = 28,
    codec: Annotated[
        CodecOption, Parameter(group=[ENCODING_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT])
    ] = "h264",
    encoder: Annotated[
        EncoderOption, Parameter(group=[ENCODING_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT])
    ] = None,
    video_topic_options: Annotated[
        VideoTopicOptionsOption,
        Parameter(group=[ENCODING_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT]),
    ] = None,
    ffmpeg_args: Annotated[
        FfmpegArgsOption,
        Parameter(group=[ENCODING_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT]),
    ] = None,
    video_topic_ffmpeg_args: Annotated[
        VideoTopicFfmpegArgsOption,
        Parameter(group=[ENCODING_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT]),
    ] = None,
    resolution: Annotated[
        ResolutionOption, Parameter(group=[POINTCLOUD_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT])
    ] = 0.01,
    pointcloud_topic_options: Annotated[
        PointCloudTopicOptionsOption,
        Parameter(group=[POINTCLOUD_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT]),
    ] = None,
    pc_format: Annotated[
        PointCloudFormatOption,
        Parameter(group=[POINTCLOUD_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT]),
    ] = "cloudini",
    pc_schema: Annotated[
        PointCloudSchemaOption,
        Parameter(group=[POINTCLOUD_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT]),
    ] = "auto",
    pc_encoding: Annotated[
        PointCloudEncodingOption,
        Parameter(group=[POINTCLOUD_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT]),
    ] = "lossy",
    pc_compression: Annotated[
        PointCloudCompressionOption,
        Parameter(group=[POINTCLOUD_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT]),
    ] = "zstd",
    draco_compression_level: Annotated[
        DracoCompressionLevelOption,
        Parameter(group=[POINTCLOUD_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT]),
    ] = 7,
    scale: Annotated[
        ScaleOption, Parameter(group=[ENCODING_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT])
    ] = None,
    image_format: Annotated[
        ImageFormatOption, Parameter(group=[ENCODING_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT])
    ] = "video",
    jpeg_quality: Annotated[
        JpegQualityOption, Parameter(group=[ENCODING_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT])
    ] = 90,
    backend: Annotated[
        BackendOption, Parameter(group=[ENCODING_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT])
    ] = "auto",
    pointcloud: Annotated[
        PointCloudOption, Parameter(group=[POINTCLOUD_GROUP, IMAGE_POINTCLOUD_MODE_CONSTRAINT])
    ] = True,
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
    video_topic_options
        Per-topic video overrides using ``PATTERN:key=value[,key=value...]``. The
        pattern is a case-insensitive, full-match regex. Repeatable.
    ffmpeg_args
        Extra ffmpeg output arguments as one shell-style string. Requires ffmpeg-cli.
    video_topic_ffmpeg_args
        Per-topic extra ffmpeg arguments using ``PATTERN:ARGS``. The pattern is
        a case-insensitive, full-match regex. Repeatable.
    scale
        Cap the maximum image dimension (width or height) while preserving aspect ratio.
        When None, use original resolution.
    resolution
        Resolution for lossy point cloud compression. Default: 0.01.
    pointcloud_topic_options
        Per-topic point-cloud overrides using ``PATTERN:key=value[,key=value...]``.
        The pattern is a case-insensitive, full-match regex. Repeatable.
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
    backend
        Video encoder backend. Default: auto.
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

    if (video_topic_options or ffmpeg_args or video_topic_ffmpeg_args) and image_format != "video":
        logger.error("video topic and FFmpeg options require --image-format video")
        return 1
    if pointcloud_topic_options and not pointcloud:
        logger.error("--pointcloud-topic-options requires --pointcloud enabled")
        return 1

    try:
        defaults = RoscompressConfig(
            image_format=image_format,
            codec=codec,
            quality=quality,
            encoder=encoder,
            backend=backend,
            scale=scale,
            jpeg_quality=jpeg_quality,
            pointcloud=pointcloud,
            resolution=resolution,
            pc_format=pc_format,
            pc_schema=pc_schema,
            pc_encoding=pc_encoding,
            pc_compression=pc_compression,
            draco_compression_level=draco_compression_level,
            pointcloud_drop_invalid=pointcloud_drop_invalid,
            pointcloud_sort_field=pointcloud_sort_field,
            ffmpeg_args=_parse_ffmpeg_args(ffmpeg_args),
        )
        video_topic_settings = _resolve_video_topic_options(
            video_topic_options,
            defaults,
            video_topic_ffmpeg_args,
        )
        pointcloud_topic_settings = _resolve_pointcloud_topic_options(
            pointcloud_topic_options, defaults
        )
        cleanup = resolve_cleanup(defaults)
    except ValueError as exc:
        logger.error(str(exc))  # noqa: TRY400
        return 1

    overwrite_policy = resolve_overwrite_policy(force=force, no_clobber=False)
    assert overwrite_policy is not None  # no_clobber is fixed False here

    # Build the transcode processor chain. Constructors do the real work
    # (encoder probing, codec dependency import), so a missing optional
    # dependency or unavailable encoder surfaces here as a clean CLI error.
    extras: list[InputProcessor] = []
    profiled: list[_ProfiledProcessor] = []
    try:
        if image_format == "video":
            video_entries = _profile_entries(video_topic_settings, defaults)
            for entry in video_entries:
                processor = create_video_compress_processor(
                    entry.settings,
                    topics=entry.topics,
                    shared_by=len(video_entries),
                )
                extras.append(processor)
                profiled.append(_ProfiledProcessor("video", entry.pattern, processor))
        elif image_format in {"jpeg", "png"}:
            extras.append(create_image_processor(defaults))

        if not pointcloud:
            cleanup_only = create_pointcloud_cleanup_processor(defaults)
            if cleanup_only is not None:
                extras.append(cleanup_only)
        else:
            pointcloud_entries = _profile_entries(pointcloud_topic_settings, defaults)
            workers = pointcloud_worker_count()
            for entry in pointcloud_entries:
                # Cloudini fuses cleanup into its native encode; every other
                # format needs the standalone pass ahead of its compressor.
                if not fuses_cloudini_cleanup(entry.settings):
                    cleanup_processor = create_pointcloud_cleanup_processor(
                        defaults, topics=entry.topics
                    )
                    if cleanup_processor is not None:
                        extras.append(cleanup_processor)
                processor = create_pointcloud_compress_processor(
                    entry.settings,
                    workers=workers,
                    topics=entry.topics,
                )
                extras.append(processor)
                profiled.append(_ProfiledProcessor("point-cloud", entry.pattern, processor))
    except ImportError:
        uses_draco = pc_format == "draco" or any(
            settings.pc_format == "draco" for settings in pointcloud_topic_settings.values()
        )
        extra = "draco" if uses_draco else "pointcloud"
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
        logger.info(f"Image mode: video ({encoder or 'auto'}, {codec}, backend={backend})")
        logger.info(f"Quality (CRF): {quality}")
        if defaults.ffmpeg_args:
            logger.info("FFmpeg args: %s", shlex.join(defaults.ffmpeg_args))
        for pattern, settings in video_topic_settings.items():
            logger.info(
                "Video topics matching %s: %s q%d scale=%s encoder=%s backend=%s",
                pattern,
                settings.codec,
                settings.quality,
                settings.scale or "original",
                settings.encoder or "auto",
                settings.backend,
            )
            if settings.ffmpeg_args:
                logger.info(
                    "Video topics matching %s ffmpeg args: %s",
                    pattern,
                    shlex.join(settings.ffmpeg_args),
                )
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
        for pattern, settings in pointcloud_topic_settings.items():
            logger.info(
                "Point cloud topics matching %s: %s (schema=%s, resolution=%g, encoding=%s, "
                "compression=%s, draco-level=%d)",
                pattern,
                settings.pc_format,
                settings.pc_schema,
                settings.resolution,
                settings.pc_encoding,
                settings.pc_compression,
                settings.draco_compression_level,
            )
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

    topics_by_kind: dict[str, set[str]] = {}
    for profile in profiled:
        topics_by_kind.setdefault(profile.kind, set()).update(profile.processor.matched_topics)
    for profile in profiled:
        if profile.pattern is None or profile.processor.matched_topics:
            continue
        pattern_selection = TopicSelection.from_patterns(include=[profile.pattern])
        if any(pattern_selection.selects(topic) for topic in topics_by_kind[profile.kind]):
            logger.warning(
                "%s profile %r was fully shadowed by an earlier profile; "
                "its per-topic options were not applied",
                profile.kind,
                profile.pattern,
            )
        else:
            logger.warning(
                "No %s input topics matched %r; its per-topic options were not applied",
                profile.kind,
                profile.pattern,
            )

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
