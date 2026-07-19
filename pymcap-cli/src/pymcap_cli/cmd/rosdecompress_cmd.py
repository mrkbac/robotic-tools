"""Command to decompress CompressedVideo and CompressedPointCloud2 topics in MCAP files.

Thin preset over the processing pipeline: builds the decompress processors
(video / point cloud) and runs them through ``run_processor`` — the inverse of
``roscompress``. The decompression logic lives in the reusable processors
(``core/processors/decompress.py``).
"""

import logging
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Group, Parameter, validators
from mcap_codec_support.video import EncoderMode
from rich.console import Console

from pymcap_cli.cmd._arg_constraints import constraint_group, requires_value
from pymcap_cli.cmd._cli_options import (
    ENCODING_GROUP,
    BackendOption,
    ForceOverwriteOption,
    OutputPathOption,
)
from pymcap_cli.cmd._run_processor import resolve_overwrite_policy, run_processor
from pymcap_cli.core.mcap_processor import InputOptions, OutputOptions
from pymcap_cli.core.mcap_transform import print_size_comparison
from pymcap_cli.utils import output_overwrites_input

logger = logging.getLogger(__name__)
console = Console()

VIDEO_GROUP = Group("Video")
POINTCLOUD_GROUP = Group("Point Cloud")

# The video knobs only apply under --video, and --jpeg-quality only to --video-format compressed.
_MODE_CONSTRAINT = constraint_group(
    requires_value("--video-format", "--video", True, hint="--video enabled"),
    requires_value("--jpeg-quality", "--video", True, hint="--video enabled"),
    requires_value("--backend", "--video", True, hint="--video enabled"),
    requires_value(
        "--jpeg-quality", "--video-format", "compressed", hint="--video-format compressed"
    ),
)


def rosdecompress(
    file: str,
    output: OutputPathOption,
    *,
    force: ForceOverwriteOption = False,
    video: Annotated[
        bool,
        Parameter(
            name=["--video"],
            group=[VIDEO_GROUP, _MODE_CONSTRAINT],
        ),
    ] = True,
    video_format: Annotated[
        Literal["compressed", "raw"],
        Parameter(
            name=["--video-format"],
            group=[VIDEO_GROUP, _MODE_CONSTRAINT],
        ),
    ] = "compressed",
    jpeg_quality: Annotated[
        int,
        Parameter(
            name=["--jpeg-quality"],
            group=[VIDEO_GROUP, _MODE_CONSTRAINT],
            validator=validators.Number(gte=1, lte=100),
        ),
    ] = 90,
    backend: Annotated[BackendOption, Parameter(group=[ENCODING_GROUP, _MODE_CONSTRAINT])] = "auto",
    pointcloud: Annotated[
        bool,
        Parameter(
            name=["--pointcloud"],
            group=POINTCLOUD_GROUP,
        ),
    ] = True,
) -> int:
    """Decompress ROS MCAP by converting compressed topics back to standard formats.

    Converts CompressedVideo topics back to CompressedImage (JPEG) or raw Image,
    CompressedPointCloud2 and Foxglove CompressedPointCloud topics back to PointCloud2.

    Parameters
    ----------
    file
        Input MCAP file (local file or HTTP/HTTPS URL).
    output
        Output filename.
    force
        Force overwrite of output file without confirmation.
    video
        Enable video decompression. Default: True.
    video_format
        Output format for video topics: "compressed" (JPEG) or "raw" (uncompressed Image).
        Default: compressed.
    jpeg_quality
        JPEG quality (1-100) when video_format=compressed. Default: 90.
    backend
        Video decoder backend: auto, pyav, or ffmpeg-cli. Default: auto.
    pointcloud
        Enable point cloud decompression. Default: True.
    """
    if output_overwrites_input(file, output):
        logger.error("Output path is the same file as the input; choose a different output file.")
        return 1

    overwrite_policy = resolve_overwrite_policy(force=force, no_clobber=False)
    assert overwrite_policy is not None

    encoder_mode = EncoderMode(backend)
    console.print(f"[cyan]Input:[/cyan] {file}")
    console.print(f"[cyan]Output:[/cyan] {output}")
    if video:
        console.print(f"[cyan]Video format:[/cyan] {video_format} (backend={encoder_mode.value})")
    else:
        console.print("[cyan]Video decompression:[/cyan] disabled")
    pc_state = "enabled" if pointcloud else "disabled"
    console.print(f"[cyan]Point cloud decompression:[/cyan] {pc_state}")

    extras = []
    if video:
        from pymcap_cli.core.processors.decompress import (  # noqa: PLC0415
            VideoDecompressProcessor,
        )

        extras.append(
            VideoDecompressProcessor(
                video_format=video_format,
                jpeg_quality=jpeg_quality,
                backend=encoder_mode,
            )
        )
    if pointcloud:
        from pymcap_cli.core.processors.decompress import (  # noqa: PLC0415
            PointcloudDecompressProcessor,
        )

        extras.append(PointcloudDecompressProcessor())

    input_options = InputOptions.from_args(extra_processors=extras or None)
    output_options = OutputOptions(output_processors=[], overwrite_policy=overwrite_policy)

    try:
        result = run_processor(
            files=[file],
            output=output,
            input_options=input_options,
            output_options=output_options,
        )
    except Exception:
        logger.exception("Error during decompression")
        output.unlink(missing_ok=True)
        return 1

    console.print()
    written = result.stats.writer_statistics.message_count
    console.print(f"[green]Messages written:[/green] {written:,}")

    input_size = _local_size(file)
    if input_size and output.exists():
        print_size_comparison(input_size, output.stat().st_size)

    return 0


def _local_size(file: str) -> int:
    try:
        return Path(file).stat().st_size
    except OSError:
        return 0
