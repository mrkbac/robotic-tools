"""Video encoding command for pymcap-cli — one MP4 per image topic.

``--output`` is a directory; one ``<safe_topic>.mp4`` is written per topic.
Wraps :class:`pymcap_cli.exporters.video_exporter.VideoExporter`.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Annotated

from cyclopts import Group, Parameter
from mcap_codec_support.video import (
    EncoderBackend,
    EncoderMode,
    VideoCodec,
    VideoEncoderError,
)
from rich.console import Console

from pymcap_cli.cmd._message_filter_options import (
    EarlyBailOption,
    EndTimeOption,
    ExcludeTopicOption,
    StartTimeOption,
    TopicOption,
    create_message_filter,
)
from pymcap_cli.exporters import run_export
from pymcap_cli.exporters.video_exporter import VideoExporter

logger = logging.getLogger(__name__)
console = Console()

OUTPUT_GROUP = Group("Output Options")
ENCODING_GROUP = Group("Encoding Options")


class QualityPreset(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# CRF-style quality values per codec for each preset (lower = better).
_QUALITY_PRESETS: dict[VideoCodec, dict[QualityPreset, int]] = {
    VideoCodec.H264: {QualityPreset.HIGH: 32, QualityPreset.MEDIUM: 35, QualityPreset.LOW: 40},
    VideoCodec.H265: {QualityPreset.HIGH: 32, QualityPreset.MEDIUM: 35, QualityPreset.LOW: 40},
    VideoCodec.VP9: {QualityPreset.HIGH: 42, QualityPreset.MEDIUM: 45, QualityPreset.LOW: 50},
    VideoCodec.AV1: {QualityPreset.HIGH: 37, QualityPreset.MEDIUM: 40, QualityPreset.LOW: 45},
}


def video(
    file: str,
    *,
    topic: TopicOption = None,
    exclude_topic: ExcludeTopicOption = None,
    start: StartTimeOption = "",
    end: EndTimeOption = "",
    early_bail: EarlyBailOption = False,
    output: Annotated[
        Path,
        Parameter(name=["-o", "--output"], group=OUTPUT_GROUP),
    ],
    codec: Annotated[
        VideoCodec,
        Parameter(name=["--codec"], group=ENCODING_GROUP),
    ] = VideoCodec.H264,
    quality: Annotated[
        QualityPreset,
        Parameter(name=["--quality"], group=ENCODING_GROUP),
    ] = QualityPreset.MEDIUM,
    crf: Annotated[
        int | None,
        Parameter(name=["--crf"], group=ENCODING_GROUP),
    ] = None,
    encoder: Annotated[
        EncoderBackend,
        Parameter(name=["--encoder"], group=ENCODING_GROUP),
    ] = EncoderBackend.AUTO,
    mode: Annotated[
        EncoderMode,
        Parameter(name=["--mode"], group=ENCODING_GROUP),
    ] = EncoderMode.AUTO,
    force: Annotated[
        bool,
        Parameter(name=["-f", "--force"], group=OUTPUT_GROUP),
    ] = False,
) -> int:
    """Encode video from image topics in an MCAP file.

    One MP4 is written to ``<output>/<safe_topic>.mp4`` per matched topic.

    Examples
    --------
        pymcap-cli video data.mcap -t /camera/front -o ./out
        pymcap-cli video data.mcap -t /cam/left -t /cam/right -o ./out
    """
    try:
        message_filter = create_message_filter(
            topic=topic,
            exclude_topic=exclude_topic,
            start=start,
            end=end,
            early_bail=early_bail,
        )
    except ValueError as exc:
        logger.error(str(exc))  # noqa: TRY400
        return 1

    if not message_filter.has_positive_topics:
        logger.error("At least one --topic is required.")
        return 1

    quality_value = crf if crf is not None else _QUALITY_PRESETS[codec][quality]

    try:
        exporter = VideoExporter(
            codec=codec,
            encoder_backend=encoder,
            quality=quality_value,
            mode=mode,
        )
    except VideoEncoderError as exc:
        logger.error(str(exc))  # noqa: TRY400
        return 1

    return run_export(
        file=file,
        output=output,
        exporter=exporter,
        message_filter=message_filter,
        force=force,
    )
