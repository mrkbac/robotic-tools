"""Video exporter — one MP4 per image topic, written into an output directory.

The CLI's ``--output`` is a directory; each topic writes to
``<output>/<safe_topic>.mp4``. This matches the rest of the export family
(images / pcd / csv / json / parquet) — every exporter writes per-topic
files into a single directory and the default
:meth:`Exporter.validate_output` handles directory creation + ``--force``.

Cross-topic grid composition (the legacy ``video`` behaviour for >1 topic)
is intentionally not supported — it requires a shared cross-topic writer
that the per-topic :class:`Exporter` protocol does not model. Re-add when
the protocol grows a cross-topic shape.

Two encoder paths, selected by ``mode`` (:class:`EncoderMode`):

* ``PYAV`` — in-process via PyAV bindings to libav/ffmpeg. Default.
* ``FFMPEG_CLI`` — spawns ``ffmpeg`` as a subprocess and pipes raw RGB
  frames to it. Useful when PyAV's encoder open fails (e.g. videotoolbox
  on small streams) or when you want the system's newer ffmpeg build.
* ``AUTO`` — try PyAV; on first-frame open failure, fall back transparently
  to ``FFMPEG_CLI``.

The underlying ffmpeg encoder name (``libx264`` / ``h264_videotoolbox`` /
``h264_nvenc`` / …) is still chosen by ``--encoder`` (:class:`EncoderBackend`)
and applies to both modes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from mcap_codec_support.video import (
    IMAGE_SCHEMAS,
    EncoderBackend,
    EncoderMode,
    VideoCodec,
    VideoEncoderError,
)

from pymcap_cli.exporters._common import (
    message_timestamps_ns,
    normalize_schema_name,
    prepare_output_file,
    schema_name_in,
)
from pymcap_cli.exporters.base import Ros2Exporter, TopicWriter
from pymcap_cli.exporters.video_file_writer import (
    VideoFileWriterSession,
    create_video_file_writer,
)

if TYPE_CHECKING:
    from pathlib import Path

    from small_mcap import DecodedMessage, Schema

    from pymcap_cli.exporters.base import TopicContext

logger = logging.getLogger(__name__)

_CANONICAL_IMAGE_SCHEMAS = frozenset(normalize_schema_name(s) for s in IMAGE_SCHEMAS)


# ---------------------------------------------------------------------------
# Topic writer — picks a strategy per topic
# ---------------------------------------------------------------------------


class _VideoTopicWriter(TopicWriter):
    """Encodes one image topic to one MP4 file via the chosen strategy."""

    def __init__(
        self,
        path: Path,
        *,
        codec: VideoCodec,
        encoder_backend: EncoderBackend,
        quality: int,
        mode: EncoderMode,
    ) -> None:
        self.path = path
        self._codec = codec
        self._encoder_backend = encoder_backend
        self._quality = quality
        self._mode = mode
        self._writer: VideoFileWriterSession = create_video_file_writer(
            path,
            codec=codec,
            encoder_backend=encoder_backend,
            quality=quality,
            mode=mode,
            on_fallback=logger.warning,
        )
        self._closed = False

    def write(self, msg: DecodedMessage) -> None:
        schema_name = normalize_schema_name(msg.schema.name) if msg.schema else ""
        if schema_name not in _CANONICAL_IMAGE_SCHEMAS:
            raise VideoEncoderError(f"Unexpected schema {schema_name!r} on {msg.channel.topic}")

        log_time_ns, _ = message_timestamps_ns(msg)
        try:
            self._writer.write_message(msg.decoded_message, schema_name, log_time_ns)
        except VideoEncoderError as exc:
            raise VideoEncoderError(f"Encoding failed on {msg.channel.topic}: {exc}") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            count = self._writer.close()
        except VideoEncoderError as exc:
            raise VideoEncoderError(f"Flush failed for {self.path}: {exc}") from exc
        logger.info(f"[green]✓[/green] {self.path}  ({count} frame(s))")


class VideoExporter(Ros2Exporter):
    """Per-topic MP4 exporter — writes ``<output>/<safe_topic>.mp4`` per topic.

    Uses the default :meth:`Exporter.validate_output` (directory mode).
    """

    name: ClassVar[str] = "video"

    def __init__(
        self,
        *,
        codec: VideoCodec,
        encoder_backend: EncoderBackend,
        quality: int,
        mode: EncoderMode = EncoderMode.AUTO,
    ) -> None:
        self._codec = codec
        self._encoder_backend = encoder_backend
        self._quality = quality
        self._mode = mode

    def accepts(self, schema: Schema | None) -> bool:
        return schema_name_in(schema, _CANONICAL_IMAGE_SCHEMAS)

    def open_topic(self, ctx: TopicContext) -> _VideoTopicWriter:
        path = prepare_output_file(ctx.output_path / f"{ctx.safe_filename}.mp4", force=ctx.force)
        return _VideoTopicWriter(
            path,
            codec=self._codec,
            encoder_backend=self._encoder_backend,
            quality=self._quality,
            mode=self._mode,
        )
