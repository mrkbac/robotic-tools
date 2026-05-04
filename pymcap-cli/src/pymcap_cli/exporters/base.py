"""Base class and shared types for pluggable exporters.

The driver (:func:`pymcap_cli.exporters.driver.run_export`) calls into an
:class:`Exporter` to obtain per-topic :class:`TopicWriter` instances and
forwards decoded MCAP messages to them. A new export format is one new
:class:`Exporter` subclass.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from small_mcap import Channel, DecodedMessage, Schema

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TopicContext:
    """Context handed to :meth:`Exporter.open_topic` when a topic first appears."""

    topic: str
    schema: Schema | None
    channel: Channel
    writer_key: int  # Stable per-channel key used by the driver.
    output_path: Path  # Resolved by :meth:`Exporter.validate_output` — dir or file.
    safe_filename: str  # Topic name sanitised for use as a filesystem path component.
    force: bool


class TopicWriter(ABC):
    """Per-topic writer. Lives for the duration of one export run."""

    @abstractmethod
    def write(self, msg: DecodedMessage) -> None:
        """Persist a single decoded message."""

    @abstractmethod
    def close(self) -> None:
        """Flush and release resources."""


class Exporter(ABC):
    """Base class for per-format exporters.

    Subclasses must implement :meth:`decoder_factories`, :meth:`accepts`, and
    :meth:`open_topic`. The lifecycle hooks :meth:`setup` and :meth:`finish`
    default to no-ops; override them when a format needs global state (e.g.
    a shared writer-thread pool, an end-of-run index file).
    """

    name: ClassVar[str]
    """Short format identifier, e.g. ``"csv"``."""

    @abstractmethod
    def decoder_factories(self) -> list[Any]:
        """Decoder factories to plug into ``small_mcap.read_message_decoded``.

        Order matters — first-matching factory wins. Implementations should
        return any format-specific decoders followed by the standard ROS2 CDR
        decoder factory.
        """

    @abstractmethod
    def accepts(self, schema: Schema | None) -> bool:
        """Return True if this exporter can handle messages with this schema.

        Called by the driver to filter messages at the reader level — the
        chunk decoder skips entire channels whose schema this method rejects,
        so unsupported / blob schemas never get CDR-decoded.
        """

    @abstractmethod
    def open_topic(self, ctx: TopicContext) -> TopicWriter:
        """Create a per-topic writer. Called once per topic on first message."""

    def validate_output(self, output: str | Path | None, *, force: bool) -> Path | None:
        """Resolve and validate the user-provided output path.

        Default treats ``output`` as a directory (the historic behaviour for
        per-topic-file exporters). Override when the exporter writes a single
        output file — e.g. video MP4 or plot HTML — or accepts ``None`` for
        interactive output.
        """
        from pymcap_cli.exporters._common import validate_output_dir  # noqa: PLC0415

        if output is None:
            logger.error("this exporter requires an output directory.")
            return None
        return validate_output_dir(output, force=force)

    def setup(self, output_path: Path) -> None:  # noqa: B027
        """Called once before iteration. Default is a no-op."""

    def finish(  # noqa: B027
        self,
        output_path: Path,
        counts: Mapping[int, int],
    ) -> None:
        """Called once after every :class:`TopicWriter` has closed.

        ``counts`` maps each ``TopicContext.writer_key`` to the number of
        messages successfully written. Default is a no-op.
        """


class Ros2Exporter(Exporter):
    """Exporter whose only decoder is the standard ROS2 CDR decoder."""

    def decoder_factories(self) -> list[Any]:
        from mcap_ros2_support_fast.decoder import (  # noqa: PLC0415
            DecoderFactory as Ros2DecoderFactory,
        )

        return [Ros2DecoderFactory()]


class JsonRos2Exporter(Exporter):
    """Exporter that accepts JSON-encoded messages plus standard ROS2 CDR."""

    def decoder_factories(self) -> list[Any]:
        from mcap_ros2_support_fast.decoder import (  # noqa: PLC0415
            DecoderFactory as Ros2DecoderFactory,
        )
        from small_mcap import JSONDecoderFactory  # noqa: PLC0415

        return [JSONDecoderFactory(), Ros2DecoderFactory()]
