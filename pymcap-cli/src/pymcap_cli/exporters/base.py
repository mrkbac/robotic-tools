"""Base class and shared types for pluggable exporters.

The driver (:func:`pymcap_cli.exporters.driver.run_export`) calls into an
:class:`Exporter` to obtain per-topic :class:`Writer` instances and forwards
decoded MCAP messages to them. Artifact exporters use this low-level API
directly; record-oriented formats extend ``StructuredExporter`` instead.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from small_mcap import Channel, DecodedMessage, Schema, Summary

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


WrittenValue_contra = TypeVar("WrittenValue_contra", contravariant=True)


class Writer(Protocol[WrittenValue_contra]):
    """Destination that consumes values for the duration of one export run."""

    def write(self, value: WrittenValue_contra, /) -> None:
        """Persist one value."""
        ...

    def close(self) -> None:
        """Flush and release resources."""
        ...


class Exporter(ABC):
    """Low-level base for artifact and transforming exporters.

    Subclasses implement :meth:`open_topic` and optionally specialize decoder
    factories, channel acceptance, output validation, or end-of-run handling.
    """

    name: ClassVar[str]
    """Short format identifier, e.g. ``"csv"``."""

    def decoder_factories(self) -> list[Any]:
        """Decoder factories to plug into ``small_mcap.read_message_decoded``.

        Order matters — first-matching factory wins. Implementations should
        return any format-specific decoders followed by the standard ROS2 CDR
        decoder factory.
        """
        from mcap_ros2_support_fast.decoder import (  # noqa: PLC0415
            DecoderFactory as Ros2DecoderFactory,
        )

        return [Ros2DecoderFactory()]

    def accepts(self, channel: Channel, schema: Schema | None) -> bool:  # noqa: ARG002
        """Return whether this exporter can handle a channel and schema."""
        return True

    @abstractmethod
    def open_topic(self, ctx: TopicContext) -> Writer[DecodedMessage]:
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

    def validate_input(self, summary: Summary | None) -> None:  # noqa: B027
        """Validate command-specific input requirements before creating output."""

    def finish(  # noqa: B027
        self,
        output_path: Path,
        counts: Mapping[int, int],
    ) -> None:
        """Called once after every per-topic :class:`Writer` has closed.

        ``counts`` maps each ``TopicContext.writer_key`` to the number of
        messages successfully written. Default is a no-op.
        """
