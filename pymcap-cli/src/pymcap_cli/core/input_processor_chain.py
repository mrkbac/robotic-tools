"""Build the per-input ``InputProcessor`` chain implied by an ``InputOptions``.

Keeps the dispatcher (``mcap_processor.py``) free of any concrete-processor
imports: it only knows about the ``InputProcessor`` interface from
``pymcap_cli.core.processors.base``. Concrete classes are wired in here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymcap_cli.core.processors.always_decode import AlwaysDecodeProcessor
from pymcap_cli.core.processors.attachment_filter import AttachmentFilterProcessor
from pymcap_cli.core.processors.latching import LatchingProcessor
from pymcap_cli.core.processors.metadata_filter import MetadataFilterProcessor
from pymcap_cli.core.processors.time_filter import TimeFilterProcessor
from pymcap_cli.core.processors.topic_filter import TopicFilterProcessor
from pymcap_cli.utils import compile_topic_patterns

if TYPE_CHECKING:
    from pymcap_cli.core.input_options import InputOptions
    from pymcap_cli.core.processors.base import InputProcessor


def build_input_processors(options: InputOptions) -> list[InputProcessor]:
    """Construct the per-input processor chain implied by ``options``.

    The chain is:

    1. Positive-vote / observer processors (``LatchingProcessor``) — must run
       first so they see every record before downstream filters apply.
    2. ``AlwaysDecodeProcessor`` if requested.
    3. Standard filters: time, topic, metadata, attachment.
    4. Caller-supplied ``extra_processors`` appended last.
    """
    procs: list[InputProcessor] = []

    if options.latch_topics or options.latch_from_metadata:
        procs.append(
            LatchingProcessor(
                patterns=compile_topic_patterns(options.latch_topics),
                from_metadata=options.latch_from_metadata,
            )
        )

    if options.always_decode_chunk:
        procs.append(AlwaysDecodeProcessor())

    if options.start_time_ns is not None or options.end_time_ns is not None:
        procs.append(
            TimeFilterProcessor(
                options.start_time_ns,
                options.end_time_ns,
                invert=options.invert_time,
            )
        )

    if options.include_topics or options.exclude_topics:
        procs.append(
            TopicFilterProcessor(
                options.include_topics,
                options.exclude_topics,
                invert=options.invert_topics,
            )
        )

    if not options.include_metadata:
        procs.append(MetadataFilterProcessor(include=False))

    if not options.include_attachments:
        procs.append(AttachmentFilterProcessor(include=False))

    procs.extend(options.extra_processors)

    return procs
