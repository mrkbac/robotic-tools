"""End-to-end pipeline test for ``TopicAliasProcessor``.

Verifies that the framework's ``register_channel`` + fan-out ``on_message``
flow actually produce an output file with both the original and the alias
topics, each carrying the same message count and payloads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymcap_cli.constants import DEFAULT_CHUNK_SIZE
from pymcap_cli.core.mcap_processor import (
    InputFile,
    InputOptions,
    McapProcessor,
    OutputOptions,
    ProcessingOptions,
)
from pymcap_cli.core.processors.topic_alias import TopicAliasProcessor
from small_mcap import get_summary, read_message_decoded

from tests.fixtures.mcap_generator import create_multi_topic_mcap

if TYPE_CHECKING:
    from pathlib import Path

    from pymcap_cli.core.processors.base import InputProcessor


@pytest.mark.e2e
def test_topic_alias_writes_both_original_and_alias(tmp_path: Path) -> None:
    src = tmp_path / "input.mcap"
    src.write_bytes(
        create_multi_topic_mcap(
            topics=["/sensor/raw"],
            messages_per_topic=20,
            chunk_size=4096,
        )
    )
    out = tmp_path / "out.mcap"

    alias: list[InputProcessor] = [TopicAliasProcessor({r"^/sensor/(.+)$": r"/sensor_mirror/\1"})]

    with src.open("rb") as fh:
        per_input = InputOptions.from_args(extra_processors=alias)
        options = ProcessingOptions(
            inputs=[InputFile(stream=fh, size=src.stat().st_size, options=per_input)],
            input_options=InputOptions.from_args(),
            output_options=OutputOptions(
                compression="zstd",
                chunk_size=DEFAULT_CHUNK_SIZE,
            ),
        )
        with out.open("wb") as out_fh:
            McapProcessor(options).process(output_stream=out_fh)

    with out.open("rb") as fh:
        summary = get_summary(fh)
    assert summary is not None
    topics = {ch.topic for ch in summary.channels.values()}
    assert "/sensor/raw" in topics
    assert "/sensor_mirror/raw" in topics

    # Each topic should have exactly 20 messages.
    counts: dict[str, int] = {}
    payloads_by_topic: dict[str, list[bytes]] = {}
    with out.open("rb") as fh:
        for decoded in read_message_decoded(fh):
            topic = decoded.channel.topic
            counts[topic] = counts.get(topic, 0) + 1
            payloads_by_topic.setdefault(topic, []).append(bytes(decoded.message.data))
    assert counts["/sensor/raw"] == 20
    assert counts["/sensor_mirror/raw"] == 20
    # And every payload on the mirror matches the original (same order).
    assert payloads_by_topic["/sensor/raw"] == payloads_by_topic["/sensor_mirror/raw"]


@pytest.mark.e2e
def test_topic_alias_multiple_aliases_per_channel(tmp_path: Path) -> None:
    src = tmp_path / "input.mcap"
    src.write_bytes(
        create_multi_topic_mcap(
            topics=["/raw"],
            messages_per_topic=10,
            chunk_size=4096,
        )
    )
    out = tmp_path / "out.mcap"

    alias: list[InputProcessor] = [
        TopicAliasProcessor({r"^/raw$": ["/copy_a", "/copy_b", "/copy_c"]})
    ]

    with src.open("rb") as fh:
        per_input = InputOptions.from_args(extra_processors=alias)
        options = ProcessingOptions(
            inputs=[InputFile(stream=fh, size=src.stat().st_size, options=per_input)],
            input_options=InputOptions.from_args(),
            output_options=OutputOptions(compression="zstd", chunk_size=DEFAULT_CHUNK_SIZE),
        )
        with out.open("wb") as out_fh:
            McapProcessor(options).process(output_stream=out_fh)

    with out.open("rb") as fh:
        summary = get_summary(fh)
    assert summary is not None
    topics_by_id = {ch.id: ch.topic for ch in summary.channels.values()}
    by_topic = {
        topics_by_id[ch_id]: count
        for ch_id, count in summary.statistics.channel_message_counts.items()
    }
    print("by_topic:", by_topic)
    assert {"/raw", "/copy_a", "/copy_b", "/copy_c"} <= set(by_topic.keys())
    # Each topic gets the full 10 messages.
    assert by_topic["/raw"] == 10
    assert by_topic["/copy_a"] == 10
    assert by_topic["/copy_b"] == 10
    assert by_topic["/copy_c"] == 10
