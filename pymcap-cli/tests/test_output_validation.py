from __future__ import annotations

import io
from typing import TYPE_CHECKING

from pymcap_cli.core.output_validation import validate_mcap_outputs
from small_mcap import McapWriter

if TYPE_CHECKING:
    from pathlib import Path


def _write_topic_counts(
    path: Path,
    counts: dict[str, int],
    *,
    use_statistics: bool = True,
) -> None:
    output = io.BytesIO()
    writer = McapWriter(output, use_statistics=use_statistics)
    writer.start()
    writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")
    for channel_id, topic in enumerate(counts, start=1):
        writer.add_channel(
            channel_id=channel_id,
            topic=topic,
            message_encoding="json",
            schema_id=1,
        )
        for index in range(counts[topic]):
            writer.add_message(
                channel_id=channel_id,
                log_time=index,
                publish_time=index,
                data=b"{}",
            )
    writer.finish()
    path.write_bytes(output.getvalue())


def test_validate_mcap_outputs_detects_per_topic_loss_when_total_is_unchanged(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mcap"
    output = tmp_path / "output.mcap"
    _write_topic_counts(source, {"/keep": 2, "/other": 2})
    _write_topic_counts(output, {"/keep": 1, "/other": 3})

    error = validate_mcap_outputs([source], [output])

    assert error == "output lost messages on preserved topics: /keep (2 -> 1)"


def test_validate_mcap_outputs_allows_loss_only_on_matching_topics(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    output = tmp_path / "output.mcap"
    _write_topic_counts(source, {"/keep": 2, "/lossy/camera": 4})
    _write_topic_counts(output, {"/keep": 2, "/lossy/camera": 1})

    error = validate_mcap_outputs(
        [source],
        [output],
        lossy_topic_patterns=[r"/lossy/.*"],
    )

    assert error is None


def test_validate_mcap_outputs_preserves_selected_topics_only(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    output = tmp_path / "output.mcap"
    _write_topic_counts(source, {"/selected": 3, "/unselected": 3})
    _write_topic_counts(output, {"/selected": 2})

    error = validate_mcap_outputs(
        [source],
        [output],
        preserved_topic_patterns=["/selected"],
        lossy_topic_patterns=(),
    )

    assert error == "output lost messages on preserved topics: /selected (3 -> 2)"


def test_validate_mcap_outputs_does_not_claim_per_topic_validation_without_counts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mcap"
    output = tmp_path / "output.mcap"
    _write_topic_counts(source, {"/keep": 2}, use_statistics=False)
    _write_topic_counts(output, {"/keep": 2})

    error = validate_mcap_outputs(
        [source],
        [output],
        lossy_topic_patterns=(),
    )

    assert error is not None
    assert "per-topic message counts unavailable" in error


def test_validate_mcap_outputs_allows_explicit_loss_without_counts(tmp_path: Path) -> None:
    source = tmp_path / "source.mcap"
    output = tmp_path / "output.mcap"
    _write_topic_counts(source, {"/lossy": 2}, use_statistics=False)
    _write_topic_counts(output, {}, use_statistics=False)

    error = validate_mcap_outputs(
        [source],
        [output],
        lossy_topic_patterns=[".*"],
    )

    assert error is None
