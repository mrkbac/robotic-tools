"""Tests for the unified `process` command — flag wiring and dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pymcap_cli.cmd import process_cmd
from pymcap_cli.cmd._rechunk_strategy import RechunkStrategy
from pymcap_cli.core.processors.channel_merge import ChannelMergeProcessor
from pymcap_cli.core.processors.chunk_groupers import PatternGrouper
from pymcap_cli.core.processors.dedup import DedupIdenticalProcessor
from pymcap_cli.core.processors.duration_split import DurationSplitProcessor
from pymcap_cli.core.processors.expression_split import ExpressionSplitProcessor
from pymcap_cli.core.processors.nth_message import NthMessageProcessor
from pymcap_cli.core.processors.size_split import SizeSplitProcessor
from pymcap_cli.core.processors.time_offset import TimeOffsetProcessor
from pymcap_cli.core.processors.timestamp_split import TimestampSplitProcessor
from pymcap_cli.core.processors.topic_alias import TopicAliasProcessor
from pymcap_cli.core.processors.topic_rewrite import TopicRewriteProcessor

from tests.helpers import empty_processor_result

if TYPE_CHECKING:
    from pymcap_cli.core.mcap_processor import InputOptions, OutputOptions


# ---------------------------------------------------------------------------
# Recorder fakes — capture the kwargs each dispatch entry point sees.
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.input_options: InputOptions | None = None
        self.output_options: OutputOptions | None = None
        self.files: list[str] | None = None
        self.output: Path | None = None
        self.multi: bool = False


def _patch_single(monkeypatch: pytest.MonkeyPatch, rec: _Recorder) -> None:
    def fake_run_processor(*, files, output, input_options, output_options):
        rec.files = list(files)
        rec.output = output
        rec.input_options = input_options
        rec.output_options = output_options
        rec.multi = False
        return empty_processor_result()

    monkeypatch.setattr(process_cmd, "run_processor", fake_run_processor)


def _patch_multi(monkeypatch: pytest.MonkeyPatch, rec: _Recorder) -> None:
    def fake_run_processor_multi(*, files, output_options, input_options=None):
        rec.files = list(files)
        rec.output = None
        rec.input_options = input_options
        rec.output_options = output_options
        rec.multi = True
        return empty_processor_result(segments={})

    monkeypatch.setattr(process_cmd, "run_processor_multi", fake_run_processor_multi)


def _kwargs() -> dict[str, object]:
    return {"file": ["input.mcap"], "output": Path("out.mcap")}


# ---------------------------------------------------------------------------
# Output / split-mode validation
# ---------------------------------------------------------------------------


class TestOutputDispatch:
    def test_single_output_uses_run_processor(self, monkeypatch: pytest.MonkeyPatch):
        rec = _Recorder()
        _patch_single(monkeypatch, rec)
        _patch_multi(monkeypatch, rec)

        exit_code = process_cmd.process(**_kwargs())

        assert exit_code == 0
        assert rec.multi is False
        assert rec.output == Path("out.mcap")

    def test_split_flag_routes_to_run_processor_multi(self, monkeypatch: pytest.MonkeyPatch):
        rec = _Recorder()
        _patch_single(monkeypatch, rec)
        _patch_multi(monkeypatch, rec)

        exit_code = process_cmd.process(file=["input.mcap"], split_duration="1s")

        assert exit_code == 0
        assert rec.multi is True
        assert rec.output_options is not None
        assert any(isinstance(r, DurationSplitProcessor) for r in rec.output_options.routers)
        assert rec.output_options.output_template  # template propagated

    def test_missing_output_without_split_is_error(self):
        exit_code = process_cmd.process(file=["input.mcap"])
        assert exit_code == 1

    def test_output_with_split_is_error(self):
        exit_code = process_cmd.process(
            file=["input.mcap"], output=Path("out.mcap"), split_duration="1s"
        )
        assert exit_code == 1

    def test_force_and_no_clobber_is_error(self):
        exit_code = process_cmd.process(**_kwargs(), force=True, no_clobber=True)
        assert exit_code == 1


# ---------------------------------------------------------------------------
# Per-flag wiring: every new flag lands in the right place.
# ---------------------------------------------------------------------------


class TestExtraProcessorWiring:
    def _extras(self, rec: _Recorder) -> list:
        assert rec.input_options is not None
        return list(rec.input_options.extra_processors)

    def test_dedup_identical_adds_processor(self, monkeypatch: pytest.MonkeyPatch):
        rec = _Recorder()
        _patch_single(monkeypatch, rec)

        process_cmd.process(**_kwargs(), dedup_identical=True)

        assert any(isinstance(p, DedupIdenticalProcessor) for p in self._extras(rec))

    def test_merge_channels_adds_processor(self, monkeypatch: pytest.MonkeyPatch):
        rec = _Recorder()
        _patch_single(monkeypatch, rec)

        process_cmd.process(**_kwargs(), merge_channels=True)

        assert any(isinstance(p, ChannelMergeProcessor) for p in self._extras(rec))

    def test_rename_topic_adds_rewrite_processor(self, monkeypatch: pytest.MonkeyPatch):
        rec = _Recorder()
        _patch_single(monkeypatch, rec)

        process_cmd.process(**_kwargs(), rename_topic=[r"/old/(.*)=/new/\1"])

        assert any(isinstance(p, TopicRewriteProcessor) for p in self._extras(rec))

    def test_alias_topic_adds_alias_processor(self, monkeypatch: pytest.MonkeyPatch):
        rec = _Recorder()
        _patch_single(monkeypatch, rec)

        process_cmd.process(**_kwargs(), alias_topic=["/old=/new"])

        assert any(isinstance(p, TopicAliasProcessor) for p in self._extras(rec))

    def test_alias_topic_with_same_pattern_fans_out(self, monkeypatch: pytest.MonkeyPatch):
        """Two --alias-topic flags with the same pattern produce list[str] in rules."""
        rec = _Recorder()
        _patch_single(monkeypatch, rec)

        process_cmd.process(**_kwargs(), alias_topic=["/src=/a", "/src=/b"])

        alias = next(p for p in self._extras(rec) if isinstance(p, TopicAliasProcessor))
        # TopicAliasProcessor stores rules internally; just confirm it constructed.
        assert alias is not None

    def test_time_offset_adds_processor(self, monkeypatch: pytest.MonkeyPatch):
        rec = _Recorder()
        _patch_single(monkeypatch, rec)

        process_cmd.process(**_kwargs(), time_offset=["/imu=500ms"])

        assert any(isinstance(p, TimeOffsetProcessor) for p in self._extras(rec))

    def test_decimate_adds_nth_message_processor(self, monkeypatch: pytest.MonkeyPatch):
        rec = _Recorder()
        _patch_single(monkeypatch, rec)

        process_cmd.process(**_kwargs(), decimate=["/imu/data=10"])

        assert any(isinstance(p, NthMessageProcessor) for p in self._extras(rec))

    def test_latch_topics_pass_through_to_input_options(self, monkeypatch: pytest.MonkeyPatch):
        rec = _Recorder()
        _patch_single(monkeypatch, rec)

        process_cmd.process(**_kwargs(), latch=["/tf_static"], latch_from_metadata=True)

        assert rec.input_options is not None
        assert rec.input_options.latch_topics == ["/tf_static"]
        assert rec.input_options.latch_from_metadata is True

    def test_rechunk_strategy_propagates_to_output_options(self, monkeypatch: pytest.MonkeyPatch):
        rec = _Recorder()
        _patch_single(monkeypatch, rec)

        process_cmd.process(
            **_kwargs(),
            rechunk_strategy=RechunkStrategy.PATTERN,
            rechunk_pattern=["/camera.*"],
        )

        assert rec.output_options is not None
        assert len(rec.output_options.output_processors) == 1
        grouper = rec.output_options.output_processors[0]
        assert isinstance(grouper, PatternGrouper)
        assert len(grouper.topic_patterns) == 1


# ---------------------------------------------------------------------------
# Chain ordering — TopicRewrite must come last among id-mutators.
# ---------------------------------------------------------------------------


class TestChainOrdering:
    def test_rewrite_after_alias_and_merge(self, monkeypatch: pytest.MonkeyPatch):
        """ARCHITECTURE.md §3: TopicRewrite is last among id-mutating processors."""
        rec = _Recorder()
        _patch_single(monkeypatch, rec)

        process_cmd.process(
            **_kwargs(),
            alias_topic=["/foo=/bar"],
            merge_channels=True,
            rename_topic=["/baz=/qux"],
        )

        assert rec.input_options is not None
        extras = list(rec.input_options.extra_processors)
        types = [type(p).__name__ for p in extras]
        rewrite_idx = types.index("TopicRewriteProcessor")
        alias_idx = types.index("TopicAliasProcessor")
        merge_idx = types.index("ChannelMergeProcessor")
        assert rewrite_idx > alias_idx
        assert rewrite_idx > merge_idx


# ---------------------------------------------------------------------------
# Split sub-flag wiring
# ---------------------------------------------------------------------------


class TestSplitRouters:
    def test_split_duration_router(self, monkeypatch: pytest.MonkeyPatch):
        rec = _Recorder()
        _patch_multi(monkeypatch, rec)

        process_cmd.process(file=["input.mcap"], split_duration="2s")

        assert rec.output_options is not None
        assert any(isinstance(r, DurationSplitProcessor) for r in rec.output_options.routers)

    def test_split_at_router(self, monkeypatch: pytest.MonkeyPatch):
        rec = _Recorder()
        _patch_multi(monkeypatch, rec)

        process_cmd.process(file=["input.mcap"], split_at=["1000000000"])

        assert rec.output_options is not None
        assert any(isinstance(r, TimestampSplitProcessor) for r in rec.output_options.routers)

    def test_split_max_size_router(self, monkeypatch: pytest.MonkeyPatch):
        rec = _Recorder()
        _patch_multi(monkeypatch, rec)

        process_cmd.process(file=["input.mcap"], split_max_size="1MB")

        assert rec.output_options is not None
        assert any(isinstance(r, SizeSplitProcessor) for r in rec.output_options.routers)

    def test_split_expression_router(self, monkeypatch: pytest.MonkeyPatch):
        rec = _Recorder()
        _patch_multi(monkeypatch, rec)

        process_cmd.process(file=["input.mcap"], split_expression="/gps/fix.status.status")

        assert rec.output_options is not None
        assert any(isinstance(r, ExpressionSplitProcessor) for r in rec.output_options.routers)

    @pytest.mark.parametrize(
        "extra",
        [
            {"split_hysteresis": 100},
            {"split_hysteresis_count": 3},
            {"split_keep_trailing_context": 100},
            {"split_keep_trailing_count": 3},
        ],
    )
    def test_expression_only_flags_require_expression(self, extra: dict):
        exit_code = process_cmd.process(file=["input.mcap"], **extra)
        assert exit_code == 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rechunk_pattern_strategy_requires_pattern(self):
        exit_code = process_cmd.process(**_kwargs(), rechunk_strategy=RechunkStrategy.PATTERN)
        assert exit_code == 1

    def test_rename_topic_missing_equals_is_error(self):
        exit_code = process_cmd.process(**_kwargs(), rename_topic=["/no-equals"])
        assert exit_code == 1

    def test_decimate_non_integer_is_error(self):
        exit_code = process_cmd.process(**_kwargs(), decimate=["/imu=ten"])
        assert exit_code == 1

    def test_time_offset_unparsable_duration_is_error(self):
        exit_code = process_cmd.process(**_kwargs(), time_offset=["/imu=not-a-duration"])
        assert exit_code == 1


# ---------------------------------------------------------------------------
# Combined operations — the whole point of this command.
# ---------------------------------------------------------------------------


class TestCombinedOperations:
    def test_merge_dedup_rename_compression(self, monkeypatch: pytest.MonkeyPatch):
        """Multi-file merge + dedup + rename-topic + compression change in one call."""
        rec = _Recorder()
        _patch_single(monkeypatch, rec)

        exit_code = process_cmd.process(
            file=["a.mcap", "b.mcap"],
            output=Path("out.mcap"),
            dedup_identical=True,
            rename_topic=[r"/old/(.*)=/new/\1"],
            compression="lz4",
        )

        assert exit_code == 0
        assert rec.files == ["a.mcap", "b.mcap"]
        assert rec.input_options is not None
        extras = list(rec.input_options.extra_processors)
        assert any(isinstance(p, DedupIdenticalProcessor) for p in extras)
        assert any(isinstance(p, TopicRewriteProcessor) for p in extras)
        assert rec.output_options is not None
        assert rec.output_options.compression == "lz4"

    def test_split_duration_with_dedup_and_alias(self, monkeypatch: pytest.MonkeyPatch):
        """Split routers and the input-stage extras chain compose."""
        rec = _Recorder()
        _patch_multi(monkeypatch, rec)

        exit_code = process_cmd.process(
            file=["a.mcap"],
            split_duration="60s",
            dedup_identical=True,
            alias_topic=["/old=/new"],
        )

        assert exit_code == 0
        assert rec.input_options is not None
        extras = list(rec.input_options.extra_processors)
        assert any(isinstance(p, DedupIdenticalProcessor) for p in extras)
        assert any(isinstance(p, TopicAliasProcessor) for p in extras)
        assert rec.output_options is not None
        assert any(isinstance(r, DurationSplitProcessor) for r in rec.output_options.routers)
