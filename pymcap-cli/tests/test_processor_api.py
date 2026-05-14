from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import pytest
from pymcap_cli.core.input_options import InputOptions
from pymcap_cli.core.mcap_processor import (
    InputFile,
    McapProcessor,
    OutputOptions,
    ProcessingOptions,
)
from pymcap_cli.core.processors.base import (
    ChunkContext,
    ChunkDecision,
    InputProcessor,
    MessageScope,
)

from tests.helpers import chunk_context, lazy_chunk


@dataclass(frozen=True)
class _StubMessageIndex:
    channel_id: int


class _NotAProcessor:
    pass


class _NoMessageProcessor(InputProcessor):
    def message_scope(self, context: ChunkContext) -> MessageScope:
        _ = context
        return MessageScope.none()


class _ChannelScopedProcessor(InputProcessor):
    def message_scope(self, context: ChunkContext) -> MessageScope:
        _ = context
        return MessageScope.channels({1})


def _processing_options(
    *,
    input_options: InputOptions | None = None,
    output_options: OutputOptions | None = None,
) -> ProcessingOptions:
    return ProcessingOptions(
        inputs=[
            InputFile(
                stream=BytesIO(b""),
                size=0,
                options=input_options or InputOptions.from_args(),
            )
        ],
        input_options=InputOptions.from_args(),
        output_options=output_options or OutputOptions(),
    )


def test_input_processor_role_is_validated_before_processing() -> None:
    input_options = InputOptions.from_args(extra_processors=[_NotAProcessor()])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="InputProcessor"):
        McapProcessor(_processing_options(input_options=input_options))


def test_output_router_role_is_validated_before_processing() -> None:
    output_options = OutputOptions(routers=[_NotAProcessor()])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="OutputRouter"):
        McapProcessor(_processing_options(output_options=output_options))


def test_default_message_scope_decodes_for_correctness() -> None:
    proc = InputProcessor()

    assert proc.on_chunk(chunk_context(), lazy_chunk(0, 1)) is ChunkDecision.DECODE


def test_none_message_scope_fast_copies_without_indexes() -> None:
    proc = _NoMessageProcessor()

    assert proc.on_chunk(chunk_context(), lazy_chunk(0, 1)) is ChunkDecision.CONTINUE


def test_channel_message_scope_decodes_when_indexes_are_missing() -> None:
    proc = _ChannelScopedProcessor()

    assert proc.on_chunk(chunk_context(), lazy_chunk(0, 1)) is ChunkDecision.DECODE


def test_channel_message_scope_fast_copies_unmatched_indexes() -> None:
    proc = _ChannelScopedProcessor()

    assert (
        proc.on_chunk(chunk_context([_StubMessageIndex(channel_id=2)]), lazy_chunk(0, 1))
        is ChunkDecision.CONTINUE
    )


def test_channel_message_scope_decodes_matching_indexes() -> None:
    proc = _ChannelScopedProcessor()

    assert (
        proc.on_chunk(chunk_context([_StubMessageIndex(channel_id=1)]), lazy_chunk(0, 1))
        is ChunkDecision.DECODE
    )
