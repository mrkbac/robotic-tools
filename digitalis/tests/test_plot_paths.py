"""Path parsing and validation for the plot panel."""

from __future__ import annotations

import pytest
from digitalis.ui.panels.plot import PlotPathValidator, parse_plot_path
from ros_parser.message_path import MessagePathError


def test_validator_accepts_plain_field_path():
    assert PlotPathValidator().validate(".pose.position.x").is_valid


def test_validator_accepts_stream_transform_path():
    assert PlotPathValidator().validate(".pose.position.x.@@delta").is_valid


def test_validator_rejects_partial_stream_modifier():
    # Intermediate state while typing '.x.@@delta' — must fail validation, not raise.
    assert not PlotPathValidator().validate(".x.@@d").is_valid


def test_validator_rejects_double_reducer():
    assert not PlotPathValidator().validate(".x.@@max.@@min").is_valid


def test_validator_rejects_legacy_single_at_stream_operation():
    result = PlotPathValidator().validate(".x.@delta")
    assert not result.is_valid
    assert "@@delta" in result.failure_descriptions[0]


def test_validator_rejects_stream_reducer():
    result = PlotPathValidator().validate(".x.@@max")
    assert not result.is_valid
    assert "cannot be plotted" in result.failure_descriptions[0]


@pytest.mark.parametrize("field_path", [".x.@@d", ".x.@@max.@@min", ".x.@@max", ".x.@bogus"])
def test_parse_plot_path_raises_message_path_error(field_path: str):
    with pytest.raises(MessagePathError):
        parse_plot_path(field_path)


def test_parse_plot_path_returns_evaluator():
    _path, evaluator = parse_plot_path(".value.@@delta")
    assert evaluator.observe({"value": 1}, 0) is not None
