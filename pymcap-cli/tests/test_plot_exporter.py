"""Unit tests for PlotExporter internals: array expansion, title, output routing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pymcap_cli.exporters.plot_exporter import (
    _MAX_ARRAY_COLUMNS,
    PlotExporter,
    SeriesData,
    _expand_series,
    _PlotTopicWriter,
)
from ros_parser.message_path import parse_message_path

if TYPE_CHECKING:
    from pathlib import Path


def _series(path_str: str, label: str | None = None) -> SeriesData:
    return SeriesData(
        label=label or path_str,
        path_str=path_str,
        parsed=parse_message_path(path_str),
    )


@dataclass
class _FakeMcapMessage:
    log_time: int


@dataclass
class _FakeDecoded:
    message: _FakeMcapMessage
    decoded_message: Any
    schema: None = None


def _feed(writer: _PlotTopicWriter, samples: list[tuple[int, Any]]) -> None:
    for log_time, decoded in samples:
        writer.write(_FakeDecoded(_FakeMcapMessage(log_time), decoded))


class TestExpandSeries:
    def test_scalar_series_expands_to_one_trace(self):
        series = _series("/odom.x")
        series.times_ns = [0, 1, 2]
        series.values = [1.0, 2.0, 3.0]

        expanded = _expand_series(series)

        assert len(expanded) == 1
        assert expanded[0].label == "/odom.x"
        assert expanded[0].values == [1.0, 2.0, 3.0]

    def test_array_series_expands_per_index(self):
        series = _series("/joints.position[:]")
        series.array_times_ns = {0: [0, 1], 1: [0, 1]}
        series.array_values = {0: [10.0, 11.0], 1: [20.0, 21.0]}

        expanded = _expand_series(series)

        assert [ps.label for ps in expanded] == ["/joints.position[:][0]", "/joints.position[:][1]"]
        assert expanded[0].values == [10.0, 11.0]
        assert expanded[1].values == [20.0, 21.0]

    def test_array_columns_are_capped_with_warning(self, caplog):
        series = _series("/wide.data[:]")
        over = _MAX_ARRAY_COLUMNS + 5
        series.array_times_ns = {i: [0] for i in range(over)}
        series.array_values = {i: [float(i)] for i in range(over)}

        with caplog.at_level("WARNING"):
            expanded = _expand_series(series)

        assert len(expanded) == _MAX_ARRAY_COLUMNS
        assert "array columns" in caplog.text


class TestWriterArrayExpansion:
    def test_list_value_fans_out_into_indexed_columns(self):
        series = _series("/joints.position[:]")
        writer = _PlotTopicWriter([series])

        _feed(writer, [(100, {"position": [1.0, 2.0, 3.0]}), (200, {"position": [4.0, 5.0, 6.0]})])

        assert series.times_ns == []
        assert series.array_values == {0: [1.0, 4.0], 1: [2.0, 5.0], 2: [3.0, 6.0]}
        assert series.array_times_ns == {0: [100, 200], 1: [100, 200], 2: [100, 200]}

    def test_element_wise_modifier_over_slice(self):
        series = _series("/joints.position[:].@degrees")
        writer = _PlotTopicWriter([series])

        _feed(writer, [(100, {"position": [0.0, math.pi]})])

        assert series.array_values[0] == [0.0]
        assert series.array_values[1][0] == math.degrees(math.pi)

    def test_scalar_path_still_scalar(self):
        series = _series("/odom.x")
        writer = _PlotTopicWriter([series])

        _feed(writer, [(10, {"x": 1.5}), (20, {"x": 2.5})])

        assert series.values == [1.5, 2.5]
        assert series.array_values == {}


class TestComposeTitle:
    def test_source_name_appended_as_subtitle(self):
        exporter = PlotExporter(output=None, paths=["/odom.x"], source_name="drive.mcap")
        title = exporter._compose_title("default")
        assert "drive.mcap" in title
        assert title.startswith("default")

    def test_no_source_name_leaves_title_plain(self):
        exporter = PlotExporter(output=None, paths=["/odom.x"])
        assert exporter._compose_title("default") == "default"

    def test_explicit_title_wins_over_default_but_keeps_source(self):
        exporter = PlotExporter(
            output=None, paths=["/odom.x"], title="My Plot", source_name="drive.mcap"
        )
        title = exporter._compose_title("labels-here")
        assert title.startswith("My Plot")
        assert "drive.mcap" in title
        assert "labels-here" not in title


class TestEmitRouting:
    """`_emit` picks the renderer from the output suffix."""

    class _FakeFig:
        def __init__(self) -> None:
            self.html_calls: list[str] = []
            self.image_calls: list[str] = []

        def write_html(self, path: str) -> None:
            self.html_calls.append(path)

        def write_image(self, path: str) -> None:
            self.image_calls.append(path)

    def test_html_suffix_routes_to_write_html(self, tmp_path: Path):
        exporter = PlotExporter(output=tmp_path / "p.html", paths=["/odom.x"])
        fig = self._FakeFig()
        exporter._emit(fig)  # type: ignore[arg-type]
        assert fig.html_calls
        assert not fig.image_calls

    def test_svg_suffix_routes_to_write_image(self, tmp_path: Path):
        exporter = PlotExporter(output=tmp_path / "p.svg", paths=["/odom.x"])
        fig = self._FakeFig()
        exporter._emit(fig)  # type: ignore[arg-type]
        assert fig.image_calls
        assert not fig.html_calls

    def test_png_suffix_routes_to_write_image(self, tmp_path: Path):
        exporter = PlotExporter(output=tmp_path / "p.PNG", paths=["/odom.x"])
        fig = self._FakeFig()
        exporter._emit(fig)  # type: ignore[arg-type]
        assert fig.image_calls
        assert not fig.html_calls
