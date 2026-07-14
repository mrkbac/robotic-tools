"""Unit tests for PlotExporter internals: array expansion, title, output routing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
import pytest
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


class TestRendering:
    @staticmethod
    def _finish_and_capture(
        exporter: PlotExporter,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> go.Figure:
        figures: list[go.Figure] = []
        monkeypatch.setattr(exporter, "_emit", figures.append)
        exporter.finish(tmp_path, {})
        assert len(figures) == 1
        return figures[0]

    def test_time_kind_renders_scatter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exporter = PlotExporter(output=None, paths=["/odom.x"], kind="time")
        exporter._series[0].times_ns = [0, 1_000_000_000]
        exporter._series[0].values = [1.0, 2.0]

        figure = self._finish_and_capture(exporter, monkeypatch, tmp_path)

        assert isinstance(figure.data[0], go.Scattergl)

    def test_xy_downsampling_keeps_samples_paired(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exporter = PlotExporter(
            output=None,
            paths=["/pose.x", "/pose.y"],
            kind="xy",
            downsample=3,
        )
        times = list(range(20))
        exporter._series[0].times_ns = times
        exporter._series[0].values = [10.0 if index == 3 else 0.0 for index in times]
        exporter._series[1].times_ns = times
        exporter._series[1].values = [10.0 if index == 12 else 0.0 for index in times]

        figure = self._finish_and_capture(exporter, monkeypatch, tmp_path)

        trace = figure.data[0]
        assert isinstance(trace, go.Scattergl)
        assert len(trace.x) == 3
        assert len(trace.y) == 3
        assert any(x != 0.0 or y != 0.0 for x, y in zip(trace.x, trace.y, strict=True))

    def test_numeric_histogram_caps_bins_at_distinct_value_count(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exporter = PlotExporter(
            output=None,
            paths=["/odom.x"],
            kind="histogram",
            bins=7,
            normalize="probability",
        )
        exporter._series[0].times_ns = [0, 1, 2, 3]
        exporter._series[0].values = [1.0, 2.0, 2.0, 3.0]

        figure = self._finish_and_capture(exporter, monkeypatch, tmp_path)

        trace = figure.data[0]
        assert isinstance(trace, go.Histogram)
        assert trace.nbinsx == 3
        assert trace.histnorm == "probability"

    def test_numeric_histogram_keeps_smaller_bin_limit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exporter = PlotExporter(
            output=None,
            paths=["/odom.x"],
            kind="histogram",
            bins=2,
        )
        exporter._series[0].times_ns = [0, 1, 2]
        exporter._series[0].values = [1.0, 2.0, 3.0]

        figure = self._finish_and_capture(exporter, monkeypatch, tmp_path)

        assert figure.data[0].nbinsx == 2

    def test_constant_numeric_histogram_renders_as_point_mass(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exporter = PlotExporter(output=None, paths=["/joint.position"], kind="histogram")
        exporter._series[0].times_ns = [0, 1, 2]
        exporter._series[0].values = [0.0, 0.0, 0.0]

        figure = self._finish_and_capture(exporter, monkeypatch, tmp_path)

        trace = figure.data[0]
        assert isinstance(trace, go.Bar)
        assert list(trace.x) == [0.0]
        assert list(trace.y) == [3]
        assert list(figure.layout.xaxis.tickvals) == [0.0]
        assert figure.layout.annotations[0].text == "/joint.position — constant 0"

    def test_numeric_histogram_supports_density(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exporter = PlotExporter(
            output=None,
            paths=["/odom.x"],
            kind="histogram",
            normalize="density",
        )
        exporter._series[0].times_ns = [0, 1, 2]
        exporter._series[0].values = [1.0, 2.0, 3.0]

        figure = self._finish_and_capture(exporter, monkeypatch, tmp_path)

        assert figure.data[0].histnorm == "probability density"

    def test_categorical_histogram_uses_deterministic_frequency_bars(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exporter = PlotExporter(
            output=None,
            paths=["/state.name"],
            kind="histogram",
            normalize="probability",
        )
        exporter._series[0].times_ns = [0, 1, 2, 3]
        exporter._series[0].values = ["warn", "ok", "warn", "ok"]

        figure = self._finish_and_capture(exporter, monkeypatch, tmp_path)

        trace = figure.data[0]
        assert isinstance(trace, go.Bar)
        assert list(trace.x) == ["ok", "warn"]
        assert list(trace.y) == [0.5, 0.5]

    def test_boolean_histogram_orders_false_before_true(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exporter = PlotExporter(output=None, paths=["/state.ready"], kind="histogram")
        exporter._series[0].times_ns = [0, 1, 2]
        exporter._series[0].values = [True, False, True]

        figure = self._finish_and_capture(exporter, monkeypatch, tmp_path)

        trace = figure.data[0]
        assert isinstance(trace, go.Bar)
        assert list(trace.x) == [False, True]
        assert list(trace.y) == [1, 2]

    def test_histogram_creates_one_subplot_per_expanded_series(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exporter = PlotExporter(
            output=None,
            paths=["/joints.position[:]"],
            kind="histogram",
        )
        series = exporter._series[0]
        series.array_times_ns = {0: [0, 1], 1: [0, 1]}
        series.array_values = {0: [1.0, 2.0], 1: [10.0, 20.0]}

        figure = self._finish_and_capture(exporter, monkeypatch, tmp_path)

        assert len(figure.data) == 2
        assert [annotation.text for annotation in figure.layout.annotations] == [
            "/joints.position[:][0]",
            "/joints.position[:][1]",
        ]
        assert figure.layout.height == 600
        assert figure.layout.yaxis.domain[0] - figure.layout.yaxis2.domain[1] >= 0.1

    def test_histogram_rejects_mixed_value_types(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exporter = PlotExporter(output=None, paths=["/mixed.value"], kind="histogram")
        exporter._series[0].times_ns = [0, 1]
        exporter._series[0].values = [1.0, "one"]
        monkeypatch.setattr(exporter, "_emit", lambda _figure: None)

        with pytest.raises(RuntimeError, match="mixed numeric and categorical"):
            exporter.finish(tmp_path, {})

    def test_density_rejects_categorical_series(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exporter = PlotExporter(
            output=None,
            paths=["/state.name"],
            kind="histogram",
            normalize="density",
        )
        exporter._series[0].times_ns = [0, 1]
        exporter._series[0].values = ["ok", "warn"]
        monkeypatch.setattr(exporter, "_emit", lambda _figure: None)

        with pytest.raises(RuntimeError, match=r"density.*categorical"):
            exporter.finish(tmp_path, {})

    def test_numeric_histogram_drops_non_finite_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        exporter = PlotExporter(output=None, paths=["/odom.x"], kind="histogram")
        exporter._series[0].times_ns = [0, 1, 2, 3]
        exporter._series[0].values = [1.0, math.nan, math.inf, 2.0]

        with caplog.at_level("WARNING"):
            figure = self._finish_and_capture(exporter, monkeypatch, tmp_path)

        assert list(figure.data[0].x) == [1.0, 2.0]
        assert "Dropped 2 non-finite values" in caplog.text


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
