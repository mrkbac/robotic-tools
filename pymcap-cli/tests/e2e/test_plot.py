"""E2E tests for the plot command."""

import io
from pathlib import Path

import pytest
from pymcap_cli.cmd.plot_cmd import plot
from pymcap_cli.exporters.plot_exporter import downsample_lttb, parse_path_arg
from small_mcap import CompressionType, McapWriter


def _make_array_mcap(path: Path, *, num_messages: int = 5, width: int = 3) -> None:
    """Write a JSON MCAP whose ``/arr`` messages carry a ``position`` array."""
    output = io.BytesIO()
    writer = McapWriter(output, chunk_size=4096, compression=CompressionType.ZSTD)
    writer.start()
    writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")
    writer.add_channel(channel_id=1, topic="/arr", message_encoding="json", schema_id=1)
    for i in range(num_messages):
        values = ",".join(str(i * 10 + j) for j in range(width))
        writer.add_message(
            channel_id=1,
            log_time=i * 1_000_000,
            data=f'{{"position": [{values}]}}'.encode(),
            publish_time=i * 1_000_000,
        )
    writer.finish()
    path.write_bytes(output.getvalue())


class TestParsePathArg:
    """Test the named extraction parser."""

    def test_plain_path(self):
        label, path = parse_path_arg("/odom.pose.position.x")
        assert label == "/odom.pose.position.x"
        assert path == "/odom.pose.position.x"

    def test_named_path(self):
        label, path = parse_path_arg("Vel X=/odom.twist.linear.x")
        assert label == "Vel X"
        assert path == "/odom.twist.linear.x"

    def test_named_path_with_equals_in_path(self):
        """The first = is the separator, rest belongs to path."""
        label, path = parse_path_arg("Name=/topic.field{x==5}")
        assert label == "Name"
        assert path == "/topic.field{x==5}"

    def test_path_starting_with_slash_no_name(self):
        """Paths starting with / are never treated as named."""
        label, path = parse_path_arg("/a=b")
        assert label == "/a=b"
        assert path == "/a=b"


class TestDownsample:
    """Test the downsampling function."""

    def test_no_downsample_when_below_threshold(self):
        times = [0.0, 1.0, 2.0]
        values: list[float | bool | str] = [1.0, 2.0, 3.0]
        out_t, out_v = downsample_lttb(times, values, 10)
        assert out_t == times
        assert out_v == values

    def test_exact_threshold(self):
        times = [0.0, 1.0, 2.0, 3.0, 4.0]
        values: list[float | bool | str] = [1.0, 2.0, 3.0, 4.0, 5.0]
        out_t, out_v = downsample_lttb(times, values, 5)
        assert out_t == times
        assert out_v == values

    def test_downsample_preserves_endpoints(self):
        n = 100
        times = [float(i) for i in range(n)]
        values: list[float | bool | str] = [float(i * i) for i in range(n)]
        out_t, out_v = downsample_lttb(times, values, 10)

        assert len(out_t) == 10
        assert out_t[0] == times[0]
        assert out_t[-1] == times[-1]
        assert out_v[0] == values[0]
        assert out_v[-1] == values[-1]

    def test_downsample_reduces_count(self):
        n = 1000
        times = [float(i) for i in range(n)]
        values: list[float | bool | str] = [float(i) for i in range(n)]
        out_t, out_v = downsample_lttb(times, values, 50)

        assert len(out_t) == 50
        assert len(out_v) == 50

    def test_downsample_preserves_spike(self):
        """LTTB should preserve visually important points like spikes."""
        n = 100
        times = [float(i) for i in range(n)]
        values: list[float | bool | str] = [0.0] * n
        values[50] = 100.0  # spike

        _, out_v = downsample_lttb(times, values, 20)

        assert 100.0 in out_v


@pytest.mark.e2e
class TestPlot:
    """Test plot command functionality."""

    def test_plot_basic(self, simple_mcap: Path, tmp_path: Path):
        """Test basic plot to HTML file."""
        output = tmp_path / "plot.html"
        exit_code = plot(file=str(simple_mcap), paths=["/test.i"], output=str(output))

        assert exit_code == 0
        assert output.exists()
        content = output.read_text()
        assert "plotly" in content.lower()

    def test_plot_named_extraction(self, simple_mcap: Path, tmp_path: Path):
        """Test named extraction appears in output."""
        output = tmp_path / "plot.html"
        exit_code = plot(
            file=str(simple_mcap),
            paths=["Counter=/test.i"],
            output=str(output),
        )

        assert exit_code == 0
        content = output.read_text()
        assert "Counter" in content

    def test_plot_invalid_path_syntax(self, simple_mcap: Path, capsys):
        """Test invalid path syntax fails fast."""
        exit_code = plot(file=str(simple_mcap), paths=["/test[bad"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Invalid path syntax" in captured.err

    def test_plot_nonexistent_topic(self, simple_mcap: Path, capsys):
        """Nonexistent topic reports a clear error and a non-zero exit code."""
        exit_code = plot(file=str(simple_mcap), paths=["/nonexistent.field"])

        assert exit_code == 1
        captured = capsys.readouterr()
        # Driver now reports "no matching topics" when the topic filter
        # eliminates every message; either message is acceptable.
        combined = captured.err + captured.out
        assert ("not found" in combined) or ("No messages exported" in combined)

    def test_plot_existing_topic_with_no_plottable_data(
        self, simple_mcap: Path, tmp_path: Path, capsys
    ):
        """Existing topic but missing field should fail instead of writing an empty plot."""
        output = tmp_path / "plot.html"
        exit_code = plot(file=str(simple_mcap), paths=["/test.missing"], output=str(output))

        assert exit_code == 1
        assert not output.exists()
        captured = capsys.readouterr()
        assert "No plottable data" in captured.err + captured.out

    def test_plot_xy_requires_two_paths(self, simple_mcap: Path, capsys):
        """Test --xy mode requires exactly 2 paths."""
        exit_code = plot(file=str(simple_mcap), paths=["/test.i"], xy=True)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "exactly 2" in captured.err

    def test_plot_downsample(self, simple_mcap: Path, tmp_path: Path, capsys):
        """Test downsampling output."""
        output = tmp_path / "plot.html"
        exit_code = plot(
            file=str(simple_mcap),
            paths=["/test.i"],
            output=str(output),
            downsample=10,
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Downsampled" in captured.err

    def test_plot_nonexistent_file(self, capsys):
        """Test nonexistent file reports error."""
        exit_code = plot(file="nonexistent.mcap", paths=["/test.i"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_plot_empty_paths(self, simple_mcap: Path, capsys):
        """Test empty paths list fails."""
        exit_code = plot(file=str(simple_mcap), paths=[])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "required" in captured.err

    def test_plot_array_expands_to_indexed_traces(self, tmp_path: Path):
        """An array-valued path fans out into one trace per element index."""
        mcap = tmp_path / "arr.mcap"
        _make_array_mcap(mcap, width=3)
        output = tmp_path / "plot.html"

        exit_code = plot(file=str(mcap), paths=["/arr.position[:]"], output=str(output))

        assert exit_code == 0
        content = output.read_text()
        assert "position[:][0]" in content
        assert "position[:][2]" in content

    def test_plot_reports_per_path_matches(self, simple_mcap: Path, tmp_path: Path, capsys):
        """A valid path reports its point count; a mistyped one is flagged for the user."""
        output = tmp_path / "plot.html"
        exit_code = plot(file=str(simple_mcap), paths=["/test.i", "/test.typo"], output=str(output))

        assert exit_code == 0  # the one good path still plots
        combined = capsys.readouterr().err
        assert "points" in combined  # positive confirmation for /test.i
        assert "'/test.typo'" in combined
        assert "check the field names" in combined

    def test_plot_bad_field_warns_before_full_scan(self, tmp_path: Path, capsys):
        """A bad field is reported from the first message, not only after scanning everything."""
        mcap = tmp_path / "big.mcap"
        output = io.BytesIO()
        writer = McapWriter(output, chunk_size=8192, compression=CompressionType.ZSTD)
        writer.start()
        writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")
        writer.add_channel(channel_id=1, topic="/test", message_encoding="json", schema_id=1)
        for i in range(5000):
            writer.add_message(
                channel_id=1,
                log_time=i * 1_000_000,
                data=f'{{"i":{i}}}'.encode(),
                publish_time=i * 1_000_000,
            )
        writer.finish()
        mcap.write_bytes(output.getvalue())

        exit_code = plot(file=str(mcap), paths=["/test.typo"], output=str(tmp_path / "o.html"))

        assert exit_code == 1  # nothing plottable
        assert "found no value in the first" in capsys.readouterr().err

    def test_plot_filter_path_maps_field(self, tmp_path: Path):
        """A filtered array path plots the field of matching elements."""
        mcap = tmp_path / "diag.mcap"
        output = io.BytesIO()
        writer = McapWriter(output, chunk_size=4096, compression=CompressionType.ZSTD)
        writer.start()
        writer.add_schema(schema_id=1, name="test", encoding="json", data=b"{}")
        writer.add_channel(channel_id=1, topic="/diag", message_encoding="json", schema_id=1)
        for i in range(20):
            data = f'{{"status":[{{"level":1,"v":{i}}},{{"level":2,"v":{i * 10}}}]}}'
            writer.add_message(
                channel_id=1, log_time=i * 1_000_000, data=data.encode(), publish_time=i * 1_000_000
            )
        writer.finish()
        mcap.write_bytes(output.getvalue())

        out = tmp_path / "plot.html"
        exit_code = plot(file=str(mcap), paths=["warn=/diag.status{level==2}.v"], output=str(out))

        assert exit_code == 0
        assert "warn" in out.read_text()

    def test_plot_filename_in_title(self, simple_mcap: Path, tmp_path: Path):
        """The source filename is embedded in the plot for back-reference."""
        output = tmp_path / "plot.html"
        exit_code = plot(file=str(simple_mcap), paths=["/test.i"], output=str(output))

        assert exit_code == 0
        assert Path(simple_mcap).name in output.read_text()

    def test_plot_svg_output(self, simple_mcap: Path, tmp_path: Path):
        """An .svg output renders a static image (no browser needed)."""
        output = tmp_path / "plot.svg"
        try:
            exit_code = plot(file=str(simple_mcap), paths=["/test.i"], output=str(output))
        except Exception as exc:  # noqa: BLE001 - kaleido/browser may be unavailable in CI
            pytest.skip(f"static image export unavailable: {exc}")

        assert exit_code == 0
        assert output.exists()
        assert output.read_text().lstrip().startswith("<")
