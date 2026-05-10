"""E2E tests for the plot command."""

from pathlib import Path

import pytest
from pymcap_cli.cmd.plot_cmd import plot
from pymcap_cli.exporters.plot_exporter import downsample_lttb, parse_path_arg


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
