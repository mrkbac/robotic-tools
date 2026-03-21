"""E2E tests for the diag command."""

import json
from pathlib import Path

import pytest
from pymcap_cli.cmd.diag_cmd import diag

from tests.fixtures.diag_mcap_generator import ensure_diag_fixtures


@pytest.fixture(scope="session")
def diag_fixtures(fixtures_dir) -> dict[str, Path]:
    """Generate diagnostic MCAP fixtures."""
    return ensure_diag_fixtures(fixtures_dir)


@pytest.fixture
def diag_mcap(diag_fixtures) -> Path:
    return diag_fixtures["diagnostics"]


@pytest.mark.e2e
class TestDiag:
    """Test diag command functionality."""

    def test_default_shows_non_ok(self, diag_mcap: Path, capsys):
        """Default output shows only components with worst_level >= WARN."""
        result = diag(file=str(diag_mcap))

        assert result == 0
        captured = capsys.readouterr()
        # encoder_top had ERROR, camera_front had ERROR — both should appear
        assert "encoder_top" in captured.out
        assert "camera_front" in captured.out
        # radar_front was always OK — should NOT appear in default view
        assert "radar_front" not in captured.out

    def test_all_shows_ok(self, diag_mcap: Path, capsys):
        """--all flag shows OK components too."""
        result = diag(file=str(diag_mcap), show_all=True)

        assert result == 0
        captured = capsys.readouterr()
        assert "radar_front" in captured.out
        assert "encoder_top" in captured.out
        assert "camera_front" in captured.out

    def test_level_filter(self, diag_mcap: Path, capsys):
        """--level 2 shows only ERROR and above."""
        result = diag(file=str(diag_mcap), level=2)

        assert result == 0
        captured = capsys.readouterr()
        assert "encoder_top" in captured.out
        assert "camera_front" in captured.out

    def test_name_filter(self, diag_mcap: Path, capsys):
        """--name filters by regex on component name."""
        result = diag(file=str(diag_mcap), show_all=True, name="radar")

        assert result == 0
        captured = capsys.readouterr()
        assert "radar_front" in captured.out
        assert "encoder_top" not in captured.out

    def test_hardware_id_filter(self, diag_mcap: Path, capsys):
        """--hardware-id filters by hardware ID."""
        result = diag(file=str(diag_mcap), show_all=True, hardware_id="cam_front")

        assert result == 0
        captured = capsys.readouterr()
        assert "camera_front" in captured.out
        assert "encoder_top" not in captured.out

    def test_inspect(self, diag_mcap: Path, capsys):
        """--inspect shows detailed view with level changes and values."""
        result = diag(file=str(diag_mcap), inspect="encoder")

        assert result == 0
        captured = capsys.readouterr()
        assert "encoder_top" in captured.out
        assert "Level Changes" in captured.out
        assert "Latest Values" in captured.out
        assert "encoder_reset_count" in captured.out

    def test_inspect_no_match(self, diag_mcap: Path, capsys):
        """--inspect with no match prints warning."""
        result = diag(file=str(diag_mcap), inspect="nonexistent_xyz")

        assert result == 0
        captured = capsys.readouterr()
        assert "No components matching" in captured.err

    def test_tree_view(self, diag_mcap: Path, capsys):
        """--tree shows hierarchical view grouped by hardware_id."""
        result = diag(file=str(diag_mcap), tree=True)

        assert result == 0
        captured = capsys.readouterr()
        assert "Diagnostics" in captured.out
        assert "encoder_top" in captured.out

    def test_json_output(self, diag_mcap: Path, capsys):
        """--json outputs valid JSON with summary and components."""
        result = diag(file=str(diag_mcap), json_output=True)

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)

        assert "summary" in data
        assert "components" in data
        assert data["summary"]["total_components"] == 3

        # Check component structure
        comp = data["components"][0]
        assert "name" in comp
        assert "worst_level" in comp
        assert "worst_level_name" in comp
        assert "level_counts" in comp
        assert "values" in comp
        assert "level_changes" in comp

    def test_json_all(self, diag_mcap: Path, capsys):
        """--json --all includes all components."""
        result = diag(file=str(diag_mcap), json_output=True, show_all=True)

        assert result == 0
        data = json.loads(capsys.readouterr().out)
        assert len(data["components"]) == 3

    def test_no_diagnostics_topic(self, simple_mcap: Path, capsys):
        """Graceful handling when diagnostics topic doesn't exist."""
        result = diag(file=str(simple_mcap))

        assert result == 0
        captured = capsys.readouterr()
        assert "No diagnostics found" in captured.err

    def test_level_changes_tracked(self, diag_mcap: Path, capsys):
        """Level transitions are recorded correctly."""
        result = diag(file=str(diag_mcap), json_output=True, show_all=True)

        assert result == 0
        data = json.loads(capsys.readouterr().out)

        # Find encoder component
        encoder = next(c for c in data["components"] if "Encoder" in c["name"])
        # Should have transitions: OK -> WARN -> ERROR -> OK -> OK (last OK not recorded since same)
        assert len(encoder["level_changes"]) >= 3
        # First was OK
        assert encoder["level_changes"][0]["level"] == 0
        # Should have gone through WARN and ERROR
        levels = [c["level"] for c in encoder["level_changes"]]
        assert 1 in levels  # WARN
        assert 2 in levels  # ERROR
