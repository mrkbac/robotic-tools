"""Tests for OSC 9;4 progress column."""

import gc

from pymcap_cli.osc_utils import Osc94States, OSCProgressColumn
from rich.progress import Progress
from rich.text import Text


def mock_set_progress(monkeypatch):
    """Mock set_progress and return list of (state, progress) calls."""
    # Force garbage collection to clear any pending __del__ calls from previous tests
    gc.collect()

    progress_calls = []

    def mock_progress(state, progress):
        progress_calls.append((state, progress))

    monkeypatch.setattr("pymcap_cli.osc_utils.set_progress", mock_progress)

    return progress_calls


class TestOSCProgressColumn:
    """Test OSC 9;4 progress column."""

    def test_osc_emission_with_support(self, monkeypatch):
        """Test OSC sequences are emitted when supported."""
        monkeypatch.setattr("pymcap_cli.osc_utils.supports_osc_9_4", lambda: True)
        progress_calls = mock_set_progress(monkeypatch)

        column = OSCProgressColumn()
        progress = Progress(column)

        with progress:
            task = progress.add_task("Test", total=100)
            progress.update(task, completed=50)
            progress.refresh()

        # Check progress calls
        # Should contain state PROGRESS with 50%
        assert (Osc94States.PROGRESS, 50) in progress_calls

        # Explicitly close to prevent __del__ interference with other tests
        column.close()

    def test_no_osc_without_support(self, monkeypatch):
        """Test OSC sequences are NOT emitted when unsupported."""
        monkeypatch.setattr("pymcap_cli.osc_utils.supports_osc_9_4", lambda: False)
        progress_calls = mock_set_progress(monkeypatch)

        column = OSCProgressColumn()
        progress = Progress(column)

        with progress:
            task = progress.add_task("Test", total=100)
            progress.update(task, completed=50)
            progress.refresh()

        # Should NOT contain progress calls
        assert len(progress_calls) == 0

    def test_indeterminate_progress(self, monkeypatch):
        """Test state 3 (indeterminate) when total is None."""
        monkeypatch.setattr("pymcap_cli.osc_utils.supports_osc_9_4", lambda: True)
        progress_calls = mock_set_progress(monkeypatch)

        column = OSCProgressColumn()
        progress = Progress(column)

        with progress:
            progress.add_task("Test", total=None)  # No total
            progress.refresh()

        # Should contain state INDETERMINATE
        assert (Osc94States.INDETERMINATE, 0) in progress_calls

    def test_completion_resets_progress(self, monkeypatch):
        """Test state 0 (reset) when task is finished."""
        monkeypatch.setattr("pymcap_cli.osc_utils.supports_osc_9_4", lambda: True)
        progress_calls = mock_set_progress(monkeypatch)

        column = OSCProgressColumn()
        progress = Progress(column)

        with progress:
            task = progress.add_task("Test", total=100)
            progress.update(task, completed=100)
            progress.refresh()

        # Should contain state RESET when finished
        assert (Osc94States.RESET, 0) in progress_calls

    def test_percentage_calculation(self, monkeypatch):
        """Test percentage calculation is correct."""
        monkeypatch.setattr("pymcap_cli.osc_utils.supports_osc_9_4", lambda: True)
        progress_calls = mock_set_progress(monkeypatch)

        column = OSCProgressColumn()
        progress = Progress(column)

        with progress:
            task = progress.add_task("Test", total=200)
            progress.update(task, completed=150)  # 75%
            progress.refresh()

        # Should contain 75% progress
        assert (Osc94States.PROGRESS, 75) in progress_calls

    def test_percentage_clamping(self, monkeypatch):
        """Test percentage is clamped to 0-100 range."""
        monkeypatch.setattr("pymcap_cli.osc_utils.supports_osc_9_4", lambda: True)
        progress_calls = mock_set_progress(monkeypatch)

        column = OSCProgressColumn()
        progress = Progress(column)

        with progress:
            task = progress.add_task("Test", total=100)
            # Simulate over-completion (shouldn't happen normally)
            progress.update(task, completed=150)
            progress.refresh()

        # When completed >= total, task is marked as finished and emits state RESET
        # This is expected behavior - the task is complete
        assert (Osc94States.RESET, 0) in progress_calls

    def test_zero_total_indeterminate(self, monkeypatch):
        """Test that zero total triggers indeterminate state."""
        monkeypatch.setattr("pymcap_cli.osc_utils.supports_osc_9_4", lambda: True)
        progress_calls = mock_set_progress(monkeypatch)

        column = OSCProgressColumn()
        progress = Progress(column)

        with progress:
            progress.add_task("Test", total=0)
            progress.refresh()

        # Should use state INDETERMINATE for zero total
        assert (Osc94States.INDETERMINATE, 0) in progress_calls

    def test_render_returns_empty_text(self, monkeypatch):
        """Test that render() returns empty Text."""
        # Mock to ensure column has OSC support (just for testing render method)
        monkeypatch.setattr("pymcap_cli.osc_utils.supports_osc_9_4", lambda: True)

        column = OSCProgressColumn()

        # Create a mock task
        class MockTask:
            finished = False
            total = 100
            completed = 50

        result = column.render(MockTask())
        assert isinstance(result, Text)
        assert str(result) == ""

    def test_write_error_handling(self, monkeypatch):
        """Test that write errors are handled gracefully."""
        monkeypatch.setattr("pymcap_cli.osc_utils.supports_osc_9_4", lambda: True)

        progress_attempts = []

        def mock_progress_with_error(state, progress):
            progress_attempts.append((state, progress))
            raise OSError("Broken pipe")

        monkeypatch.setattr("pymcap_cli.osc_utils.set_progress", mock_progress_with_error)

        column = OSCProgressColumn()
        progress = Progress(column)

        # Should not raise an exception even though set_progress fails
        with progress:
            task = progress.add_task("Test", total=100)
            progress.update(task, completed=50)
            progress.refresh()

        # Verify set_progress was attempted
        assert len(progress_attempts) > 0

    def test_terminal_title_updates(self, monkeypatch):
        """Test terminal title is updated with progress percentage."""
        monkeypatch.setattr("pymcap_cli.osc_utils.supports_osc_9_4", lambda: True)

        title_calls = []
        progress_calls = []

        def mock_set_title(title):
            title_calls.append(title)

        def mock_set_progress(state, progress):
            progress_calls.append((state, progress))

        monkeypatch.setattr("pymcap_cli.osc_utils.set_window_title", mock_set_title)
        monkeypatch.setattr("pymcap_cli.osc_utils.set_progress", mock_set_progress)

        column = OSCProgressColumn(title="Processing file.mcap")
        progress = Progress(column)

        with progress:
            task = progress.add_task("Test", total=100)
            progress.update(task, completed=50)
            progress.refresh()

        # Should update title with percentage
        assert "50% - Processing file.mcap" in title_calls
        # Should have progress calls too
        assert (Osc94States.PROGRESS, 50) in progress_calls

    def test_terminal_title_reset_on_completion(self, monkeypatch):
        """Test terminal title is reset to empty when task completes."""
        monkeypatch.setattr("pymcap_cli.osc_utils.supports_osc_9_4", lambda: True)

        title_calls = []

        def mock_set_title(title):
            title_calls.append(title)

        monkeypatch.setattr("pymcap_cli.osc_utils.set_window_title", mock_set_title)
        monkeypatch.setattr("pymcap_cli.osc_utils.set_progress", lambda _s, _p: None)

        column = OSCProgressColumn(title="Processing file.mcap")
        progress = Progress(column)

        with progress:
            task = progress.add_task("Test", total=100)
            progress.update(task, completed=100)
            progress.refresh()

        # Should reset title to empty on completion
        assert "" in title_calls

    def test_no_title_updates_without_title(self, monkeypatch):
        """Test no title updates when title is not provided."""
        monkeypatch.setattr("pymcap_cli.osc_utils.supports_osc_9_4", lambda: True)

        title_calls = []

        def mock_set_title(title):
            title_calls.append(title)

        monkeypatch.setattr("pymcap_cli.osc_utils.set_window_title", mock_set_title)
        monkeypatch.setattr("pymcap_cli.osc_utils.set_progress", lambda _s, _p: None)

        column = OSCProgressColumn()  # No title provided
        progress = Progress(column)

        with progress:
            task = progress.add_task("Test", total=100)
            progress.update(task, completed=50)
            progress.refresh()

        # Should not update title
        assert len(title_calls) == 0
