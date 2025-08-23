from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import HorizontalGroup, VerticalGroup
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Select, Static
from textual_slider import Slider

from digitalis.utilities import (
    nanoseconds_duration,
    nanoseconds_to_iso,
)


def _build_tooltip(start: int, end: int) -> Text:
    duration_ns = end - start
    return Text.assemble(
        Text.from_markup("[dim]Start:[/dim]\n"),
        nanoseconds_to_iso(start),
        "\n",
        f"{start}ns\n",
        Text.from_markup("[dim]End:[/dim]\n"),
        nanoseconds_to_iso(end),
        "\n",
        f"{end}ns\n",
        Text.from_markup("[dim]Duration:[/dim]\n"),
        nanoseconds_duration(duration_ns),
    )


_speeds: list[float] = [
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
    20.0,
    50.0,
]


class PlayGroup(HorizontalGroup):
    DEFAULT_CSS = """
        PlayGroup {
            align: center middle;

            & > #pause {
                display: none;
            }

            &.started > #play {
                display: none;
            }

            &.started > #pause {
                display: block;
            }

        }
    """

    class Changed(Message):
        def __init__(self, value: bool) -> None:
            super().__init__()
            self.value: bool = value

    def on_button_pressed(self, _event: Button.Pressed) -> None:
        """Event handler called when a button is pressed."""
        self.toggle()

    def toggle(self) -> None:
        """Toggle playback with space bar."""
        if self.has_class("started"):
            self.remove_class("started")
            self.post_message(self.Changed(False))
        else:
            self.add_class("started")
            self.post_message(self.Changed(True))

    def compose(self) -> ComposeResult:
        """Create child widgets of a stopwatch."""
        yield Button("Play", id="play", variant="success", compact=True)
        yield Button("Pause", id="pause", variant="error", compact=True)


class TimeControl(VerticalGroup):
    DEFAULT_CSS = """
        TimeControl {
            Slider {
                width: 100%;
            }
            Select {
                width: 25;
            }

            &.non-seeking {
                PlayGroup {
                    display: none;
                }
                Select {
                    display: none;
                }
            }
            #time-info {
                width: 3;
                padding: 0 1;
                background: $warning;
            }
        }
    """

    current_time: reactive[int] = reactive(0)
    start_end: reactive[tuple[int, int]] = reactive((0, 10))
    speed: reactive[float] = reactive(1.0)
    can_seek: reactive[bool] = reactive(True)

    class TimeChanged(Message):
        def __init__(self, value: int) -> None:
            super().__init__()
            self.value: int = value

    class PlaybackChanged(Message):
        def __init__(self, playing: bool) -> None:
            super().__init__()
            self.playing: bool = playing

    class SpeedChanged(Message):
        def __init__(self, speed: float) -> None:
            super().__init__()
            self.speed: float = speed

    def compose(self) -> ComposeResult:
        """Compose the TimeControl widget."""
        yield Slider(id="time-slider", min=0, max=10)
        with HorizontalGroup(id="time-control"):
            yield Static("I", id="time-info")
            yield Button("", id="time-current", compact=True, tooltip="Click to copy")
            yield PlayGroup()
            yield Select(
                ((f"{line}x", line) for line in _speeds),
                prompt="Speed",
                value=1.0,
                compact=True,
            )

    @on(Button.Pressed, "#time-current")
    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.app.copy_to_clipboard(str(event.button.label))
        self.app.notify(f"Copied {event.button.label} to clipboard")

    def watch_can_seek(self, can_seek: bool) -> None:
        """Update UI based on seeking capability."""
        if can_seek:
            self.remove_class("non-seeking")
        else:
            self.add_class("non-seeking")

    def watch_start_end(self, se: tuple[int, int]) -> None:
        """Update slider range when start/end times change."""
        slider = self.query_one(Slider)
        info = self.query_one("#time-info", Static)
        if se == (0, 0):
            slider.min = 0
            slider.max = 10
            slider.display = False
            info.tooltip = None
        else:
            slider.min = se[0]
            slider.max = se[1]
            slider.display = True

            info.tooltip = _build_tooltip(se[0], se[1])

    def watch_current_time(self, ct: int) -> None:
        """Update time display when current time changes."""
        label = self.query_one("#time-current", Button)
        label.label = nanoseconds_to_iso(ct)

        # Update slider position
        slider = self.query_one(Slider)
        with slider.prevent(Slider.Changed):
            slider.value = ct

    @on(Slider.Changed)
    def time_change(self, change: Slider.Changed) -> None:
        """Handle slider time changes."""
        self.post_message(self.TimeChanged(change.value))

    @on(PlayGroup.Changed)
    def play_pause(self, event: PlayGroup.Changed) -> None:
        self.post_message(self.PlaybackChanged(event.value))

    @on(Select.Changed)
    def speed_changed(self, event: Select.Changed) -> None:
        val = event.value
        if isinstance(val, float):
            self.post_message(self.SpeedChanged(val))

    def toggle(self) -> None:
        """Toggle playback with space bar."""
        button_group = self.query_one(PlayGroup)
        button_group.toggle()
