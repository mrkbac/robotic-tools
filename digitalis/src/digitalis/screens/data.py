import logging
from collections.abc import Callable
from functools import partial
from typing import ClassVar

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.command import Hit, Hits, Provider
from textual.constants import MAX_FPS
from textual.containers import Horizontal, Vertical
from textual.messages import ExitApp
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Input

from digitalis.reader import create_source
from digitalis.reader.source import PlaybackSource, SourceStatus
from digitalis.reader.types import MessageEvent, SourceInfo, Topic
from digitalis.ui.datapanel import DataView
from digitalis.ui.search import TopicSearch
from digitalis.ui.timecontrol import TimeControl
from digitalis.ui.widgets import CustomHeader
from digitalis.utilities import get_file_paths


class PanelSetTopic(Provider):
    def _set_topic(self, screen: "DataScreen", topic: Topic) -> None:
        """Set the selected topic in the DataScreen."""
        screen.selected_topic = topic

    async def search(self, query: str) -> Hits:
        """Search for topics."""
        matcher = self.matcher(query)

        datascreen = None
        for screen in self.app.screen_stack:
            if isinstance(screen, DataScreen):
                datascreen = screen
                break
        else:
            logging.warning("No DataScreen found in app screen stack")
            return

        for topic in datascreen.topics:
            topic_name = topic.name
            score = matcher.match(topic_name)

            if score > 0:
                help_text = f"{topic.schema_name}"
                yield Hit(
                    score,
                    matcher.highlight(topic_name),
                    partial(self._set_topic, datascreen, topic),
                    help=help_text,
                )


class PanelType(Provider):
    def _set_topic(self, screen: "DataScreen", topic: Topic) -> None:
        """Set the selected topic in the DataScreen."""
        screen.selected_topic = topic

    async def search(self, query: str) -> Hits:
        """Search for Python files."""
        matcher = self.matcher(query)

        datascreen = None
        for screen in self.app.screen_stack:
            if isinstance(screen, DataScreen):
                datascreen = screen
                break
        else:
            logging.warning("No DataScreen found in app screen stack")
            return

        dataview = datascreen.query_one(DataView)

        for panel in dataview.available_panels:
            name = f"Panel: {panel.__name__}"
            score = matcher.match(name)

            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(name),
                    partial(dataview.action_switch_panel, panel),
                    help=f"Switch to {panel.__name__} panel",
                )


class DataScreen(Screen):
    COMMANDS: ClassVar[set[type[Provider] | Callable[[], type[Provider]]]] = {
        PanelSetTopic,
        PanelType,
    }

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("space", "toggle_playback", "Play/Pause", priority=True),
        Binding("slash", "focus_search", "Search"),
        Binding("t", "toggle_topic_search", "Toggle Topics"),
    ]
    TITLE = "Digitalis 🪻"

    selected_topic: reactive[Topic | None] = reactive(None)
    start_end: reactive[tuple[int, int]] = reactive((0, 0))
    current_time: reactive[int] = reactive(0)
    speed: reactive[float] = reactive(1.0)
    source_status: reactive[SourceStatus] = reactive(SourceStatus.INITIALIZING)

    topics: reactive[list[Topic]] = reactive([])
    _previous_topic: Topic | None = None

    def __init__(self, file_or_url: str) -> None:
        super().__init__()
        self.title = f"Digitalis 🪻 - {file_or_url} - FPS: {MAX_FPS}"

        self.source = create_source(
            file_or_url,
            on_message=self.on_message,
            on_source_info=self.on_source_info,
            on_time=self.on_time_update,
            on_status=self.on_status_update,
        )

    def on_source_info(self, source_info: SourceInfo) -> None:
        """Handle source info updates from the source."""
        self.topics = source_info.topics
        logging.info(f"Topics discovered: {[t.name for t in source_info.topics]}")

        # Update time range if available
        if source_info.start_time_ns is not None and source_info.end_time_ns is not None:
            self.start_end = (source_info.start_time_ns, source_info.end_time_ns)

    def on_message(self, message: MessageEvent) -> None:
        """Handle messages from the source."""
        if self.selected_topic and self.selected_topic.name == message.topic:
            data_panel = self.query_one(DataView)
            data_panel.data = message

    def on_status_update(self, status: SourceStatus) -> None:
        """Handle source status updates."""
        self.source_status = status
        self.title = f"Digitalis 🪻 - {status.value} - FPS: {MAX_FPS}"
        logging.debug(f"Source status changed to: {status.value}")

    def compose(self) -> ComposeResult:
        """Compose our UI."""
        # Custom header with integrated connection status
        yield CustomHeader(
            icon="🪻",
        ).data_bind(status=DataScreen.source_status)

        with Vertical():
            with Horizontal(id="main-panel"):
                yield TopicSearch().data_bind(topics=DataScreen.topics)
                yield DataView().data_bind(topic=DataScreen.selected_topic)
            yield TimeControl().data_bind(start_end=DataScreen.start_end)

        yield Footer()

    def load_data(self) -> None:
        """Load data source and extract information."""
        if isinstance(self.source, PlaybackSource):
            time_range = self.source.time_range
            if time_range:
                self.start_end = time_range

    def watch_start_end(self, start_end: tuple[int, int]) -> None:
        """Update slider range when start/end times change."""
        time_control = self.query_one(TimeControl)
        time_control.start_end = start_end
        self.current_time = start_end[0]

    def watch_current_time(self, current_time: int) -> None:
        """Update time display when current time changes."""
        time_control = self.query_one(TimeControl)
        time_control.current_time = current_time

    def watch_selected_topic(self, topic: Topic | None) -> None:
        # Unsubscribe from previous topic if any
        if self._previous_topic and self._previous_topic != topic:
            self.run_worker(self.source.unsubscribe(self._previous_topic.name))

        # Subscribe to the selected topic
        if topic:
            data_panel = self.query_one(DataView)
            data_panel.data = None
            self.run_worker(self.source.subscribe(topic.name))
            self._previous_topic = topic

    def on_mount(self) -> None:
        """Initialize the app when mounted."""
        self.load_data()

        # Configure time control based on source capabilities
        time_control = self.query_one(TimeControl)
        time_control.can_seek = isinstance(self.source, PlaybackSource)

        self.run_worker(self.source.initialize(), start=True, exit_on_error=True)

    def on_time_update(self, timestamp_ns: int) -> None:
        """Handle time updates from the backend via callback."""
        self.current_time = timestamp_ns

    @on(TopicSearch.Changed)
    def topic_selected(self, event: TopicSearch.Changed) -> None:
        self.selected_topic = event.selected
        logging.info(f"New selection: {self.selected_topic}")

    @on(TimeControl.TimeChanged)
    def time_changed(self, event: TimeControl.TimeChanged) -> None:
        # For seeking sources, use explicit seeking for user-initiated time changes
        if isinstance(self.source, PlaybackSource):
            self.call_later(self.source.seek_to_time, event.value)
        self.current_time = event.value

    @on(TimeControl.PlaybackChanged)
    def playback_changed(self, event: TimeControl.PlaybackChanged) -> None:
        if event.playing:
            self.source.start_playback()
        else:
            self.source.pause_playback()

    @on(TimeControl.SpeedChanged)
    def speed_changed(self, event: TimeControl.SpeedChanged) -> None:
        self.speed = event.speed
        # Update source playback speed
        if isinstance(self.source, PlaybackSource):
            self.source.set_playback_speed(event.speed)

    def action_toggle_playback(self) -> None:
        # Only allow playback for seeking sources
        if isinstance(self.source, PlaybackSource):
            time_control = self.query_one(TimeControl)
            time_control.toggle()

    def action_focus_search(self) -> None:
        """Focus the search input."""
        search_input = self.query_one(Input)
        search_input.focus()
        topic_search = self.query_one(TopicSearch)
        topic_search.display = True

    def action_toggle_topic_search(self) -> None:
        """Toggle the TopicSearch visibility."""
        topic_search = self.query_one(TopicSearch)
        topic_search.display = not topic_search.display

    @on(ExitApp)
    async def on_exit(self) -> None:
        """Handle exit cleanup."""

        await self.source.close()

    def on_paste(self, event: events.Paste) -> None:
        """Handle paste events to return to the previous screen."""
        for path in get_file_paths(event.text):
            if path.suffix == ".mcap":
                self.app.switch_screen(DataScreen(str(path)))
                return
