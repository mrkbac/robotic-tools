"""Path-safe recording library and shared Foxglove playback sessions."""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias
from urllib.parse import parse_qs, urlsplit

from robo_ws_bridge import (
    ConnectionState,
    PlaybackCommand,
    PlaybackControlRequest,
    PlaybackStatus,
    StatusLevel,
    WebSocketBridgeEndpoint,
    install_invalid_handshake_log_filter,
)
from robo_ws_bridge import (
    PlaybackState as FoxglovePlaybackState,
)
from small_mcap import McapFile
from typing_extensions import Self
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response
from websockets.typing import Subprotocol

from pymcap_cli.cmd.bridge._playback import (
    PlaybackController,
    PlaybackError,
    PlaybackStats,
    PreparedPlayback,
    prepare_playback,
    publish_playback_snapshot,
    run_playback,
)
from pymcap_cli.cmd.bridge._playback_transforms import (
    create_playback_preset_config,
    create_playback_transform_plan,
)
from pymcap_cli.cmd.bridge.serve import BridgeServerPlaybackSink
from pymcap_cli.core.rosbag2_layout import find_bag_splits

if TYPE_CHECKING:
    from pymcap_cli.cmd.bridge._playback import PlaybackSink, PlaybackTransformPlan
    from pymcap_cli.cmd.bridge._playback_transforms import PlaybackTransformConfig
    from pymcap_cli.core.message_filter import MessageFilterOptions

_MAX_FILES_PER_SESSION = 32
_PLAYBACK_STATUS_ID = "playback"
_PLAYBACK_DEGRADED_STATUS_ID = "playback-degraded"
JsonValue: TypeAlias = (
    str | int | float | bool | None | Sequence["JsonValue"] | Mapping[str, "JsonValue"]
)
SessionPreset = Literal["none", "compress", "decompress", "fast", "low"]


@dataclass(frozen=True, slots=True)
class RecordingRequest:
    files: tuple[Path, ...]
    preset: SessionPreset | None


class PlaybackRunner(Protocol):
    async def __call__(
        self,
        prepared: PreparedPlayback,
        sink: PlaybackSink,
        *,
        speed: float,
        loop: bool,
        show_status: bool,
        transform_plan: PlaybackTransformPlan | None = None,
        controller: PlaybackController | None = None,
        stats: PlaybackStats | None = None,
        start_time_ns: int | None = None,
        recordings: tuple[McapFile, ...] | None = None,
    ) -> PlaybackStats: ...


def _open_recordings(files: tuple[Path, ...]) -> tuple[McapFile, ...]:
    with ExitStack() as stack:
        recordings = tuple(stack.enter_context(McapFile.open(path)) for path in files)
        stack.pop_all()
    return recordings


def _open_local_recordings(files: tuple[str, ...]) -> tuple[McapFile, ...] | None:
    paths: list[Path] = []
    for file in files:
        parsed = urlsplit(file)
        if parsed.scheme not in {"", "file"}:
            return None
        paths.append(Path(parsed.path))
    return _open_recordings(tuple(paths))


_INDEX_HTML = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>MCAP recordings</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body>
  <main>
    <h1>MCAP recordings</h1>
    <p class="muted">Open one recording, or select several to merge them by log time.</p>
    <div class="toolbar">
      <a id="open" class="button" aria-disabled="true">Open selected in Foxglove</a>
      <span id="status" class="muted">No recording selected</span>
    </div>
    <div id="recordings" class="recordings"></div>
  </main>
  <script src="/app.js"></script>
</body>
</html>
"""

_STYLE_CSS = """\
:root {
  color-scheme: light dark;
  --bg: #fafafa;
  --fg: #191919;
  --muted: #707070;
  --card: #fff;
  --border: #ddd;
  --accent: #2368d8;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #111;
    --fg: #eee;
    --muted: #999;
    --card: #191919;
    --border: #333;
    --accent: #79adff;
  }
}
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font: 15px/1.5 system-ui, sans-serif;
}
main { max-width: 900px; margin: 0 auto; padding: 2rem 1rem; }
h1 { margin: 0; }
.muted { color: var(--muted); }
.toolbar {
  position: sticky;
  top: 0;
  display: flex;
  flex-wrap: wrap;
  gap: .5rem;
  align-items: center;
  padding: 1rem 0;
  background: var(--bg);
}
.button {
  display: inline-block;
  padding: .45rem .75rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--card);
  color: var(--fg);
  cursor: pointer;
  text-decoration: none;
}
.button:hover { border-color: var(--accent); }
.button[aria-disabled="true"] { opacity: .45; pointer-events: none; }
#status { margin-left: .5rem; }
.recordings { display: grid; gap: .4rem; }
.recording {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: .75rem;
  align-items: center;
  padding: .7rem .8rem;
  border: 1px solid var(--border);
  border-radius: 7px;
  background: var(--card);
}
.path {
  color: var(--fg);
  font: 13px/1.4 ui-monospace, monospace;
  overflow-wrap: anywhere;
  text-decoration: none;
}
.path:hover { color: var(--accent); text-decoration: underline; }
.size { color: var(--muted); white-space: nowrap; }
"""

_APP_JS = """\
(() => {
  const list = document.getElementById("recordings");
  const status = document.getElementById("status");
  const open = document.getElementById("open");

  const selected = () =>
    [...document.querySelectorAll('input[name="recording"]:checked')].map((item) => item.value);

  const query = (files) => {
    const params = new URLSearchParams();
    for (const file of files) params.append("file", file);
    return params;
  };

  const websocketUrl = (files) => {
    const websocket = new URL("/ws", window.location.href);
    websocket.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    websocket.search = query(files).toString();
    return websocket;
  };

  const foxgloveUrl = (websocket) => {
    const foxglove = new URL("foxglove://open");
    foxglove.searchParams.set("ds", "foxglove-websocket");
    foxglove.searchParams.set("ds.url", websocket);
    return foxglove;
  };

  const openFoxglove = (event) => {
    event.preventDefault();
    window.location.href = foxgloveUrl(event.currentTarget.href).toString();
  };

  open.addEventListener("click", openFoxglove);

  const updateOpenLink = () => {
    const files = selected();
    if (!files.length) {
      open.removeAttribute("href");
      open.setAttribute("aria-disabled", "true");
      status.textContent = "No recording selected";
      return;
    }
    open.href = websocketUrl(files).toString();
    open.setAttribute("aria-disabled", "false");
    status.textContent = files.length === 1
      ? "1 recording selected"
      : `${files.length} recordings selected`;
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
    if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
    return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
  };

  fetch("/api/recordings")
    .then((response) => response.json())
    .then((value) => {
      if (!value.recordings.length) {
        list.innerHTML = '<p class="muted">No .mcap files found.</p>';
        return;
      }
      for (const recording of value.recordings) {
        const label = document.createElement("label");
        label.className = "recording";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.name = "recording";
        checkbox.value = recording.path;
        checkbox.onchange = updateOpenLink;
        const path = document.createElement("a");
        path.className = "path";
        path.textContent = recording.path;
        path.href = websocketUrl([recording.path]).toString();
        path.addEventListener("click", openFoxglove);
        const size = document.createElement("span");
        size.className = "size";
        size.textContent = formatSize(recording.sizeBytes);
        label.append(checkbox, path, size);
        list.append(label);
      }
      updateOpenLink();
    })
    .catch(() => { list.innerHTML = '<p class="muted">Could not load recordings.</p>'; });

  updateOpenLink();
})();
"""


@dataclass(frozen=True, slots=True)
class RecordingEntry:
    path: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _SelectedRecordings:
    paths: dict[str, Path]
    entries: tuple[RecordingEntry, ...]


class RecordingLibrary:
    """Discover and safely resolve MCAP files beneath one root directory."""

    def __init__(self, root: Path) -> None:
        if not root.is_dir():
            raise ValueError(f"{root} is not a directory")
        self.root = root.resolve()
        self._selected: _SelectedRecordings | None = None

    @classmethod
    def from_paths(cls, paths: tuple[Path, ...]) -> Self:
        if not paths:
            raise ValueError("At least one MCAP input is required")
        resolved_paths = tuple(path.resolve() for path in paths)
        root = Path(os.path.commonpath(path.parent for path in resolved_paths))
        library = cls(root)
        selected_paths: dict[str, Path] = {}
        entries: list[RecordingEntry] = []
        for path in resolved_paths:
            splits = find_bag_splits(path) if path.is_dir() else []
            is_mcap = path.is_file() and path.suffix.lower() == ".mcap"
            if not is_mcap and not splits:
                raise ValueError(f"MCAP file does not exist: {path}")
            relative = path.relative_to(root).as_posix()
            if relative in selected_paths:
                raise ValueError(f"duplicate file selection: {path}")
            selected_paths[relative] = path
            entries.append(
                RecordingEntry(
                    path=relative,
                    size_bytes=(
                        sum(split.stat().st_size for split in splits)
                        if splits
                        else path.stat().st_size
                    ),
                )
            )
        library._selected = _SelectedRecordings(selected_paths, tuple(entries))
        return library

    def recordings(self) -> tuple[RecordingEntry, ...]:
        if self._selected is not None:
            return self._selected.entries
        entries: dict[str, RecordingEntry] = {}
        candidates: list[Path] = []
        for candidate in self.root.rglob("*.mcap"):
            resolved = candidate.resolve()
            try:
                resolved.relative_to(self.root)
            except ValueError:
                continue
            if not resolved.is_file():
                continue
            candidates.append(resolved)

        grouped_splits: set[Path] = set()
        for parent in {candidate.parent for candidate in candidates}:
            if parent == self.root:
                continue
            splits = find_bag_splits(parent)
            if not splits:
                continue
            resolved_splits = {split.resolve() for split in splits}
            grouped_splits.update(resolved_splits)
            relative = parent.relative_to(self.root)
            path = relative.as_posix()
            entries[path] = RecordingEntry(
                path=path,
                size_bytes=sum(split.stat().st_size for split in resolved_splits),
            )

        for resolved in candidates:
            if resolved in grouped_splits:
                continue
            relative = resolved.relative_to(self.root)
            path = relative.as_posix()
            entries[path] = RecordingEntry(path=path, size_bytes=resolved.stat().st_size)
        return tuple(entries[path] for path in sorted(entries, key=str.casefold))

    def resolve(self, requested: list[str]) -> tuple[Path, ...]:
        if not requested:
            raise ValueError("At least one file is required")
        if len(requested) > _MAX_FILES_PER_SESSION:
            raise ValueError(f"At most {_MAX_FILES_PER_SESSION} files may be opened together")

        if self._selected is not None:
            resolved_files: list[Path] = []
            seen: set[Path] = set()
            for raw_path in requested:
                candidate = self._selected.paths.get(raw_path)
                if candidate is None:
                    raise ValueError(f"Recording is not available: {raw_path}")
                if candidate in seen:
                    raise ValueError(f"duplicate file selection: {raw_path}")
                seen.add(candidate)
                resolved_files.append(candidate)
            return tuple(resolved_files)

        resolved_files: list[Path] = []
        seen: set[Path] = set()
        for raw_path in requested:
            relative = Path(raw_path)
            if relative.is_absolute():
                raise ValueError(
                    "Only relative MCAP file paths and rosbag2 directories are allowed"
                )
            candidate = (self.root / relative).resolve()
            try:
                candidate.relative_to(self.root)
            except ValueError as exc:
                raise ValueError(f"File is outside the recording root: {raw_path}") from exc
            is_mcap = candidate.is_file() and relative.suffix.lower() == ".mcap"
            is_bag = candidate.is_dir() and bool(find_bag_splits(candidate))
            if not is_mcap and not is_bag:
                if candidate.is_dir():
                    raise ValueError(f"Directory is not a rosbag2 MCAP recording: {raw_path}")
                raise ValueError(f"MCAP file does not exist: {raw_path}")
            if candidate in seen:
                raise ValueError(f"duplicate file selection: {raw_path}")
            seen.add(candidate)
            resolved_files.append(candidate)
        return tuple(resolved_files)


class RecordingSession:
    """One independent playback session for an ordered set of recordings."""

    def __init__(
        self,
        files: tuple[Path, ...],
        prepared: PreparedPlayback,
        transform_plan: PlaybackTransformPlan | None,
        endpoint: WebSocketBridgeEndpoint,
        sink: BridgeServerPlaybackSink,
        *,
        speed: float,
        loop: bool,
        show_status: bool = False,
        playback_runner: PlaybackRunner = run_playback,
        recordings: tuple[McapFile, ...] | None = None,
    ) -> None:
        self.files = files
        self.prepared = prepared
        self.transform_plan = transform_plan
        self.endpoint = endpoint
        self.sink = sink
        self.speed = speed
        self.loop = loop
        self.show_status = show_status
        self._playback_runner = playback_runner
        self.controller = PlaybackController(is_looping=loop, speed=speed)
        self.stats = PlaybackStats()
        self.timeline_start_ns = max(
            prepared.recording_start_ns,
            prepared.resolved_filter.start_time_ns,
        )
        self.timeline_end_ns = min(
            prepared.recording_end_ns,
            prepared.resolved_filter.end_time_ns,
        )
        self._start_time_ns = self.timeline_start_ns
        self.error: str | None = None
        self._snapshot_time_ns: int | None = None
        self._task: asyncio.Task[None] | None = None
        self._has_degraded_status = False
        self._control_lock = asyncio.Lock()
        self.endpoint.on_playback_control(self.handle_playback_control)
        self.sink.on_timeline_started(self._handle_timeline_started)

        output_channels = prepared.channels if transform_plan is None else transform_plan.channels
        topics_by_channel_id = {
            sink.channel_ids[channel]: channel.topic for channel in output_channels
        }

        async def publish_subscription_snapshot(
            _state: ConnectionState,
            _subscription_id: int,
            channel_id: int,
        ) -> None:
            topic = topics_by_channel_id.get(channel_id)
            if topic is None:
                return
            if self.controller.state != "Paused" and self.stats.state != "Finished":
                return
            current_time_ns = self.sink.current_time_ns
            if current_time_ns is None:
                current_time_ns = max(self._start_time_ns, self.stats.playhead_ns)
            await publish_playback_snapshot(
                self.prepared,
                self.sink,
                timestamp_ns=current_time_ns,
                transform_plan=self.transform_plan,
                topics={topic},
                recordings=self._recordings,
            )
            await self.endpoint.publish_time(current_time_ns)

        self.endpoint.on_subscribe(publish_subscription_snapshot)
        self._recordings = (
            _open_local_recordings(prepared.files) if recordings is None else recordings
        )

    @classmethod
    async def create(
        cls,
        files: tuple[Path, ...],
        *,
        message_filter: MessageFilterOptions,
        transform_config: PlaybackTransformConfig,
        speed: float,
        loop: bool,
    ) -> RecordingSession:
        prepared = await asyncio.to_thread(
            prepare_playback,
            [str(path) for path in files],
            message_filter,
        )
        recording_paths = tuple(Path(path) for path in prepared.files)
        recordings = await asyncio.to_thread(_open_recordings, recording_paths)
        try:
            transform_plan = create_playback_transform_plan(transform_config, prepared.channels)
            output_channels = (
                prepared.channels if transform_plan is None else transform_plan.channels
            )
            timeline_start_ns = max(
                prepared.recording_start_ns,
                prepared.resolved_filter.start_time_ns,
            )
            timeline_end_ns = min(
                prepared.recording_end_ns,
                prepared.resolved_filter.end_time_ns,
            )
            endpoint = WebSocketBridgeEndpoint(
                name=f"pymcap-cli: {', '.join(path.name for path in files)}",
                capabilities=["time"],
                supported_encodings=sorted(
                    {channel.message_encoding for channel in output_channels}
                ),
                metadata={"source": "pymcap-cli"},
                playback_time_range=(timeline_start_ns, timeline_end_ns),
            )
            sink = BridgeServerPlaybackSink(
                "127.0.0.1",
                0,
                endpoint=endpoint,
                url="/ws",
            )
            await sink.start(output_channels)
            session = cls(
                files,
                prepared,
                transform_plan,
                endpoint,
                sink,
                speed=speed,
                loop=loop,
                recordings=recordings,
            )
            session.play()
            session.broadcast_playback_state()
        except BaseException:
            for recording in recordings:
                recording.close()
            raise
        return session

    def play(self) -> None:
        if self._task is not None and not self._task.done():
            self.controller.play()
            return
        self.controller = PlaybackController(is_looping=self.loop, speed=self.speed)
        self.stats = PlaybackStats(playhead_ns=self._start_time_ns)
        self._task = asyncio.create_task(self._run())

    def pause(self) -> None:
        if self._task is not None and not self._task.done():
            self.controller.pause()

    def broadcast_playback_state(self) -> None:
        """Notify Foxglove after a playback state change outside a client request."""
        self.endpoint.broadcast_playback_state(self._foxglove_playback_state(did_seek=False))

    def _handle_timeline_started(self) -> None:
        self.stats.state = "Playing"
        self.broadcast_playback_state()

    def set_speed(self, speed: float) -> None:
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be finite and positive")
        self.speed = speed
        self.controller.set_speed(speed)

    async def handle_playback_control(
        self,
        request: PlaybackControlRequest,
    ) -> FoxglovePlaybackState:
        """Apply one Foxglove playback request to this session."""
        async with self._control_lock:
            should_play = request.playback_command is PlaybackCommand.PLAY
            if request.playback_speed > 0:
                self.set_speed(request.playback_speed)
            elif should_play or request.playback_speed != 0:
                raise ValueError("playback speed must be finite and positive while playing")
            if request.seek_time is not None:
                await self._seek_to_timestamp(request.seek_time, should_play=should_play)
            elif should_play:
                self.play()
            else:
                self.pause()
            return self._foxglove_playback_state(did_seek=request.seek_time is not None)

    def _foxglove_playback_state(self, *, did_seek: bool) -> FoxglovePlaybackState:
        current_time = self.sink.current_time_ns
        if current_time is None:
            playhead_ns = self.stats.playhead_ns
            current_time = self._start_time_ns if playhead_ns is None else playhead_ns
        if self.controller.state == "Paused":
            status = PlaybackStatus.PAUSED
        elif self.stats.state == "Finished":
            status = PlaybackStatus.ENDED
        elif self.stats.state in {"Preparing", "Waiting"}:
            status = PlaybackStatus.BUFFERING
        elif self.controller.state == "Playing":
            status = PlaybackStatus.PLAYING
        else:
            status = PlaybackStatus.BUFFERING
        return FoxglovePlaybackState(
            status=status,
            current_time=current_time,
            playback_speed=self.speed,
            did_seek=did_seek,
        )

    async def _seek_to_timestamp(self, timestamp_ns: int, *, should_play: bool) -> None:
        # Foxglove's step buttons can intentionally seek past either boundary.
        timestamp_ns = min(max(timestamp_ns, self.timeline_start_ns), self.timeline_end_ns)
        self.controller.stop()
        await self._cancel_task()
        # Drop frames still queued for slow clients so the post-seek stream is
        # not preceded by stale pre-seek frames (which look like a jump back).
        self.endpoint.clear_pending_frames()
        self._start_time_ns = timestamp_ns
        self.controller = PlaybackController(
            start_paused=not should_play,
            is_looping=self.loop,
            speed=self.speed,
        )
        self.stats = PlaybackStats(
            state="Preparing" if should_play else "Paused",
            playhead_ns=self._start_time_ns,
        )
        self._snapshot_time_ns = timestamp_ns - 1 if should_play else timestamp_ns
        self._task = asyncio.create_task(self._run())

    async def _cancel_task(self) -> None:
        if self._task is None:
            return
        if not self._task.done():
            self._task.cancel()
        with suppress(asyncio.CancelledError, KeyboardInterrupt):
            await self._task

    async def wait(self) -> PlaybackStats:
        """Wait for the current playback run, following task replacement after seeks."""
        while self._task is not None:
            task = self._task
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise
                if self._task is task:
                    break
                continue
            if self._task is task:
                break
        return self.stats

    async def close(self) -> None:
        self.controller.stop()
        try:
            await self._cancel_task()
            await self.endpoint.close_connections()
        finally:
            for recording in self._recordings or ():
                recording.close()

    async def _run(self) -> None:
        try:
            if self.error is not None:
                await self.endpoint.remove_status((_PLAYBACK_STATUS_ID,))
                self.error = None
            snapshot_time_ns = self._snapshot_time_ns
            self._snapshot_time_ns = None
            if snapshot_time_ns is not None:
                await publish_playback_snapshot(
                    self.prepared,
                    self.sink,
                    timestamp_ns=snapshot_time_ns,
                    transform_plan=self.transform_plan,
                    recordings=self._recordings,
                )
                await self.endpoint.publish_time(self._start_time_ns)
            await self._playback_runner(
                self.prepared,
                self.sink,
                speed=self.speed,
                loop=self.loop,
                show_status=self.show_status,
                transform_plan=self.transform_plan,
                controller=self.controller,
                stats=self.stats,
                start_time_ns=self._start_time_ns,
                recordings=self._recordings,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - expose background failures in session status
            self.error = str(exc)
            self.stats.state = "Error"
            self.controller.pause()
            await self.endpoint.send_status(
                StatusLevel.ERROR,
                f"Playback failed: {self.error}",
                status_id=_PLAYBACK_STATUS_ID,
            )
            self.broadcast_playback_state()
        else:
            if self.stats.dropped_messages:
                count = self.stats.dropped_messages
                unit = "frame" if count == 1 else "frames"
                await self.endpoint.send_status(
                    StatusLevel.WARNING,
                    f"Playback dropped {count:,} {unit} to stay synchronized",
                    status_id=_PLAYBACK_DEGRADED_STATUS_ID,
                )
                self._has_degraded_status = True
            elif self._has_degraded_status:
                await self.endpoint.remove_status((_PLAYBACK_DEGRADED_STATUS_ID,))
                self._has_degraded_status = False
            self.broadcast_playback_state()


class RecordingSessionManager:
    def __init__(
        self,
        *,
        message_filter: MessageFilterOptions,
        transform_config: PlaybackTransformConfig,
        speed: float,
        loop: bool,
    ) -> None:
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be finite and positive")
        self.message_filter = message_filter
        self.transform_config = transform_config
        self.speed = speed
        self.loop = loop
        self._sessions: set[RecordingSession] = set()
        self._lock = asyncio.Lock()

    async def create(
        self,
        files: tuple[Path, ...],
        *,
        preset: SessionPreset | None = None,
    ) -> RecordingSession:
        async with self._lock:
            session = await RecordingSession.create(
                files,
                message_filter=self.message_filter,
                transform_config=_session_transform_config(self.transform_config, preset),
                speed=self.speed,
                loop=self.loop,
            )
            self._sessions.add(session)
            return session

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)

    async def remove(self, session: RecordingSession) -> None:
        async with self._lock:
            if session not in self._sessions:
                return
            self._sessions.remove(session)
        await session.close()

    async def close(self) -> None:
        async with self._lock:
            sessions = tuple(self._sessions)
            self._sessions.clear()
        for session in sessions:
            await session.close()


class RecordingLibraryServer:
    """Serve a minimal recording UI and per-connection Foxglove sessions."""

    def __init__(
        self,
        library: RecordingLibrary,
        *,
        host: str,
        port: int,
        message_filter: MessageFilterOptions,
        transform_config: PlaybackTransformConfig,
        speed: float,
        loop: bool,
    ) -> None:
        self.library = library
        self.host = host
        self.port = port
        self.manager = RecordingSessionManager(
            message_filter=message_filter,
            transform_config=transform_config,
            speed=speed,
            loop=loop,
        )
        self._server: Server | None = None

    async def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("server already running")
        install_invalid_handshake_log_filter()
        self._server = await serve(
            self._handle_connection,
            self.host,
            self.port,
            subprotocols=[
                Subprotocol("foxglove.sdk.v1"),
                Subprotocol("foxglove.websocket.v1"),
            ],
            process_request=self._process_request,
        )
        socket = next(iter(self._server.sockets), None)
        if socket is not None:
            self.port = int(socket.getsockname()[1])

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        await self._server.serve_forever()

    async def stop(self) -> None:
        await self.manager.close()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        request = websocket.request
        if request is None:
            await websocket.close(code=1008, reason="Missing request")
            return
        parsed = urlsplit(request.path)
        if parsed.path != "/ws":
            await websocket.close(code=1008, reason="Unknown WebSocket path")
            return
        try:
            request = self._resolve_query(parsed.query)
            session = await self.manager.create(request.files, preset=request.preset)
        except (OSError, PlaybackError, ValueError) as exc:
            await websocket.close(code=1008, reason=str(exc)[:120])
            return
        try:
            await session.endpoint.handle_connection(websocket)
        finally:
            await self.manager.remove(session)

    async def _process_request(
        self,
        _connection: ServerConnection,
        request: Request,
    ) -> Response | None:
        parsed = urlsplit(request.path)
        if parsed.path == "/ws":
            return None

        try:
            if parsed.path == "/":
                return _response(200, "text/html; charset=utf-8", _INDEX_HTML)
            if parsed.path == "/style.css":
                return _response(200, "text/css; charset=utf-8", _STYLE_CSS)
            if parsed.path == "/app.js":
                return _response(200, "text/javascript; charset=utf-8", _APP_JS)
            if parsed.path == "/favicon.ico":
                return _response(204, "image/x-icon", "")
            if parsed.path == "/api/recordings":
                recordings = [
                    {"path": entry.path, "sizeBytes": entry.size_bytes}
                    for entry in self.library.recordings()
                ]
                return _json_response(200, {"recordings": recordings})
        except (OSError, PlaybackError, ValueError) as exc:
            return _json_response(400, {"error": str(exc)})
        return _json_response(404, {"error": "not found"})

    def _resolve_query(self, query_string: str) -> RecordingRequest:
        query = parse_qs(query_string, keep_blank_values=True)
        unknown = sorted(query.keys() - {"file", "preset"})
        if unknown:
            raise ValueError(f"Unknown WebSocket query parameter: {unknown[0]}")
        return RecordingRequest(
            files=self.library.resolve(query.get("file", [])),
            preset=_parse_session_preset(query.get("preset", [])),
        )


def _parse_session_preset(values: list[str]) -> SessionPreset | None:
    if not values:
        return None
    if len(values) > 1:
        raise ValueError("preset may be provided at most once")
    match values[0]:
        case "none":
            return "none"
        case "compress":
            return "compress"
        case "decompress":
            return "decompress"
        case "fast":
            return "fast"
        case "low":
            return "low"
        case value:
            raise ValueError(f"Unknown playback preset: {value}")


def _session_transform_config(
    default: PlaybackTransformConfig,
    preset: SessionPreset | None,
) -> PlaybackTransformConfig:
    if preset is None:
        return default
    if preset == "none":
        return None
    return create_playback_preset_config(preset, adaptive_quality=True)


def _response(status: int, content_type: str, body: str) -> Response:
    payload = body.encode()
    headers = Headers()
    headers["Content-Type"] = content_type
    headers["Content-Length"] = str(len(payload))
    headers["Cache-Control"] = "no-store"
    return Response(status, HTTPStatus(status).phrase, headers, payload)


def _json_response(status: int, body: Mapping[str, JsonValue]) -> Response:
    return _response(status, "application/json; charset=utf-8", json.dumps(body))
