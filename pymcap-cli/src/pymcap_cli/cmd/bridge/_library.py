"""Path-safe recording library and shared Foxglove playback sessions."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias
from urllib.parse import parse_qs, urlsplit

from robo_ws_bridge import WebSocketBridgeEndpoint, install_invalid_handshake_log_filter
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response
from websockets.typing import Subprotocol

from pymcap_cli.cmd.bridge._playback import (
    PlaybackController,
    PlaybackError,
    PlaybackState,
    PlaybackStats,
    PreparedPlayback,
    prepare_playback,
    run_playback,
)
from pymcap_cli.cmd.bridge._playback_transforms import create_playback_transform_plan
from pymcap_cli.cmd.bridge.serve import BridgeServerPlaybackSink
from pymcap_cli.core.rosbag2_layout import find_bag_splits

if TYPE_CHECKING:
    from pymcap_cli.cmd.bridge._playback import PlaybackTransformPlan
    from pymcap_cli.cmd.bridge._playback_transforms import PlaybackTransformConfig
    from pymcap_cli.core.message_filter import MessageFilterOptions

_MAX_FILES_PER_SESSION = 32
_SESSION_IDLE_TIMEOUT_SECONDS = 30.0
JsonValue: TypeAlias = (
    str | int | float | bool | None | Sequence["JsonValue"] | Mapping[str, "JsonValue"]
)

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
    <p class="muted">Select one or more files. Selected recordings are merged by log time.</p>
    <div class="toolbar">
      <button id="open">Open in Foxglove</button>
      <button id="play-pause">Play</button>
      <label class="playback-control">Speed x
        <input id="speed" type="number" min="0.1" step="0.25" value="1">
      </label>
      <label class="playback-control"><input id="loop" type="checkbox" checked> Loop</label>
      <span id="status" class="muted">No recording selected</span>
    </div>
    <div id="recordings" class="recordings"></div>
    <section id="sessions">
      <h2>Active sessions</h2>
      <div id="active-sessions"></div>
    </section>
  </main>
  <script src="/app.js"></script>
</body>
</html>
"""

_CONTROL_HTML = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>MCAP playback</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body>
  <main class="controller">
    <h1>MCAP playback</h1>
    <div id="files" class="selected-files"></div>
    <div class="toolbar">
      <button id="foxglove">Open Foxglove</button>
      <button id="play-pause">Play</button>
      <label class="playback-control">Speed x
        <input id="speed" type="number" min="0.1" step="0.25" value="1">
      </label>
      <button class="speed-preset" data-speed="1">1x</button>
      <button class="speed-preset" data-speed="5">5x</button>
      <button class="speed-preset" data-speed="10">10x</button>
      <label class="playback-control"><input id="loop" type="checkbox" checked> Loop</label>
    </div>
    <div class="timeline">
      <input id="seek" type="range" min="0" max="0" value="0" step="0.001">
      <div class="timeline-labels">
        <span id="position">0:00</span>
        <span id="duration">0:00</span>
      </div>
    </div>
    <p id="status" class="muted">Connecting…</p>
  </main>
  <script src="/control.js"></script>
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
button {
  padding: .45rem .75rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--card);
  color: var(--fg);
  cursor: pointer;
}
button:hover { border-color: var(--accent); }
button:disabled { opacity: .45; cursor: default; }
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
.controller { max-width: 460px; }
.controller h1 { font-size: 1.35rem; }
.selected-files {
  margin-top: .6rem;
  color: var(--muted);
  font: 12px/1.5 ui-monospace, monospace;
  overflow-wrap: anywhere;
}
.timeline { margin-top: .5rem; }
.timeline input { width: 100%; }
.timeline-labels {
  display: flex;
  justify-content: space-between;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.playback-control { display: inline-flex; gap: .35rem; align-items: center; }
#speed { width: 4.5rem; }
#play-pause { min-width: 5rem; text-align: center; }
.speed-preset.active { border-color: var(--accent); color: var(--accent); }
#sessions { margin-top: 2rem; }
#sessions h2 { font-size: 1.1rem; margin: 0 0 .5rem; }
#active-sessions { display: grid; gap: .4rem; }
.session {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: .75rem;
  align-items: center;
  padding: .6rem .8rem;
  border: 1px solid var(--border);
  border-radius: 7px;
  background: var(--card);
}
.session .files {
  font: 13px/1.4 ui-monospace, monospace;
  overflow-wrap: anywhere;
}
.session .meta { color: var(--muted); white-space: nowrap; }
"""

_APP_JS = """\
(() => {
  const list = document.getElementById("recordings");
  const activeSessions = document.getElementById("active-sessions");
  const status = document.getElementById("status");
  const open = document.getElementById("open");
  const playPause = document.getElementById("play-pause");
  const speed = document.getElementById("speed");
  const loop = document.getElementById("loop");
  const controls = [open, playPause, speed, loop];
  let speedChangeTimer = null;

  const selected = () =>
    [...document.querySelectorAll('input[name="recording"]:checked')].map((item) => item.value);

  const query = (files) => {
    const params = new URLSearchParams();
    for (const file of files) params.append("file", file);
    return params;
  };

  const foxgloveUrl = (files) => {
    const websocket = new URL("/ws", window.location.href);
    websocket.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    websocket.search = query(files).toString();
    const foxglove = new URL("foxglove://open");
    foxglove.searchParams.set("ds", "foxglove-websocket");
    foxglove.searchParams.set("ds.url", websocket.toString());
    return foxglove;
  };

  const controllerUrl = (files) => {
    const controller = new URL("/control", window.location.href);
    controller.search = query(files).toString();
    return controller;
  };

  const openSession = (files) => {
    window.location.href = controllerUrl(files).toString();
  };

  const updateButtons = () => {
    const disabled = selected().length === 0;
    for (const control of controls) control.disabled = disabled;
    if (disabled) status.textContent = "No recording selected";
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
    if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
    return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
  };

  const refreshStatus = async () => {
    const files = selected();
    if (!files.length) return;
    try {
      const response = await fetch(`/api/session?${query(files)}`);
      const value = await response.json();
      const viewers = value.viewers === 1 ? "1 viewer" : `${value.viewers} viewers`;
      const droppedCount = (value.droppedMessages || 0) + (value.droppedFrames || 0);
      const dropped = droppedCount ? ` · ${droppedCount} dropped` : "";
      status.textContent = `${value.state} · ${viewers} · ${value.messages} messages${dropped}`;
      playPause.textContent = value.isPlaying ? "Pause" : "Play";
      if (document.activeElement !== speed) speed.value = String(value.speed);
      loop.checked = value.loop;
    } catch (_error) {
      status.textContent = "Status unavailable";
    }
  };

  const control = async (action, options = {}) => {
    const params = query(selected());
    params.set("action", action);
    for (const [key, value] of Object.entries(options)) params.set(key, String(value));
    const response = await fetch(`/api/control?${params}`);
    const value = await response.json();
    status.textContent = value.error || value.state;
    await refreshStatus();
  };

  const formatTime = (seconds) => {
    const total = Math.max(0, Math.round(seconds));
    const minutes = Math.floor(total / 60);
    const remainder = total % 60;
    return `${minutes}:${String(remainder).padStart(2, "0")}`;
  };

  const refreshSessions = async () => {
    try {
      const response = await fetch("/api/sessions");
      const value = await response.json();
      activeSessions.replaceChildren();
      if (!value.sessions.length) {
        const empty = document.createElement("p");
        empty.className = "muted";
        empty.textContent = "No active sessions";
        activeSessions.append(empty);
        return;
      }
      for (const session of value.sessions) {
        const row = document.createElement("div");
        row.className = "session";
        const files = document.createElement("span");
        files.className = "files";
        files.textContent = session.files.join(" + ");
        const viewers = session.viewers === 1 ? "1 viewer" : `${session.viewers} viewers`;
        const meta = document.createElement("span");
        meta.className = "meta";
        const position = formatTime(session.positionSeconds);
        const duration = formatTime(session.durationSeconds);
        meta.textContent = `${session.state} · ${viewers} · ${position} / ${duration}`;
        row.append(files, meta);
        activeSessions.append(row);
      }
    } catch (_error) {
      activeSessions.replaceChildren();
    }
  };

  playPause.onclick = () => control("toggle");
  speed.oninput = () => {
    clearTimeout(speedChangeTimer);
    const value = Number(speed.value);
    if (Number.isFinite(value) && value > 0) {
      speedChangeTimer = setTimeout(() => control("speed", {speed: value}), 250);
    }
  };
  loop.onchange = () => control("loop", {enabled: loop.checked});
  open.onclick = () => openSession(selected());

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
        checkbox.onchange = () => { updateButtons(); refreshStatus(); };
        const path = document.createElement("a");
        path.className = "path";
        path.textContent = recording.path;
        path.href = controllerUrl([recording.path]);
        path.target = "_blank";
        path.onclick = (event) => {
          event.preventDefault();
          openSession([recording.path]);
        };
        const size = document.createElement("span");
        size.className = "size";
        size.textContent = formatSize(recording.sizeBytes);
        label.append(checkbox, path, size);
        list.append(label);
      }
      updateButtons();
    })
    .catch(() => { list.innerHTML = '<p class="muted">Could not load recordings.</p>'; });

  updateButtons();
  refreshSessions();
  setInterval(() => { refreshStatus(); refreshSessions(); }, 1000);
})();
"""

_CONTROL_JS = """\
(() => {
  const files = new URLSearchParams(window.location.search).getAll("file");
  const seek = document.getElementById("seek");
  const position = document.getElementById("position");
  const duration = document.getElementById("duration");
  const status = document.getElementById("status");
  const playPause = document.getElementById("play-pause");
  const speed = document.getElementById("speed");
  const speedPresets = [...document.querySelectorAll(".speed-preset")];
  const loop = document.getElementById("loop");
  let isSeeking = false;
  let speedChangeTimer = null;

  const query = () => {
    const params = new URLSearchParams();
    for (const file of files) params.append("file", file);
    return params;
  };

  const foxgloveUrl = () => {
    const websocket = new URL("/ws", window.location.href);
    websocket.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    websocket.search = query().toString();
    const foxglove = new URL("foxglove://open");
    foxglove.searchParams.set("ds", "foxglove-websocket");
    foxglove.searchParams.set("ds.url", websocket.toString());
    return foxglove;
  };

  const formatTime = (seconds) => {
    const total = Math.max(0, Math.round(seconds));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const remainder = total % 60;
    return hours
      ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
      : `${minutes}:${String(remainder).padStart(2, "0")}`;
  };

  const renderPosition = () => {
    position.textContent = formatTime(Number(seek.value));
  };

  const refresh = async () => {
    try {
      const response = await fetch(`/api/session?${query()}`);
      const value = await response.json();
      const viewers = value.viewers === 1 ? "1 viewer" : `${value.viewers} viewers`;
      const droppedCount = (value.droppedMessages || 0) + (value.droppedFrames || 0);
      const dropped = droppedCount ? ` · ${droppedCount} dropped` : "";
      status.textContent = `${value.state} · ${viewers} · ${value.messages} messages${dropped}`;
      playPause.textContent = value.isPlaying ? "Pause" : "Play";
      if (document.activeElement !== speed) speed.value = String(value.speed);
      for (const preset of speedPresets) {
        preset.classList.toggle("active", Number(preset.dataset.speed) === value.speed);
      }
      loop.checked = value.loop;
      seek.max = String(value.durationSeconds);
      duration.textContent = formatTime(value.durationSeconds);
      if (!isSeeking) {
        seek.value = String(value.positionSeconds);
        renderPosition();
      }
    } catch (_error) {
      status.textContent = "Status unavailable";
    }
  };

  const control = async (action, options = {}) => {
    const params = query();
    params.set("action", action);
    for (const [key, value] of Object.entries(options)) params.set(key, String(value));
    const response = await fetch(`/api/control?${params}`);
    const value = await response.json();
    status.textContent = value.error || value.state;
    await refresh();
  };

  document.getElementById("files").textContent = files.join(" + ");
  document.getElementById("foxglove").onclick = () => {
    window.location.href = foxgloveUrl();
  };
  playPause.onclick = () => control("toggle");
  speed.oninput = () => {
    clearTimeout(speedChangeTimer);
    const value = Number(speed.value);
    if (Number.isFinite(value) && value > 0) {
      speedChangeTimer = setTimeout(() => control("speed", {speed: value}), 250);
    }
  };
  for (const preset of speedPresets) {
    preset.onclick = () => {
      const value = Number(preset.dataset.speed);
      speed.value = String(value);
      control("speed", {speed: value});
    };
  }
  loop.onchange = () => control("loop", {enabled: loop.checked});
  seek.oninput = () => {
    isSeeking = true;
    renderPosition();
  };
  seek.onchange = async () => {
    await control("seek", {offset: Number(seek.value)});
    isSeeking = false;
  };

  refresh();
  setInterval(refresh, 500);
})();
"""


@dataclass(frozen=True, slots=True)
class RecordingEntry:
    path: str
    size_bytes: int


class RecordingLibrary:
    """Discover and safely resolve MCAP files beneath one root directory."""

    def __init__(self, root: Path) -> None:
        if not root.is_dir():
            raise ValueError(f"{root} is not a directory")
        self.root = root.resolve()

    def recordings(self) -> tuple[RecordingEntry, ...]:
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
    """One shared playback room for an ordered set of recordings."""

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
    ) -> None:
        self.files = files
        self.prepared = prepared
        self.transform_plan = transform_plan
        self.endpoint = endpoint
        self.sink = sink
        self.speed = speed
        self.loop = loop
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
        self._task: asyncio.Task[None] | None = None

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
        transform_plan = create_playback_transform_plan(transform_config, prepared.channels)
        output_channels = prepared.channels if transform_plan is None else transform_plan.channels
        endpoint = WebSocketBridgeEndpoint(
            name=f"pymcap-cli: {', '.join(path.name for path in files)}",
            capabilities=["time"],
            supported_encodings=sorted({channel.message_encoding for channel in output_channels}),
            metadata={"source": "pymcap-cli"},
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
        )
        session.play()
        return session

    @property
    def state(self) -> PlaybackState:
        if self.error is not None:
            return "Error"
        if self.controller.state == "Paused":
            return self.controller.state
        return self.stats.state

    def status(self) -> dict[str, JsonValue]:
        playhead_ns = self.stats.playhead_ns or self._start_time_ns
        return {
            "state": self.state.lower(),
            "viewers": len(self.endpoint.connections),
            "messages": self.stats.messages,
            "droppedMessages": self.stats.dropped_messages,
            "droppedFrames": self.endpoint.dropped_frames,
            "payloadBytes": self.stats.payload_bytes,
            "playheadNs": playhead_ns,
            "positionSeconds": (playhead_ns - self.timeline_start_ns) / 1_000_000_000,
            "durationSeconds": (self.timeline_end_ns - self.timeline_start_ns) / 1_000_000_000,
            "isPlaying": self.is_playing,
            "speed": self.speed,
            "loop": self.loop,
            "files": [path.name for path in self.files],
            "error": self.error,
        }

    def play(self) -> None:
        if self._task is not None and not self._task.done():
            self.controller.play()
            return
        self.controller = PlaybackController(is_looping=self.loop, speed=self.speed)
        self.stats = PlaybackStats(playhead_ns=self._start_time_ns)
        self.error = None
        self._task = asyncio.create_task(self._run())

    def pause(self) -> None:
        if self._task is not None and not self._task.done():
            self.controller.pause()

    @property
    def is_playing(self) -> bool:
        return (
            self._task is not None and not self._task.done() and self.controller.state == "Playing"
        )

    def toggle_playback(self) -> None:
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def set_looping(self, is_looping: bool) -> None:
        self.loop = is_looping
        self.controller.set_looping(is_looping)

    def set_speed(self, speed: float) -> None:
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be finite and positive")
        self.speed = speed
        self.controller.set_speed(speed)

    async def seek(self, offset_seconds: float) -> None:
        if not math.isfinite(offset_seconds):
            raise ValueError("seek offset must be finite")
        duration_seconds = (self.timeline_end_ns - self.timeline_start_ns) / 1_000_000_000
        if not 0 <= offset_seconds <= duration_seconds:
            raise ValueError(f"seek offset must be between 0 and {duration_seconds:g} seconds")

        should_play = self.state in {"Preparing", "Waiting", "Playing"}
        self.controller.stop()
        await self._cancel_task()
        # Drop frames still queued for slow clients so the post-seek stream is
        # not preceded by stale pre-seek frames (which look like a jump back).
        self.endpoint.clear_pending_frames()
        self._start_time_ns = self.timeline_start_ns + round(offset_seconds * 1_000_000_000)
        self.controller = PlaybackController(
            start_paused=not should_play,
            is_looping=self.loop,
            speed=self.speed,
        )
        self.stats = PlaybackStats(
            state="Preparing" if should_play else "Paused",
            playhead_ns=self._start_time_ns,
        )
        self.error = None
        self._task = asyncio.create_task(self._run())

    async def _cancel_task(self) -> None:
        if self._task is None:
            return
        if not self._task.done():
            self._task.cancel()
        with suppress(asyncio.CancelledError, KeyboardInterrupt):
            await self._task

    async def close(self) -> None:
        self.controller.stop()
        await self._cancel_task()
        await self.endpoint.close_connections()

    async def _run(self) -> None:
        try:
            await run_playback(
                self.prepared,
                self.sink,
                speed=self.speed,
                loop=self.loop,
                show_status=False,
                transform_plan=self.transform_plan,
                controller=self.controller,
                stats=self.stats,
                start_time_ns=self._start_time_ns,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - expose background failures in session status
            self.error = str(exc)
            self.stats.state = "Error"


class RecordingSessionManager:
    def __init__(
        self,
        *,
        message_filter: MessageFilterOptions,
        transform_config: PlaybackTransformConfig,
        speed: float,
        loop: bool,
        session_idle_timeout: float,
    ) -> None:
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be finite and positive")
        if not math.isfinite(session_idle_timeout) or session_idle_timeout < 0:
            raise ValueError("session idle timeout must be finite and non-negative")
        self.message_filter = message_filter
        self.transform_config = transform_config
        self.speed = speed
        self.loop = loop
        self.session_idle_timeout = session_idle_timeout
        self._sessions: dict[tuple[Path, ...], RecordingSession] = {}
        self._cleanup_tasks: dict[tuple[Path, ...], asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._is_closing = False

    async def get_or_create(self, files: tuple[Path, ...]) -> RecordingSession:
        async with self._lock:
            session = self._sessions.get(files)
            if session is None:
                session = await RecordingSession.create(
                    files,
                    message_filter=self.message_filter,
                    transform_config=self.transform_config,
                    speed=self.speed,
                    loop=self.loop,
                )
                self._sessions[files] = session
                session.sink.on_activity_change(
                    lambda is_active, files=files, session=session: self._activity_changed(
                        files,
                        session,
                        is_active,
                    )
                )
            return session

    def get(self, files: tuple[Path, ...]) -> RecordingSession | None:
        return self._sessions.get(files)

    def active_sessions(self) -> tuple[dict[str, JsonValue], ...]:
        return tuple(session.status() for session in self._sessions.values())

    def _activity_changed(
        self,
        files: tuple[Path, ...],
        session: RecordingSession,
        is_active: bool,
    ) -> None:
        if self._is_closing:
            return
        cleanup_task = self._cleanup_tasks.pop(files, None)
        if cleanup_task is not None:
            cleanup_task.cancel()
        if not is_active:
            self._cleanup_tasks[files] = asyncio.create_task(
                self._cleanup_after_idle(files, session)
            )

    async def _cleanup_after_idle(
        self,
        files: tuple[Path, ...],
        session: RecordingSession,
    ) -> None:
        cleanup_task = asyncio.current_task()
        try:
            await asyncio.sleep(self.session_idle_timeout)
            async with self._lock:
                if self._sessions.get(files) is not session or session.sink.has_subscriptions:
                    return
                self._sessions.pop(files)
            await session.close()
        finally:
            if self._cleanup_tasks.get(files) is cleanup_task:
                self._cleanup_tasks.pop(files)

    async def close(self) -> None:
        self._is_closing = True
        try:
            cleanup_tasks = tuple(self._cleanup_tasks.values())
            self._cleanup_tasks.clear()
            for cleanup_task in cleanup_tasks:
                cleanup_task.cancel()
            for cleanup_task in cleanup_tasks:
                with suppress(asyncio.CancelledError):
                    await cleanup_task
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
            for session in sessions:
                await session.close()
        finally:
            self._is_closing = False


class RecordingLibraryServer:
    """Serve a minimal recording UI and path-isolated Foxglove sessions."""

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
        session_idle_timeout: float = _SESSION_IDLE_TIMEOUT_SECONDS,
    ) -> None:
        self.library = library
        self.host = host
        self.port = port
        self.manager = RecordingSessionManager(
            message_filter=message_filter,
            transform_config=transform_config,
            speed=speed,
            loop=loop,
            session_idle_timeout=session_idle_timeout,
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
            subprotocols=[Subprotocol("foxglove.websocket.v1")],
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
            files = self._resolve_query(parsed.query)
            session = await self.manager.get_or_create(files)
        except (OSError, PlaybackError, ValueError) as exc:
            await websocket.close(code=1008, reason=str(exc)[:120])
            return
        await session.endpoint.handle_connection(websocket)

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
            if parsed.path == "/control":
                self._resolve_query(parsed.query)
                return _response(200, "text/html; charset=utf-8", _CONTROL_HTML)
            if parsed.path == "/control.js":
                return _response(200, "text/javascript; charset=utf-8", _CONTROL_JS)
            if parsed.path == "/favicon.ico":
                return _response(204, "image/x-icon", "")
            if parsed.path == "/api/recordings":
                recordings = [
                    {"path": entry.path, "sizeBytes": entry.size_bytes}
                    for entry in self.library.recordings()
                ]
                return _json_response(200, {"recordings": recordings})
            if parsed.path == "/api/sessions":
                return _json_response(
                    200,
                    {"sessions": list(self.manager.active_sessions())},
                )
            if parsed.path == "/api/session":
                files = self._resolve_query(parsed.query)
                session = self.manager.get(files)
                if session is None:
                    return _json_response(
                        200,
                        {
                            "state": "idle",
                            "viewers": 0,
                            "messages": 0,
                            "droppedMessages": 0,
                            "droppedFrames": 0,
                            "payloadBytes": 0,
                            "playheadNs": 0,
                            "positionSeconds": 0,
                            "durationSeconds": 0,
                            "isPlaying": False,
                            "speed": self.manager.speed,
                            "loop": self.manager.loop,
                            "files": [path.name for path in files],
                            "error": None,
                        },
                    )
                return _json_response(200, session.status())
            if parsed.path == "/api/control":
                query = parse_qs(parsed.query, keep_blank_values=True)
                action = query.get("action", [""])[0]
                if action not in {"toggle", "seek", "loop", "speed"}:
                    return _json_response(
                        400,
                        {"error": "action must be toggle, seek, loop, or speed"},
                    )
                files = self._resolve_query(parsed.query)
                session = self.manager.get(files)
                if action == "toggle":
                    if session is None:
                        session = await self.manager.get_or_create(files)
                    else:
                        session.toggle_playback()
                elif action == "seek":
                    offsets = query.get("offset", [])
                    if len(offsets) != 1:
                        return _json_response(
                            400,
                            {"error": "seek requires one offset in seconds"},
                        )
                    try:
                        offset_seconds = float(offsets[0])
                    except ValueError:
                        return _json_response(
                            400,
                            {"error": "seek offset must be a number"},
                        )
                    session = await self.manager.get_or_create(files)
                    await session.seek(offset_seconds)
                elif action == "loop":
                    enabled_values = query.get("enabled", [])
                    if len(enabled_values) != 1 or enabled_values[0] not in {"true", "false"}:
                        return _json_response(
                            400,
                            {"error": "loop requires enabled=true or enabled=false"},
                        )
                    session = await self.manager.get_or_create(files)
                    session.set_looping(enabled_values[0] == "true")
                else:
                    speed_values = query.get("speed", [])
                    if len(speed_values) != 1:
                        return _json_response(
                            400,
                            {"error": "speed requires one positive multiplier"},
                        )
                    try:
                        speed = float(speed_values[0])
                    except ValueError:
                        return _json_response(
                            400,
                            {"error": "speed must be a number"},
                        )
                    if not math.isfinite(speed) or speed <= 0:
                        return _json_response(
                            400,
                            {"error": "speed must be finite and positive"},
                        )
                    session = await self.manager.get_or_create(files)
                    session.set_speed(speed)
                assert session is not None
                return _json_response(200, session.status())
        except (OSError, PlaybackError, ValueError) as exc:
            return _json_response(400, {"error": str(exc)})
        return _json_response(404, {"error": "not found"})

    def _resolve_query(self, query_string: str) -> tuple[Path, ...]:
        query = parse_qs(query_string, keep_blank_values=True)
        return self.library.resolve(query.get("file", []))


def _response(status: int, content_type: str, body: str) -> Response:
    payload = body.encode()
    headers = Headers()
    headers["Content-Type"] = content_type
    headers["Content-Length"] = str(len(payload))
    headers["Cache-Control"] = "no-store"
    return Response(status, HTTPStatus(status).phrase, headers, payload)


def _json_response(status: int, body: Mapping[str, JsonValue]) -> Response:
    return _response(status, "application/json; charset=utf-8", json.dumps(body))
