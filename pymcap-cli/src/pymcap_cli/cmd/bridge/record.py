"""`pymcap-cli bridge record` — capture live bridge messages into MCAP."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from re import Pattern
from typing import Annotated

from cyclopts import Group as CycloptsGroup
from cyclopts import Parameter
from rich.console import Group, RenderableType
from rich.live import Live
from rich.table import Table
from rich.text import Text
from robo_ws_bridge import WebSocketBridgeClient
from robo_ws_bridge.ws_types import ChannelInfo
from small_mcap import McapWriter

from pymcap_cli.cmd.bridge._shared import (
    CONNECTION_GROUP,
    DISPLAY_GROUP,
    _format_byte_size,
    console,
    to_ws_url,
)
from pymcap_cli.constants import DEFAULT_CHUNK_SIZE, DEFAULT_COMPRESSION
from pymcap_cli.display.display_utils import _format_parts_with_colors
from pymcap_cli.log_setup import ERR
from pymcap_cli.types.types_manual import (
    ChunkSizeOption,
    CompressionName,
    CompressionOption,
    ForceOverwriteOption,
    OutputPathOption,
)
from pymcap_cli.utils import (
    McapWriterOptions,
    compile_topic_patterns,
    confirm_output_overwrite,
    create_mcap_writer,
)

logger = logging.getLogger(__name__)

RECORD_GROUP = CycloptsGroup("Record Options")


@dataclass(frozen=True)
class _RecorderChannelKey:
    topic: str
    schema_name: str
    schema_encoding: str
    schema_data: bytes
    message_encoding: str


@dataclass(frozen=True)
class TopicSelector:
    """Decide which topics to record, mirroring `ros2 bag record` semantics."""

    all_topics: bool = False
    exact_topics: frozenset[str] = field(default_factory=frozenset)
    include_patterns: tuple[Pattern[str], ...] = ()
    exclude_topics: frozenset[str] = field(default_factory=frozenset)
    exclude_patterns: tuple[Pattern[str], ...] = ()

    def matches(self, topic: str) -> bool:
        if topic in self.exclude_topics:
            return False
        if any(pattern.search(topic) for pattern in self.exclude_patterns):
            return False
        if self.all_topics:
            return True
        if topic in self.exact_topics:
            return True
        return any(pattern.search(topic) for pattern in self.include_patterns)


@dataclass
class BridgeRecorder:
    """Translate `(channel, timestamp, payload)` callbacks into MCAP records."""

    writer: McapWriter
    selector: TopicSelector
    message_limit: int | None = None
    done_event: asyncio.Event | None = None
    schema_ids: dict[tuple[str, str, bytes], int] = field(default_factory=dict)
    channel_ids: dict[_RecorderChannelKey, int] = field(default_factory=dict)
    message_counts: dict[str, int] = field(default_factory=dict)
    payload_bytes: int = 0
    total_messages: int = 0
    _next_schema_id: int = 1
    _next_channel_id: int = 1

    def matches_topic(self, topic: str) -> bool:
        return self.selector.matches(topic)

    def schema_id_for(self, channel: ChannelInfo) -> int:
        schema_name = channel.get("schemaName", "")
        schema_data = channel.get("schema", "").encode("utf-8")
        schema_encoding = channel.get("schemaEncoding", "")
        if not schema_name and not schema_data:
            return 0
        key = (schema_name, schema_encoding, schema_data)
        existing = self.schema_ids.get(key)
        if existing is not None:
            return existing
        schema_id = self._next_schema_id
        self._next_schema_id += 1
        self.schema_ids[key] = schema_id
        self.writer.add_schema(
            schema_id=schema_id,
            name=schema_name,
            encoding=schema_encoding,
            data=schema_data,
        )
        return schema_id

    def channel_id_for(self, channel: ChannelInfo) -> int:
        key = _RecorderChannelKey(
            topic=channel["topic"],
            schema_name=channel.get("schemaName", ""),
            schema_encoding=channel.get("schemaEncoding", ""),
            schema_data=channel.get("schema", "").encode("utf-8"),
            message_encoding=channel["encoding"],
        )
        existing = self.channel_ids.get(key)
        if existing is not None:
            return existing
        channel_id = self._next_channel_id
        self._next_channel_id += 1
        schema_id = self.schema_id_for(channel)
        self.channel_ids[key] = channel_id
        self.writer.add_channel(
            channel_id=channel_id,
            topic=key.topic,
            message_encoding=key.message_encoding,
            schema_id=schema_id,
        )
        return channel_id

    def on_message(self, channel: ChannelInfo, timestamp: int, payload: bytes) -> None:
        if not self.matches_topic(channel["topic"]):
            return
        if self.message_limit is not None and self.total_messages >= self.message_limit:
            return
        channel_id = self.channel_id_for(channel)
        self.writer.add_message(
            channel_id=channel_id,
            log_time=timestamp,
            data=payload,
            publish_time=timestamp,
        )
        self.total_messages += 1
        self.payload_bytes += len(payload)
        self.message_counts[channel["topic"]] = self.message_counts.get(channel["topic"], 0) + 1
        if (
            self.done_event is not None
            and self.message_limit is not None
            and self.total_messages >= self.message_limit
        ):
            self.done_event.set()


def _build_record_status(
    *,
    url: str,
    output: Path,
    recorder: BridgeRecorder,
    elapsed: float,
    duration: float | None,
    message_limit: int | None,
) -> RenderableType:
    summary = Table.grid(padding=(0, 1))
    summary.add_column(style="bold blue")
    summary.add_column()
    summary.add_row("Bridge:", f"[green]{url}[/]")
    summary.add_row("Output:", f"[green]{output}[/]")
    elapsed_str = f"{elapsed:.1f}s"
    if duration is not None:
        elapsed_str = f"{elapsed:.1f}/{duration:.1f}s"
    summary.add_row("Elapsed:", elapsed_str)
    counter = f"[green]{recorder.total_messages:,}[/]"
    if message_limit is not None:
        counter = f"{counter} / {message_limit:,}"
    summary.add_row("Messages:", counter)
    summary.add_row("Payload:", _format_byte_size(recorder.payload_bytes))

    table = Table(title="Per-topic", title_justify="left", title_style="bold cyan")
    table.add_column("Topic")
    table.add_column("Messages", justify="right")
    for topic in sorted(recorder.message_counts):
        count = recorder.message_counts[topic]
        table.add_row(_format_parts_with_colors(topic), f"{count:,}")

    if not recorder.message_counts:
        return Group(summary, Text(""), Text("Waiting for messages...", style="dim"))
    return Group(summary, Text(""), table)


async def _subscribe_if_matching(
    client: WebSocketBridgeClient, recorder: BridgeRecorder, channel: ChannelInfo
) -> None:
    if recorder.matches_topic(channel["topic"]):
        await client.subscribe(channel["topic"])


async def _subscribe_matching_channels(
    client: WebSocketBridgeClient, recorder: BridgeRecorder
) -> None:
    async def _on_advertise(channel: ChannelInfo) -> None:
        await _subscribe_if_matching(client, recorder, channel)

    client.on_advertised_channel(_on_advertise)
    for channel in list(client.channels.values()):
        await _subscribe_if_matching(client, recorder, channel)


async def _record_async(
    *,
    url: str,
    output: Path,
    selector: TopicSelector,
    duration: float | None,
    message_limit: int | None,
    chunk_size: int,
    compression_choice: CompressionName,
    connect_timeout: float,
    refresh_interval: float,
    show_status: bool,
) -> int:
    client = WebSocketBridgeClient(url, min_retry_delay=0.2, max_retry_delay=2.0)
    server_info_event = asyncio.Event()
    client.on_server_info(lambda *_: server_info_event.set())

    await client.connect()
    try:
        try:
            await asyncio.wait_for(server_info_event.wait(), timeout=connect_timeout)
        except asyncio.TimeoutError:
            ERR.print(f"[red]Error:[/] Timed out connecting to {url}")
            return 1

        try:
            with output.open("wb") as out:
                writer = create_mcap_writer(
                    out,
                    McapWriterOptions(chunk_size=chunk_size, compression=compression_choice),
                )
                writer.start(profile="", library="pymcap-cli bridge record")

                done = asyncio.Event()
                recorder = BridgeRecorder(
                    writer=writer,
                    selector=selector,
                    message_limit=message_limit,
                    done_event=done,
                )

                client.on_message(recorder.on_message)
                await _subscribe_matching_channels(client, recorder)

                start = time.monotonic()
                background: list[asyncio.Task[None]] = []
                if duration is not None:

                    async def _stop_after_duration() -> None:
                        await asyncio.sleep(duration)
                        done.set()

                    background.append(asyncio.create_task(_stop_after_duration()))

                async def _wait_for_done_with_status() -> None:
                    if not show_status:
                        await done.wait()
                        return
                    with Live(console=console, refresh_per_second=4) as live:
                        while not done.is_set():
                            elapsed = time.monotonic() - start
                            live.update(
                                _build_record_status(
                                    url=url,
                                    output=output,
                                    recorder=recorder,
                                    elapsed=elapsed,
                                    duration=duration,
                                    message_limit=message_limit,
                                )
                            )
                            try:
                                await asyncio.wait_for(
                                    asyncio.shield(done.wait()), timeout=refresh_interval
                                )
                            except asyncio.TimeoutError:
                                continue

                try:
                    await _wait_for_done_with_status()
                finally:
                    for task in background:
                        task.cancel()
                    if background:
                        await asyncio.gather(*background, return_exceptions=True)
                    writer.finish()
        except OSError as exc:
            ERR.print(f"[red]Error:[/] Failed to write to {output}: {exc}")
            return 1

        console.print(f"[green]Wrote {recorder.total_messages:,} messages to[/] [bold]{output}[/]")
        return 0
    finally:
        await client.disconnect()


def record(
    target: str,
    *,
    output: OutputPathOption,
    topics: Annotated[
        list[str],
        Parameter(
            name=["--topics"],
            group=RECORD_GROUP,
            help="Space-delimited list of topics to record (exact names).",
        ),
    ] = [],  # noqa: B006
    all_topics: Annotated[
        bool,
        Parameter(
            name=["--all", "-a"],
            group=RECORD_GROUP,
            negative="--no-all",
            help="Record every advertised topic.",
        ),
    ] = False,
    regex: Annotated[
        str | None,
        Parameter(
            name=["--regex", "-e"],
            group=RECORD_GROUP,
            help=(
                "Record topics matching this regex (case-insensitive, ``re.search``)."
                " Note: --all overrides --regex."
            ),
        ),
    ] = None,
    exclude_topics: Annotated[
        list[str],
        Parameter(
            name=["--exclude-topics"],
            group=RECORD_GROUP,
            help="Space-delimited list of topics to skip (exact names).",
        ),
    ] = [],  # noqa: B006
    exclude_regex: Annotated[
        str | None,
        Parameter(
            name=["--exclude-regex"],
            group=RECORD_GROUP,
            help="Skip topics matching this regex (case-insensitive). Wins over includes.",
        ),
    ] = None,
    duration: Annotated[
        float | None,
        Parameter(
            name=["--duration", "-d"],
            group=RECORD_GROUP,
            help=(
                "Stop after this many seconds. (Differs from ros2 bag's -d, which splits the bag.)"
            ),
        ),
    ] = None,
    message_limit: Annotated[
        int | None,
        Parameter(
            name=["--message-limit"],
            group=RECORD_GROUP,
            help="Stop after writing this many messages.",
        ),
    ] = None,
    connect_timeout: Annotated[
        float,
        Parameter(name=["--connect-timeout"], group=CONNECTION_GROUP),
    ] = 5.0,
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    force: ForceOverwriteOption = False,
    progress: Annotated[
        bool,
        Parameter(
            name=["--progress"],
            group=DISPLAY_GROUP,
            negative="--no-progress",
            help="Show a live status panel while recording.",
        ),
    ] = True,
) -> int:
    """Record messages from a live Foxglove WebSocket bridge into an MCAP file.

    Mirrors the ``ros2 bag record`` topic-selection surface: ``--topics`` for
    a list of exact names, ``-e/--regex`` for a regex include, ``-a/--all`` for
    everything, with ``--exclude-topics`` and ``--exclude-regex`` filtering on
    top. Stops on ``--duration`` / ``--message-limit`` or Ctrl+C.

    Parameters
    ----------
    target
        Bridge address — same forms accepted by ``bridge`` (URL, host, or
        ``host:port``); defaults to port 8765 when none is given.
    output
        MCAP file to write.
    topics
        Space-delimited list of topics to record (exact names).
    all_topics
        Record every advertised topic.
    regex
        Record topics matching this regex. ``--all`` overrides this.
    exclude_topics
        Topics to skip (exact names).
    exclude_regex
        Skip topics matching this regex. Wins over includes.
    duration
        Stop after this many seconds.
    message_limit
        Stop after writing this many messages.
    connect_timeout
        Seconds to wait for the bridge's ``serverInfo`` (default: 5.0).
    chunk_size
        MCAP chunk size in bytes.
    compression
        Output compression algorithm.
    force
        Overwrite ``output`` without prompting.
    progress
        Show a live status panel while recording.

    Examples
    --------
    ```
    pymcap-cli bridge record ws://localhost:8765 -a -o capture.mcap
    pymcap-cli bridge record localhost --topics /chatter /imu/data -o capture.mcap
    pymcap-cli bridge record localhost -e '^/camera/' -o capture.mcap -d 30
    pymcap-cli bridge record localhost -a --exclude-regex '^/debug/' -o capture.mcap
    ```
    """
    if duration is not None and duration <= 0:
        ERR.print("[red]Error:[/] --duration must be positive")
        return 1
    if message_limit is not None and message_limit <= 0:
        ERR.print("[red]Error:[/] --message-limit must be positive")
        return 1
    if not all_topics and not topics and regex is None:
        ERR.print("[red]Error:[/] specify --topics, --all, or --regex.")
        return 1

    output_path = Path(output)
    confirm_output_overwrite(output_path, force)

    try:
        include_patterns = (
            tuple(compile_topic_patterns([regex])) if regex is not None and not all_topics else ()
        )
        exclude_patterns = (
            tuple(compile_topic_patterns([exclude_regex])) if exclude_regex is not None else ()
        )
    except ValueError as exc:
        ERR.print(f"[red]Error:[/] {exc}")
        return 1

    selector = TopicSelector(
        all_topics=all_topics,
        exact_topics=frozenset(topics),
        include_patterns=include_patterns,
        exclude_topics=frozenset(exclude_topics),
        exclude_patterns=exclude_patterns,
    )

    url = to_ws_url(target)
    refresh_interval = 0.25

    try:
        return asyncio.run(
            _record_async(
                url=url,
                output=output_path,
                selector=selector,
                duration=duration,
                message_limit=message_limit,
                chunk_size=chunk_size,
                compression_choice=compression,
                connect_timeout=connect_timeout,
                refresh_interval=refresh_interval,
                show_status=progress,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Recording stopped.[/]")
        return 0
    except OSError as exc:
        ERR.print(f"[red]Error:[/] Recording failed for {url}: {exc}")
        return 1
