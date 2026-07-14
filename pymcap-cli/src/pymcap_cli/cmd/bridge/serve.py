"""`pymcap-cli bridge serve` — host MCAP playback over Foxglove WebSocket."""

import asyncio

from robo_ws_bridge import ConnectionState, ServerConnection, WebSocketBridgeServer
from robo_ws_bridge.server import Channel as ServerChannel

from pymcap_cli.cmd._cli_options import (
    EarlyBailOption,
    EndTimeOption,
    ExcludeTopicOption,
    ProgressOption,
    ServerHostOption,
    ServerPortOption,
    StartTimeOption,
    TopicOption,
)
from pymcap_cli.cmd._message_filter_options import create_message_filter
from pymcap_cli.cmd.bridge._playback import (
    _SETTLE_SECONDS,
    PlaybackChannel,
    PlaybackClock,
    PlaybackError,
    prepare_playback,
    run_playback,
)
from pymcap_cli.cmd.bridge._shared import console
from pymcap_cli.cmd.bridge.play import LoopOption, SpeedOption
from pymcap_cli.log_setup import ERR

_CLOCK_INTERVAL_SECONDS = 1 / 30


class BridgeServerPlaybackSink:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.url = f"ws://{host}:{port}"
        self.server: WebSocketBridgeServer | None = None
        self.channel_ids: dict[PlaybackChannel, int] = {}
        self._subscriptions: set[tuple[ServerConnection, int]] = set()
        self._has_subscription = asyncio.Event()
        self.started = asyncio.Event()
        self._clock_done = asyncio.Event()
        self._clock_task: asyncio.Task[None] | None = None

    async def start(self, channels: tuple[PlaybackChannel, ...]) -> None:
        encodings = sorted({channel.message_encoding for channel in channels})
        server = WebSocketBridgeServer(
            host=self.host,
            port=self.port,
            name="pymcap-cli bridge serve",
            capabilities=["time"],
            supported_encodings=encodings,
            metadata={"source": "pymcap-cli"},
        )
        self.server = server
        for channel_id, channel in enumerate(channels, start=1):
            self.channel_ids[channel] = channel_id
            server.register_channel(
                ServerChannel(
                    id=channel_id,
                    topic=channel.topic,
                    encoding=channel.message_encoding,
                    schema_name=channel.schema_name,
                    schema=channel.schema_text,
                    schema_encoding=channel.schema_encoding or None,
                )
            )

        def on_subscribe(state: ConnectionState, subscription_id: int, _channel_id: int) -> None:
            websocket = state.websocket
            self._subscriptions.add((websocket, subscription_id))
            self._has_subscription.set()

        def on_unsubscribe(state: ConnectionState, subscription_id: int, _channel_id: int) -> None:
            websocket = state.websocket
            self._subscriptions.discard((websocket, subscription_id))
            if not self._subscriptions:
                self._has_subscription.clear()

        def on_disconnect(state: ConnectionState) -> None:
            websocket = state.websocket
            self._subscriptions = {
                entry for entry in self._subscriptions if entry[0] is not websocket
            }
            if not self._subscriptions:
                self._has_subscription.clear()

        server.on_subscribe(on_subscribe)
        server.on_unsubscribe(on_unsubscribe)
        server.on_disconnect(on_disconnect)
        await server.start()
        self.started.set()
        console.print(f"[green]Serving playback at[/] [bold]{self.url}[/]")

    async def wait_until_ready(self) -> None:
        console.print("[dim]Waiting for a client subscription...[/]")
        while True:
            await self._has_subscription.wait()
            await asyncio.sleep(_SETTLE_SECONDS)
            if self._subscriptions:
                return

    async def publish(
        self, channel: PlaybackChannel, timestamp_ns: int, payload: bytes | memoryview
    ) -> None:
        server = self.server
        assert server is not None
        await server.publish_message(
            self.channel_ids[channel], bytes(payload), timestamp_ns=timestamp_ns
        )

    async def timeline_started(self, clock: PlaybackClock) -> None:
        server = self.server
        assert server is not None
        self._clock_done.clear()

        async def publish_clock() -> None:
            while not self._clock_done.is_set():
                await server.publish_time(clock.current_time_ns())
                try:
                    await asyncio.wait_for(self._clock_done.wait(), timeout=_CLOCK_INTERVAL_SECONDS)
                except asyncio.TimeoutError:
                    continue

        await server.publish_time(clock.record_origin_ns)
        self._clock_task = asyncio.create_task(publish_clock())

    async def timeline_finished(self, timestamp_ns: int) -> None:
        self._clock_done.set()
        if self._clock_task is not None:
            await self._clock_task
            self._clock_task = None
        server = self.server
        assert server is not None
        await server.publish_time(timestamp_ns)

    async def close(self) -> None:
        self._clock_done.set()
        if self._clock_task is not None:
            await self._clock_task
            self._clock_task = None
        if self.server is not None:
            await self.server.stop()

    def status_rows(self) -> tuple[tuple[str, str], ...]:
        connections = 0 if self.server is None else len(self.server.connections)
        return (
            ("Server", self.url),
            ("Clients", str(connections)),
            ("Subscriptions", str(len(self._subscriptions))),
        )


def serve(
    files: list[str],
    *,
    host: ServerHostOption = "127.0.0.1",
    port: ServerPortOption = 8765,
    speed: SpeedOption = 1.0,
    loop: LoopOption = False,
    topic: TopicOption = None,
    exclude_topic: ExcludeTopicOption = None,
    start: StartTimeOption = "",
    end: EndTimeOption = "",
    early_bail: EarlyBailOption = False,
    progress: ProgressOption = True,
) -> int:
    """Host one or more MCAP files as a Foxglove WebSocket data source.

    Playback begins one second after the first client subscribes. All clients
    share one chronological playback timeline.

    Examples
    --------
    ```
    pymcap-cli bridge serve recording.mcap
    pymcap-cli bridge serve part1.mcap part2.mcap --speed 2 --loop
    pymcap-cli bridge serve recording.mcap --host 0.0.0.0 --port 8765
    ```
    """
    if not 1 <= port <= 65535:
        ERR.print("[red]Error:[/] --port must be in [1, 65535]")
        return 1
    try:
        message_filter = create_message_filter(
            topic=topic,
            exclude_topic=exclude_topic,
            start=start,
            end=end,
            early_bail=early_bail,
        )
        prepared = prepare_playback(files, message_filter)
        stats = asyncio.run(
            run_playback(
                prepared,
                BridgeServerPlaybackSink(host, port),
                speed=speed,
                loop=loop,
                show_status=progress and console.is_terminal,
            )
        )
    except (PlaybackError, OSError, ValueError) as exc:
        ERR.print(f"[red]Error:[/] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[dim]Server stopped.[/]")
        return 0
    console.print(f"[green]Served[/] {stats.messages:,} messages.")
    return 0
