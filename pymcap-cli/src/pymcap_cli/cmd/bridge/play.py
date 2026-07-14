"""`pymcap-cli bridge play` — publish MCAP playback into a live bridge."""

import asyncio
from typing import Annotated

from cyclopts import Group, Parameter
from robo_ws_bridge import WebSocketBridgeClient
from robo_ws_bridge.ws_types import ServerCapabilities

from pymcap_cli.cmd._cli_options import (
    BridgeTarget,
    ConnectTimeoutOption,
    EarlyBailOption,
    EndTimeOption,
    ExcludeTopicOption,
    ProgressOption,
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
from pymcap_cli.cmd.bridge._shared import console, to_ws_url
from pymcap_cli.log_setup import ERR

PLAYBACK_GROUP = Group("Playback")
SpeedOption = Annotated[
    float,
    Parameter(name=["--speed"], group=PLAYBACK_GROUP, help="Playback speed multiplier."),
]
LoopOption = Annotated[
    bool,
    Parameter(name=["--loop"], group=PLAYBACK_GROUP, help="Repeat playback until Ctrl+C."),
]


class BridgeClientPlaybackSink:
    def __init__(self, url: str, *, connect_timeout: float) -> None:
        self.url = url
        self.connect_timeout = connect_timeout
        self.client = WebSocketBridgeClient(url, min_retry_delay=0.2, max_retry_delay=2.0)
        self.channel_ids: dict[PlaybackChannel, int] = {}
        self._server_info = asyncio.Event()
        self._connection_lost = asyncio.Event()
        self.client.on_server_info(lambda *_: self._server_info.set())
        self.client.on_reconnecting(lambda *_: self._connection_lost.set())

    async def start(self, channels: tuple[PlaybackChannel, ...]) -> None:
        await self.client.connect()
        try:
            await asyncio.wait_for(self._server_info.wait(), timeout=self.connect_timeout)
        except asyncio.TimeoutError as exc:
            detail = (
                f"Timed out after {self.connect_timeout:.1f}s waiting for "
                f"serverInfo from {self.url}"
            )
            raise PlaybackError(detail) from exc
        server_info = self.client.server_info
        if server_info is None:
            raise PlaybackError(f"No serverInfo received from {self.url}")
        if ServerCapabilities.CLIENT_PUBLISH.value not in server_info["capabilities"]:
            raise PlaybackError(f"Bridge at {self.url} does not advertise clientPublish")
        supported = set(server_info.get("supportedEncodings", []))
        unsupported = sorted({channel.message_encoding for channel in channels} - supported)
        if unsupported:
            raise PlaybackError(
                f"Bridge at {self.url} does not support message encoding(s): "
                + ", ".join(unsupported)
            )
        for channel in channels:
            channel_id = await self.client.advertise(
                channel.topic,
                encoding=channel.message_encoding,
                schema_name=channel.schema_name,
                schema=channel.schema_text,
                schema_encoding=channel.schema_encoding or None,
            )
            self.channel_ids[channel] = channel_id
        console.print(f"[green]Connected to[/] [bold]{self.url}[/]")

    async def wait_until_ready(self) -> None:
        await asyncio.sleep(_SETTLE_SECONDS)

    async def publish(
        self, channel: PlaybackChannel, timestamp_ns: int, payload: bytes | memoryview
    ) -> None:
        del timestamp_ns
        if self._connection_lost.is_set():
            raise PlaybackError(f"Connection to {self.url} was lost during playback")
        await self.client.publish(self.channel_ids[channel], bytes(payload))

    async def timeline_started(self, clock: PlaybackClock) -> None:
        del clock

    async def timeline_finished(self, timestamp_ns: int) -> None:
        del timestamp_ns

    async def close(self) -> None:
        for channel_id in self.channel_ids.values():
            try:
                await self.client.unadvertise(channel_id)
            except (OSError, RuntimeError):
                break
        await self.client.disconnect()

    def status_rows(self) -> tuple[tuple[str, str], ...]:
        return (("Bridge", self.url),)


def play(
    files: list[str],
    *,
    target: BridgeTarget,
    speed: SpeedOption = 1.0,
    loop: LoopOption = False,
    topic: TopicOption = None,
    exclude_topic: ExcludeTopicOption = None,
    start: StartTimeOption = "",
    end: EndTimeOption = "",
    early_bail: EarlyBailOption = False,
    connect_timeout: ConnectTimeoutOption = 5.0,
    progress: ProgressOption = True,
) -> int:
    """Play one or more MCAP files into a live Foxglove WebSocket bridge.

    Inputs are merged chronologically by log time. The bridge must advertise
    ``clientPublish`` and support every selected message encoding.

    Examples
    --------
    ```
    pymcap-cli bridge play recording.mcap --target localhost
    pymcap-cli bridge play part1.mcap part2.mcap --target localhost --speed 2
    PYMCAP_BRIDGE=localhost pymcap-cli bridge play recording.mcap -t '/camera/.*'
    ```
    """
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
                BridgeClientPlaybackSink(to_ws_url(target), connect_timeout=connect_timeout),
                speed=speed,
                loop=loop,
                show_status=progress and console.is_terminal,
            )
        )
    except (PlaybackError, OSError, ValueError) as exc:
        ERR.print(f"[red]Error:[/] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[dim]Playback stopped.[/]")
        return 0
    console.print(f"[green]Published[/] {stats.messages:,} messages.")
    return 0
