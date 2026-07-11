"""Tests for `pymcap-cli bridge proxy` low-latency routing helpers."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pymcap_cli.core.processors.video_compress as video_compress
import pytest
from mcap_codec_support.pointcloud.schemas import POINTCLOUD2
from mcap_codec_support.video.common import EncoderConfig
from pymcap_cli.cmd.bridge._proxy_dashboard import ProxyDashboard
from pymcap_cli.cmd.bridge._proxy_runtime import (
    IncomingMessage,
    OutboundMessage,
    ProxyMetrics,
    SendManager,
    TransformResult,
    TransformWorker,
)
from pymcap_cli.cmd.bridge._proxy_transforms import (
    ImageConfig,
    PointCloudConfig,
    ProcessorRule,
    ProxyConfig,
    VideoTransformer,
    build_transform_rules,
    create_transformer,
    is_video_keyframe,
)
from pymcap_cli.cmd.bridge.proxy import BridgeProxy
from pymcap_cli.core.processors.message_transform import (
    MessageTransformProcessor,
    TransformOutput,
)
from pymcap_cli.core.processors.video_compress import ResolvedVideoCompressionBackend
from rich.console import Console
from robo_ws_bridge.server import Channel as ServerChannel
from robo_ws_bridge.server import WebSocketBridgeServer

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcap_codec_support.video import EncoderMode
    from robo_ws_bridge.ws_types import ChannelInfo


async def _wait_for(predicate: Callable[[], bool]) -> None:
    for _ in range(50):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not reached")


def _proxy_config() -> ProxyConfig:
    return ProxyConfig(
        image=ImageConfig(
            image_format="none",
            codec="h264",
            quality=28,
            encoder=None,
            backend="auto",
            scale=None,
            jpeg_quality=90,
        ),
        pointcloud=PointCloudConfig(
            enabled=False,
            pc_format="cloudini",
            pc_schema="auto",
            pc_encoding="lossy",
            pc_compression="zstd",
            resolution=0.01,
            draco_compression_level=7,
        ),
        transform_queue_size=1,
        send_queue_size=1,
        throttle_hz=0,
        max_message_size=None,
    )


def test_h264_keyframe_detection() -> None:
    assert is_video_keyframe(b"\x00\x00\x00\x01\x65\x88", "h264")
    assert not is_video_keyframe(b"\x00\x00\x01\x41\x9a", "h264")


def test_h265_keyframe_detection() -> None:
    assert is_video_keyframe(b"\x00\x00\x01\x26\x01", "h265")
    assert not is_video_keyframe(b"\x00\x00\x01\x02\x01", "h265")


def test_vp9_keyframe_detection() -> None:
    # VP9 uncompressed header, MSB-first: frame_marker=0b10, profile 0,
    # show_existing_frame=0, then frame_type (0=key). 0x80 = key, 0x84 = inter.
    assert is_video_keyframe(b"\x80", "vp9")
    assert not is_video_keyframe(b"\x84", "vp9")


def test_av1_keyframe_detection() -> None:
    # A sequence-header OBU (type 1) precedes every AV1 key frame; a lone
    # frame OBU (type 6) marks an inter frame. Header byte layout:
    # forbidden(1) type(4) extension(1) has_size(1) reserved(1).
    seq_header_obu = b"\x0a\x01\x00"  # type=1, has_size=1, size=1
    frame_obu = b"\x32\x01\x00"  # type=6, has_size=1, size=1
    assert is_video_keyframe(seq_header_obu, "av1")
    assert not is_video_keyframe(frame_obu, "av1")


def test_send_manager_latest_only_drops_stale_messages(monkeypatch) -> None:
    async def run() -> None:
        server = WebSocketBridgeServer()
        metrics = ProxyMetrics()
        manager = SendManager(server, metrics, queue_size=1)
        sent: list[bytes] = []
        release = asyncio.Event()
        websocket = object()

        async def fake_send(
            _websocket: object,
            _subscription_id: int,
            payload: bytes,
            *,
            timestamp_ns: int | None = None,
        ) -> None:
            del timestamp_ns
            sent.append(payload)
            if payload == b"slow":
                await release.wait()

        monkeypatch.setattr(server, "send_message_to_subscription", fake_send)

        manager.enqueue(websocket, 1, OutboundMessage(1, 1, b"slow"))
        await _wait_for(lambda: sent == [b"slow"])

        manager.enqueue(websocket, 1, OutboundMessage(1, 2, b"stale"))
        manager.enqueue(websocket, 1, OutboundMessage(1, 3, b"latest"))

        release.set()
        await _wait_for(lambda: sent == [b"slow", b"latest"])
        await manager.close()

        assert metrics.send_queue_drops == 1

    asyncio.run(run())


def test_send_manager_waits_for_keyframe_after_video_drop(monkeypatch) -> None:
    async def run() -> None:
        server = WebSocketBridgeServer()
        metrics = ProxyMetrics()
        manager = SendManager(server, metrics, queue_size=1)
        sent: list[bytes] = []
        release = asyncio.Event()
        websocket = object()

        async def fake_send(
            _websocket: object,
            _subscription_id: int,
            payload: bytes,
            *,
            timestamp_ns: int | None = None,
        ) -> None:
            del timestamp_ns
            sent.append(payload)
            if payload == b"first-key":
                await release.wait()

        monkeypatch.setattr(server, "send_message_to_subscription", fake_send)

        manager.enqueue(
            websocket,
            1,
            OutboundMessage(1, 1, b"first-key", is_compressed_video=True, is_keyframe=True),
        )
        await _wait_for(lambda: sent == [b"first-key"])

        manager.enqueue(
            websocket,
            1,
            OutboundMessage(1, 2, b"stale-delta", is_compressed_video=True, is_keyframe=False),
        )
        manager.enqueue(
            websocket,
            1,
            OutboundMessage(1, 3, b"latest-delta", is_compressed_video=True, is_keyframe=False),
        )

        release.set()
        await _wait_for(lambda: metrics.video_packets_waiting_for_keyframe == 1)
        assert sent == [b"first-key"]

        manager.enqueue(
            websocket,
            1,
            OutboundMessage(1, 4, b"next-key", is_compressed_video=True, is_keyframe=True),
        )
        await _wait_for(lambda: sent == [b"first-key", b"next-key"])
        await manager.close()

        assert metrics.send_queue_drops == 1

    asyncio.run(run())


def test_bridge_proxy_advertises_transformed_channel(monkeypatch) -> None:
    class DummyTransformer:
        output_schema_name = "std_msgs/msg/String"
        output_schema_text = "string data"
        output_schema_encoding = "ros2msg"
        output_message_encoding = "cdr"
        worker_count = 1

        def transform(self, decoded: object, timestamp_ns: int) -> TransformResult:
            del decoded, timestamp_ns
            return TransformResult({"data": "ok"})

        def close(self) -> None:
            return

    async def run() -> None:
        bridge = BridgeProxy(
            upstream_url="ws://upstream:8765",
            listen_host="127.0.0.1",
            listen_port=8766,
            config=_proxy_config(),
        )
        advertised: list[ServerChannel] = []

        async def fake_advertise(channel: ServerChannel, *, update_registry: bool = True) -> None:
            del update_registry
            advertised.append(channel)

        monkeypatch.setattr(bridge.downstream_server, "advertise_channel", fake_advertise)
        monkeypatch.setattr(bridge, "_create_transformer", lambda _channel: DummyTransformer())

        channel: ChannelInfo = {
            "id": 7,
            "topic": "/chatter",
            "encoding": "cdr",
            "schemaName": "std_msgs/msg/String",
            "schema": "string data",
            "schemaEncoding": "ros2msg",
        }
        await bridge.handle_upstream_advertise(channel)
        try:
            assert len(advertised) == 1
            assert advertised[0].id == 10000
            assert advertised[0].topic == "/chatter"
            assert advertised[0].schema_name == "std_msgs/msg/String"
            assert bridge.metrics.transformed_channel_count == 1
        finally:
            await bridge.stop()

    asyncio.run(run())


class _FakeUpstreamClient:
    def __init__(self) -> None:
        self.is_connected = False
        self.subscribed: list[tuple[int, int]] = []
        self.unsubscribed: list[int] = []

    async def subscribe_to_channel(self, subscription_id: int, channel_id: int) -> None:
        self.subscribed.append((subscription_id, channel_id))

    async def unsubscribe_from_channel(self, subscription_id: int) -> None:
        self.unsubscribed.append(subscription_id)


def test_bridge_proxy_snapshot_reports_state() -> None:
    async def run() -> None:
        bridge = BridgeProxy(
            upstream_url="ws://upstream:8765",
            listen_host="127.0.0.1",
            listen_port=8766,
            config=_proxy_config(),
        )
        try:
            snap = bridge.snapshot()
            assert snap.upstream_channels == 0
            assert snap.transformed_channels == 0
            assert snap.client_count == 0
            assert snap.upstream_connected is False
        finally:
            await bridge.stop()

    asyncio.run(run())


def test_proxy_dashboard_renders_without_error() -> None:
    async def run() -> None:
        bridge = BridgeProxy(
            upstream_url="ws://upstream:8765",
            listen_host="127.0.0.1",
            listen_port=8766,
            config=_proxy_config(),
        )
        try:
            bridge.metrics.upstream_messages_received = 5
            console = Console()
            panel = ProxyDashboard(bridge, console)._render()
            with console.capture() as capture:
                console.print(panel)
            assert "bridge proxy" in capture.get()
            assert "disconnected" in capture.get()
        finally:
            await bridge.stop()

    asyncio.run(run())


def test_bridge_proxy_defers_upstream_subscribe_until_connected() -> None:
    async def run() -> None:
        bridge = BridgeProxy(
            upstream_url="ws://upstream:8765",
            listen_host="127.0.0.1",
            listen_port=8766,
            config=_proxy_config(),
        )
        fake_upstream = _FakeUpstreamClient()
        bridge.upstream_client = fake_upstream
        websocket = object()

        await bridge.handle_client_subscribe(websocket, 11, 7)

        assert fake_upstream.subscribed == []
        assert bridge._upstream_subscriptions == {7: (1, 1)}

        fake_upstream.is_connected = True
        await bridge.handle_upstream_reconnected()
        await bridge.handle_client_unsubscribe(websocket, 11)

        assert fake_upstream.subscribed == [(1, 7)]
        assert fake_upstream.unsubscribed == [1]

    asyncio.run(run())


class _AppendLiveProcessor(MessageTransformProcessor):
    def matches(self, _channel, schema) -> bool:
        return schema is not None and schema.name == "std_msgs/msg/String"

    def transform(self, channel, schema, decoded):
        return [
            TransformOutput(
                topic=channel.topic,
                schema_name=schema.name,
                schema_encoding=schema.encoding,
                schema_data=schema.data,
                data={"data": f"{decoded.data}!"},
            )
        ]


def test_processor_rule_wraps_message_transform_processor() -> None:
    rule = ProcessorRule(processor_factory=_AppendLiveProcessor)

    transformer = rule.create_transformer(
        {
            "id": 3,
            "topic": "/chatter",
            "encoding": "cdr",
            "schemaName": "std_msgs/msg/String",
            "schema": "string data",
            "schemaEncoding": "ros2msg",
        }
    )

    assert transformer is not None
    result = transformer.transform(SimpleNamespace(data="ok"), 0)
    assert result is not None
    assert result.payload == {"data": "ok!"}
    assert transformer.output_schema_name == "std_msgs/msg/String"


def test_pointcloud_rule_drops_invalid_points_before_compression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "pymcap_cli.cmd._pointcloud_cleanup.os.cpu_count",
        lambda: 14,
    )
    config = replace(
        _proxy_config(),
        pointcloud=PointCloudConfig(
            enabled=True,
            pc_format="cloudini",
            pc_schema="auto",
            pc_encoding="lossy",
            pc_compression="zstd",
            resolution=0.01,
            draco_compression_level=7,
        ),
    )
    transformer = create_transformer(
        build_transform_rules(config),
        {
            "id": 9,
            "topic": "/lidar/points",
            "encoding": "cdr",
            "schemaName": "sensor_msgs/msg/PointCloud2",
            "schema": POINTCLOUD2,
            "schemaEncoding": "ros2msg",
        },
    )
    assert transformer is not None
    assert transformer.worker_count == 8

    dtype = np.dtype(
        {
            "names": ["x", "y", "z", "line"],
            "formats": ["<f4", "<f4", "<f4", "u1"],
            "itemsize": 16,
        }
    )
    points = np.zeros(4, dtype=dtype)
    points["x"][:2] = [1.0, 2.0]
    points["y"][:2] = [1.0, 2.0]
    points["z"][:2] = [1.0, 2.0]
    cloud = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=1, nanosec=0),
            frame_id="lidar",
        ),
        height=1,
        width=4,
        fields=[
            SimpleNamespace(name="x", offset=0, datatype=7, count=1),
            SimpleNamespace(name="y", offset=4, datatype=7, count=1),
            SimpleNamespace(name="z", offset=8, datatype=7, count=1),
            SimpleNamespace(name="line", offset=12, datatype=2, count=1),
        ],
        is_bigendian=False,
        point_step=16,
        row_step=64,
        data=points.tobytes(),
        is_dense=True,
    )

    result = transformer.transform(cloud, 0)
    transformer.close()

    assert result is not None
    assert result.payload["width"] == 2


def test_transform_worker_uses_transformer_worker_count(monkeypatch: pytest.MonkeyPatch) -> None:
    class ConcurrentTransformer:
        output_schema_name = "std_msgs/msg/String"
        output_schema_text = "string data"
        output_schema_encoding = "ros2msg"
        output_message_encoding = "cdr"
        worker_count = 2

        def __init__(self) -> None:
            self.release = threading.Event()
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def transform(self, decoded: object, timestamp_ns: int) -> TransformResult:
            del timestamp_ns
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            self.release.wait(timeout=2)
            with self.lock:
                self.active -= 1
            return TransformResult({"data": decoded})

        def close(self) -> None:
            return

    async def run() -> None:
        monkeypatch.setattr(
            TransformWorker,
            "_decoder_for",
            lambda _self, _channel: lambda payload: payload,
        )
        monkeypatch.setattr(
            TransformWorker,
            "_encoder_for",
            lambda _self, _transformer: lambda payload: payload["data"],
        )
        transformer = ConcurrentTransformer()
        emitted: list[OutboundMessage] = []
        worker = TransformWorker(
            channel={
                "id": 3,
                "topic": "/points",
                "encoding": "cdr",
                "schemaName": "std_msgs/msg/String",
                "schema": "string data",
                "schemaEncoding": "ros2msg",
            },
            downstream_id=10,
            transformer=transformer,
            queue_size=1,
            emit=emitted.append,
            metrics=ProxyMetrics(),
        )
        try:
            worker.enqueue(IncomingMessage(1, b"first"))
            await _wait_for(lambda: transformer.active == 1)
            worker.enqueue(IncomingMessage(2, b"second"))
            await _wait_for(lambda: transformer.max_active == 2)
        finally:
            transformer.release.set()
            await worker.close()

    asyncio.run(run())


def test_transform_worker_drops_stale_parallel_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OutOfOrderTransformer:
        output_schema_name = "std_msgs/msg/String"
        output_schema_text = "string data"
        output_schema_encoding = "ros2msg"
        output_message_encoding = "cdr"
        worker_count = 2

        def __init__(self) -> None:
            self.first_started = threading.Event()
            self.release_first = threading.Event()

        def transform(self, decoded: object, timestamp_ns: int) -> TransformResult:
            if timestamp_ns == 1:
                self.first_started.set()
                self.release_first.wait(timeout=2)
            return TransformResult({"data": decoded})

        def close(self) -> None:
            return

    async def run() -> None:
        monkeypatch.setattr(
            TransformWorker,
            "_decoder_for",
            lambda _self, _channel: lambda payload: payload,
        )
        monkeypatch.setattr(
            TransformWorker,
            "_encoder_for",
            lambda _self, _transformer: lambda payload: payload["data"],
        )
        transformer = OutOfOrderTransformer()
        metrics = ProxyMetrics()
        emitted: list[OutboundMessage] = []
        worker = TransformWorker(
            channel={
                "id": 3,
                "topic": "/points",
                "encoding": "cdr",
                "schemaName": "std_msgs/msg/String",
                "schema": "string data",
                "schemaEncoding": "ros2msg",
            },
            downstream_id=10,
            transformer=transformer,
            queue_size=1,
            emit=emitted.append,
            metrics=metrics,
        )
        try:
            worker.enqueue(IncomingMessage(1, b"first"))
            await _wait_for(transformer.first_started.is_set)
            worker.enqueue(IncomingMessage(2, b"second"))
            await _wait_for(lambda: [message.timestamp_ns for message in emitted] == [2])
            transformer.release_first.set()
            await _wait_for(lambda: metrics.transform_stale_drops == 1)
        finally:
            transformer.release_first.set()
            await worker.close()

        assert [message.timestamp_ns for message in emitted] == [2]

    asyncio.run(run())


class _FakeVideoEncoder:
    def __init__(self, codec_name: str) -> None:
        self.config = EncoderConfig(width=4, height=4, codec_name=codec_name)

    def encode(self, _frame: object) -> bytes:
        return b"\x00\x00\x00\x01\x65"

    def close(self) -> None:
        pass


class _FakeVideoBackend:
    def test_encoder(self, encoder_name: str) -> bool:
        return encoder_name != "missing"

    def resolve_encoder(self, codec: str) -> str:
        return f"{codec}_resolved"

    def decode_image(self, _msg: object, _schema_name: str) -> tuple[bytes, int, int]:
        return b"frame", 4, 4

    def create_encoder(
        self,
        _width: int,
        _height: int,
        codec_name: str,
        _quality: int,
        *,
        input_pix_fmt: str | None = None,
        scale: tuple[int, int] | None = None,
    ) -> _FakeVideoEncoder:
        del input_pix_fmt, scale
        return _FakeVideoEncoder(codec_name)

    def get_pix_fmt(self, _topic: str) -> str | None:
        return None


@pytest.mark.parametrize("backend", ["auto", "pyav", "ffmpeg-cli", "gstreamer"])
def test_video_transformer_uses_roscompress_backend_resolver(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    calls: list[tuple[str, str | None, str]] = []

    def fake_resolve_video_compression_backend(
        *,
        codec: str,
        encoder: str | None,
        backend: EncoderMode,
    ) -> ResolvedVideoCompressionBackend:
        calls.append((codec, encoder, backend.value))
        return ResolvedVideoCompressionBackend(_FakeVideoBackend(), "selected_encoder")

    monkeypatch.setattr(
        video_compress,
        "resolve_video_compression_backend",
        fake_resolve_video_compression_backend,
    )

    transformer = VideoTransformer(
        ImageConfig(
            image_format="video",
            codec="h264",
            quality=28,
            encoder=None,
            backend=backend,
            scale=None,
            jpeg_quality=90,
        ),
        "/cam",
    )

    assert calls == [("h264", None, backend)]
    assert transformer._session._encoder_name == "selected_encoder"
