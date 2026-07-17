"""Live recording-contract checks for Foxglove WebSocket bridges."""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, cast

from mcap_ros2_support_fast.decoder import DecoderFactory
from robo_ws_bridge import ConnectionGraph, WebSocketBridgeClient
from robo_ws_bridge.ws_types import ChannelInfo, ServerCapabilities
from ros_parser.message_path import MessagePathError
from small_mcap import Channel, JSONDecoderFactory, Schema

from pymcap_cli.check import (
    ERROR,
    OK,
    CheckReport,
    CheckResult,
    CheckSpec,
    CheckSpecError,
    EndpointRule,
    LiveNodeRule,
    ObservationValue,
    TopicRule,
    _build_results,
    _create_runtime,
    _TopicObservation,
    _TopicRuntime,
    load_check_spec,
)
from pymcap_cli.cmd._cli_options import (
    BridgeTarget,
    CheckSpecOption,
    ConnectTimeoutOption,
    DiscoverSecondsOption,
    SampleDurationOption,
)
from pymcap_cli.cmd.bridge._shared import (
    BridgeFetchError,
    ChannelSubscriptionManager,
    collect_graph_nodes,
    to_ws_url,
)
from pymcap_cli.cmd.check_cmd import print_check_report
from pymcap_cli.display.cat_helpers import SchemaCache
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)


class DecodedPayload(Protocol):
    """Marker protocol for dynamically decoded JSON and ROS messages."""


Decoder: TypeAlias = Callable[[bytes | memoryview], DecodedPayload]


@dataclass(slots=True)
class _LiveEvaluator:
    spec: CheckSpec
    start_ns: int
    end_ns: int
    observations: dict[int, dict[str, _TopicObservation]] = field(init=False)
    runtimes: dict[tuple[int, str], _TopicRuntime] = field(default_factory=dict)
    _predicates: list[Callable[[Channel, Schema | None], bool]] = field(init=False)
    _records: dict[int, tuple[Channel, Schema | None]] = field(default_factory=dict)
    _matched_indexes: dict[int, tuple[int, ...]] = field(default_factory=dict)
    _decoders: dict[int, Decoder | None] = field(default_factory=dict)
    _validated_paths: set[tuple[int, int]] = field(default_factory=set)
    _schema_cache: SchemaCache = field(default_factory=SchemaCache)

    def __post_init__(self) -> None:
        self.observations = {index: {} for index in range(len(self.spec.topics))}
        self._predicates = [rule.selector.create_channel_predicate() for rule in self.spec.topics]

    def add_channel(self, info: ChannelInfo) -> None:
        channel, schema = _channel_records(info)
        self._records[channel.id] = (channel, schema)
        matched = tuple(
            index for index, predicate in enumerate(self._predicates) if predicate(channel, schema)
        )
        self._matched_indexes[channel.id] = matched
        for index in matched:
            observation = self.observations[index].setdefault(channel.topic, _TopicObservation())
            if channel.id not in observation.channels:
                observation.add_channel(channel, schema, 0)

    def wants_channel(self, info: ChannelInfo) -> bool:
        self.add_channel(info)
        return any(
            self.spec.topics[index].expected and self.spec.topics[index].needs_messages
            for index in self._matched_indexes[info["id"]]
        )

    def observe(
        self,
        info: ChannelInfo,
        payload: bytes,
        timestamp_ns: int,
        publish_time_ns: int | None = None,
    ) -> None:
        if info["id"] not in self._records:
            self.add_channel(info)
        channel, schema = self._records[info["id"]]
        matched = self._matched_indexes[info["id"]]
        decoded: DecodedPayload | None = None
        decode_attempted = False
        decode_error: str | None = None

        for index in matched:
            rule = self.spec.topics[index]
            self.observations[index][channel.topic].message_count += 1
            if not rule.expected or not rule.needs_messages:
                continue

            runtime = self.runtimes.get((index, channel.topic))
            if runtime is None:
                runtime = _create_runtime(rule)
                self.runtimes[index, channel.topic] = runtime
            runtime.observe_timestamp(timestamp_ns, self.start_ns, self.end_ns)
            if not rule.values:
                continue

            self._validate_paths(index, runtime, channel, schema)
            active = [tracker for tracker in runtime.values if tracker.evaluation_error is None]
            if not active:
                continue
            if not decode_attempted:
                decode_attempted = True
                try:
                    decoded = self._decode(info, schema, payload)
                except Exception as exc:  # noqa: BLE001 - decoder plugins have no common exception
                    decode_error = f"payload decode failed: {exc}"
            if decode_error is not None:
                for tracker in active:
                    tracker.evaluation_error = decode_error
                continue
            for tracker in active:
                try:
                    tracker.evaluate(
                        decoded,
                        timestamp_ns,
                        {
                            "log_time_ns": timestamp_ns,
                            "publish_time_ns": (
                                timestamp_ns if publish_time_ns is None else publish_time_ns
                            ),
                            "recording_start_ns": self.start_ns,
                            "recording_end_ns": self.end_ns,
                        },
                    )
                except (MessagePathError, TypeError, ValueError) as exc:
                    tracker.evaluation_error = f"MessagePath evaluation failed: {exc}"

    def report(self, url: str, graph: ConnectionGraph | None) -> CheckReport:
        results = _build_results(
            self.spec,
            self.observations,
            self.runtimes,
            self.start_ns,
            self.end_ns,
        )
        results.extend(_graph_results(self.spec, self.observations, graph))
        return CheckReport(path=url, results=results)

    def _validate_paths(
        self,
        index: int,
        runtime: _TopicRuntime,
        channel: Channel,
        schema: Schema | None,
    ) -> None:
        if schema is None or schema.encoding not in ("ros1msg", "ros2msg"):
            return
        key = (index, schema.id)
        if key in self._validated_paths:
            return
        self._validated_paths.add(key)
        for value_index, tracker in enumerate(runtime.values):
            if not self._schema_cache.validate_query(
                tracker.rule.path,
                schema,
                channel.topic,
                query_repr=tracker.rule.path_source,
            ):
                tracker.evaluation_error = (
                    f"value[{value_index}] path is incompatible with schema {schema.name!r}"
                )

    def _decode(
        self,
        info: ChannelInfo,
        schema: Schema | None,
        payload: bytes,
    ) -> DecodedPayload:
        channel_id = info["id"]
        if channel_id not in self._decoders:
            decoder: Decoder | None = None
            for factory in (JSONDecoderFactory(), DecoderFactory()):
                candidate = factory.decoder_for(info["encoding"], schema)
                if candidate is not None:
                    decoder = cast("Decoder", candidate)
                    break
            self._decoders[channel_id] = decoder
        decoder = self._decoders[channel_id]
        if decoder is None:
            raise ValueError(
                f"no decoder for encoding {info['encoding']!r} and "
                f"schema {info.get('schemaName', '')!r}"
            )
        return decoder(payload)


def _channel_records(info: ChannelInfo) -> tuple[Channel, Schema | None]:
    has_schema = bool(info.get("schemaName") or info.get("schemaEncoding") or info.get("schema"))
    schema_id = info["id"] if has_schema else 0
    channel = Channel(
        id=info["id"],
        schema_id=schema_id,
        topic=info["topic"],
        message_encoding=info["encoding"],
        metadata={},
    )
    if not has_schema:
        return channel, None
    return channel, Schema(
        id=schema_id,
        name=info.get("schemaName", ""),
        encoding=info.get("schemaEncoding", ""),
        data=info.get("schema", "").encode(),
    )


def _endpoint_result(
    rule: TopicRule,
    topic: str,
    kind: str,
    endpoint: EndpointRule,
    node_ids: tuple[str, ...],
) -> CheckResult:
    count = len(node_ids)
    is_valid = True
    if endpoint.minimum is not None and count < endpoint.minimum:
        is_valid = False
    if endpoint.maximum is not None and count > endpoint.maximum:
        is_valid = False
    matching_nodes = (
        tuple(node_id for node_id in node_ids if endpoint.node_selector.matches_topic(node_id))
        if endpoint.node_selector is not None
        else ()
    )
    if endpoint.node_selector is not None and not matching_nodes:
        is_valid = False
    level = OK if is_valid else rule.violation_level
    summary = (
        f"{kind} satisfy live requirements" if is_valid else f"{kind} violate live requirements"
    )
    values: dict[str, ObservationValue] = {"count": count}
    if node_ids:
        values["nodes"] = ", ".join(sorted(node_ids))
    if endpoint.node is not None:
        values["required_node"] = endpoint.node
    return CheckResult(level, f"{rule.name}:{topic}/{kind}", summary, values)


def _node_result(rule: LiveNodeRule, node_ids: set[str]) -> CheckResult:
    matches = sorted(node_id for node_id in node_ids if rule.selector.matches_topic(node_id))
    exists = bool(matches)
    is_valid = exists if rule.expected else not exists
    if rule.expected:
        summary = "expected live node is present" if exists else "expected live node is missing"
    else:
        summary = "forbidden live node is present" if exists else "forbidden live node is absent"
    values = {"nodes": ", ".join(matches)} if matches else {}
    return CheckResult(
        OK if is_valid else rule.violation_level, f"{rule.name}/node", summary, values
    )


def _graph_results(
    spec: CheckSpec,
    observations: dict[int, dict[str, _TopicObservation]],
    graph: ConnectionGraph | None,
) -> list[CheckResult]:
    if not spec.has_live_rules:
        return []
    if graph is None:
        return [
            CheckResult(
                ERROR,
                "live/connection_graph",
                "bridge does not provide the connectionGraph capability",
            )
        ]

    publishers = {item["name"]: tuple(item["publisherIds"]) for item in graph.published_topics}
    subscribers = {item["name"]: tuple(item["subscriberIds"]) for item in graph.subscribed_topics}
    graph_topics = set(publishers) | set(subscribers)
    results: list[CheckResult] = []
    for index, rule in enumerate(spec.topics):
        if rule.live is None:
            continue
        topics = set(observations[index]) | {
            topic for topic in graph_topics if rule.selector.matches_topic(topic)
        }
        if not topics:
            topics.add(rule.topic)
        for topic in sorted(topics):
            if rule.live.publishers is not None:
                results.append(
                    _endpoint_result(
                        rule,
                        topic,
                        "publishers",
                        rule.live.publishers,
                        publishers.get(topic, ()),
                    )
                )
            if rule.live.subscribers is not None:
                results.append(
                    _endpoint_result(
                        rule,
                        topic,
                        "subscribers",
                        rule.live.subscribers,
                        subscribers.get(topic, ()),
                    )
                )

    node_ids = set(collect_graph_nodes(graph))
    results.extend(_node_result(rule, node_ids) for rule in spec.live_nodes)
    return results


async def _collect_async(
    url: str,
    spec: CheckSpec,
    *,
    duration: float,
    connect_timeout: float,
    discover_seconds: float,
) -> CheckReport:
    has_dynamic_rules = any(rule.expected and rule.needs_messages for rule in spec.topics)
    if has_dynamic_rules and duration <= 0:
        raise ValueError(
            "duration must be greater than zero for frequency, timeout, or value checks"
        )
    if duration < 0:
        raise ValueError("duration must be non-negative")
    client = WebSocketBridgeClient(url, min_retry_delay=0.2, max_retry_delay=1.0)
    server_info_event = asyncio.Event()
    client.on_server_info(lambda *_: server_info_event.set())
    await client.connect()
    try:
        try:
            await asyncio.wait_for(server_info_event.wait(), timeout=connect_timeout)
        except asyncio.TimeoutError as exc:
            raise BridgeFetchError(
                f"Timed out after {connect_timeout:.1f}s waiting for serverInfo from {url}"
            ) from exc
        server_info = client.server_info
        if server_info is None:
            raise BridgeFetchError(f"No serverInfo received from {url}")
        graph_supported = ServerCapabilities.CONNECTION_GRAPH.value in server_info["capabilities"]
        if graph_supported and spec.has_live_rules:
            await client.subscribe_connection_graph()
        await asyncio.sleep(discover_seconds)
        # Sampling subscriptions must not satisfy the contract's subscriber requirements.
        graph = client.connection_graph if graph_supported else None

        start_ns = time.monotonic_ns()
        end_ns = start_ns + int(duration * 1_000_000_000)
        evaluator = _LiveEvaluator(spec, start_ns, end_ns)
        for channel in client.channels.values():
            evaluator.add_channel(channel)

        if has_dynamic_rules:
            subscriber = ChannelSubscriptionManager(client, evaluator.wants_channel)
            subscriber.install()
            client.on_message(
                lambda channel, bridge_timestamp, payload: evaluator.observe(
                    channel, payload, time.monotonic_ns(), bridge_timestamp
                )
            )
            await subscriber.subscribe_existing()
            await asyncio.sleep(duration)

        return evaluator.report(url, graph)
    finally:
        await client.disconnect()


def collect_bridge_check(
    url: str,
    spec: CheckSpec,
    *,
    duration: float = 5.0,
    connect_timeout: float = 5.0,
    discover_seconds: float = 1.5,
) -> CheckReport:
    """Collect a live bridge snapshot and optional message sample, then evaluate it."""
    return asyncio.run(
        _collect_async(
            url,
            spec,
            duration=duration,
            connect_timeout=connect_timeout,
            discover_seconds=discover_seconds,
        )
    )


def check(
    target: BridgeTarget,
    *,
    spec: CheckSpecOption,
    duration: SampleDurationOption = 5.0,
    connect_timeout: ConnectTimeoutOption = 5.0,
    discover_seconds: DiscoverSecondsOption = 1.5,
) -> int:
    """Check a live bridge against the same contract used for recorded MCAP files."""
    url = to_ws_url(target)
    try:
        report = collect_bridge_check(
            url,
            load_check_spec(spec),
            duration=duration,
            connect_timeout=connect_timeout,
            discover_seconds=discover_seconds,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 1
    except CheckSpecError as exc:
        ERR.print(f"[red]Invalid check spec:[/red] {exc}")
        return 1
    except (BridgeFetchError, OSError, ValueError) as exc:
        ERR.print(f"[red]Bridge check failed:[/red] {exc}")
        return 1
    except Exception as exc:
        ERR.print(f"[red]Bridge check failed:[/red] {exc}")
        logger.exception("Bridge check failed")
        return 1

    print_check_report(report)
    return 1 if report.error_count else 0
