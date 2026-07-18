"""High-performance recording contract checks for MCAP files."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, TypeAlias, cast

import yaml
from mcap_ros2_support_fast.decoder import DecoderFactory
from ros_parser.message_path import (
    NO_OUTPUT,
    Filter,
    MessagePath,
    MessagePathError,
    MessagePathEvaluator,
    parse_message_path,
)
from small_mcap import JSONDecoderFactory, get_summary, read_message_decoded

from pymcap_cli.core.input_handler import open_input
from pymcap_cli.core.message_filter import MessageFilterOptions
from pymcap_cli.display.cat_helpers import SchemaCache, query_result_is_empty
from pymcap_cli.types.check_spec_types import (
    CheckSpecInput,
    ComparableValue,
    EndpointRuleSpec,
    LiveNodeRuleSpec,
    LiveRootSpec,
    LiveTopicRuleSpec,
    SchemaRuleSpec,
    Severity,
    TopicRuleSpec,
    ValueRuleMappingSpec,
)
from pymcap_cli.types.duration import parse_duration_ns
from pymcap_cli.utils import ProgressTrackingIO, file_progress

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path
    from typing import IO

    from small_mcap import Channel, DecodedMessage, Schema, Summary

YamlScalar: TypeAlias = None | bool | int | float | str
YamlValue: TypeAlias = YamlScalar | list["YamlValue"] | dict[str, "YamlValue"]
ObservationValue: TypeAlias = int | float | str

OK = 0
WARN = 1
ERROR = 2
_FAILURE_SAMPLE_LIMIT = 3


class CheckSpecError(ValueError):
    """A recording check specification is invalid."""


@dataclass(frozen=True, slots=True)
class SchemaRule:
    name: str | None = None
    encoding: str | None = None


@dataclass(frozen=True, slots=True)
class FrequencyRule:
    minimum_hz: float | None
    maximum_hz: float | None
    tolerance: float
    window_ns: int

    @property
    def effective_minimum_hz(self) -> float | None:
        if self.minimum_hz is None:
            return None
        return self.minimum_hz * (1 - self.tolerance)

    @property
    def effective_maximum_hz(self) -> float | None:
        if self.maximum_hz is None:
            return None
        return self.maximum_hz * (1 + self.tolerance)


@dataclass(frozen=True, slots=True)
class ValueRule:
    path_source: str
    path: MessagePath
    minimum: float | None = None
    maximum: float | None = None
    equals: ComparableValue | None = None
    one_of: tuple[ComparableValue, ...] = ()


@dataclass(frozen=True, slots=True)
class EndpointRule:
    minimum: int | None = None
    maximum: int | None = None
    node: str | None = None
    node_selector: MessageFilterOptions | None = None


@dataclass(frozen=True, slots=True)
class LiveTopicRule:
    publishers: EndpointRule | None = None
    subscribers: EndpointRule | None = None


@dataclass(frozen=True, slots=True)
class LiveNodeRule:
    name: str
    node: str
    selector: MessageFilterOptions
    expected: bool = True
    severity: Severity = "error"

    @property
    def violation_level(self) -> int:
        return WARN if self.severity == "warn" else ERROR


@dataclass(frozen=True, slots=True)
class TopicRule:
    name: str
    topic: str
    selector: MessageFilterOptions
    expected: bool = True
    severity: Severity = "error"
    schema: SchemaRule | None = None
    message_encoding: str | None = None
    frequency: FrequencyRule | None = None
    timeout_ns: int | None = None
    values: tuple[ValueRule, ...] = ()
    live: LiveTopicRule | None = None

    @property
    def violation_level(self) -> int:
        return WARN if self.severity == "warn" else ERROR

    @property
    def needs_messages(self) -> bool:
        return self.frequency is not None or self.timeout_ns is not None or bool(self.values)


@dataclass(frozen=True, slots=True)
class CheckSpec:
    topics: tuple[TopicRule, ...]
    live_nodes: tuple[LiveNodeRule, ...] = ()
    version: int = 1

    @property
    def has_live_rules(self) -> bool:
        return bool(self.live_nodes) or any(rule.live is not None for rule in self.topics)


@dataclass(frozen=True, slots=True)
class CheckResult:
    level: int
    name: str
    summary: str
    values: dict[str, ObservationValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CheckReport:
    path: str
    results: list[CheckResult]

    @property
    def ok_count(self) -> int:
        return sum(result.level == OK for result in self.results)

    @property
    def warning_count(self) -> int:
        return sum(result.level == WARN for result in self.results)

    @property
    def error_count(self) -> int:
        return sum(result.level >= ERROR for result in self.results)


def parse_check_spec(text: str, *, source: str = "<spec>") -> CheckSpec:
    """Parse and strictly validate a version 1 recording check specification."""
    try:
        loaded = cast("YamlValue", yaml.safe_load(text))
    except yaml.YAMLError as exc:
        raise CheckSpecError(f"{source}: invalid YAML: {exc}") from exc

    root = _mapping(loaded, source, set(CheckSpecInput.__annotations__))
    version = _integer(root.get("version"), f"{source}.version")
    if version != 1:
        raise CheckSpecError(f"{source}: version must be 1, got {version}")

    topics_value = root.get("topics")
    if not isinstance(topics_value, dict):
        raise CheckSpecError(f"{source}: topics must be a mapping")
    topics = cast("dict[str, YamlValue]", topics_value)
    rules = tuple(
        _parse_topic_rule(name, value, f"{source}.topics.{name}") for name, value in topics.items()
    )
    live_nodes = _parse_live_root(root["live"], f"{source}.live") if "live" in root else ()
    return CheckSpec(topics=rules, live_nodes=live_nodes)


def load_check_spec(path: Path) -> CheckSpec:
    """Load a recording check specification from a YAML file."""
    try:
        text = path.read_text()
    except OSError as exc:
        raise CheckSpecError(f"{path}: could not read spec: {exc}") from exc
    return parse_check_spec(text, source=str(path))


def _mapping(value: YamlValue, path: str, allowed: set[str]) -> dict[str, YamlValue]:
    if not isinstance(value, dict):
        raise CheckSpecError(f"{path} must be a mapping")
    mapping = cast("dict[str, YamlValue]", value)
    non_string_keys = [key for key in mapping if not isinstance(key, str)]
    if non_string_keys:
        raise CheckSpecError(f"{path} keys must be strings")
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise CheckSpecError(f"{path}: unknown key {unknown[0]!r}")
    return mapping


def _string(value: YamlValue, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CheckSpecError(f"{path} must be a non-empty string")
    return value


def _boolean(value: YamlValue, path: str) -> bool:
    if not isinstance(value, bool):
        raise CheckSpecError(f"{path} must be true or false")
    return value


def _integer(value: YamlValue, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CheckSpecError(f"{path} must be an integer")
    return value


def _number(value: YamlValue, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckSpecError(f"{path} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise CheckSpecError(f"{path} must be finite")
    return number


def _optional_number(mapping: dict[str, YamlValue], key: str, path: str) -> float | None:
    value = mapping.get(key)
    return None if value is None else _number(value, f"{path}.{key}")


def _duration(value: YamlValue, path: str) -> int:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise CheckSpecError(f"{path} must be a duration such as '500ms' or '2s'")
    try:
        duration_ns = parse_duration_ns(str(value))
    except ValueError as exc:
        raise CheckSpecError(f"{path}: {exc}") from exc
    if duration_ns <= 0:
        raise CheckSpecError(f"{path} must be greater than zero")
    return duration_ns


def _parse_topic_rule(name: str, value: YamlValue, path: str) -> TopicRule:
    if not isinstance(name, str) or not name:
        raise CheckSpecError(f"{path}: topic rule names must be non-empty strings")
    mapping = _mapping(value, path, set(TopicRuleSpec.__annotations__))
    topic = _string(mapping.get("topic"), f"{path}.topic")
    try:
        selector = MessageFilterOptions.from_args(topic=[topic])
    except ValueError as exc:
        raise CheckSpecError(f"{path}: invalid topic regex: {exc}") from exc

    expected_value = mapping.get("expected", True)
    expected = _boolean(expected_value, f"{path}.expected")
    severity_value = mapping.get("severity", "error")
    severity = _string(severity_value, f"{path}.severity")
    if severity not in ("warn", "error"):
        raise CheckSpecError(f"{path}.severity must be 'warn' or 'error'")

    schema = _parse_schema(mapping["schema"], f"{path}.schema") if "schema" in mapping else None
    message_encoding = (
        _string(mapping["message_encoding"], f"{path}.message_encoding")
        if "message_encoding" in mapping
        else None
    )
    frequency = (
        _parse_frequency(mapping["frequency"], f"{path}.frequency")
        if "frequency" in mapping
        else None
    )
    timeout_ns = _duration(mapping["timeout"], f"{path}.timeout") if "timeout" in mapping else None
    values = _parse_values(mapping["values"], f"{path}.values") if "values" in mapping else ()
    live = _parse_live_topic(mapping["live"], f"{path}.live") if "live" in mapping else None

    if not expected and any(
        item is not None and item != ()
        for item in (schema, message_encoding, frequency, timeout_ns, values, live)
    ):
        raise CheckSpecError(
            f"{path}: forbidden topic rule (expected: false) cannot define other checks"
        )

    return TopicRule(
        name=name,
        topic=topic,
        selector=selector,
        expected=expected,
        severity=cast("Severity", severity),
        schema=schema,
        message_encoding=message_encoding,
        frequency=frequency,
        timeout_ns=timeout_ns,
        values=values,
        live=live,
    )


def _parse_live_topic(value: YamlValue, path: str) -> LiveTopicRule:
    mapping = _mapping(value, path, set(LiveTopicRuleSpec.__annotations__))
    publishers = (
        _parse_endpoint_rule(mapping["publishers"], f"{path}.publishers")
        if "publishers" in mapping
        else None
    )
    subscribers = (
        _parse_endpoint_rule(mapping["subscribers"], f"{path}.subscribers")
        if "subscribers" in mapping
        else None
    )
    if publishers is None and subscribers is None:
        raise CheckSpecError(f"{path} must define publishers or subscribers")
    return LiveTopicRule(publishers=publishers, subscribers=subscribers)


def _parse_endpoint_rule(value: YamlValue, path: str) -> EndpointRule:
    mapping = _mapping(value, path, set(EndpointRuleSpec.__annotations__))
    minimum = _optional_non_negative_integer(mapping, "min", path)
    maximum = _optional_non_negative_integer(mapping, "max", path)
    if minimum is not None and maximum is not None and minimum > maximum:
        raise CheckSpecError(f"{path}.min must not exceed max")
    node = _string(mapping["node"], f"{path}.node") if "node" in mapping else None
    if minimum is None and maximum is None and node is None:
        raise CheckSpecError(f"{path} must define min, max, or node")
    selector = _selector(node, path) if node is not None else None
    return EndpointRule(minimum, maximum, node, selector)


def _optional_non_negative_integer(
    mapping: dict[str, YamlValue], key: str, path: str
) -> int | None:
    if key not in mapping:
        return None
    value = _integer(mapping[key], f"{path}.{key}")
    if value < 0:
        raise CheckSpecError(f"{path}.{key} must be non-negative")
    return value


def _parse_live_root(value: YamlValue, path: str) -> tuple[LiveNodeRule, ...]:
    mapping = _mapping(value, path, set(LiveRootSpec.__annotations__))
    nodes_value = mapping.get("nodes")
    if not isinstance(nodes_value, dict) or not nodes_value:
        raise CheckSpecError(f"{path}.nodes must be a non-empty mapping")
    nodes = cast("dict[str, YamlValue]", nodes_value)
    return tuple(
        _parse_live_node(name, node, f"{path}.nodes.{name}") for name, node in nodes.items()
    )


def _parse_live_node(name: str, value: YamlValue, path: str) -> LiveNodeRule:
    if not isinstance(name, str) or not name:
        raise CheckSpecError(f"{path}: live node rule names must be non-empty strings")
    mapping = _mapping(value, path, set(LiveNodeRuleSpec.__annotations__))
    node = _string(mapping.get("node"), f"{path}.node")
    expected = _boolean(mapping.get("expected", True), f"{path}.expected")
    severity = _string(mapping.get("severity", "error"), f"{path}.severity")
    if severity not in ("warn", "error"):
        raise CheckSpecError(f"{path}.severity must be 'warn' or 'error'")
    return LiveNodeRule(
        name=name,
        node=node,
        selector=_selector(node, path),
        expected=expected,
        severity=cast("Severity", severity),
    )


def _selector(pattern: str, path: str) -> MessageFilterOptions:
    try:
        return MessageFilterOptions.from_args(topic=[pattern])
    except ValueError as exc:
        raise CheckSpecError(f"{path}: invalid regex: {exc}") from exc


def _parse_schema(value: YamlValue, path: str) -> SchemaRule:
    mapping = _mapping(value, path, set(SchemaRuleSpec.__annotations__))
    name = _string(mapping["name"], f"{path}.name") if "name" in mapping else None
    encoding = _string(mapping["encoding"], f"{path}.encoding") if "encoding" in mapping else None
    if name is None and encoding is None:
        raise CheckSpecError(f"{path} must define name or encoding")
    return SchemaRule(name=name, encoding=encoding)


def _parse_frequency(value: YamlValue, path: str) -> FrequencyRule:
    mapping = _mapping(value, path, {"min", "max", "tolerance", "window"})
    minimum = _optional_number(mapping, "min", path)
    maximum = _optional_number(mapping, "max", path)
    if minimum is None and maximum is None:
        raise CheckSpecError(f"{path} must define min or max")
    if minimum is not None and minimum < 0:
        raise CheckSpecError(f"{path}.min must be non-negative")
    if maximum is not None and maximum < 0:
        raise CheckSpecError(f"{path}.max must be non-negative")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise CheckSpecError(f"{path}.min must not exceed max")
    tolerance = _optional_number(mapping, "tolerance", path) or 0.0
    if tolerance < 0 or tolerance >= 1:
        raise CheckSpecError(f"{path}.tolerance must be at least 0 and less than 1")
    if "window" not in mapping:
        raise CheckSpecError(f"{path}.window is required")
    window_ns = _duration(mapping["window"], f"{path}.window")
    return FrequencyRule(minimum, maximum, tolerance, window_ns)


def _parse_values(value: YamlValue, path: str) -> tuple[ValueRule, ...]:
    if not isinstance(value, list) or not value:
        raise CheckSpecError(f"{path} must be a non-empty list")
    return tuple(_parse_value_rule(item, f"{path}[{index}]") for index, item in enumerate(value))


def _parse_value_rule(value: YamlValue, path: str) -> ValueRule:
    if isinstance(value, str):
        mapping: dict[str, YamlValue] = {}
        path_source = value
    else:
        mapping = _mapping(value, path, set(ValueRuleMappingSpec.__annotations__))
        path_source = _string(mapping.get("path"), f"{path}.path")
    if not path_source.startswith((".", "{")):
        raise CheckSpecError(f"{path}.path must be relative and start with '.' or '{{'")
    try:
        parsed_path = parse_message_path(f"/__check__{path_source}")
        if parsed_path.has_stream:
            MessagePathEvaluator(parsed_path)
    except Exception as exc:
        raise CheckSpecError(f"{path}.path is not a valid MessagePath: {exc}") from exc

    minimum = _optional_number(mapping, "min", path)
    maximum = _optional_number(mapping, "max", path)
    if minimum is not None and maximum is not None and minimum > maximum:
        raise CheckSpecError(f"{path}.min must not exceed max")

    equals = _comparable(mapping["equals"], f"{path}.equals") if "equals" in mapping else None
    one_of: tuple[ComparableValue, ...] = ()
    if "one_of" in mapping:
        one_of_value = mapping["one_of"]
        if not isinstance(one_of_value, list) or not one_of_value:
            raise CheckSpecError(f"{path}.one_of must be a non-empty list")
        one_of = tuple(
            _comparable(item, f"{path}.one_of[{index}]") for index, item in enumerate(one_of_value)
        )

    comparator_families = sum(
        (
            minimum is not None or maximum is not None,
            "equals" in mapping,
            bool(one_of),
        )
    )
    is_predicate = bool(parsed_path.segments) and isinstance(parsed_path.segments[-1], Filter)
    if comparator_families == 0 and not is_predicate:
        raise CheckSpecError(f"{path} must end in a filter or define min/max, equals, or one_of")
    if comparator_families > 1:
        raise CheckSpecError(f"{path} cannot combine numeric bounds with equals or one_of")
    return ValueRule(path_source, parsed_path, minimum, maximum, equals, one_of)


def _comparable(value: YamlValue, path: str) -> ComparableValue:
    if not isinstance(value, (bool, int, float, str)):
        raise CheckSpecError(f"{path} must be a boolean, number, or string")
    if isinstance(value, float) and not math.isfinite(value):
        raise CheckSpecError(f"{path} must be finite")
    return value


@dataclass(slots=True)
class TopicObservation:
    message_count: int = 0
    channels: dict[int, tuple[Channel, Schema | None]] = field(default_factory=dict)

    def add_channel(self, channel: Channel, schema: Schema | None, count: int) -> None:
        self.channels[channel.id] = (channel, schema)
        self.message_count += count


@dataclass(slots=True)
class _RateTracker:
    rule: FrequencyRule
    timestamps: deque[int] = field(default_factory=deque)
    minimum_count: int | None = None
    maximum_count: int | None = None
    minimum_start_ns: int = 0
    maximum_start_ns: int = 0
    first_window_recorded: bool = False
    last_timestamp_ns: int | None = None
    pending_records: list[tuple[int, int]] = field(default_factory=list)

    def observe(self, timestamp_ns: int, recording_start_ns: int) -> None:
        window_ns = self.rule.window_ns
        first_end_ns = recording_start_ns + window_ns

        if self.last_timestamp_ns is not None and timestamp_ns > self.last_timestamp_ns:
            self._flush_pending()

        if not self.first_window_recorded and timestamp_ns >= first_end_ns:
            self._discard_before(recording_start_ns)
            self._record(len(self.timestamps), recording_start_ns)
            self.first_window_recorded = True

        if timestamp_ns >= first_end_ns:
            window_start_ns = timestamp_ns - window_ns
            self._discard_before(window_start_ns)
            self._record(len(self.timestamps), window_start_ns)
            while self.timestamps and self.timestamps[0] == window_start_ns:
                self.timestamps.popleft()

        self.timestamps.append(timestamp_ns)
        if timestamp_ns >= first_end_ns:
            # The window ending at this message only counts if the recording
            # extends past it; a later message or finish() decides.
            self.pending_records.append((len(self.timestamps), timestamp_ns - window_ns + 1))
        self.last_timestamp_ns = timestamp_ns

    def finish(self, recording_start_ns: int, recording_end_ns: int) -> bool:
        if recording_end_ns - recording_start_ns < self.rule.window_ns:
            return False
        if self.last_timestamp_ns is not None and self.last_timestamp_ns < recording_end_ns:
            self._flush_pending()
        self.pending_records.clear()
        first_end_ns = recording_start_ns + self.rule.window_ns
        if not self.first_window_recorded:
            count = sum(timestamp < first_end_ns for timestamp in self.timestamps)
            self._record(count, recording_start_ns)
            self.first_window_recorded = True
        if self.last_timestamp_ns != recording_end_ns:
            window_start_ns = recording_end_ns - self.rule.window_ns
            self._discard_before(window_start_ns)
            count = sum(timestamp < recording_end_ns for timestamp in self.timestamps)
            self._record(count, window_start_ns)
        return True

    def _flush_pending(self) -> None:
        for count, start_ns in self.pending_records:
            self._record(count, start_ns)
        self.pending_records.clear()

    def _discard_before(self, timestamp_ns: int) -> None:
        while self.timestamps and self.timestamps[0] < timestamp_ns:
            self.timestamps.popleft()

    def _record(self, count: int, start_ns: int) -> None:
        if self.minimum_count is None or count < self.minimum_count:
            self.minimum_count = count
            self.minimum_start_ns = start_ns
        if self.maximum_count is None or count > self.maximum_count:
            self.maximum_count = count
            self.maximum_start_ns = start_ns


@dataclass(slots=True)
class _ValueTracker:
    rule: ValueRule
    evaluator: MessagePathEvaluator = field(init=False)
    observed_count: int = 0
    observed_minimum: float | int | None = None
    observed_maximum: float | int | None = None
    failure_count: int = 0
    failure_samples: list[str] = field(default_factory=list)
    evaluation_error: str | None = None
    is_finalized: bool = False

    def __post_init__(self) -> None:
        self.evaluator = MessagePathEvaluator(self.rule.path)

    def evaluate(
        self,
        message: object,
        timestamp_ns: int,
        variables: dict[str, int],
    ) -> None:
        result = self.evaluator.observe(message, timestamp_ns, variables)
        if result is not NO_OUTPUT:
            self.observe(result, timestamp_ns)

    def finalize(self, timestamp_ns: int, variables: dict[str, int]) -> None:
        if self.is_finalized:
            return
        self.is_finalized = True
        result = self.evaluator.finalize(variables)
        if result is not NO_OUTPUT:
            self.observe(result, timestamp_ns)
        elif self.rule.path.has_stream and self.observed_count == 0 and self.failure_count == 0:
            self._fail(timestamp_ns, "stream produced no value")

    def observe(self, result: object, timestamp_ns: int) -> None:
        if query_result_is_empty(result):
            self._fail(timestamp_ns, "empty result")
            return
        values = tuple(_flatten_values(result))
        if not values:
            self._fail(timestamp_ns, "empty result")
            return
        for value in values:
            self.observed_count += 1
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                self.observed_minimum = (
                    value if self.observed_minimum is None else min(self.observed_minimum, value)
                )
                self.observed_maximum = (
                    value if self.observed_maximum is None else max(self.observed_maximum, value)
                )
            if not self._matches(value):
                self._fail(timestamp_ns, repr(value))

    def _matches(self, value: object) -> bool:
        if self.rule.minimum is not None or self.rule.maximum is not None:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                return False
            if self.rule.minimum is not None and value < self.rule.minimum:
                return False
            return self.rule.maximum is None or value <= self.rule.maximum
        if self.rule.equals is not None:
            return value == self.rule.equals
        if self.rule.one_of:
            return value in self.rule.one_of
        return True

    def _fail(self, timestamp_ns: int, value: str) -> None:
        self.failure_count += 1
        if len(self.failure_samples) < _FAILURE_SAMPLE_LIMIT:
            self.failure_samples.append(f"{timestamp_ns}: {value}")


def _flatten_values(value: object) -> Iterable[object]:
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_values(item)
        return
    yield value


@dataclass(slots=True)
class _TopicRuntime:
    rate: _RateTracker | None
    values: list[_ValueTracker]
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    maximum_internal_gap_ns: int = 0

    def observe_timestamp(self, timestamp_ns: int, recording_start_ns: int) -> None:
        if self.last_timestamp_ns is not None:
            self.maximum_internal_gap_ns = max(
                self.maximum_internal_gap_ns,
                timestamp_ns - self.last_timestamp_ns,
            )
        else:
            self.first_timestamp_ns = timestamp_ns
        self.last_timestamp_ns = timestamp_ns
        if self.rate is not None:
            self.rate.observe(timestamp_ns, recording_start_ns)


def _decoded_payload(message: DecodedMessage) -> object:
    """Access one lazy decoded payload; kept separate for regression instrumentation."""
    return message.decoded_message


def check_mcap(file: str, spec: CheckSpec, *, num_workers: int = 4) -> CheckReport:
    """Evaluate one MCAP file against a compiled recording check specification."""
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    with open_input(file) as (stream, file_size):
        summary = get_summary(stream)
        observations, has_complete_summary = _summary_observations(summary, spec)
        scan_rule_indexes = {
            index for index, rule in enumerate(spec.topics) if rule.expected and rule.needs_messages
        }
        needs_scan = bool(scan_rule_indexes) or not has_complete_summary
        runtimes: dict[tuple[int, str], _TopicRuntime] = {}
        recording_start_ns, recording_end_ns = _summary_time_bounds(summary)

        if needs_scan:
            scanned: IO[bytes] = stream
            if file_size:
                progress = file_progress("[bold blue]Checking MCAP...")
                progress.start()
                try:
                    task = progress.add_task("Scanning", total=file_size)
                    scanned = ProgressTrackingIO(stream, task, progress, stream.tell())
                    recording_start_ns, recording_end_ns = _scan_messages(
                        scanned,
                        spec,
                        observations,
                        runtimes,
                        scan_rule_indexes,
                        has_complete_summary,
                        recording_start_ns,
                        recording_end_ns,
                        num_workers,
                    )
                finally:
                    progress.stop()
            else:
                recording_start_ns, recording_end_ns = _scan_messages(
                    scanned,
                    spec,
                    observations,
                    runtimes,
                    scan_rule_indexes,
                    has_complete_summary,
                    recording_start_ns,
                    recording_end_ns,
                    num_workers,
                )

    results = build_results(
        spec,
        observations,
        runtimes,
        recording_start_ns,
        recording_end_ns,
    )
    return CheckReport(path=file, results=results)


def _summary_observations(
    summary: Summary | None,
    spec: CheckSpec,
) -> tuple[dict[int, dict[str, TopicObservation]], bool]:
    observations = {index: {} for index in range(len(spec.topics))}
    if summary is None or summary.statistics is None:
        return observations, False
    counts = summary.statistics.channel_message_counts
    if not counts and summary.statistics.message_count:
        return observations, False

    predicates = [rule.selector.create_channel_predicate() for rule in spec.topics]
    for channel in summary.channels.values():
        count = counts.get(channel.id, 0)
        if count <= 0:
            continue
        schema = summary.schemas.get(channel.schema_id)
        for index, predicate in enumerate(predicates):
            if predicate(channel, schema):
                observation = observations[index].setdefault(channel.topic, TopicObservation())
                observation.add_channel(channel, schema, count)
    return observations, True


def _summary_time_bounds(summary: Summary | None) -> tuple[int | None, int | None]:
    statistics = summary.statistics if summary is not None else None
    if statistics is None or statistics.message_count == 0:
        return None, None
    return statistics.message_start_time, statistics.message_end_time


def _scan_messages(
    stream: IO[bytes],
    spec: CheckSpec,
    observations: dict[int, dict[str, TopicObservation]],
    runtimes: dict[tuple[int, str], _TopicRuntime],
    scan_rule_indexes: set[int],
    has_complete_summary: bool,
    recording_start_ns: int | None,
    recording_end_ns: int | None,
    num_workers: int,
) -> tuple[int | None, int | None]:
    predicates = [rule.selector.create_channel_predicate() for rule in spec.topics]
    if has_complete_summary:
        active_predicates = [predicates[index] for index in scan_rule_indexes]

        def should_include(channel: Channel, schema: Schema | None) -> bool:
            return any(predicate(channel, schema) for predicate in active_predicates)

    else:

        def should_include(_channel: Channel, _schema: Schema | None) -> bool:
            return True

    matched_by_channel: dict[int, tuple[int, ...]] = {}
    known_channels: set[tuple[int, int]] = set()
    evaluator = MessageRuleEvaluator(spec, runtimes)
    # Without summary statistics the recording end is unknown until the scan
    # completes; rules referencing $recording_end_ns then fail explicitly
    # instead of comparing against a bogus per-message value.
    known_end_ns = recording_end_ns

    for message in read_message_decoded(
        stream,
        should_include=should_include,
        decoder_factories=[JSONDecoderFactory(), DecoderFactory()],
        num_workers=num_workers,
    ):
        timestamp_ns = message.message.log_time
        if recording_start_ns is None:
            recording_start_ns = timestamp_ns
        if known_end_ns is None and (recording_end_ns is None or timestamp_ns > recording_end_ns):
            recording_end_ns = timestamp_ns

        channel = message.channel
        schema = message.schema
        matched_indexes = matched_by_channel.get(channel.id)
        if matched_indexes is None:
            matched_indexes = tuple(
                index for index, predicate in enumerate(predicates) if predicate(channel, schema)
            )
            matched_by_channel[channel.id] = matched_indexes

        variables: dict[str, int] | None = None
        for index in matched_indexes:
            rule = spec.topics[index]
            if not has_complete_summary:
                observation = observations[index].setdefault(channel.topic, TopicObservation())
                channel_key = (index, channel.id)
                if channel_key not in known_channels:
                    observation.add_channel(channel, schema, 0)
                    known_channels.add(channel_key)
                observation.message_count += 1
            if index not in scan_rule_indexes or not rule.expected:
                continue

            if variables is None:
                variables = {
                    "log_time_ns": message.message.log_time,
                    "publish_time_ns": message.message.publish_time,
                    "recording_start_ns": recording_start_ns,
                }
                if known_end_ns is not None:
                    variables["recording_end_ns"] = known_end_ns
            evaluator.observe(
                index,
                channel,
                schema,
                timestamp_ns,
                recording_start_ns,
                variables,
                partial(_decoded_payload, message),
            )

    return recording_start_ns, recording_end_ns


def _create_runtime(rule: TopicRule) -> _TopicRuntime:
    return _TopicRuntime(
        rate=_RateTracker(rule.frequency) if rule.frequency is not None else None,
        values=[_ValueTracker(value_rule) for value_rule in rule.values],
    )


@dataclass(slots=True)
class MessageRuleEvaluator:
    """Per-message rule evaluation shared by the MCAP scan and the live bridge check."""

    spec: CheckSpec
    runtimes: dict[tuple[int, str], _TopicRuntime] = field(default_factory=dict)
    _validated_paths: set[tuple[int, int]] = field(default_factory=set)
    _schema_cache: SchemaCache = field(default_factory=SchemaCache)

    def observe(
        self,
        index: int,
        channel: Channel,
        schema: Schema | None,
        timestamp_ns: int,
        recording_start_ns: int,
        variables: dict[str, int],
        decoded_supplier: Callable[[], object],
    ) -> None:
        """Feed one matched message into the rule's frequency and value trackers."""
        rule = self.spec.topics[index]
        runtime = self.runtimes.get((index, channel.topic))
        if runtime is None:
            runtime = _create_runtime(rule)
            self.runtimes[index, channel.topic] = runtime
        runtime.observe_timestamp(timestamp_ns, recording_start_ns)

        if not rule.values:
            return
        self._validate_paths(index, runtime, channel, schema)
        active_values = [tracker for tracker in runtime.values if tracker.evaluation_error is None]
        if not active_values:
            return
        try:
            decoded = decoded_supplier()
        except Exception as exc:  # noqa: BLE001 - decoder plugins have no common exception
            for tracker in active_values:
                tracker.evaluation_error = f"payload decode failed: {exc}"
            return
        for tracker in active_values:
            try:
                tracker.evaluate(decoded, timestamp_ns, variables)
            except (MessagePathError, TypeError, ValueError) as exc:
                tracker.evaluation_error = f"MessagePath evaluation failed: {exc}"

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


def build_results(
    spec: CheckSpec,
    observations: dict[int, dict[str, TopicObservation]],
    runtimes: dict[tuple[int, str], _TopicRuntime],
    recording_start_ns: int | None,
    recording_end_ns: int | None,
) -> list[CheckResult]:
    final_timestamp_ns = recording_end_ns or 0
    final_variables: dict[str, int] = {}
    if recording_start_ns is not None:
        final_variables["recording_start_ns"] = recording_start_ns
    if recording_end_ns is not None:
        final_variables["recording_end_ns"] = recording_end_ns
    for runtime in runtimes.values():
        for tracker in runtime.values:
            if tracker.evaluation_error is None:
                try:
                    tracker.finalize(final_timestamp_ns, final_variables)
                except (MessagePathError, TypeError, ValueError) as exc:
                    tracker.evaluation_error = f"MessagePath finalization failed: {exc}"

    results: list[CheckResult] = []
    for index, rule in enumerate(spec.topics):
        matched = observations[index]
        exists = bool(matched)
        if rule.expected:
            level = OK if exists else rule.violation_level
            summary = "expected topic is present" if exists else "expected topic is missing"
        else:
            level = rule.violation_level if exists else OK
            summary = "forbidden topic is present" if exists else "forbidden topic is absent"
        values: dict[str, ObservationValue] = {}
        if matched:
            values["topics"] = ", ".join(sorted(matched))
            values["message_count"] = sum(item.message_count for item in matched.values())
        results.append(CheckResult(level, f"{rule.name}/expected", summary, values))

        if not rule.expected or not matched:
            continue
        for topic, observation in sorted(matched.items()):
            if rule.schema is not None or rule.message_encoding is not None:
                results.append(_schema_result(rule, topic, observation))
            runtime = runtimes.get((index, topic))
            if rule.frequency is not None:
                results.append(
                    _frequency_result(
                        rule,
                        topic,
                        runtime,
                        recording_start_ns,
                        recording_end_ns,
                    )
                )
            if rule.timeout_ns is not None:
                results.append(
                    _timeout_result(
                        rule,
                        topic,
                        runtime,
                        recording_start_ns,
                        recording_end_ns,
                    )
                )
            results.extend(
                _value_result(rule, topic, runtime, value_index)
                for value_index in range(len(rule.values))
            )
    return results


def _schema_result(
    rule: TopicRule,
    topic: str,
    observation: TopicObservation,
) -> CheckResult:
    mismatches: list[str] = []
    for channel, schema in observation.channels.values():
        if rule.message_encoding is not None and channel.message_encoding != rule.message_encoding:
            mismatches.append(f"channel {channel.id} message encoding {channel.message_encoding!r}")
        if rule.schema is None:
            continue
        if schema is None:
            mismatches.append(f"channel {channel.id} has no schema")
            continue
        if rule.schema.name is not None and schema.name != rule.schema.name:
            mismatches.append(f"channel {channel.id} schema name {schema.name!r}")
        if rule.schema.encoding is not None and schema.encoding != rule.schema.encoding:
            mismatches.append(f"channel {channel.id} schema encoding {schema.encoding!r}")
    if mismatches:
        return CheckResult(
            rule.violation_level,
            f"{rule.name}:{topic}/schema",
            "schema or message encoding does not match",
            {"observed": "; ".join(mismatches)},
        )
    return CheckResult(OK, f"{rule.name}:{topic}/schema", "schema and encoding match")


def _frequency_result(
    rule: TopicRule,
    topic: str,
    runtime: _TopicRuntime | None,
    recording_start_ns: int | None,
    recording_end_ns: int | None,
) -> CheckResult:
    name = f"{rule.name}:{topic}/frequency"
    if (
        runtime is None
        or runtime.rate is None
        or recording_start_ns is None
        or recording_end_ns is None
        or not runtime.rate.finish(recording_start_ns, recording_end_ns)
        or runtime.rate.minimum_count is None
        or runtime.rate.maximum_count is None
    ):
        return CheckResult(
            rule.violation_level,
            name,
            "recording is too short to evaluate frequency",
        )

    frequency = cast("FrequencyRule", rule.frequency)
    window_seconds = frequency.window_ns / 1_000_000_000
    minimum_hz = runtime.rate.minimum_count / window_seconds
    maximum_hz = runtime.rate.maximum_count / window_seconds
    effective_minimum = frequency.effective_minimum_hz
    effective_maximum = frequency.effective_maximum_hz
    is_low = effective_minimum is not None and minimum_hz < effective_minimum
    is_high = effective_maximum is not None and maximum_hz > effective_maximum
    level = rule.violation_level if is_low or is_high else OK
    summary = "frequency is within bounds" if level == OK else "frequency is outside bounds"
    values: dict[str, ObservationValue] = {
        "minimum_hz": minimum_hz,
        "maximum_hz": maximum_hz,
        "window_ns": frequency.window_ns,
    }
    if effective_minimum is not None:
        values["required_minimum_hz"] = effective_minimum
        values["minimum_window_start_ns"] = runtime.rate.minimum_start_ns
    if effective_maximum is not None:
        values["required_maximum_hz"] = effective_maximum
        values["maximum_window_start_ns"] = runtime.rate.maximum_start_ns
    return CheckResult(level, name, summary, values)


def _timeout_result(
    rule: TopicRule,
    topic: str,
    runtime: _TopicRuntime | None,
    recording_start_ns: int | None,
    recording_end_ns: int | None,
) -> CheckResult:
    name = f"{rule.name}:{topic}/timeout"
    if (
        runtime is None
        or runtime.first_timestamp_ns is None
        or runtime.last_timestamp_ns is None
        or recording_start_ns is None
        or recording_end_ns is None
    ):
        return CheckResult(rule.violation_level, name, "timeout could not be evaluated")
    maximum_gap_ns = max(
        runtime.first_timestamp_ns - recording_start_ns,
        runtime.maximum_internal_gap_ns,
        recording_end_ns - runtime.last_timestamp_ns,
    )
    timeout_ns = cast("int", rule.timeout_ns)
    level = rule.violation_level if maximum_gap_ns > timeout_ns else OK
    summary = "message timeout is within limit" if level == OK else "message timeout exceeded"
    return CheckResult(
        level,
        name,
        summary,
        {"maximum_gap_ns": maximum_gap_ns, "timeout_ns": timeout_ns},
    )


def _value_result(
    rule: TopicRule,
    topic: str,
    runtime: _TopicRuntime | None,
    value_index: int,
) -> CheckResult:
    name = f"{rule.name}:{topic}/value[{value_index}]"
    if runtime is None:
        return CheckResult(rule.violation_level, name, "value check could not be evaluated")
    tracker = runtime.values[value_index]
    if tracker.evaluation_error is not None:
        return CheckResult(rule.violation_level, name, tracker.evaluation_error)
    level = rule.violation_level if tracker.failure_count else OK
    summary = (
        f"{tracker.rule.path_source} values are within bounds"
        if level == OK
        else f"{tracker.rule.path_source} has {tracker.failure_count} failing values"
    )
    values: dict[str, ObservationValue] = {"observed_count": tracker.observed_count}
    if tracker.observed_minimum is not None:
        values["observed_minimum"] = tracker.observed_minimum
    if tracker.observed_maximum is not None:
        values["observed_maximum"] = tracker.observed_maximum
    if tracker.failure_samples:
        values["failure_samples"] = "; ".join(tracker.failure_samples)
    return CheckResult(level, name, summary, values)
