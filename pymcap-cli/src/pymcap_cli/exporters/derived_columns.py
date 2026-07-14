"""Message-path column projection for structured exporters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ros_parser import MessageDefinition, Type, parse_schema_to_definitions
from ros_parser.message_path import MathModifier, MessagePathError, ValidationError

from pymcap_cli.core.named_message_path import NamedMessagePath, parse_named_columns
from pymcap_cli.types.to_plain import to_plain

if TYPE_CHECKING:
    from small_mcap import DecodedMessage, Schema, Summary

_RESERVED_COLUMNS = frozenset({"_log_time_ns", "_publish_time_ns"})
_HISTORY_MODIFIERS = frozenset({"delta", "derivative", "timedelta"})
ColumnScalar = bool | int | float | str | bytes
ColumnValue = ColumnScalar | list[ColumnScalar] | None


@dataclass(frozen=True, slots=True)
class ResolvedColumn:
    column: NamedMessagePath
    result_type: Type
    definitions: dict[str, MessageDefinition]


class ColumnSelection:
    def __init__(self, select: list[str] | None) -> None:
        self.columns = parse_named_columns(select)
        by_topic: dict[str, list[NamedMessagePath]] = {}
        for column in self.columns:
            by_topic.setdefault(column.path.topic, []).append(column)
        self._by_topic = {topic: tuple(values) for topic, values in by_topic.items()}

    @property
    def is_enabled(self) -> bool:
        return bool(self.columns)

    def includes_topic(self, topic: str) -> bool:
        return not self.is_enabled or topic in self._by_topic

    def values_for(self, topic: str, msg: DecodedMessage) -> dict[str, ColumnValue]:
        values: dict[str, ColumnValue] = {}
        for column in self._by_topic.get(topic, ()):
            try:
                value = column.path.apply(msg.decoded_message)
            except MessagePathError as exc:
                raise ValueError(
                    f"Cannot evaluate column {column.name!r} from {column.source!r}: {exc}"
                ) from exc
            values[column.name] = to_plain(value)
        return values

    def validate_input(self, summary: Summary | None) -> None:
        if not self.columns or summary is None:
            return
        channels_by_topic: dict[str, list[int]] = {}
        for channel in summary.channels.values():
            channels_by_topic.setdefault(channel.topic, []).append(channel.schema_id)
        for topic in self._by_topic:
            schema_ids = channels_by_topic.get(topic)
            if not schema_ids:
                raise ValueError(f"Selected topic {topic!r} was not found in the MCAP")
            for schema_id in set(schema_ids):
                schema = summary.schemas.get(schema_id)
                if schema is None:
                    raise ValueError(f"Selected topic {topic!r} has no schema for validation")
                self.resolve_for_schema(topic, schema)

    def resolve_for_schema(self, topic: str, schema: Schema) -> tuple[ResolvedColumn, ...]:
        if schema.encoding != "ros2msg":
            raise ValueError(
                f"Selected columns currently require a ROS2 message schema; "
                f"topic {topic!r} uses {schema.encoding!r}"
            )
        definitions = parse_schema_to_definitions(schema.name, schema.data)
        root = definitions.get(schema.name) or definitions.get(schema.name.replace("/msg/", "/"))
        if root is None:
            raise ValueError(f"Cannot resolve root schema {schema.name!r} for topic {topic!r}")

        resolved: list[ResolvedColumn] = []
        for column in self._by_topic.get(topic, ()):
            if column.name in _RESERVED_COLUMNS:
                raise ValueError(f"Column name {column.name!r} is reserved")
            for segment in column.path.segments:
                if isinstance(segment, MathModifier) and segment.operation in _HISTORY_MODIFIERS:
                    raise ValueError(
                        f"Column {column.name!r} uses history-dependent modifier "
                        f"@{segment.operation}, which export does not support"
                    )
            try:
                result_type, _result_definition = column.path.resolve_type(root, definitions)
            except ValidationError as exc:
                raise ValueError(
                    f"Invalid path for column {column.name!r} on {topic!r}: {exc}"
                ) from exc
            if not result_type.is_primitive:
                raise ValueError(
                    f"Column {column.name!r} resolves to complex type {result_type}; "
                    "select a primitive field or apply a reducing modifier"
                )
            resolved.append(
                ResolvedColumn(
                    column=column,
                    result_type=result_type,
                    definitions=definitions,
                )
            )
        return tuple(resolved)
