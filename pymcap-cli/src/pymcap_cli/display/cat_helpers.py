"""Shared schema-cache helpers for `cat` and `bridge cat`."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from ros_parser import parse_schema_to_definitions
from ros_parser.message_path import ValidationError

from pymcap_cli.display.message_render import build_enum_plan, resolve_msgdef_by_name

if TYPE_CHECKING:
    from ros_parser.message_path import MessagePath
    from ros_parser.models import MessageDefinition

    from pymcap_cli.display.message_render import EnumPlan

logger = logging.getLogger(__name__)


class SchemaLike(Protocol):
    id: int
    name: str
    data: bytes


@dataclass
class SchemaCache:
    parsed: dict[int, dict[str, MessageDefinition] | None] = field(default_factory=dict)
    plans: dict[int, EnumPlan | None] = field(default_factory=dict)

    def parsed_schema(self, schema: SchemaLike) -> dict[str, MessageDefinition] | None:
        if schema.id not in self.parsed:
            try:
                self.parsed[schema.id] = parse_schema_to_definitions(schema.name, schema.data)
            except Exception:
                logger.exception(f"Failed to parse schema '{schema.name}'")
                self.parsed[schema.id] = None
        return self.parsed[schema.id]

    def enum_plan(self, schema: SchemaLike) -> EnumPlan | None:
        if schema.id not in self.plans:
            parsed = self.parsed_schema(schema)
            self.plans[schema.id] = build_enum_plan(schema.name, parsed) if parsed else None
        return self.plans[schema.id]

    def validate_query(
        self, query_path: MessagePath, schema: SchemaLike, topic: str, *, query_repr: str
    ) -> bool:
        defs = self.parsed_schema(schema)
        if defs is None:
            return True
        try:
            root = resolve_msgdef_by_name(schema.name, defs)
            if root is None:
                logger.warning(f"Could not find message definition for schema '{schema.name}'")
            else:
                query_path.validate(root, defs)
        except ValidationError:
            logger.exception(f"Query validation error for topic '{topic}'")
            logger.exception(f"Query: {query_repr}  Schema: {schema.name}")
            return False
        return True
