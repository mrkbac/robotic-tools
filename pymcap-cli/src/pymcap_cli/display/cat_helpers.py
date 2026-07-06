"""Shared schema-cache helpers for `cat` and `bridge cat`."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from ros_parser import parse_schema_to_definitions
from ros_parser.message_path import ArrayIndex, ArraySlice, FieldAccess, ValidationError

from pymcap_cli.display.message_render import EnumPlan, build_enum_plan, resolve_msgdef_by_name

if TYPE_CHECKING:
    from ros_parser.message_path import MessagePath
    from ros_parser.models import MessageDefinition

logger = logging.getLogger(__name__)


def query_result_is_empty(result: object) -> bool:
    """True when a query produced nothing to show — ``None`` or an empty sequence.

    Filter/slice queries (e.g. ``.transforms[:]{child_frame_id=="base_link"}``) return
    an empty list when nothing matched. Callers skip those so ``--limit`` counts real
    hits and empty frames don't clutter the stream. A falsy *scalar* (``0``, ``False``,
    ``""``) is a genuine value and is NOT treated as empty.
    """
    if result is None:
        return True
    return isinstance(result, (list, tuple)) and len(result) == 0


def plan_for_query(root_plan: EnumPlan | None, parsed_query: MessagePath | None) -> EnumPlan | None:
    """Return the render plan matching the sub-object a query extracts.

    The root plan describes the whole message. A field-path query renders a sub-value,
    so we walk the query's segments into the plan's ``nested_plans``:
    ``.field`` descends into that field's nested plan, array index/slice keep the
    element plan. Anything the plan can't follow (a scalar/enum leaf, a filter, a math
    modifier) yields None, so that sub-value simply renders undecorated.
    """
    if parsed_query is None:
        return root_plan
    plan = root_plan
    for segment in parsed_query.segments:
        if isinstance(segment, (ArrayIndex, ArraySlice)):
            continue
        if isinstance(segment, FieldAccess):
            if plan is None:
                return None
            plan = plan.nested_plans.get(segment.field_name)
            continue
        return None
    return plan


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
