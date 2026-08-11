"""Named Foxglove message paths shared by cat, plotting, and tabular export."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

from ros_parser.message_path import (
    NO_OUTPUT,
    MessagePath,
    MessagePathError,
    MessagePathEvaluator,
    MessagePathVariables,
    parse_message_path,
)

_COLUMN_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True, slots=True)
class NamedMessagePath:
    name: str
    source: str
    path: MessagePath


@dataclass(frozen=True, slots=True)
class CatQuery:
    """One parsed cat query and its output key."""

    source: str
    output_name: str
    path: MessagePath


class CatQueryRuntime:
    """Stateful evaluation of cat-style queries for one message stream."""

    __slots__ = ("_evaluators", "_queries")

    def __init__(self, queries: Mapping[str, tuple[CatQuery, ...]]) -> None:
        self._queries = dict(queries)
        self._evaluators = {
            topic: tuple(
                MessagePathEvaluator(query.path) if query.path.has_stream else None
                for query in topic_queries
            )
            for topic, topic_queries in self._queries.items()
        }

    def evaluate(
        self,
        topic: str,
        decoded_message: Any,
        timestamp_ns: int,
        variables: MessagePathVariables,
    ) -> Any:
        """Evaluate the queries for one message on ``topic``."""
        queries = self._queries[topic]
        evaluators = self._evaluators[topic]
        if len(queries) == 1:
            query = queries[0]
            evaluator = evaluators[0]
            result = (
                evaluator.observe(decoded_message, timestamp_ns, variables)
                if evaluator is not None
                else query.path.apply(decoded_message, variables)
            )
            return NO_OUTPUT if result is NO_OUTPUT or query_result_is_empty(result) else result

        projected: dict[str, Any] = {}
        for query, evaluator in zip(queries, evaluators, strict=True):
            try:
                result = (
                    evaluator.observe(decoded_message, timestamp_ns, variables)
                    if evaluator is not None
                    else query.path.apply(decoded_message, variables)
                )
            except MessagePathError as exc:
                raise MessagePathError(f"{query.source}: {exc}") from exc
            if result is not NO_OUTPUT and not query_result_is_empty(result):
                projected[query.output_name] = result
        return projected or NO_OUTPUT

    def reducers(self) -> Iterator[tuple[str, str, MessagePathEvaluator]]:
        """Yield each stream evaluator with its topic and display name."""
        for topic, evaluators in self._evaluators.items():
            queries = self._queries[topic]
            for query, evaluator in zip(queries, evaluators, strict=True):
                if evaluator is not None:
                    display_name = query.output_name if len(queries) > 1 else query.source
                    yield topic, display_name, evaluator


def parse_path_arg(arg: str) -> tuple[str, str]:
    """Parse ``Label=/path`` or a bare ``/path`` into its label and path."""
    if "=" in arg and not arg.startswith("/"):
        label, _, path_str = arg.partition("=")
        return label, path_str
    return arg, arg


def parse_cat_queries(queries: list[str] | None) -> dict[str, tuple[CatQuery, ...]]:
    """Parse repeatable cat queries and group them by selected topic."""
    parsed: dict[str, list[CatQuery]] = {}
    for argument in queries or ():
        label, path_source = parse_path_arg(argument)
        if label == path_source:
            path = parse_message_path(path_source)
            output_name = path_source.removeprefix(path.topic) or "."
        else:
            column = parse_named_columns([argument])[0]
            path_source = column.source
            path = column.path
            output_name = column.name
        topic_queries = parsed.setdefault(path.topic, [])
        if any(query.output_name == output_name for query in topic_queries):
            raise ValueError(
                f"Duplicate query output name {output_name!r} for topic {path.topic!r}; "
                "use LABEL=/topic.path to choose unique names"
            )
        topic_queries.append(CatQuery(source=path_source, output_name=output_name, path=path))
    return {topic: tuple(topic_queries) for topic, topic_queries in parsed.items()}


def query_result_is_empty(result: object) -> bool:
    """True for no result, but not for genuine falsy scalars such as zero or false."""
    if result is None:
        return True
    return isinstance(result, (list, tuple)) and len(result) == 0


def parse_named_columns(args: list[str] | None) -> tuple[NamedMessagePath, ...]:
    columns: list[NamedMessagePath] = []
    seen: set[tuple[str, str]] = set()
    for arg in args or ():
        name, path_str = parse_path_arg(arg)
        name = name.strip()
        path_str = path_str.strip()
        if name == path_str:
            raise ValueError(f"Column expression {arg!r} must use NAME=/topic.path syntax")
        if _COLUMN_NAME_RE.fullmatch(name) is None:
            raise ValueError(f"Invalid column name {name!r}; use letters, numbers, and underscores")
        if not path_str.startswith("/"):
            raise ValueError(f"Column {name!r} must use a topic-qualified message path")
        try:
            path = parse_message_path(path_str)
        except Exception as exc:
            raise ValueError(f"Invalid message path for column {name!r}: {exc}") from exc
        key = (path.topic, name)
        if key in seen:
            raise ValueError(f"Duplicate column name {name!r} for topic {path.topic!r}")
        seen.add(key)
        columns.append(NamedMessagePath(name=name, source=path_str, path=path))
    return tuple(columns)
