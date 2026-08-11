"""``pymcap-cli index stats`` — reduce MessagePaths once per indexed file."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypeAlias, cast

from cyclopts import Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory
from mcap_ros2_support_fast.writer import Schema
from rich.table import Table
from ros_parser.message_path import (
    NO_OUTPUT,
    LarkError,
    MessagePathError,
    MessagePathEvaluator,
    MessagePathVariables,
)
from small_mcap import Channel, JSONDecoderFactory, McapError, get_summary, read_message_decoded

from pymcap_cli.cmd._cli_options import (
    IndexDbOption,
    IndexFolderOption,
    IndexTableJsonFormatOption,
    MessagePathVariablesOption,
    QueryOption,
)
from pymcap_cli.cmd._message_path_options import create_message_path_variables
from pymcap_cli.cmd.index._helpers import (
    _path_prefix_predicate,
    _print_db_needs_migration,
    _resolve_db,
    _stdout_line,
    console,
)
from pymcap_cli.core.named_message_path import (
    CatQuery,
    parse_cat_queries,
)
from pymcap_cli.display.cat_helpers import SchemaCache
from pymcap_cli.display.display_utils import _format_parts_with_colors
from pymcap_cli.display.message_render import BytesMode, message_to_dict
from pymcap_cli.index.db import IndexDbNeedsMigrationError, open_db

JsonValue: TypeAlias = (
    str | int | float | bool | Sequence["JsonValue"] | Mapping[str, "JsonValue"] | None
)


@dataclass(frozen=True, slots=True)
class _FileStats:
    path: str
    values: dict[str, JsonValue]
    error: str | None = None


def _query_columns(
    parsed_queries: dict[str, tuple[CatQuery, ...]],
) -> list[CatQuery]:
    columns: list[CatQuery] = []
    seen: set[str] = set()
    for topic_queries in parsed_queries.values():
        for query in topic_queries:
            key = query.output_name
            if key in seen:
                raise ValueError(
                    f"Duplicate result name {key!r}; use LABEL=/topic.path to make it unique"
                )
            seen.add(key)
            columns.append(query)
    return columns


def _matching_paths(
    db_path: Path,
    topics: tuple[str, ...],
    folder: Path | None,
    limit: int,
) -> list[str]:
    placeholders = ", ".join("?" for _ in topics)
    sql = (
        "SELECT cf.abs_path "  # noqa: S608
        "FROM current_file cf "
        "JOIN content_channel cc ON cc.content_id = cf.content_id "
        "JOIN channel_signature sig ON sig.id = cc.channel_signature_id "
        "JOIN topic t ON t.id = sig.topic_id "
        f"WHERE t.name IN ({placeholders}) "
    )
    params: list[str | int] = list(topics)
    if folder is not None:
        predicate, path_params = _path_prefix_predicate(folder)
        sql += f"AND ({predicate.replace('abs_path', 'cf.abs_path')}) "
        params.extend(path_params)
    sql += "GROUP BY cf.abs_path HAVING COUNT(DISTINCT t.name) = ? ORDER BY cf.abs_path "
    params.append(len(topics))
    if limit > 0:
        sql += "LIMIT ?"
        params.append(limit)
    with open_db(db_path, read_only=True) as conn:
        return [row[0] for row in conn.execute(sql, params)]


def _reduce_file(
    path: str,
    parsed_queries: dict[str, tuple[CatQuery, ...]],
    variables: MessagePathVariables,
    validated_queries: set[tuple[str, str, bytes, str]],
) -> _FileStats:
    try:
        reducers = {
            topic: tuple((query, MessagePathEvaluator(query.path)) for query in topic_queries)
            for topic, topic_queries in parsed_queries.items()
        }
        schema_cache = SchemaCache()
        validated_topics: set[str] = set()
        with Path(path).open("rb") as stream:
            summary = get_summary(stream)
            if summary is not None:
                available_topics = {channel.topic for channel in summary.channels.values()}
                missing_topics = parsed_queries.keys() - available_topics
                if missing_topics:
                    return _FileStats(
                        path, {}, f"topics no longer present: {', '.join(missing_topics)}"
                    )

            def should_include(channel: Channel, _schema: Schema | None) -> bool:
                return channel.topic in parsed_queries

            for message in read_message_decoded(
                stream,
                decoder_factories=[JSONDecoderFactory(), DecoderFactory()],
                should_include=should_include,
            ):
                topic = message.channel.topic
                topic_queries = parsed_queries[topic]
                if topic not in validated_topics:
                    validated_topics.add(topic)
                    if message.schema is not None and message.schema.encoding in {
                        "ros1msg",
                        "ros2msg",
                    }:
                        for query in topic_queries:
                            validation_key = (
                                message.schema.name,
                                message.schema.encoding,
                                message.schema.data,
                                query.source,
                            )
                            if validation_key in validated_queries:
                                continue
                            if not schema_cache.validate_query(
                                query.path,
                                message.schema,
                                topic,
                                query_repr=query.source,
                            ):
                                return _FileStats(path, {}, f"query validation failed for {topic}")
                            validated_queries.add(validation_key)
                for _, evaluator in reducers[topic]:
                    evaluator.observe(
                        message.decoded_message,
                        message.message.log_time,
                        variables,
                    )

        values: dict[str, JsonValue] = {}
        for topic_reducers in reducers.values():
            for query, evaluator in topic_reducers:
                reduced = evaluator.finalize(variables)
                values[query.output_name] = (
                    None
                    if reduced is NO_OUTPUT
                    else cast("JsonValue", message_to_dict(reduced, bytes_mode=BytesMode.SMART))
                )
        return _FileStats(path, values)
    except (OSError, McapError, MessagePathError, ValueError) as exc:
        return _FileStats(path, {}, f"{type(exc).__name__}: {exc}")


def stats_cmd(
    folder: IndexFolderOption = None,
    *,
    query: QueryOption = None,
    var: MessagePathVariablesOption = None,
    limit: Annotated[
        int,
        Parameter(help="Maximum files to read; use 0 for every matching file."),
    ] = 0,
    format: IndexTableJsonFormatOption = "table",
    db: IndexDbOption = None,
) -> int:
    """Run stream reducers over every indexed file containing their topics.

    Each query must contain a terminal stream reducer such as ``@@max``,
    ``@@mean``, or ``@@count``. Files must contain every queried topic.
    """
    db_path = _resolve_db(db)
    if not db_path.exists():
        console.print(f"[red]Error:[/] no index DB at {db_path}")
        return 1
    if not query:
        console.print("[red]Error:[/] at least one --query is required")
        return 1
    if limit < 0:
        console.print("[red]Error:[/] --limit must be non-negative")
        return 1

    try:
        parsed_queries = parse_cat_queries(query)
        columns = _query_columns(parsed_queries)
        variables = create_message_path_variables(var)
    except (LarkError, MessagePathError, ValueError) as exc:
        console.print(f"[red]Error:[/] {exc}")
        return 1

    for parsed in columns:
        if not parsed.path.has_stream_reducer:
            console.print(
                f"[red]Error:[/] query {parsed.source!r} must end in a stream reducer such as @@max"
            )
            return 1

    try:
        paths = _matching_paths(db_path, tuple(parsed_queries), folder, limit)
    except IndexDbNeedsMigrationError as exc:
        _print_db_needs_migration(exc)
        return 1

    validated_queries: set[tuple[str, str, bytes, str]] = set()
    results = [_reduce_file(path, parsed_queries, variables, validated_queries) for path in paths]
    rows: list[dict[str, JsonValue]] = []
    for result in results:
        row: dict[str, JsonValue] = {"path": result.path, "stats": result.values}
        if result.error is not None:
            row["error"] = result.error
        rows.append(row)

    if format == "json":
        _stdout_line(json.dumps(rows))
    else:
        table = Table(title=f"Per-file statistics ({len(rows):,})")
        table.add_column("Path", overflow="fold")
        for query_column in columns:
            table.add_column(query_column.output_name, justify="right")
        table.add_column("Error", style="red", overflow="fold")
        for result in results:
            table.add_row(
                _format_parts_with_colors(result.path),
                *(json.dumps(result.values.get(column.output_name)) for column in columns),
                result.error or "",
            )
        console.print(table)
    return 1 if any(result.error is not None for result in results) else 0
