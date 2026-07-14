"""`pymcap-cli bridge cat` — stream decoded live bridge messages to stdout."""

import asyncio
import json
import logging
import re
import sys
from collections.abc import Callable
from typing import Annotated, Any

from cyclopts import Group as CycloptsGroup
from cyclopts import Parameter
from mcap_ros2_support_fast.decoder import DecoderFactory
from rich.panel import Panel
from rich.text import Text
from robo_ws_bridge import WebSocketBridgeClient
from robo_ws_bridge.ws_types import ChannelInfo
from ros_parser.message_path import (
    MessagePath,
    MessagePathError,
    MessagePathVariables,
    parse_message_path,
)
from small_mcap import JSONDecoderFactory, Schema

from pymcap_cli.cmd._message_path_options import (
    MessagePathVariablesOption,
    create_message_path_variables,
)
from pymcap_cli.cmd.bridge._shared import (
    CONNECTION_GROUP,
    BridgeTarget,
    ChannelSubscriptionManager,
    channel_to_schema,
    console,
    to_ws_url,
)
from pymcap_cli.display.cat_helpers import SchemaCache, plan_for_query, query_result_is_empty
from pymcap_cli.display.message_render import (
    SMART_BYTES_INLINE_LIMIT,
    TTY_BYTES_TRUNCATE,
    BytesMode,
    changed_leaf_paths,
    message_matches_grep,
    message_to_dict,
    render_message_flat,
    render_message_tree,
)
from pymcap_cli.log_setup import ERR

logger = logging.getLogger(__name__)

CAT_FILTER_GROUP = CycloptsGroup("Filtering")
CAT_OUTPUT_GROUP = CycloptsGroup("Output")


async def _cat_async(
    *,
    url: str,
    topics: list[str],
    exclude_topics: list[str],
    query: str | None,
    grep: str | None,
    grep_ignore_case: bool,
    limit: int | None,
    duration: float | None,
    bytes_mode: BytesMode,
    connect_timeout: float,
    flat: bool = False,
    changed: bool = False,
    variables: MessagePathVariables | None = None,
) -> int:
    path_variables = dict(variables or {})
    parsed_query: MessagePath | None = None
    if query:
        try:
            parsed_query = parse_message_path(query)
        except Exception:
            logger.exception("Invalid --query syntax")
            return 1

    grep_pattern: re.Pattern[str] | None = None
    if grep:
        try:
            grep_pattern = re.compile(grep, re.IGNORECASE if grep_ignore_case else 0)
        except re.error:
            logger.exception("Invalid --grep regex")
            return 1

    try:
        topic_patterns = [re.compile(p) for p in topics]
        exclude_topic_patterns = [re.compile(p) for p in exclude_topics]
    except re.error:
        logger.exception("Invalid topic regex")
        return 1

    is_tty = sys.stdout.isatty()
    factories = [JSONDecoderFactory(), DecoderFactory()]

    decoders: dict[int, Callable[[bytes | memoryview], Any] | None] = {}
    schemas_by_channel: dict[int, Schema] = {}
    schema_cache = SchemaCache()
    previous_by_topic: dict[str, Any] = {}
    validated_topics: set[str] = set()
    warned_channels: set[int] = set()
    total = 0
    return_code = 0
    done = asyncio.Event()

    def _topic_matches(topic: str) -> bool:
        if parsed_query is not None:
            return topic == parsed_query.topic and not any(
                p.search(topic) for p in exclude_topic_patterns
            )
        if topic_patterns and not any(p.search(topic) for p in topic_patterns):
            return False
        return not any(p.search(topic) for p in exclude_topic_patterns)

    def _decoder_for(channel: ChannelInfo) -> Callable[[bytes | memoryview], Any] | None:
        cid = channel["id"]
        if cid in decoders:
            return decoders[cid]
        schema = channel_to_schema(channel)
        message_encoding = channel["encoding"]
        decoder: Callable[[bytes | memoryview], Any] | None = None
        for factory in factories:
            try:
                decoder = factory.decoder_for(message_encoding, schema)
            except Exception:
                # `mcap_ros2_support_fast.DecoderFactory` parses ROS2 schemas eagerly
                # and raises on malformed ros2msg. Treat as "no decoder" so a single
                # bad schema doesn't take down the whole cat session.
                logger.exception(
                    f"Decoder construction failed for {channel['topic']} "
                    f"(schema={schema.name!r}, encoding={message_encoding!r})"
                )
                decoder = None
            if decoder is not None:
                break
        decoders[cid] = decoder
        if decoder is not None:
            schemas_by_channel[cid] = schema
        return decoder

    def _emit_jsonl(channel: ChannelInfo, log_time_ns: int, data: Any) -> None:
        entry: dict[str, Any] = {
            "topic": channel["topic"],
            "log_time": log_time_ns,
        }
        schema_name = channel.get("schemaName")
        if schema_name:
            entry["schema"] = schema_name
        entry["message"] = message_to_dict(data, bytes_mode=bytes_mode)
        sys.stdout.write(json.dumps(entry, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    def _emit_tty(channel: ChannelInfo, log_time_ns: int, data: Any) -> None:
        schema_name = channel.get("schemaName") or "unknown"
        header = Text()
        header.append(channel["topic"], style="bold cyan")
        header.append(" @ ", style="dim")
        header.append(str(log_time_ns), style="green")
        header.append(" [", style="dim")
        header.append(schema_name, style="yellow")
        header.append("]", style="dim")
        schema = schemas_by_channel.get(channel["id"])
        root_plan = schema_cache.enum_plan(schema) if schema is not None else None
        plan = plan_for_query(root_plan, parsed_query)

        changed_paths = None
        if changed:
            topic = channel["topic"]
            previous = previous_by_topic.get(topic)
            changed_paths = changed_leaf_paths(previous, data) if previous is not None else None
            previous_by_topic[topic] = data

        if flat:
            console.print(header)
            for flat_line in render_message_flat(
                data,
                plan,
                bytes_mode=bytes_mode,
                truncate_bytes=TTY_BYTES_TRUNCATE,
                changed_paths=changed_paths,
            ):
                console.print(flat_line)
        else:
            tree = render_message_tree(
                data,
                plan,
                title=header,
                bytes_mode=bytes_mode,
                truncate_bytes=TTY_BYTES_TRUNCATE,
                changed_paths=changed_paths,
            )
            console.print(Panel(tree, border_style="blue", expand=False))

    def _on_message(channel: ChannelInfo, log_time_ns: int, payload: bytes) -> None:
        nonlocal total, return_code
        if done.is_set():
            return
        decoder = _decoder_for(channel)
        if decoder is None:
            cid = channel["id"]
            if cid not in warned_channels:
                warned_channels.add(cid)
                logger.warning(
                    f"No decoder for {channel['topic']} "
                    f"(message_encoding={channel['encoding']!r}, "
                    f"schemaEncoding={channel.get('schemaEncoding', '')!r})"
                )
            return
        try:
            decoded = decoder(payload)
        except Exception:
            logger.exception(f"Failed to decode message on {channel['topic']}")
            return

        topic = channel["topic"]
        if parsed_query is not None and topic not in validated_topics:
            validated_topics.add(topic)
            schema = schemas_by_channel.get(channel["id"])
            if schema is not None and not schema_cache.validate_query(
                parsed_query, schema, topic, query_repr=query or ""
            ):
                return_code = 1
                done.set()
                return

        if parsed_query is not None:
            try:
                data = parsed_query.apply(decoded, path_variables)
            except MessagePathError as exc:
                logger.warning(f"Filter error on {topic}: {exc}")
                return
            if query_result_is_empty(data):
                return
        else:
            data = decoded

        if grep_pattern is not None and not message_matches_grep(data, grep_pattern):
            return

        if is_tty:
            _emit_tty(channel, log_time_ns, data)
        else:
            _emit_jsonl(channel, log_time_ns, data)

        total += 1
        if limit is not None and total >= limit:
            done.set()

    client = WebSocketBridgeClient(url, min_retry_delay=0.2, max_retry_delay=2.0)
    server_info_event = asyncio.Event()
    client.on_server_info(lambda *_: server_info_event.set())

    subscriber = ChannelSubscriptionManager(
        client,
        lambda channel: _topic_matches(channel["topic"]) and _decoder_for(channel) is not None,
    )
    subscriber.install()
    client.on_message(_on_message)

    await client.connect()
    try:
        try:
            await asyncio.wait_for(server_info_event.wait(), timeout=connect_timeout)
        except asyncio.TimeoutError:
            ERR.print(f"[red]Error:[/] Timed out connecting to {url}")
            return 1

        await subscriber.subscribe_existing()

        background: list[asyncio.Task[None]] = []
        if duration is not None:

            async def _stop_after_duration() -> None:
                await asyncio.sleep(duration)
                done.set()

            background.append(asyncio.create_task(_stop_after_duration()))

        try:
            await done.wait()
        finally:
            for task in background:
                task.cancel()
            if background:
                await asyncio.gather(*background, return_exceptions=True)

        return return_code
    finally:
        await client.disconnect()


def cat(
    target: BridgeTarget,
    *,
    topics: Annotated[
        list[str],
        Parameter(
            name=["-t", "--topics"],
            group=CAT_FILTER_GROUP,
            help="Topic regex(es) to include. Repeat or pass space-delimited.",
        ),
    ] = [],  # noqa: B006
    exclude_topics: Annotated[
        list[str],
        Parameter(
            name=["-x", "--exclude-topics", "-n"],
            group=CAT_FILTER_GROUP,
            help="Topic regex(es) to exclude. Wins over --topics and --query.",
        ),
    ] = [],  # noqa: B006
    query: Annotated[
        str | None,
        Parameter(
            name=["-q", "--query"],
            group=CAT_FILTER_GROUP,
            help="MessagePath expression scoping output to one topic and/or subfield.",
        ),
    ] = None,
    var: MessagePathVariablesOption = None,
    grep: Annotated[
        str | None,
        Parameter(
            name=["-g", "--grep"],
            group=CAT_FILTER_GROUP,
            help=(
                "Regex applied to every scalar value in the decoded message. "
                "Messages with no match are skipped. Bytes-like fields are not "
                "searched."
            ),
        ),
    ] = None,
    grep_ignore_case: Annotated[
        bool,
        Parameter(name=["-i", "--grep-ignore-case"], group=CAT_FILTER_GROUP),
    ] = False,
    limit: Annotated[
        int | None,
        Parameter(name=["-l", "--limit"], group=CAT_OUTPUT_GROUP, help="Stop after N messages."),
    ] = None,
    duration: Annotated[
        float | None,
        Parameter(
            name=["-d", "--duration"],
            group=CAT_OUTPUT_GROUP,
            help="Stop after this many seconds.",
        ),
    ] = None,
    bytes_mode: Annotated[
        BytesMode,
        Parameter(
            name=["--bytes"],
            group=CAT_OUTPUT_GROUP,
            help=(
                "How to render `bytes` fields. `smart` (default) inlines payloads "
                f"≤{SMART_BYTES_INLINE_LIMIT} bytes as int lists and collapses "
                "larger ones to `<N bytes>`. `ints` for the full list, `base64` "
                "for a string, or `skip` to drop the payload."
            ),
        ),
    ] = BytesMode.SMART,
    flat: Annotated[
        bool,
        Parameter(
            name=["--flat"],
            group=CAT_OUTPUT_GROUP,
            help=(
                "In a terminal, print one `dotted.path: value` line per leaf "
                "instead of a tree. Greppable, and handy with --query."
            ),
        ),
    ] = False,
    changed: Annotated[
        bool,
        Parameter(
            name=["--changed"],
            group=CAT_OUTPUT_GROUP,
            help=(
                "In a terminal, highlight values that changed since the previous "
                "message on the same topic. The full message is still shown."
            ),
        ),
    ] = False,
    connect_timeout: Annotated[
        float,
        Parameter(name=["--connect-timeout"], group=CONNECTION_GROUP),
    ] = 5.0,
) -> int:
    """Stream live decoded messages from a Foxglove WebSocket bridge to stdout.

    TTY: each message is rendered as a Rich panel with a per-field tree.
    Pipe: each message becomes one JSONL line `{topic, log_time, schema, message}`.
    Use `bridge record` instead if you want an MCAP file.

    Parameters
    ----------
    target
        Bridge address. Same forms accepted by ``bridge`` (URL, host, ``host:port``);
        defaults to port 8765.
    topics
        Topic regex(es) to include (case-sensitive ``re.search``). Empty means all.
    exclude_topics
        Topic regex(es) to skip. Wins over includes and over ``--query``.
    query
        MessagePath expression (e.g. ``/odom.pose.position.x``); restricts output
        to its topic and projects the addressed subfield.
    grep
        Regex applied to scalar values in the decoded message.
    grep_ignore_case
        Make ``--grep`` case-insensitive.
    limit
        Stop after this many messages.
    duration
        Stop after this many seconds.
    bytes_mode
        How to serialise `bytes` fields. See ``--bytes``.
    connect_timeout
        Seconds to wait for the bridge's ``serverInfo`` (default: 5.0).

    Examples
    --------
    ```
    pymcap-cli bridge cat ws://localhost:8765 -t '^/chatter' -l 3
    pymcap-cli bridge cat localhost:8765 -t '.*' -l 1 | jq
    pymcap-cli bridge cat localhost:8765 -q '/odom.pose.position.x' -l 5
    pymcap-cli bridge cat localhost:8765 -t '.*' -g 'error' -i -d 10
    ```
    """
    if duration is not None and duration <= 0:
        ERR.print("[red]Error:[/] --duration must be positive")
        return 1
    if limit is not None and limit <= 0:
        ERR.print("[red]Error:[/] --limit must be positive")
        return 1

    try:
        variables = create_message_path_variables(var) if query or var else {}
    except ValueError as exc:
        ERR.print(f"[red]Error:[/] {exc}")
        return 1

    url = to_ws_url(target)

    try:
        return asyncio.run(
            _cat_async(
                url=url,
                topics=topics,
                exclude_topics=exclude_topics,
                query=query,
                grep=grep,
                grep_ignore_case=grep_ignore_case,
                limit=limit,
                duration=duration,
                bytes_mode=bytes_mode,
                flat=flat,
                changed=changed,
                connect_timeout=connect_timeout,
                variables=variables,
            )
        )
    except KeyboardInterrupt:
        return 0
    except OSError as exc:
        ERR.print(f"[red]Error:[/] Failed to connect to {url}: {exc}")
        return 1
