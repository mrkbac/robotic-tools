"""Unified process command — every streaming MCAP→MCAP transform in one pass."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypeVar

from cyclopts import Group, Parameter, validators
from rich.console import Console
from ros_parser.message_path import MessagePathError

from pymcap_cli.cmd._run_processor import (
    finalize_delete_source,
    resolve_overwrite_policy,
    run_processor,
)
from pymcap_cli.cmd._run_processor_multi import run_processor_multi
from pymcap_cli.constants import DEFAULT_CHUNK_SIZE, DEFAULT_COMPRESSION
from pymcap_cli.core.mcap_processor import (
    InputOptions,
    OutputOptions,
    RechunkStrategy,
)
from pymcap_cli.core.processors.channel_merge import ChannelMergeProcessor
from pymcap_cli.core.processors.dedup import DedupIdenticalProcessor
from pymcap_cli.core.processors.duration_split import DurationSplitProcessor
from pymcap_cli.core.processors.expression_split import ExpressionSplitProcessor
from pymcap_cli.core.processors.nth_message import NthMessageProcessor
from pymcap_cli.core.processors.size_split import SizeSplitProcessor
from pymcap_cli.core.processors.time_offset import TimeOffsetProcessor
from pymcap_cli.core.processors.timestamp_split import TimestampSplitProcessor
from pymcap_cli.core.processors.topic_alias import TopicAliasProcessor
from pymcap_cli.core.processors.topic_rewrite import TopicRewriteProcessor
from pymcap_cli.types.duration import parse_duration_ns
from pymcap_cli.types.size import parse_size_bytes
from pymcap_cli.types.types_manual import (
    OUTPUT_OPTIONS_GROUP,
    ChunkSizeOption,
    CompressionOption,
    DeleteSourceOption,
    ForceOverwriteOption,
    NoClobberOption,
)
from pymcap_cli.utils import (
    AttachmentsMode,
    MetadataMode,
    bytes_to_human,
    compile_topic_patterns,
    parse_time_arg,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cyclopts.token import Token

    from pymcap_cli.core.processors.base import InputProcessor, OutputRouter

logger = logging.getLogger(__name__)
console = Console()

# Parameter groups (declared once so --help groups every flag consistently)
RECOVERY_GROUP = Group("Recovery Options")
FILTERING_GROUP = Group("Topic Filtering")
TIME_FILTERING_GROUP = Group("Time Filtering")
CONTENT_FILTERING_GROUP = Group("Content Filtering")
DEDUP_GROUP = Group("Deduplication")
LATCHING_GROUP = Group("Latching")
TOPIC_TRANSFORM_GROUP = Group("Topic Transforms")
TIME_TRANSFORM_GROUP = Group("Time / Decimation")
RECHUNK_GROUP = Group("Rechunking")
SPLIT_GROUP = Group("Splitting (multi-output)")
SPLIT_EXPR_GROUP = Group("Splitting — Expression Options")

T = TypeVar("T")


def _parse_kv_rules(tokens: list[str], value_parser: Callable[[str], T]) -> dict[str, T]:
    """Parse repeatable ``PATTERN=VALUE`` tokens into a mapping.

    Splits on the *first* ``=`` so values may contain further ``=`` characters
    (e.g. regex backreferences in a replacement).
    """
    rules: dict[str, T] = {}
    for tok in tokens:
        if "=" not in tok:
            msg = f"Expected 'PATTERN=VALUE', got {tok!r}"
            raise ValueError(msg)
        pattern, raw = tok.split("=", 1)
        rules[pattern] = value_parser(raw)
    return rules


def _parse_alias_rules(tokens: list[str]) -> dict[str, str | list[str]]:
    """Parse repeatable ``--alias-topic PATTERN=REPLACEMENT`` tokens.

    Multiple tokens with the same pattern fan out — the rule's value becomes
    a list of all replacements supplied for that pattern.
    """
    rules: dict[str, str | list[str]] = {}
    for tok in tokens:
        if "=" not in tok:
            msg = f"Expected 'PATTERN=REPLACEMENT', got {tok!r}"
            raise ValueError(msg)
        pat, repl = tok.split("=", 1)
        existing = rules.get(pat)
        if existing is None:
            rules[pat] = repl
        elif isinstance(existing, list):
            existing.append(repl)
        else:
            rules[pat] = [existing, repl]
    return rules


def _duration_ns_converter(_type: type, tokens: Sequence[Token]) -> int:
    if len(tokens) != 1:
        msg = "Expected exactly one duration."
        raise ValueError(msg)
    return parse_duration_ns(tokens[0].value)


def process(
    file: list[str],
    output: Annotated[
        Path | None,
        Parameter(name=["-o", "--output"], group=OUTPUT_OPTIONS_GROUP),
    ] = None,
    *,
    # ----- Recovery -----
    always_decode_chunk: Annotated[
        bool,
        Parameter(name=["-a", "--always-decode-chunk"], group=RECOVERY_GROUP),
    ] = False,
    # ----- Topic filtering -----
    include_topic_regex: Annotated[
        list[str] | None,
        Parameter(
            name=["-t", "--topics", "--include-topic-regex", "-y"],
            group=FILTERING_GROUP,
        ),
    ] = None,
    exclude_topic_regex: Annotated[
        list[str] | None,
        Parameter(
            name=["-x", "--exclude-topics", "--exclude-topic-regex", "-n"],
            group=FILTERING_GROUP,
        ),
    ] = None,
    # ----- Time filtering -----
    start: Annotated[str, Parameter(name=["-S", "--start"], group=TIME_FILTERING_GROUP)] = "",
    start_secs: Annotated[int, Parameter(name=["--start-secs"], group=TIME_FILTERING_GROUP)] = 0,
    start_nsecs: Annotated[int, Parameter(name=["--start-nsecs"], group=TIME_FILTERING_GROUP)] = 0,
    end: Annotated[str, Parameter(name=["-E", "--end"], group=TIME_FILTERING_GROUP)] = "",
    end_secs: Annotated[int, Parameter(name=["--end-secs"], group=TIME_FILTERING_GROUP)] = 0,
    end_nsecs: Annotated[int, Parameter(name=["--end-nsecs"], group=TIME_FILTERING_GROUP)] = 0,
    # ----- Content filtering -----
    metadata_mode: Annotated[
        MetadataMode,
        Parameter(name=["--metadata"], group=CONTENT_FILTERING_GROUP),
    ] = MetadataMode.INCLUDE,
    attachments_mode: Annotated[
        AttachmentsMode,
        Parameter(name=["--attachments"], group=CONTENT_FILTERING_GROUP),
    ] = AttachmentsMode.INCLUDE,
    # ----- Deduplication -----
    dedup_identical: Annotated[
        bool,
        Parameter(
            name=["--dedup-identical"],
            group=DEDUP_GROUP,
            help=(
                "Drop messages whose (channel, log_time, payload-hash) was "
                "already written. Chunks whose time range doesn't overlap "
                "any other input's chunk range still fast-copy; overlapping "
                "chunks are decoded so the per-message hash check can run. "
                "Combine with --always-decode-chunk to also catch intra-input "
                "duplicates inside non-overlapping chunks."
            ),
        ),
    ] = False,
    # ----- Latching -----
    latch: Annotated[
        list[str] | None,
        Parameter(
            name=["--latch"],
            group=LATCHING_GROUP,
            help=(
                "Topic regex (repeatable) whose latest message is replayed into "
                "every output segment. Use for /tf_static and other "
                "transient-local topics that consumers need at the start of "
                "each segment."
            ),
        ),
    ] = None,
    latch_from_metadata: Annotated[
        bool,
        Parameter(
            name=["--latch-from-metadata"],
            group=LATCHING_GROUP,
            help=(
                "Auto-detect latched channels by reading the MCAP "
                "'offered_qos_profiles' metadata for durability=transient_local."
            ),
        ),
    ] = False,
    # ----- Topic transforms -----
    rename_topic: Annotated[
        list[str] | None,
        Parameter(
            name=["--rename-topic"],
            group=TOPIC_TRANSFORM_GROUP,
            help=(
                "Rewrite Channel.topic via regex substitution. Format: "
                "'PATTERN=REPLACEMENT' (re.sub semantics — backrefs like '\\1' "
                "work). Repeatable; first matching pattern wins."
            ),
        ),
    ] = None,
    alias_topic: Annotated[
        list[str] | None,
        Parameter(
            name=["--alias-topic"],
            group=TOPIC_TRANSFORM_GROUP,
            help=(
                "Emit each matched message under both its original topic and "
                "the replacement. Format: 'PATTERN=REPLACEMENT'. Repeatable; "
                "use the same pattern twice for multi-alias fan-out."
            ),
        ),
    ] = None,
    merge_channels: Annotated[
        bool,
        Parameter(
            name=["--merge-channels"],
            group=TOPIC_TRANSFORM_GROUP,
            help=(
                "Collapse duplicate channels with identical "
                "(topic, schema, encoding, metadata) onto a single channel id "
                "— useful when a recorder restart re-advertised a channel."
            ),
        ),
    ] = False,
    # ----- Time transform / decimation -----
    time_offset: Annotated[
        list[str] | None,
        Parameter(
            name=["--time-offset"],
            group=TIME_TRANSFORM_GROUP,
            help=(
                "Shift log_time / publish_time by an ns offset for matching "
                "channels. Format: 'PATTERN=DURATION' (e.g. '/imu=500ms' or "
                "'/gps=-1s'). Repeatable."
            ),
        ),
    ] = None,
    decimate: Annotated[
        list[str] | None,
        Parameter(
            name=["--decimate"],
            group=TIME_TRANSFORM_GROUP,
            help=(
                "Keep every Nth message on channels whose topic matches. "
                "Format: 'PATTERN=N'. The first message on each channel is "
                "always kept. Repeatable."
            ),
        ),
    ] = None,
    # ----- Rechunking -----
    rechunk_strategy: Annotated[
        RechunkStrategy,
        Parameter(name=["--rechunk-strategy"], group=RECHUNK_GROUP),
    ] = RechunkStrategy.NONE,
    rechunk_pattern: Annotated[
        list[str] | None,
        Parameter(
            name=["--rechunk-pattern"],
            group=RECHUNK_GROUP,
            help=(
                "Regex matched against Channel.topic (repeatable; only with "
                "--rechunk-strategy=pattern). First match wins across topic "
                "patterns then schema patterns."
            ),
        ),
    ] = None,
    rechunk_schema_pattern: Annotated[
        list[str] | None,
        Parameter(
            name=["--rechunk-schema-pattern"],
            group=RECHUNK_GROUP,
            help=(
                "Regex matched against Schema.name (e.g. "
                "'sensor_msgs/.*Image.*'). Streaming-safe — schema is known "
                "at channel registration time. Evaluated after topic patterns."
            ),
        ),
    ] = None,
    rechunk_max_groups: Annotated[
        int | None,
        Parameter(
            name=["--rechunk-max-groups"],
            group=RECHUNK_GROUP,
            help=(
                "Hard cap on concurrent chunk groups per output segment. "
                "Overflow channels share the last-created group."
            ),
        ),
    ] = None,
    rechunk_max_memory: Annotated[
        str | None,
        Parameter(
            name=["--rechunk-max-memory"],
            group=RECHUNK_GROUP,
            help=(
                "Cap on total uncompressed bytes buffered across all chunk "
                "groups in a segment (e.g. '256MB'). When exceeded, the "
                "largest in-flight chunk is flushed prematurely."
            ),
        ),
    ] = None,
    # ----- Splitting (multi-output) -----
    split_duration: Annotated[
        str | None,
        Parameter(
            name=["--split-duration"],
            group=SPLIT_GROUP,
            help="Split every N time units (e.g. '60s', '1.5m', '1h').",
        ),
    ] = None,
    split_at: Annotated[
        list[str] | None,
        Parameter(
            name=["--split-at"],
            group=SPLIT_GROUP,
            help="Split at specific timestamps (ns or RFC3339, repeatable).",
        ),
    ] = None,
    split_expression: Annotated[
        str | None,
        Parameter(
            name=["--split-expression"],
            group=SPLIT_GROUP,
            help=(
                "Split whenever a ros-parser message path changes value, "
                "e.g. '/gps/fix.status.status'."
            ),
        ),
    ] = None,
    split_max_size: Annotated[
        str | None,
        Parameter(
            name=["--split-max-size"],
            group=SPLIT_GROUP,
            help="Split when accumulated message bytes exceed N (e.g. '1G').",
        ),
    ] = None,
    split_hysteresis: Annotated[
        int | None,
        Parameter(
            name=["--split-hysteresis"],
            group=SPLIT_EXPR_GROUP,
            converter=_duration_ns_converter,
            validator=validators.Number(gt=0),
            help=(
                "Time hysteresis for --split-expression: a new value must "
                "persist for at least this duration (e.g. '500ms', '2s') "
                "before a segment cut fires."
            ),
        ),
    ] = None,
    split_hysteresis_count: Annotated[
        int | None,
        Parameter(
            name=["--split-hysteresis-count"],
            group=SPLIT_EXPR_GROUP,
            validator=validators.Number(gt=0),
            help=(
                "Count hysteresis for --split-expression: a new value must "
                "appear this many times before a cut fires."
            ),
        ),
    ] = None,
    split_keep_trailing_context: Annotated[
        int | None,
        Parameter(
            name=["--split-keep-trailing-context"],
            group=SPLIT_EXPR_GROUP,
            converter=_duration_ns_converter,
            validator=validators.Number(gt=0),
            help=(
                "After a transition, also write target-topic messages from "
                "this duration (e.g. '500ms') into the previous segment."
            ),
        ),
    ] = None,
    split_keep_trailing_count: Annotated[
        int | None,
        Parameter(
            name=["--split-keep-trailing-count"],
            group=SPLIT_EXPR_GROUP,
            validator=validators.Number(gt=0),
            help=(
                "After a transition, also write up to this many target-topic "
                "messages into the previous segment."
            ),
        ),
    ] = None,
    output_template: Annotated[
        str,
        Parameter(
            name=["--output-template"],
            group=SPLIT_GROUP,
            help=(
                "Output file naming template, used in split mode. "
                "Variables: {index}, {index1}, {key}, {start_time}, "
                "{start_time_iso}, {end_time}."
            ),
        ),
    ] = "output_{index:03d}.mcap",
    # ----- Output -----
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    force: ForceOverwriteOption = False,
    no_clobber: NoClobberOption = False,
    delete_source: DeleteSourceOption = False,
) -> int:
    """Process MCAP files — recovery, filtering, transforms, splitting in one pass.

    Single command for every streaming MCAP→MCAP operation: recovery of
    corrupt files, topic/time filtering, dedup, latching replay, rename /
    alias / channel-merge, time-offset, decimation, rechunking, and
    multi-output splitting. Multiple input files are merged chronologically.

    Parameters
    ----------
    file
        Path(s) to MCAP file(s) to process (local files or HTTP/HTTPS URLs,
        merged if multiple).
    output
        Output filename. Required unless a --split-* flag is set.
    always_decode_chunk
        Always decode chunks, never use fast copying.
    include_topic_regex
        Include messages with topic names matching this regex (repeatable).
    exclude_topic_regex
        Exclude messages with topic names matching this regex (repeatable).
    start
        Include messages at or after this time (nanoseconds or RFC3339 date).
    start_secs
        Include messages at or after this time in seconds.
    start_nsecs
        [DEPRECATED — use --start instead] Same in nanoseconds.
    end
        Include messages before this time (nanoseconds or RFC3339 date).
    end_secs
        Include messages before this time in seconds.
    end_nsecs
        [DEPRECATED — use --end instead] Same in nanoseconds.
    metadata_mode
        Metadata handling: include or exclude metadata records.
    attachments_mode
        Attachments handling: include or exclude attachment records.
    chunk_size
        Chunk size of output file in bytes.
    compression
        Compression algorithm for output file.
    force
        Force overwrite of output file without confirmation.
    no_clobber
        Fail instead of prompting if the output already exists.
    delete_source
        Delete source file(s) after the output is validated.

    Examples
    --------
    ```
    # Recover + change compression in one pass
    pymcap-cli process corrupt.mcap -o fixed.mcap --compression lz4

    # Filter by topic and time
    pymcap-cli process in.mcap -o out.mcap -y '/camera.*' --start-secs 10

    # Merge multiple files with dedup
    pymcap-cli process a.mcap b.mcap -o merged.mcap --dedup-identical

    # Rename topics and decimate
    pymcap-cli process in.mcap -o out.mcap \\
        --rename-topic '/old/(.*)=/new/\\1' --decimate '/imu/data=10'

    # Split into 60-second segments
    pymcap-cli process in.mcap --split-duration 60s \\
        --output-template 'seg_{index:03d}.mcap'
    ```
    """
    overwrite_policy = resolve_overwrite_policy(force=force, no_clobber=no_clobber)
    if overwrite_policy is None:
        logger.error("--force and --no-clobber cannot be used together.")
        return 1

    # --- Validate output / split mode coupling ---------------------------------
    any_split = bool(split_duration or split_at or split_expression or split_max_size)

    expr_only = (
        split_hysteresis is not None
        or split_hysteresis_count is not None
        or split_keep_trailing_context is not None
        or split_keep_trailing_count is not None
    )
    if expr_only and not split_expression:
        logger.error(
            "--split-hysteresis, --split-hysteresis-count, "
            "--split-keep-trailing-context and --split-keep-trailing-count "
            "require --split-expression."
        )
        return 1

    if any_split and output is not None:
        logger.error(
            "--output is for single-file mode; in split mode use --output-template instead."
        )
        return 1
    if not any_split and output is None:
        logger.error("--output is required unless a --split-* flag is set.")
        return 1

    # --- Rechunking validation -------------------------------------------------
    if rechunk_strategy == RechunkStrategy.PATTERN and not (
        rechunk_pattern or rechunk_schema_pattern
    ):
        logger.error(
            "--rechunk-strategy=pattern requires at least one "
            "--rechunk-pattern or --rechunk-schema-pattern."
        )
        return 1
    if rechunk_max_groups is not None and rechunk_max_groups < 1:
        logger.error("--rechunk-max-groups must be >= 1.")
        return 1

    rechunk_max_memory_bytes: int | None = None
    if rechunk_max_memory is not None:
        try:
            rechunk_max_memory_bytes = parse_size_bytes(rechunk_max_memory)
        except ValueError:
            logger.exception(f"Error parsing --rechunk-max-memory {rechunk_max_memory!r}")
            return 1
        if rechunk_max_memory_bytes < 1:
            logger.error("--rechunk-max-memory must be >= 1 byte.")
            return 1

    rechunk_patterns_compiled = []
    rechunk_schema_patterns_compiled = []
    try:
        if rechunk_pattern:
            rechunk_patterns_compiled = compile_topic_patterns(rechunk_pattern)
        if rechunk_schema_pattern:
            rechunk_schema_patterns_compiled = compile_topic_patterns(rechunk_schema_pattern)
    except ValueError as e:
        logger.error(str(e))  # noqa: TRY400
        return 1

    # --- Build KV rule maps ----------------------------------------------------
    try:
        rename_rules = _parse_kv_rules(rename_topic or [], str)
        time_offset_rules = _parse_kv_rules(time_offset or [], parse_duration_ns)
        decimate_rules = _parse_kv_rules(decimate or [], int)
        alias_rules = _parse_alias_rules(alias_topic or [])
    except ValueError as e:
        logger.error(str(e))  # noqa: TRY400
        return 1

    # --- Build the input processor chain --------------------------------------
    # Order follows core/processors/ARCHITECTURE.md §"Chain ordering":
    # alias before id-mutators; rewrite LAST among id-mutators (it produces
    # DECODE_VERIFY decisions); dedup needs every per-message run so it goes
    # after id-mutators that fan out / merge.
    extras: list[InputProcessor] = []
    if alias_rules:
        extras.append(TopicAliasProcessor(alias_rules))
    if merge_channels:
        extras.append(ChannelMergeProcessor())
    if time_offset_rules:
        extras.append(TimeOffsetProcessor(time_offset_rules))
    if decimate_rules:
        extras.append(NthMessageProcessor(decimate_rules))
    if dedup_identical:
        extras.append(DedupIdenticalProcessor())
    if rename_rules:
        extras.append(TopicRewriteProcessor(rename_rules))

    if dedup_identical and not always_decode_chunk:
        logger.info(
            "Dedup enabled without --always-decode-chunk: intra-input "
            "duplicates inside non-overlapping chunks may slip past."
        )

    # --- Build InputOptions ----------------------------------------------------
    try:
        input_options = InputOptions.from_args(
            include_topic_regex=include_topic_regex,
            exclude_topic_regex=exclude_topic_regex,
            start=start,
            start_nsecs=start_nsecs,
            start_secs=start_secs,
            end=end,
            end_nsecs=end_nsecs,
            end_secs=end_secs,
            include_metadata=metadata_mode == MetadataMode.INCLUDE,
            include_attachments=attachments_mode == AttachmentsMode.INCLUDE,
            always_decode_chunk=always_decode_chunk,
            latch_topics=latch,
            latch_from_metadata=latch_from_metadata,
            extra_processors=extras or None,
        )
    except ValueError as e:
        logger.error(str(e))  # noqa: TRY400
        return 1

    # --- Build split routers ---------------------------------------------------
    routers: list[OutputRouter] = []
    if split_duration:
        try:
            duration_ns = parse_duration_ns(split_duration)
        except ValueError:
            logger.exception(f"Error parsing --split-duration {split_duration!r}")
            return 1
        if duration_ns <= 0:
            logger.error("--split-duration must be positive.")
            return 1
        routers.append(DurationSplitProcessor(duration_ns))

    if split_at:
        split_points: list[int] = []
        for ts in split_at:
            try:
                split_points.append(parse_time_arg(ts))
            except ValueError:
                logger.exception(f"Error parsing --split-at {ts!r}")
                return 1
        routers.append(TimestampSplitProcessor(split_points))

    if split_max_size:
        try:
            max_size_bytes = parse_size_bytes(split_max_size)
        except ValueError:
            logger.exception(f"Error parsing --split-max-size {split_max_size!r}")
            return 1
        routers.append(SizeSplitProcessor(max_size_bytes))

    if split_expression:
        try:
            routers.append(
                ExpressionSplitProcessor(
                    split_expression,
                    hysteresis_ns=split_hysteresis,
                    hysteresis_count=split_hysteresis_count,
                    trailing_context_ns=split_keep_trailing_context,
                    trailing_context_count=split_keep_trailing_count,
                )
            )
        except MessagePathError:
            logger.exception(f"Error parsing --split-expression {split_expression!r}")
            return 1
        except ValueError:
            logger.exception("Invalid --split-expression option")
            return 1

    # --- Run ------------------------------------------------------------------
    output_options = OutputOptions(
        compression=compression,
        chunk_size=chunk_size,
        rechunk_strategy=rechunk_strategy,
        rechunk_patterns=rechunk_patterns_compiled,
        rechunk_schema_patterns=rechunk_schema_patterns_compiled,
        rechunk_max_groups=rechunk_max_groups,
        rechunk_max_memory=rechunk_max_memory_bytes,
        routers=routers,
        output_template=output_template if any_split else "",
        overwrite_policy=overwrite_policy,
    )

    try:
        if any_split:
            result = run_processor_multi(
                files=file,
                input_options=input_options,
                output_options=output_options,
            )
        else:
            assert output is not None  # validated above
            result = run_processor(
                files=file,
                output=output,
                input_options=input_options,
                output_options=output_options,
            )
    except Exception:
        logger.exception("Error during processing")
        return 1

    # --- Report ---------------------------------------------------------------
    if len(file) > 1:
        logger.info(f"[green]✓ Successfully processed {len(file)} files![/green]")
    else:
        logger.info("[green]✓ Processing completed successfully![/green]")
    console.print(result.stats)

    dedup_proc = next(
        (p for p in extras if isinstance(p, DedupIdenticalProcessor)),
        None,
    )
    if dedup_proc and dedup_proc.dropped_count:
        console.print(
            f"[dim]Dedup dropped {dedup_proc.dropped_count:,} duplicate message(s).[/dim]"
        )

    if any_split and result.processor.output_manager is not None:
        segments = result.processor.output_manager.segments
        console.print(f"Created {len(segments)} output file(s)")
        for _, segment in sorted(segments.items(), key=lambda x: x[1].index):
            seg_stats = segment.writer.statistics
            console.print(
                f"  [{segment.index}] {segment.path}: "
                f"{seg_stats.message_count:,} messages, "
                f"{bytes_to_human(seg_stats.chunk_count * chunk_size)} "
                f"({seg_stats.chunk_count} chunks)"
            )

    if delete_source:
        if any_split:
            assert result.processor.output_manager is not None
            outputs = [Path(seg.path) for seg in result.processor.output_manager.segments.values()]
        else:
            assert output is not None
            outputs = [output]
        return finalize_delete_source(sources=list(file), outputs=outputs)

    return 0
