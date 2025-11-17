"""Convert command for pymcap-cli."""

import logging
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, BinaryIO

from cyclopts import Group, Parameter
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from small_mcap.writer import CompressionType, McapWriter
from small_mcap.writer import CompressionType as WriterCompressionType

from pymcap_cli.mcap_processor import confirm_output_overwrite
from pymcap_cli.msg_resolver import ROS2Distro, get_message_definition
from pymcap_cli.types_manual import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION,
    ChunkSizeOption,
    CompressionOption,
    ForceOverwriteOption,
    OutputPathOption,
)

logger = logging.getLogger(__name__)


@dataclass
class ConvertOptions:
    """Options for DB3 to MCAP conversion."""

    distro: ROS2Distro = ROS2Distro.HUMBLE
    extra_paths: list[Path] | None = None
    chunk_size: int = 1024 * 1024 * 8  # 8MB default
    compression: CompressionType = CompressionType.ZSTD
    enable_crcs: bool = True
    use_chunking: bool = True


@dataclass
class ConversionStatistics:
    """Statistics from conversion process."""

    topic_count: int
    message_count: int
    schema_count: int
    skipped_topics: list[str]
    skipped_message_count: int


@dataclass(slots=True)
class TopicRecord:
    """Represents a topic from the DB3 topics table."""

    id: int
    name: str
    type: str
    serialization_format: str
    offered_qos_profiles: str | None = None


@dataclass(slots=True)
class MessageRecord:
    """Represents a message from the DB3 messages table."""

    topic_id: int
    timestamp: int
    data: bytes


def read_topics(db_path: Path) -> list[TopicRecord]:
    """Read all topics from DB3 file.

    Args:
        db_path: Path to the DB3 file

    Returns:
        List of topic records

    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        # Check if offered_qos_profiles column exists (newer format)
        cursor.execute("PRAGMA table_info(topics)")
        columns = [row[1] for row in cursor.fetchall()]
        has_qos = "offered_qos_profiles" in columns

        if has_qos:
            query = """
                SELECT id, name, type, serialization_format, offered_qos_profiles
                FROM topics
            """
        else:
            query = """
                SELECT id, name, type, serialization_format
                FROM topics
            """

        cursor.execute(query)
        rows = cursor.fetchall()

        topics = []
        for row in rows:
            if has_qos:
                topic_id, name, msg_type, serialization_format, qos_profiles = row
                topics.append(
                    TopicRecord(
                        id=topic_id,
                        name=name,
                        type=msg_type,
                        serialization_format=serialization_format,
                        offered_qos_profiles=qos_profiles,
                    )
                )
            else:
                topic_id, name, msg_type, serialization_format = row
                topics.append(
                    TopicRecord(
                        id=topic_id,
                        name=name,
                        type=msg_type,
                        serialization_format=serialization_format,
                    )
                )

        return topics
    finally:
        conn.close()


def read_messages_for_topic(
    db_path: Path,
    topic_id: int,
) -> Iterator[MessageRecord]:
    """Read all messages for a specific topic from DB3 file.

    This is a lazy iterator that yields messages one at a time without loading
    the entire dataset into memory, making it suitable for large bag files.

    Args:
        db_path: Path to the DB3 file
        topic_id: Topic ID to read messages for

    Yields:
        Message records ordered by timestamp

    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT topic_id, timestamp, data
            FROM messages
            WHERE topic_id = ?
            ORDER BY timestamp ASC
            """,
            (topic_id,),
        )

        # Iterate over cursor directly - SQLite will fetch rows lazily
        for row in cursor:
            topic_id_val, timestamp, data = row
            yield MessageRecord(
                topic_id=topic_id_val,
                timestamp=timestamp,
                data=data,
            )
    finally:
        conn.close()


def iter_all_messages(db_path: Path) -> Iterator[MessageRecord]:
    """Read all messages from DB3 file ordered by timestamp.

    This is a lazy iterator that yields messages one at a time without loading
    the entire dataset into memory, making it suitable for large bag files.

    Args:
        db_path: Path to the DB3 file

    Yields:
        Message records ordered by timestamp

    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT topic_id, timestamp, data
            FROM messages
            ORDER BY timestamp ASC
            """
        )

        # Iterate over cursor directly - SQLite will fetch rows lazily
        for row in cursor:
            topic_id, timestamp, data = row
            yield MessageRecord(
                topic_id=topic_id,
                timestamp=timestamp,
                data=data,
            )
    finally:
        conn.close()


def get_message_count(db_path: Path) -> int:
    """Get total number of messages in DB3 file.

    Args:
        db_path: Path to the DB3 file

    Returns:
        Total message count

    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages")
        return int(cursor.fetchone()[0])
    finally:
        conn.close()


def convert_db3_to_mcap(
    db3_path: Path,
    output: BinaryIO,
    options: ConvertOptions,
) -> ConversionStatistics:
    """Convert ROS2 DB3 bag file to MCAP format.

    Args:
        db3_path: Path to input DB3 file
        output: Output stream for MCAP data
        options: Conversion options

    Returns:
        Conversion statistics

    """

    extra_paths = tuple(options.extra_paths) if options.extra_paths else ()

    # Read topics from DB3
    logger.info(f"Reading topics from {db3_path}")
    topics = read_topics(db3_path)

    if not topics:
        logger.warning("No topics found in DB3 file")
        # Write empty MCAP
        writer = McapWriter(
            output,
            chunk_size=options.chunk_size,
            compression=options.compression,
            enable_crcs=options.enable_crcs,
            use_chunking=options.use_chunking,
        )
        writer.start(profile="ros2")
        writer.finish()
        return ConversionStatistics(
            topic_count=0,
            message_count=0,
            schema_count=0,
            skipped_topics=[],
            skipped_message_count=0,
        )

    logger.info(f"Found {len(topics)} topics")

    # Build schema definitions
    logger.info("Building message definitions...")
    schema_map: dict[str, int] = {}  # msg_type -> schema_id
    schema_definitions: dict[int, tuple[str, str]] = {}  # schema_id -> (msg_type, definition)
    skipped_types: set[str] = set()  # Track types we couldn't resolve
    next_schema_id = 1

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task("[cyan]Building message definitions...", total=len(topics))

        for topic in topics:
            if topic.type in schema_map:
                progress.advance(task)
                continue

            if topic.type in skipped_types:
                progress.advance(task)
                continue

            msg_def = get_message_definition(topic.type, options.distro, extra_paths)
            if msg_def is None:
                logger.warning(f"Skipping topic type {topic.type}: message definition not found")
                skipped_types.add(topic.type)
                progress.advance(task)
                continue

            schema_id = next_schema_id
            next_schema_id += 1

            schema_map[topic.type] = schema_id
            schema_definitions[schema_id] = (topic.type, msg_def)
            progress.advance(task)

    logger.info(f"Built {len(schema_definitions)} schema definitions")
    if skipped_types:
        logger.warning(
            f"Skipped {len(skipped_types)} message types: {', '.join(sorted(skipped_types))}"
        )

    # Create MCAP writer
    writer = McapWriter(
        output,
        chunk_size=options.chunk_size,
        compression=options.compression,
        enable_crcs=options.enable_crcs,
        use_chunking=options.use_chunking,
    )

    # Start writing MCAP
    writer.start(profile="ros2")

    # Write schemas
    for schema_id, (msg_type, definition) in schema_definitions.items():
        writer.add_schema(
            schema_id=schema_id,
            name=msg_type,
            encoding="ros2msg",
            data=definition.encode("utf-8"),
        )

    # Write channels (only for topics with resolved schemas)
    channel_map: dict[int, int] = {}  # topic_id -> channel_id
    skipped_topics: list[str] = []  # Track skipped topic names
    for topic in topics:
        # Skip topics with unresolved message types
        if topic.type not in schema_map:
            skipped_topics.append(topic.name)
            continue

        schema_id = schema_map[topic.type]
        channel_id = topic.id

        # Build metadata from QoS profiles if available
        metadata = {}
        if topic.offered_qos_profiles:
            metadata["offered_qos_profiles"] = topic.offered_qos_profiles

        writer.add_channel(
            channel_id=channel_id,
            topic=topic.name,
            message_encoding="cdr",
            schema_id=schema_id,
            metadata=metadata,
        )
        channel_map[topic.id] = channel_id

    # Read and write messages
    logger.info("Converting messages...")
    total_messages = get_message_count(db3_path)

    # Track sequence numbers per channel
    sequence_counters: defaultdict[int, int] = defaultdict(int)
    skipped_message_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task("[cyan]Converting messages...", total=total_messages)

        for msg in iter_all_messages(db3_path):
            progress.advance(task)

            # Skip messages for topics we couldn't resolve
            if msg.topic_id not in channel_map:
                skipped_message_count += 1
                continue

            channel_id = channel_map[msg.topic_id]

            # Write message
            writer.add_message(
                channel_id=channel_id,
                log_time=msg.timestamp,
                data=msg.data,
                publish_time=msg.timestamp,
                sequence=sequence_counters[channel_id],
            )
            sequence_counters[channel_id] += 1

    # Finish writing
    writer.finish()

    converted_messages = total_messages - skipped_message_count
    converted_topics = len(topics) - len(skipped_topics)

    logger.info(
        f"Conversion complete: {converted_topics}/{len(topics)} topics, "
        f"{converted_messages:,}/{total_messages:,} messages, {len(schema_definitions)} schemas"
    )
    if skipped_topics:
        logger.warning(f"Skipped {len(skipped_topics)} topics, {skipped_message_count:,} messages")

    return ConversionStatistics(
        topic_count=converted_topics,
        message_count=converted_messages,
        schema_count=len(schema_definitions),
        skipped_topics=skipped_topics,
        skipped_message_count=skipped_message_count,
    )


console = Console()

# Parameter groups
CONVERT_OPTIONS_GROUP = Group("Convert Options")


def convert(
    file: str,
    output: OutputPathOption,
    *,
    distro: Annotated[
        ROS2Distro,
        Parameter(
            name=["--distro"],
            group=CONVERT_OPTIONS_GROUP,
            help="ROS2 distribution for message definitions",
        ),
    ] = ROS2Distro.HUMBLE,
    extra_path: Annotated[
        list[Path],
        Parameter(
            name=["--extra-path"],
            group=CONVERT_OPTIONS_GROUP,
            help="Additional paths to search for custom message definitions",
        ),
    ] = [],  # noqa: B006
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    force: ForceOverwriteOption = False,
) -> None:
    """Convert ROS2 DB3 (SQLite) bag files to MCAP format.

    This command converts ROS2 bag files in DB3 (SQLite) format to MCAP format,
    preserving all message data, topic metadata, and QoS profiles.

    Message definitions are resolved from:
    1. User-provided --extra-path directories (custom messages)
    2. AMENT_PREFIX_PATH environment variable (installed ROS packages)
    3. Downloaded standard ROS2 repositories (rcl_interfaces, common_interfaces, geometry2)

    Parameters
    ----------
    file
        Path to the DB3 file to convert.
    output
        Output MCAP filename.
    distro
        ROS2 distribution to use for standard message definitions.
    extra_path
        Additional paths to search for custom message definitions.
        Can be specified multiple times for multiple directories.
    chunk_size
        Chunk size of output file in bytes.
    compression
        Compression algorithm for output file.
    force
        Force overwrite of output file without confirmation.

    Examples
    --------
    ```
    # Basic conversion
    pymcap-cli convert input.db3 -o output.mcap

    # Specify ROS distro
    pymcap-cli convert input.db3 -o output.mcap --distro jazzy

    # With custom message definitions
    pymcap-cli convert input.db3 -o output.mcap --extra-path /path/to/msgs

    # Multiple custom paths
    pymcap-cli convert input.db3 -o output.mcap \\
        --extra-path /path/to/msgs1 \\
        --extra-path /path/to/msgs2

    # With custom chunk size and compression
    pymcap-cli convert input.db3 -o output.mcap \\
        --chunk-size 8388608 \\
        --compression lz4
    ```
    """
    # Validate input file
    input_path = Path(file)
    if not input_path.exists():
        console.print(f"[red]Error: Input file '{file}' does not exist[/red]")
        sys.exit(1)

    # Confirm overwrite if needed
    confirm_output_overwrite(output, force)

    # Map compression type from CLI enum to writer enum
    compression_map = {
        "zstd": WriterCompressionType.ZSTD,
        "lz4": WriterCompressionType.LZ4,
        "none": WriterCompressionType.NONE,
    }
    writer_compression = compression_map[compression.value]

    # Build conversion options
    options = ConvertOptions(
        distro=distro,
        extra_paths=extra_path if extra_path else None,
        chunk_size=chunk_size,
        compression=writer_compression,
        enable_crcs=True,
        use_chunking=True,
    )

    console.print(f"[blue]Converting '{file}' to '{output}'[/blue]")
    console.print(f"[dim]ROS2 distro: {distro.value}[/dim]")
    if extra_path:
        console.print(f"[dim]Extra paths: {', '.join(str(p) for p in extra_path)}[/dim]")

    # Perform conversion
    with output.open("wb") as output_stream:
        try:
            stats = convert_db3_to_mcap(input_path, output_stream, options)

            console.print("[green]✓ Conversion completed successfully![/green]")
            console.print(
                f"Converted {stats.topic_count} topics, "
                f"{stats.message_count:,} messages, "
                f"{stats.schema_count} schemas"
            )

            # Report skipped topics if any
            if stats.skipped_topics:
                console.print(
                    "[yellow]"
                    f"⚠ Skipped {len(stats.skipped_topics)} topics "
                    f"({stats.skipped_message_count:,} messages) due to missing message definitions"
                    "[/]"
                )
                console.print("[dim]Skipped topics:[/dim]")
                for topic in sorted(stats.skipped_topics):
                    console.print(f"  [dim]- {topic}[/dim]")

        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error during conversion: {e}[/red]")
            sys.exit(1)
