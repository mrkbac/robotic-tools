"""Convert ROS1 bag files to MCAP format (ros1 profile)."""

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from pymcap_cli.cmd._cli_options import (
    ChunkSizeOption,
    CompressionOption,
    ForceOverwriteOption,
    OutputPathOption,
)
from pymcap_cli.constants import DEFAULT_CHUNK_SIZE, DEFAULT_COMPRESSION
from pymcap_cli.display.osc_utils import OSCProgressColumn
from pymcap_cli.utils import WriterCompression, confirm_output_overwrite, create_mcap_writer

logger = logging.getLogger(__name__)


@dataclass
class Bag2McapOptions:
    """Options for bag to MCAP conversion."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    compression: WriterCompression = DEFAULT_COMPRESSION
    enable_crcs: bool = True
    use_chunking: bool = True
    zstd_level: int | None = None


@dataclass
class Bag2McapStatistics:
    """Statistics from bag to MCAP conversion."""

    topic_count: int
    message_count: int
    schema_count: int


def convert_bag_to_mcap(
    bag_path: Path,
    output: BinaryIO,
    options: Bag2McapOptions,
) -> Bag2McapStatistics:
    """Convert a ROS1 bag file to MCAP format with ros1 profile.

    Messages are passed through as raw ROS1-serialized bytes.
    Schema encoding is "ros1msg", message encoding is "ros1".

    Args:
        bag_path: Path to the input .bag file.
        output: Output stream for MCAP data.
        options: Conversion options.

    Returns:
        Conversion statistics.

    """
    from pymcap_cli.rosbag_reader import read_bag_info, read_bag_messages  # noqa: PLC0415

    with bag_path.open("rb") as bag_file:
        info = read_bag_info(bag_file)

    if not info.connections:
        logger.warning("No connections found in bag file")
        return Bag2McapStatistics(topic_count=0, message_count=0, schema_count=0)

    logger.info(f"Found {len(info.connections)} connections, {info.message_count} messages")

    # Build schema map: deduplicate by msg_type
    schema_map: dict[str, int] = {}  # msg_type -> schema_id
    schema_definitions: dict[int, tuple[str, str]] = {}  # schema_id -> (msg_type, definition)
    next_schema_id = 1

    for conn in info.connections.values():
        if conn.msg_type not in schema_map:
            schema_id = next_schema_id
            next_schema_id += 1
            schema_map[conn.msg_type] = schema_id
            schema_definitions[schema_id] = (conn.msg_type, conn.message_definition)

    # Build channel map: deduplicate by (topic, msg_type)
    # Multiple connections can exist for the same topic (multiple publishers)
    channel_map: dict[tuple[str, str], int] = {}  # (topic, msg_type) -> channel_id
    conn_to_channel: dict[int, int] = {}  # conn_id -> channel_id
    next_channel_id = 1

    for conn in info.connections.values():
        key = (conn.topic, conn.msg_type)
        if key not in channel_map:
            channel_map[key] = next_channel_id
            next_channel_id += 1
        conn_to_channel[conn.conn_id] = channel_map[key]

    # Create MCAP writer
    writer = create_mcap_writer(output, options)
    writer.start(profile="ros1")

    # Write schemas
    for schema_id, (msg_type, definition) in schema_definitions.items():
        writer.add_schema(
            schema_id=schema_id,
            name=msg_type,
            encoding="ros1msg",
            data=definition.encode("utf-8"),
        )

    # Write channels
    for (topic, msg_type), channel_id in channel_map.items():
        schema_id = schema_map[msg_type]

        # Collect metadata from connections for this channel
        metadata: dict[str, str] = {}
        for conn in info.connections.values():
            if conn.topic == topic and conn.msg_type == msg_type:
                metadata["md5sum"] = conn.md5sum
                if conn.callerid:
                    metadata["callerid"] = conn.callerid
                break

        writer.add_channel(
            channel_id=channel_id,
            topic=topic,
            message_encoding="ros1",
            schema_id=schema_id,
            metadata=metadata,
        )

    # Read and write messages
    logger.info("Converting messages...")
    sequence_counters: defaultdict[int, int] = defaultdict(int)
    message_count = 0

    with (
        bag_path.open("rb") as bag_file,
        Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            OSCProgressColumn(title="Converting messages"),
            transient=False,
        ) as progress,
    ):
        task = progress.add_task(
            "[cyan]Converting messages...",
            total=info.message_count if info.message_count > 0 else None,
        )

        for msg in read_bag_messages(bag_file, info):
            channel_id = conn_to_channel[msg.conn_id]

            writer.add_message(
                channel_id=channel_id,
                log_time=msg.time_ns,
                data=msg.data,
                publish_time=msg.time_ns,
                sequence=sequence_counters[channel_id],
            )
            sequence_counters[channel_id] += 1
            message_count += 1
            progress.advance(task)

    writer.finish()

    logger.info(
        f"Conversion complete: {len(channel_map)} topics, "
        f"{message_count:,} messages, {len(schema_definitions)} schemas"
    )

    return Bag2McapStatistics(
        topic_count=len(channel_map),
        message_count=message_count,
        schema_count=len(schema_definitions),
    )


console = Console()


def bag2mcap(
    file: str,
    output: OutputPathOption,
    *,
    chunk_size: ChunkSizeOption = DEFAULT_CHUNK_SIZE,
    compression: CompressionOption = DEFAULT_COMPRESSION,
    force: ForceOverwriteOption = False,
) -> int:
    """Convert ROS1 bag files to MCAP format.

    Converts ROS1 bag files to MCAP with ros1 profile, preserving all
    message data as raw ROS1-serialized bytes. Schemas use ros1msg encoding
    with the full message definition from the bag file.

    Parameters
    ----------
    file
        Path to the ROS1 .bag file to convert.
    output
        Output MCAP filename.
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
    pymcap-cli bag2mcap recording.bag -o recording.mcap

    # With custom compression
    pymcap-cli bag2mcap recording.bag -o recording.mcap --compression lz4
    ```
    """
    input_path = Path(file)
    if not input_path.exists():
        logger.error(f"Input file '{file}' does not exist")
        return 1

    from pymcap_cli.rosbag_reader import read_bag_info  # noqa: PLC0415

    with input_path.open("rb") as bag_file:
        if not read_bag_info(bag_file).connections:
            logger.warning("No connections found in bag file; no output written")
            return 0

    confirm_output_overwrite(output, force)

    options = Bag2McapOptions(
        chunk_size=chunk_size,
        compression=compression,
    )

    logger.info(f"Converting '{file}' to '{output}'")

    with output.open("wb") as output_stream:
        try:
            stats = convert_bag_to_mcap(input_path, output_stream, options)

            logger.info("[green]Conversion completed successfully[/green]")
            console.print(
                f"Converted {stats.topic_count} topics, "
                f"{stats.message_count:,} messages, "
                f"{stats.schema_count} schemas"
            )
        except Exception:
            logger.exception("Error during conversion")
            return 1

    return 0
