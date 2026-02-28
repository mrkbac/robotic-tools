"""ROS2 PointCloud2 compression analysis command."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from mcap_ros2_support_fast.decoder import DecoderFactory
from pureini import (
    CompressionOption,
    EncodingInfo,
    EncodingOptions,
    FieldType,
    PointcloudEncoder,
    PointField,
)
from rich.console import Console
from rich.table import Table
from small_mcap import read_message

console = Console()


def bytes_to_human(size_bytes: float) -> str:
    """Convert bytes to a human-readable format."""
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PiB"


@dataclass
class CompressionStats:
    """Track compression statistics for a single topic."""

    topic: str
    schema: str
    message_count: int = 0
    original_bytes: int = 0
    compressed_bytes: int = 0

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        return self.original_bytes / self.compressed_bytes if self.compressed_bytes > 0 else 0.0

    @property
    def bytes_saved(self) -> int:
        """Calculate bytes saved."""
        return self.original_bytes - self.compressed_bytes


def handle_command(args: argparse.Namespace) -> None:
    """Handle the ros2compress command execution."""
    input_file = Path(args.file)

    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        return

    # Map string options to enums
    encoding_map = {
        "lossy": EncodingOptions.LOSSY,
        "lossless": EncodingOptions.LOSSLESS,
        "none": EncodingOptions.NONE,
    }
    compression_map = {
        "zstd": CompressionOption.ZSTD,
        "lz4": CompressionOption.LZ4,
        "none": CompressionOption.NONE,
    }

    encoding_opt = encoding_map[args.encoding]
    compression_opt = compression_map[args.compression]
    resolution = args.resolution

    # Phase 2: Process messages and compress
    decoder_factory = DecoderFactory()
    stats_by_channel: dict[int, CompressionStats] = {}

    console.print(
        f"\n[bold blue]Phase 2: Analyzing compression "
        f"({args.encoding}/{args.compression})...[/bold blue]"
    )

    with input_file.open("rb") as f:
        for schema, channel, record in read_message(
            f,
            should_include=lambda _, s: (
                s is not None
                and s.name in {"sensor_msgs/msg/PointCloud2", "sensor_msgs/PointCloud2"}
            ),
        ):
            assert schema is not None
            # Initialize stats for this channel
            if record.channel_id not in stats_by_channel:
                stats_by_channel[record.channel_id] = CompressionStats(
                    topic=channel.topic,
                    schema=schema.name,
                )

            stats = stats_by_channel[record.channel_id]

            # Decode message
            decoder = decoder_factory.decoder_for("cdr", schema)
            if not decoder:
                console.print(f"[yellow]Warning: No decoder for schema: {schema.name}[/yellow]")
                continue

            try:
                msg = decoder(record.data)

                # Build encoding info
                info = _build_encoding_info(msg, encoding_opt, compression_opt, resolution)

                # Compress
                encoder = PointcloudEncoder(info)
                compressed = encoder.encode(bytes(msg.data))

                # Track stats
                stats.message_count += 1
                stats.original_bytes += len(msg.data)
                stats.compressed_bytes += len(compressed)

            except Exception as e:
                console.print(
                    f"[yellow]Warning: Failed to process message on {channel.topic}: {e}[/yellow]"
                )
                continue

    # Display results
    _display_results(stats_by_channel, encoding_opt, compression_opt, resolution)


def _build_encoding_info(
    msg: object,
    encoding_opt: EncodingOptions,
    compression_opt: CompressionOption,
    resolution: float,
) -> EncodingInfo:
    """
    Build pureini EncodingInfo from a decoded ROS2 PointCloud2 message.

    Args:
        msg: Decoded PointCloud2 message
        encoding_opt: Pureini encoding option
        compression_opt: Pureini compression option
        resolution: Resolution for lossy float compression

    Returns:
        EncodingInfo for pureini encoder
    """
    info = EncodingInfo()
    info.width = msg.width  # type: ignore[attr-defined]
    info.height = msg.height  # type: ignore[attr-defined]
    info.point_step = msg.point_step  # type: ignore[attr-defined]
    info.encoding_opt = encoding_opt
    info.compression_opt = compression_opt

    info.fields = []
    for ros_field in msg.fields:  # type: ignore[attr-defined]
        # Map ROS2 PointField datatype to pureini FieldType (1:1 mapping!)
        field = PointField(
            name=ros_field.name,
            offset=ros_field.offset,
            type=FieldType(ros_field.datatype),
            # Only apply resolution to FLOAT32 fields in lossy mode
            resolution=resolution if ros_field.datatype == 7 else None,
        )
        info.fields.append(field)

    return info


def _display_results(
    stats_by_channel: dict[int, CompressionStats],
    encoding_opt: EncodingOptions,
    compression_opt: CompressionOption,
    resolution: float,
) -> None:
    """
    Display compression statistics in a rich table.

    Args:
        stats_by_channel: Compression statistics per channel
        encoding_opt: Encoding option used
        compression_opt: Compression option used
        resolution: Resolution used for lossy float compression
    """
    console.print()
    table = Table(title="PointCloud2 Compression Analysis", show_lines=True)
    table.add_column("Topic", style="cyan", no_wrap=True)
    table.add_column("Messages", justify="right", style="green")
    table.add_column("Original Size", justify="right", style="yellow")
    table.add_column("Compressed Size", justify="right", style="yellow")
    table.add_column("Ratio", justify="right", style="magenta")
    table.add_column("Saved", justify="right", style="green")

    total_original = 0
    total_compressed = 0
    total_messages = 0

    for stats in stats_by_channel.values():
        table.add_row(
            stats.topic,
            f"{stats.message_count:,}",
            bytes_to_human(stats.original_bytes),
            bytes_to_human(stats.compressed_bytes),
            f"{stats.compression_ratio:.2f}x",
            bytes_to_human(stats.bytes_saved),
        )
        total_original += stats.original_bytes
        total_compressed += stats.compressed_bytes
        total_messages += stats.message_count

    # Add totals row
    table.add_section()
    total_ratio = total_original / total_compressed if total_compressed > 0 else 0.0
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total_messages:,}[/bold]",
        f"[bold]{bytes_to_human(total_original)}[/bold]",
        f"[bold]{bytes_to_human(total_compressed)}[/bold]",
        f"[bold]{total_ratio:.2f}x[/bold]",
        f"[bold]{bytes_to_human(total_original - total_compressed)}[/bold]",
    )

    console.print(table)

    # Display settings
    settings_table = Table.grid(padding=(0, 1))
    settings_table.add_column(style="bold blue")
    settings_table.add_column()
    settings_table.add_row("Encoding:", f"[yellow]{encoding_opt.name}[/yellow]")
    settings_table.add_row("Compression:", f"[yellow]{compression_opt.name}[/yellow]")
    if encoding_opt == EncodingOptions.LOSSY:
        settings_table.add_row("Resolution:", f"[yellow]{resolution}[/yellow]")

    console.print()
    console.print(settings_table)


def main() -> None:
    """Add the ros2compress command parser to the subparsers."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect all PointCloud2 topics in an MCAP file"
            " and report compression statistics using pureini"
        ),
    )

    parser.add_argument("file", help="Path to MCAP file", type=str)

    parser.add_argument(
        "--encoding",
        choices=["lossy", "lossless", "none"],
        default="lossy",
        help="Encoding option (default: lossy)",
    )

    parser.add_argument(
        "--compression",
        choices=["zstd", "lz4", "none"],
        default="zstd",
        help="Compression algorithm (default: zstd)",
    )

    parser.add_argument(
        "--resolution",
        type=float,
        default=0.01,
        help="Resolution for lossy float compression (default: 0.01)",
    )

    args = parser.parse_args()
    handle_command(args)


if __name__ == "__main__":
    main()
