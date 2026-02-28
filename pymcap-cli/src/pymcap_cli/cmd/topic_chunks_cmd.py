"""Topic-chunks command - show which topics appear in which chunks."""

from collections import defaultdict

from rich.console import Console
from rich.table import Table

from pymcap_cli.input_handler import open_input
from pymcap_cli.utils import read_or_rebuild_info

console = Console()


def topic_chunks(file: str) -> int:
    """Show which topics appear in which chunks.

    Displays a table of topics sorted by the number of chunks they appear in,
    along with the percentage of total chunks.

    Parameters
    ----------
    file
        Path to the MCAP file (local file or HTTP/HTTPS URL).
    """
    with open_input(file) as (f, file_size):
        info = read_or_rebuild_info(f, file_size)

    summary = info.summary

    if not summary.chunk_indexes:
        console.print("[yellow]No chunks found[/yellow]")
        return 0

    total_chunks = len(summary.chunk_indexes)
    topic_chunk_counts: dict[str, int] = defaultdict(int)

    for chunk_index in summary.chunk_indexes:
        for channel_id in chunk_index.message_index_offsets:
            channel = summary.channels.get(channel_id)
            if channel is not None:
                topic_chunk_counts[channel.topic] += 1

    table = Table(title="Topic Chunks Table")
    table.add_column("Topic", style="bold white")
    table.add_column("Chunks", style="cyan", justify="right")
    table.add_column("% Chunks", justify="right")

    for topic, count in sorted(topic_chunk_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_chunks * 100
        table.add_row(topic, str(count), f"{pct:.0f}%")

    console.print(table)
    console.print(f"Total: {total_chunks} chunks")

    return 0
