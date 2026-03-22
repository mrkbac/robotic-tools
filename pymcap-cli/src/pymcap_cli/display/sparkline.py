"""Sparkline renderable for categorical time-series data."""

from bisect import bisect_right

from rich.text import Text

# Unicode block characters ordered by height
BLOCKS = " ▁▂▃▄▅▆▇█"


def sparkline(
    changes: list[tuple[int, int]],
    end_ns: int,
    *,
    char_map: dict[int, str],
    style_map: dict[int, str],
    width: int = 20,
    reducer: str = "max",
) -> Text:
    """Build a colored sparkline from timestamped categorical values.

    Parameters
    ----------
    changes
        List of (timestamp_ns, value) tuples representing state transitions.
        Must be sorted by timestamp.
    end_ns
        End timestamp in nanoseconds (last known data point).
    char_map
        Maps each value to a Unicode block character.
    style_map
        Maps each value to a Rich style string.
    width
        Number of characters in the output.
    reducer
        How to combine multiple values in one bucket: "max" or "min".
    """
    if not changes:
        return Text("")

    if len(changes) == 1:
        val = changes[0][1]
        char = char_map.get(val, "▁")
        style = style_map.get(val, "dim")
        return Text(char * width, style=style)

    t_start = changes[0][0]
    if end_ns <= t_start:
        return Text("")

    span = end_ns - t_start
    reduce_fn = max if reducer == "max" else min
    timestamps = [ts for ts, _ in changes]

    result = Text()
    for i in range(width):
        bucket_start = t_start + (span * i) // width
        bucket_end = t_start + (span * (i + 1)) // width

        # Active value: last change at or before bucket_start.
        idx = bisect_right(timestamps, bucket_start) - 1
        active = changes[max(idx, 0)][1]
        worst = active

        # Reduce over changes within (bucket_start, bucket_end].
        start_idx = bisect_right(timestamps, bucket_start)
        end_idx = bisect_right(timestamps, bucket_end)
        for j in range(start_idx, end_idx):
            worst = reduce_fn(worst, changes[j][1])

        char = char_map.get(worst, "▁")
        style = style_map.get(worst, "dim")
        result.append(char, style=style)

    return result
