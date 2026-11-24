"""Per-recording catalog index (topics, time bounds, GPS tracks, preview frames), cached on disk.

Sampling mirrors Digitalis's local-folder catalog so served and local clips look the same.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any

from mcap_ros2_support_fast.decoder import DecoderFactory as Ros2DecoderFactory
from rich.markup import escape
from small_mcap import JSONDecoderFactory, McapFile

from pymcap_cli.cmd.bridge._shared import console
from pymcap_cli.exporters._common import normalize_schema_name
from pymcap_cli.exporters.image_exporter import compressed_image_to_jpeg_thumbnail

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from small_mcap import Channel, ChunkIndex, Schema

    from pymcap_cli.cmd.bridge._library import JsonValue

SAMPLE_INTERVAL_NS = 2_000_000_000
MIN_SAMPLES = 8
MAX_SAMPLES = 120
MAX_IMAGE_TOPICS = 2
PREVIEW_WIDTH = 240
_SANE_START_NS = 946_684_800 * 1_000_000_000  # 2000-01-01
_DAY_NS = 86_400 * 1_000_000_000
_NAVSAT_FIX = "sensor_msgs/NavSatFix"
_FIX_SCHEMAS = frozenset({_NAVSAT_FIX, "foxglove_msgs/LocationFix", "foxglove.LocationFix"})
_IMAGE_SCHEMAS = frozenset(
    {"sensor_msgs/CompressedImage", "foxglove_msgs/CompressedImage", "foxglove.CompressedImage"}
)
_STATUS_NO_FIX = -1
_CACHE_FORMAT = "catalog-index-v2"


def sample_count(duration_ns: int) -> int:
    """One sample per ``SAMPLE_INTERVAL_NS`` of recording, within ``[MIN, MAX]_SAMPLES``."""
    return min(MAX_SAMPLES, max(MIN_SAMPLES, math.floor(duration_ns / SAMPLE_INTERVAL_NS + 0.5)))


def pick_chunks(
    chunk_indexes: Sequence[ChunkIndex], channel_ids: Sequence[int], count: int
) -> list[ChunkIndex]:
    """Up to ``count`` chunks spread evenly over the chunks holding any of ``channel_ids``.

    When the evenly spaced chunk lacks some channels, a neighbor within half the spacing that
    has more of them is taken instead, so GPS and camera samples share a chunk where possible.
    """

    def coverage(chunk: ChunkIndex) -> int:
        return sum(channel_id in chunk.message_index_offsets for channel_id in channel_ids)

    chunks = sorted(
        (chunk for chunk in chunk_indexes if coverage(chunk) > 0),
        key=lambda chunk: chunk.message_start_time,
    )
    n = min(count, len(chunks))
    if n == 0:
        return []
    spacing = len(chunks) / n
    radius = math.floor(spacing / 2)
    picked: list[int] = []
    for i in range(n):
        target = math.floor(i * spacing)
        best = target
        distance = 1
        while distance <= radius and coverage(chunks[best]) < len(channel_ids):
            for candidate in (target + distance, target - distance):
                if 0 <= candidate < len(chunks) and coverage(chunks[candidate]) > coverage(
                    chunks[best]
                ):
                    best = candidate
            distance += 1
        if best not in picked:
            picked.append(best)
    return [chunks[i] for i in picked]


def fix_to_point(
    latitude: float, longitude: float, status: int | None
) -> tuple[float, float] | None:
    """A ``(lon, lat)`` point rounded to ~1 m, or None for an unusable fix."""
    if (
        not math.isfinite(latitude)
        or not math.isfinite(longitude)
        or abs(latitude) > 90
        or abs(longitude) > 180
        or (latitude == 0 and longitude == 0)
        or status == _STATUS_NO_FIX
    ):
        return None
    return round(longitude, 5), round(latitude, 5)


def ns_to_rfc3339(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds)) + f".{nanoseconds:09d}Z"


class _Decoders:
    """ROS 2 CDR messages decode to objects, JSON messages (foxglove schemas) to dicts."""

    def __init__(self) -> None:
        self._factories = (Ros2DecoderFactory(), JSONDecoderFactory())

    def decoder_for(
        self, message_encoding: str, schema: Schema | None
    ) -> Callable[[bytes | memoryview], Any] | None:
        for factory in self._factories:
            if (decoder := factory.decoder_for(message_encoding, schema)) is not None:
                return decoder
        return None


def _fix_to_point(decoded: Any, is_navsat_fix: bool) -> tuple[float, float] | None:
    if isinstance(decoded, dict):
        latitude, longitude = decoded.get("latitude"), decoded.get("longitude")
        status = decoded.get("status")
        status = status.get("status") if isinstance(status, dict) else None
    else:
        latitude, longitude = decoded.latitude, decoded.longitude
        status = decoded.status.status if is_navsat_fix else None
    if not isinstance(latitude, int | float) or not isinstance(longitude, int | float):
        return None
    return fix_to_point(float(latitude), float(longitude), status)


def _frame_data(decoded: Any) -> bytes:
    if not isinstance(decoded, dict):
        return bytes(decoded.data)
    data = decoded.get("data")
    try:
        return base64.b64decode(data) if isinstance(data, str) else b""
    except binascii.Error:
        return b""


def _schema_name(schema: Schema | None) -> str:
    return "" if schema is None else normalize_schema_name(schema.name)


def _time_bounds(recording: McapFile, now_ns: int) -> tuple[int, int] | None:
    """Message time bounds, ignoring outlier chunk stamps (before 2000 or in the future)."""
    summary = recording.summary
    assert summary is not None
    stamps = [
        stamp
        for chunk in summary.chunk_indexes
        for stamp in (chunk.message_start_time, chunk.message_end_time)
    ]
    statistics = summary.statistics
    if not stamps and statistics is not None and statistics.message_count:
        stamps = [statistics.message_start_time, statistics.message_end_time]
    sane = [stamp for stamp in stamps if _SANE_START_NS <= stamp <= now_ns + _DAY_NS]
    bounds = sane or stamps
    return (min(bounds), max(bounds)) if bounds else None


def _topics(recording: McapFile) -> list[JsonValue]:
    summary = recording.summary
    assert summary is not None
    counts = {} if summary.statistics is None else summary.statistics.channel_message_counts
    topics: dict[str, dict[str, JsonValue]] = {}
    for channel in summary.channels.values():
        schema = summary.schemas.get(channel.schema_id)
        schema_name = "" if schema is None else schema.name
        topic = topics.setdefault(channel.topic, {"name": channel.topic, "schema": schema_name})
        if topic["schema"] != schema_name:
            topic["schema"] = "(multiple)"
        if channel.id in counts:
            messages = topic.get("messages", 0)
            assert isinstance(messages, int)
            topic["messages"] = messages + counts[channel.id]
    return [topics[name] for name in sorted(topics)]


def _sample(
    recording: McapFile, out_dir: Path, url_prefix: str, origin_ns: int | None
) -> dict[str, JsonValue]:
    """GPS tracks and preview frames from evenly picked chunks, one message per topic each."""
    summary = recording.summary
    assert summary is not None
    decoders = _Decoders()
    counts = {} if summary.statistics is None else summary.statistics.channel_message_counts

    schema_by_channel = {
        channel.id: _schema_name(summary.schemas.get(channel.schema_id))
        for channel in summary.channels.values()
    }

    def is_sampled(channel: Channel) -> bool:
        if schema_by_channel[channel.id] not in _FIX_SCHEMAS | _IMAGE_SCHEMAS:
            return False
        schema = summary.schemas.get(channel.schema_id)
        return decoders.decoder_for(channel.message_encoding, schema) is not None

    channels = [channel for channel in summary.channels.values() if is_sampled(channel)]
    gps = [channel for channel in channels if schema_by_channel[channel.id] in _FIX_SCHEMAS]
    image_messages: dict[str, int] = {}
    for channel in channels:
        if schema_by_channel[channel.id] in _IMAGE_SCHEMAS:
            image_messages[channel.topic] = image_messages.get(channel.topic, 0) + counts.get(
                channel.id, 0
            )
    cameras = sorted(image_messages, key=lambda topic: (-image_messages[topic], topic))[
        :MAX_IMAGE_TOPICS
    ]
    sampled = [*gps, *(channel for channel in channels if channel.topic in cameras)]
    channel_ids = [channel.id for channel in sampled]
    topics = {channel.topic for channel in sampled}
    relevant = [
        chunk
        for chunk in summary.chunk_indexes
        if any(channel_id in chunk.message_index_offsets for channel_id in channel_ids)
    ]
    if not relevant:
        return {}
    start_ns = min(chunk.message_start_time for chunk in relevant)
    end_ns = max(chunk.message_end_time for chunk in relevant)
    chunks = pick_chunks(relevant, channel_ids, sample_count(end_ns - start_ns))

    points: dict[str, list[list[float]]] = {channel.topic: [] for channel in gps}
    frames: dict[str, list[tuple[int, bytes]]] = {topic: [] for topic in cameras}
    for chunk in chunks:
        done: set[str] = set()
        for schema, channel, message in recording.read_message(
            lambda channel, _schema: channel.id in channel_ids,
            start_time_ns=chunk.message_start_time,
            end_time_ns=chunk.message_end_time + 1,
        ):
            if channel.topic in done:
                continue
            decoder = decoders.decoder_for(channel.message_encoding, schema)
            assert decoder is not None
            decoded = decoder(message.data)
            if channel.topic in points:
                point = _fix_to_point(decoded, _schema_name(schema) == _NAVSAT_FIX)
                if point is None:
                    continue  # a later fix in this chunk may be valid
                track = points[channel.topic]
                if not track or track[-1] != list(point):
                    track.append(list(point))
            else:
                data = _frame_data(decoded)
                if not data:
                    continue
                frames[channel.topic].append((message.log_time, data))
            done.add(channel.topic)
            if len(done) == len(topics):
                break

    fields: dict[str, JsonValue] = {}
    tracks = [{"topic": topic, "points": points[topic]} for topic in sorted(points)]
    if tracks := [track for track in tracks if track["points"]]:
        fields["tracks"] = tracks
    previews: list[JsonValue] = []
    for camera_index, topic in enumerate(cameras):
        rendered: list[JsonValue] = []
        for log_time_ns, data in frames[topic]:
            try:
                jpeg = compressed_image_to_jpeg_thumbnail(data, PREVIEW_WIDTH)
            except (OSError, ValueError):
                continue  # frames that fail to decode are left out
            name = f"{camera_index}/{len(rendered):03d}.jpg"
            (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (out_dir / name).write_bytes(jpeg)
            origin = frames[topic][0][0] if origin_ns is None else origin_ns
            rendered.append(
                {"offset": round((log_time_ns - origin) / 1e9, 3), "url": f"{url_prefix}/{name}"}
            )
        if rendered:
            previews.append({"topic": topic, "frames": rendered})
    if previews:
        fields["previews"] = previews
        first = previews[0]
        assert isinstance(first, dict)
        first_frames = first["frames"]
        assert isinstance(first_frames, list)
        thumbnail = first_frames[len(first_frames) // 4]
        assert isinstance(thumbnail, dict)
        fields["thumbnail"] = thumbnail["url"]
    return fields


def index_recording(
    path: Path, out_dir: Path, url_prefix: str, now_ns: int
) -> dict[str, JsonValue]:
    """Catalog fields for one MCAP; preview JPEGs are written below ``out_dir``."""
    with McapFile.open(path) as recording:
        summary = recording.summary
        if summary is None:
            return {}
        fields: dict[str, JsonValue] = {"topics": _topics(recording)}
        if summary.statistics is not None:
            fields["messages"] = summary.statistics.message_count
        bounds = _time_bounds(recording, now_ns)
        if bounds is not None:
            fields["start"] = ns_to_rfc3339(bounds[0])
            fields["end"] = ns_to_rfc3339(bounds[1])
        fields.update(
            _sample(recording, out_dir, url_prefix, None if bounds is None else bounds[0])
        )
        return fields


def _has_pillow() -> bool:
    try:
        compressed_image_to_jpeg_thumbnail(b"", PREVIEW_WIDTH)
    except ImportError:
        return False
    except (OSError, ValueError):
        return True
    return True


class CatalogIndex:
    """Index recordings in the background and cache each one by path, size and mtime."""

    def __init__(self, cache_dir: Path, *, workers: int = 4) -> None:
        self.cache_dir = cache_dir
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="catalog")
        self._lock = threading.Lock()
        self._ready: dict[str, dict[str, JsonValue]] = {}
        self._pending: dict[str, Future[None]] = {}
        self._failed: set[str] = set()
        self._done = 0
        # Previews depend on Pillow, so an index made without it is redone once it is installed.
        has_pillow = _has_pillow()
        self._format = f"{_CACHE_FORMAT}-{'previews' if has_pillow else 'no-previews'}"
        if not has_pillow:
            console.print("[yellow]Catalog previews need Pillow:[/] use pymcap-cli[bridge,image]")

    def get(self, path: Path, label: str) -> dict[str, JsonValue] | None:
        """Cached fields for ``path``, or None while it is (re)indexed in the background."""
        stat = path.stat()
        key = hashlib.sha256(
            f"{self._format}\0{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
        ).hexdigest()[:32]
        with self._lock:
            if key in self._ready:
                return self._ready[key]
            if key in self._pending or key in self._failed:
                return None
        summary_path = self.cache_dir / key / "summary.json"
        if summary_path.is_file():
            fields = json.loads(summary_path.read_text())
            with self._lock:
                self._ready[key] = fields
            return fields
        with self._lock:
            if key not in self._pending:
                self._pending[key] = self._executor.submit(self._index, path, key, label)
        return None

    def wait(self) -> None:
        """Block until every scheduled recording is indexed."""
        with self._lock:
            futures = list(self._pending.values())
        wait(futures)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _index(self, path: Path, key: str, label: str) -> None:
        started = time.monotonic()
        tmp_dir = self.cache_dir / f"{key}.tmp-{threading.get_ident()}"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        try:
            fields = index_recording(path, tmp_dir, f"previews/{key}", time.time_ns())
            (tmp_dir / "summary.json").write_text(json.dumps(fields))
            try:
                tmp_dir.replace(self.cache_dir / key)
            except OSError:
                shutil.rmtree(tmp_dir, ignore_errors=True)  # another server indexed it first
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the catalog
            shutil.rmtree(tmp_dir, ignore_errors=True)
            with self._lock:
                self._pending.pop(key, None)
                self._failed.add(key)
            console.print(
                f"[yellow]Catalog index failed[/] {escape(label)}: {escape(str(exc))}",
                soft_wrap=True,
            )
            return
        with self._lock:
            self._ready[key] = fields
            self._pending.pop(key, None)
            self._done += 1
            done, total = self._done, self._done + len(self._pending)
        console.print(
            f"[dim]Catalog indexed {done}/{total}[/] {escape(label)} "
            f"[dim]({time.monotonic() - started:.1f} s)[/]",
            soft_wrap=True,
        )
