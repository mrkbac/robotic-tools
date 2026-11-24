"""Tests for the cached per-recording catalog index."""

from __future__ import annotations

import base64
import io
import json
import math
from typing import TYPE_CHECKING

import pymcap_cli.cmd.bridge._catalog_index as catalog_index_module
import pytest
from mcap_ros2_support_fast.writer import ROS2EncoderFactory
from PIL import Image
from pymcap_cli.cmd.bridge._catalog_index import (
    CatalogIndex,
    fix_to_point,
    pick_chunks,
    sample_count,
)
from small_mcap import ChunkIndex, CompressionType, McapWriter, Schema

if TYPE_CHECKING:
    from pathlib import Path

_TIME = """
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""
_NAVSATFIX = (
    """std_msgs/Header header
sensor_msgs/NavSatStatus status
float64 latitude
float64 longitude
float64 altitude
float64[9] position_covariance
uint8 position_covariance_type
================================================================================
MSG: sensor_msgs/NavSatStatus
int8 status
uint16 service
"""
    + _TIME
)
_COMPRESSED_IMAGE = "std_msgs/Header header\nstring format\nuint8[] data\n" + _TIME
_START_NS = 1_700_000_000_000_000_000
_SECOND_NS = 1_000_000_000


def _jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def _write_recording(path: Path, duration_s: int) -> None:
    """GPS at 1 Hz (every 5th fix unusable), cameras at 2 Hz, 2 Hz and 0.1 Hz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    schemas = {
        1: Schema(
            id=1, name="sensor_msgs/msg/NavSatFix", encoding="ros2msg", data=_NAVSATFIX.encode()
        ),
        2: Schema(
            id=2,
            name="sensor_msgs/msg/CompressedImage",
            encoding="ros2msg",
            data=_COMPRESSED_IMAGE.encode(),
        ),
    }
    encoders = ROS2EncoderFactory()
    frame = _jpeg(480, 240)
    with path.open("wb") as stream:
        writer = McapWriter(stream, chunk_size=4096, compression=CompressionType.NONE)
        writer.start(profile="ros2")
        for schema in schemas.values():
            writer.add_schema(schema.id, schema.name, schema.encoding, schema.data)
        writer.add_channel(1, "/gps/fix", "cdr", 1)
        writer.add_channel(2, "/camera/front", "cdr", 2)
        writer.add_channel(3, "/camera/rear", "cdr", 2)
        writer.add_channel(4, "/camera/front/throttled", "cdr", 2)
        encode_fix = encoders.encoder_for(schemas[1])
        encode_image = encoders.encoder_for(schemas[2])
        assert encode_fix is not None
        assert encode_image is not None
        for tick in range(duration_s * 2):
            log_time = _START_NS + tick * _SECOND_NS // 2
            if tick % 2 == 0:
                second = tick // 2
                unusable = second % 5 == 4
                fix = {
                    "status": {"status": -1 if unusable else 0, "service": 1},
                    "latitude": 52.5 + second * 0.0001,
                    "longitude": 13.4 + second * 0.0001,
                    "position_covariance": [0.0] * 9,
                }
                writer.add_message(1, log_time, bytes(encode_fix(fix)), publish_time=log_time)
            image = bytes(encode_image({"format": "jpeg", "data": frame}))
            writer.add_message(2, log_time, image, publish_time=log_time)
            writer.add_message(3, log_time, image, publish_time=log_time)
            if tick % 20 == 0:
                writer.add_message(4, log_time, image, publish_time=log_time)
        writer.finish()


def _chunk(index: int, channel_ids: list[int]) -> ChunkIndex:
    return ChunkIndex(
        message_start_time=index * 10,
        message_end_time=index * 10 + 9,
        chunk_start_offset=index * 100,
        chunk_length=100,
        message_index_offsets=dict.fromkeys(channel_ids, 0),
        message_index_length=0,
        compression="",
        compressed_size=0,
        uncompressed_size=0,
    )


@pytest.mark.parametrize(
    ("duration_s", "expected"),
    [(0, 8), (15, 8), (17, 9), (20, 10), (61, 31), (240, 120), (3600, 120)],
)
def test_sample_count_one_per_two_seconds_within_bounds(duration_s: int, expected: int) -> None:
    assert sample_count(duration_s * 1_000_000_000) == expected


def test_pick_chunks_spreads_evenly_and_prefers_neighbors_with_more_channels() -> None:
    # Channel 1 is in every chunk, channel 2 only in odd ones.
    chunks = [_chunk(i, [1, 2] if i % 2 else [1]) for i in range(10)]

    picked = pick_chunks(chunks, [1, 2], 5)

    assert [chunk.chunk_start_offset // 100 for chunk in picked] == [1, 3, 5, 7, 9]


def test_pick_chunks_takes_every_chunk_when_there_are_few() -> None:
    chunks = [_chunk(i, [1]) for i in (2, 0, 1)]

    assert [chunk.chunk_start_offset // 100 for chunk in pick_chunks(chunks, [1], 8)] == [0, 1, 2]
    assert pick_chunks(chunks, [7], 8) == []


@pytest.mark.parametrize(
    ("latitude", "longitude", "status", "expected"),
    [
        (52.5000049, 13.4000051, 0, (13.40001, 52.5)),
        (-33.9, 151.2, None, (151.2, -33.9)),
        (52.5, 13.4, -1, None),
        (0.0, 0.0, 0, None),
        (math.nan, 13.4, 0, None),
        (52.5, math.inf, 0, None),
        (90.5, 13.4, 0, None),
        (52.5, -180.5, 0, None),
    ],
)
def test_fix_to_point_filters_unusable_fixes(
    latitude: float, longitude: float, status: int | None, expected: tuple[float, float] | None
) -> None:
    assert fix_to_point(latitude, longitude, status) == expected


def test_catalog_index_samples_tracks_and_previews(tmp_path: Path) -> None:
    recording = tmp_path / "example.mcap"
    _write_recording(recording, duration_s=60)
    index = CatalogIndex(tmp_path / "cache")

    assert index.get(recording, "example.mcap") is None
    index.wait()
    fields = index.get(recording, "example.mcap")

    assert fields is not None
    assert fields["topics"] == [
        {"name": "/camera/front", "schema": "sensor_msgs/msg/CompressedImage", "messages": 120},
        {
            "name": "/camera/front/throttled",
            "schema": "sensor_msgs/msg/CompressedImage",
            "messages": 6,
        },
        {"name": "/camera/rear", "schema": "sensor_msgs/msg/CompressedImage", "messages": 120},
        {"name": "/gps/fix", "schema": "sensor_msgs/msg/NavSatFix", "messages": 60},
    ]
    assert fields["messages"] == 306
    assert fields["start"] == "2023-11-14T22:13:20.000000000Z"
    assert fields["end"] == "2023-11-14T22:14:19.500000000Z"

    [track] = fields["tracks"]
    assert track["topic"] == "/gps/fix"
    assert 20 <= len(track["points"]) <= 30
    valid = {(round(13.4 + s * 0.0001, 5), round(52.5 + s * 0.0001, 5)) for s in range(60)}
    unusable = {(round(13.4 + s * 0.0001, 5), round(52.5 + s * 0.0001, 5)) for s in range(4, 60, 5)}
    assert {tuple(point) for point in track["points"]} <= valid - unusable

    previews = fields["previews"]
    assert [preview["topic"] for preview in previews] == ["/camera/front", "/camera/rear"]
    frames = previews[0]["frames"]
    assert 20 <= len(frames) <= 30
    offsets = [frame["offset"] for frame in frames]
    assert offsets == sorted(offsets)
    assert 0 <= offsets[0] < 3
    assert offsets[-1] <= 59.5
    assert fields["thumbnail"] == frames[len(frames) // 4]["url"]
    jpeg = (tmp_path / "cache" / frames[0]["url"].removeprefix("previews/")).read_bytes()
    assert Image.open(io.BytesIO(jpeg)).size == (240, 120)


def test_catalog_index_cache_hit_does_not_reopen_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = tmp_path / "example.mcap"
    _write_recording(recording, duration_s=20)
    first = CatalogIndex(tmp_path / "cache")
    first.get(recording, "example.mcap")
    first.wait()
    expected = first.get(recording, "example.mcap")

    def fail_open(*_args, **_kwargs):
        raise AssertionError("a cache hit must not read the recording")

    monkeypatch.setattr(catalog_index_module.McapFile, "open", fail_open)

    assert CatalogIndex(tmp_path / "cache").get(recording, "example.mcap") == expected
    summary = tmp_path / "cache" / expected["thumbnail"].removeprefix("previews/")
    assert json.loads((summary.parents[1] / "summary.json").read_text()) == expected


def test_catalog_index_reindexes_a_changed_file(tmp_path: Path) -> None:
    recording = tmp_path / "example.mcap"
    _write_recording(recording, duration_s=20)
    index = CatalogIndex(tmp_path / "cache")
    index.get(recording, "example.mcap")
    index.wait()

    _write_recording(recording, duration_s=30)

    assert index.get(recording, "example.mcap") is None
    index.wait()
    fields = index.get(recording, "example.mcap")
    assert fields is not None
    assert fields["messages"] == 30 + 60 + 60 + 3


def test_catalog_index_samples_json_encoded_foxglove_topics(tmp_path: Path) -> None:
    recording = tmp_path / "example.mcap"
    frame = base64.b64encode(_jpeg(320, 240)).decode()
    with recording.open("wb") as stream:
        writer = McapWriter(stream, chunk_size=2048, compression=CompressionType.NONE)
        writer.start()
        writer.add_schema(1, "foxglove.LocationFix", "jsonschema", b"{}")
        writer.add_schema(2, "foxglove.CompressedImage", "jsonschema", b"{}")
        writer.add_channel(1, "/gps/fix", "json", 1)
        writer.add_channel(2, "/camera/front", "json", 2)
        for second in range(20):
            log_time = _START_NS + second * _SECOND_NS
            fix = {"latitude": 48.1 + second * 0.001, "longitude": 11.5, "altitude": 500.0}
            image = {"format": "jpeg", "data": frame}
            writer.add_message(1, log_time, json.dumps(fix).encode(), publish_time=log_time)
            writer.add_message(2, log_time, json.dumps(image).encode(), publish_time=log_time)
        writer.finish()
    index = CatalogIndex(tmp_path / "cache")
    index.get(recording, "example.mcap")
    index.wait()

    fields = index.get(recording, "example.mcap")

    assert fields is not None
    [track] = fields["tracks"]
    assert track["points"][0] == [11.5, 48.1]
    assert len(track["points"]) >= 8
    [preview] = fields["previews"]
    assert preview["topic"] == "/camera/front"
    jpeg = (tmp_path / "cache" / preview["frames"][0]["url"].removeprefix("previews/")).read_bytes()
    assert Image.open(io.BytesIO(jpeg)).size == (240, 180)
