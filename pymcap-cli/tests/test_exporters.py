"""End-to-end tests for the pluggable Exporter pipeline.

Exercises each format exporter against the prebuilt
``tests/fixtures/image_compressed.mcap`` plus a synthetic ROS2 fixture for
non-image formats.
"""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from pymcap_cli.exporters import run_export
from pymcap_cli.exporters.csv_exporter import CsvExporter
from pymcap_cli.exporters.image_exporter import (
    ImageExporter,
    _format_to_extension,
    _resolve_raw_encoder,
    _supported_image_formats,
)
from pymcap_cli.exporters.json_exporter import JsonExporter
from small_mcap import CompressionType, McapWriter

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _make_pose_mcap(
    path: Path,
    *,
    num_messages: int = 5,
    log_time_ns: int | None = None,
) -> None:
    """Write a tiny ROS2 MCAP with a single ``/pose`` topic.

    Uses the ROS2 ``Pose`` schema with a ``cdr`` payload so the standard
    ``mcap_ros2_support_fast`` decoder picks it up.
    """
    schema = b"""# Geometry-style Pose
float64 x
float64 y
float64 z
"""

    buf = io.BytesIO()
    writer = McapWriter(buf, chunk_size=64 * 1024, compression=CompressionType.NONE)
    writer.start(profile="ros2", library="pymcap-cli-tests")
    writer.add_schema(schema_id=1, name="test_msgs/Pose", encoding="ros2msg", data=schema)
    writer.add_channel(channel_id=1, topic="/pose", message_encoding="cdr", schema_id=1)
    # CDR encapsulation header (4 bytes: little-endian, no options) + 3 x float64.
    cdr_header = b"\x00\x01\x00\x00"
    for i in range(num_messages):
        timestamp = i * 1_000_000_000 if log_time_ns is None else log_time_ns
        payload = cdr_header + struct.pack("<ddd", float(i), float(i + 1), float(i + 2))
        writer.add_message(
            channel_id=1,
            log_time=timestamp,
            publish_time=timestamp,
            data=payload,
        )
    writer.finish()
    path.write_bytes(buf.getvalue())


def _make_same_topic_pose_mcap(path: Path) -> None:
    """Write two channels with the same topic so exporter dispatch is channel-safe."""
    schema = b"""# Geometry-style Pose
float64 x
float64 y
float64 z
"""

    buf = io.BytesIO()
    writer = McapWriter(buf, chunk_size=64 * 1024, compression=CompressionType.NONE)
    writer.start(profile="ros2", library="pymcap-cli-tests")
    writer.add_schema(schema_id=1, name="test_msgs/Pose", encoding="ros2msg", data=schema)
    writer.add_channel(channel_id=1, topic="/pose", message_encoding="cdr", schema_id=1)
    writer.add_channel(channel_id=2, topic="/pose", message_encoding="cdr", schema_id=1)

    cdr_header = b"\x00\x01\x00\x00"
    for channel_id, x in ((1, 1.0), (2, 2.0)):
        payload = cdr_header + struct.pack("<ddd", x, x + 1.0, x + 2.0)
        writer.add_message(
            channel_id=channel_id,
            log_time=channel_id,
            publish_time=channel_id,
            data=payload,
        )
    writer.finish()
    path.write_bytes(buf.getvalue())


def test_csv_exporter_writes_one_file_per_topic(tmp_path):
    src = tmp_path / "src.mcap"
    _make_pose_mcap(src, num_messages=3)

    out = tmp_path / "out"
    rc = run_export(file=str(src), output=out, exporter=CsvExporter())
    assert rc == 0

    csvs = list(out.glob("*.csv"))
    assert len(csvs) == 1
    text = csvs[0].read_text()
    lines = text.strip().splitlines()
    assert len(lines) == 4  # header + 3 rows
    assert "x" in lines[0]
    assert "y" in lines[0]
    assert "z" in lines[0]
    assert "_log_time_ns" in lines[0]


def test_json_exporter_ndjson_line_count_matches_messages(tmp_path):
    src = tmp_path / "src.mcap"
    _make_pose_mcap(src, num_messages=4)

    out = tmp_path / "out"
    rc = run_export(file=str(src), output=out, exporter=JsonExporter())
    assert rc == 0

    ndjson_files = list(out.glob("*.ndjson"))
    assert len(ndjson_files) == 1
    lines = ndjson_files[0].read_text().strip().splitlines()
    assert len(lines) == 4
    record = json.loads(lines[0])
    assert "_log_time_ns" in record
    assert "data" in record


def test_json_exporter_per_message_writes_one_file_per_message(tmp_path):
    src = tmp_path / "src.mcap"
    _make_pose_mcap(src, num_messages=3)

    out = tmp_path / "out"
    rc = run_export(file=str(src), output=out, exporter=JsonExporter(per_message=True))
    assert rc == 0

    per_topic_dir = out / "pose"
    assert per_topic_dir.is_dir()
    files = list(per_topic_dir.glob("*.json"))
    assert len(files) == 3


def test_json_exporter_per_message_disambiguates_duplicate_log_times(tmp_path):
    src = tmp_path / "src.mcap"
    _make_pose_mcap(src, num_messages=3, log_time_ns=42)

    out = tmp_path / "out"
    rc = run_export(
        file=str(src),
        output=out,
        exporter=JsonExporter(per_message=True),
        num_workers=1,
    )
    assert rc == 0

    files = sorted(p.name for p in (out / "pose").glob("*.json"))
    assert files == ["42.json", "42_000001.json", "42_000002.json"]


def test_json_exporter_per_message_force_cleans_topic_dir(tmp_path):
    src = tmp_path / "src.mcap"
    _make_pose_mcap(src, num_messages=3)

    out = tmp_path / "out"
    assert run_export(file=str(src), output=out, exporter=JsonExporter(per_message=True)) == 0
    assert len(list((out / "pose").glob("*.json"))) == 3

    _make_pose_mcap(src, num_messages=1)
    rc = run_export(
        file=str(src),
        output=out,
        exporter=JsonExporter(per_message=True),
        force=True,
    )
    assert rc == 0

    assert [p.name for p in (out / "pose").glob("*.json")] == ["0.json"]


def test_json_exporter_splits_same_topic_by_channel(tmp_path):
    src = tmp_path / "src.mcap"
    _make_same_topic_pose_mcap(src)

    out = tmp_path / "out"
    rc = run_export(file=str(src), output=out, exporter=JsonExporter(), num_workers=1)
    assert rc == 0

    files = sorted(out.glob("*.ndjson"))
    assert [p.name for p in files] == ["pose.ndjson", "pose_2.ndjson"]
    assert [len(p.read_text().strip().splitlines()) for p in files] == [1, 1]


@pytest.mark.skipif(
    not (FIXTURE_DIR / "image_compressed.mcap").exists(),
    reason="image_compressed.mcap fixture missing",
)
def test_image_exporter_passthrough_compressed(tmp_path):
    src = FIXTURE_DIR / "image_compressed.mcap"
    out = tmp_path / "out"
    rc = run_export(file=str(src), output=out, exporter=ImageExporter())
    assert rc == 0

    topic_dirs = [p for p in out.iterdir() if p.is_dir()]
    assert len(topic_dirs) == 1
    files = list(topic_dirs[0].iterdir())
    assert len(files) > 0
    # Default output format is native, so we keep original compressed extensions.
    assert all(p.suffix in {".jpg", ".png", ".jxl", ".webp", ".bin"} for p in files)


@pytest.mark.skipif(
    not (FIXTURE_DIR / "image_compressed.mcap").exists(),
    reason="image_compressed.mcap fixture missing",
)
def test_image_exporter_converts_compressed_to_target_format(tmp_path):
    src = FIXTURE_DIR / "image_compressed.mcap"
    out = tmp_path / "out"
    rc = run_export(file=str(src), output=out, exporter=ImageExporter(output_format="jpeg"))
    assert rc == 0

    topic_dirs = [p for p in out.iterdir() if p.is_dir()]
    assert len(topic_dirs) == 1
    files = list(topic_dirs[0].iterdir())
    assert len(files) > 0
    assert all(p.suffix == ".jpg" for p in files)


def _fake_imagecodecs() -> SimpleNamespace:
    return SimpleNamespace(
        jpeg_encode=lambda _data: b"jpeg",
        jpegxl_encode=lambda _data: b"jxl",
        png_encode=lambda _data: b"png",
        zlib_encode=lambda _data: b"zlib",
    )


def test_supported_image_formats_discovers_encoder_names() -> None:
    module = _fake_imagecodecs()
    assert _supported_image_formats(module) == frozenset({"jpeg", "jpg", "jpegxl", "jxl", "png"})


def test_resolve_raw_encoder_accepts_imagecodecs_aliases() -> None:
    module = _fake_imagecodecs()
    extension, encode = _resolve_raw_encoder("JPG", imagecodecs_module=module)
    assert extension == "jpg"
    assert encode is module.jpeg_encode

    extension, encode = _resolve_raw_encoder("jxl", imagecodecs_module=module)
    assert extension == "jxl"
    assert encode is module.jpegxl_encode


def test_resolve_raw_encoder_rejects_unsupported_formats() -> None:
    module = _fake_imagecodecs()
    with pytest.raises(TypeError, match="is not supported"):
        _resolve_raw_encoder("avif", imagecodecs_module=module)


def test_resolve_raw_encoder_rejects_non_image_codecs() -> None:
    module = _fake_imagecodecs()
    with pytest.raises(TypeError, match="is not supported"):
        _resolve_raw_encoder("zlib", imagecodecs_module=module)


def test_compressed_format_to_extension_prefers_jpegxl_over_jpeg() -> None:
    assert _format_to_extension("jpegxl") == "jxl"
    assert _format_to_extension("rgb8; jpeg compressed bgr8") == "jpg"


def test_csv_exporter_skips_blob_schemas_by_default():
    """An ``Image`` schema is in DEFAULT_BLOB_SCHEMAS — exporter should refuse it."""

    class _Schema:
        name = "sensor_msgs/msg/Image"

    assert CsvExporter().accepts(_Schema()) is False  # type: ignore[arg-type]
    assert CsvExporter(include_blobs=True).accepts(_Schema()) is True  # type: ignore[arg-type]
