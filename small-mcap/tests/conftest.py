"""Shared pytest fixtures for small-mcap tests."""

import io
from pathlib import Path
from typing import Any

import pytest
from small_mcap import (
    Channel,
    Header,
    McapRecord,
    MessageEncoding,
    Schema,
    SchemaEncoding,
    stream_reader,
)

# Path to official MCAP conformance test files
CONFORMANCE_DATA = Path(__file__).parent.parent.parent / "data" / "conformance"
DEMO_MCAP = Path(__file__).parent.parent.parent / "mcap" / "testdata" / "mcap" / "demo.mcap"


@pytest.fixture
def temp_mcap_file(tmp_path: Path) -> Path:
    """Create a temporary MCAP file path."""
    return tmp_path / "test.mcap"


@pytest.fixture
def temp_mcap_stream() -> io.BytesIO:
    """Create a temporary in-memory byte stream."""
    return io.BytesIO()


@pytest.fixture
def sample_header() -> Header:
    """Create a sample MCAP header."""
    return Header(profile="test", library="small-mcap-test")


@pytest.fixture
def sample_schema() -> Schema:
    """Create a sample schema."""
    return Schema(
        id=1,
        name="test_schema",
        encoding=SchemaEncoding.PROTOBUF,
        data=b"message Test { int32 value = 1; }",
    )


@pytest.fixture
def sample_channel(sample_schema: Schema) -> Channel:
    """Create a sample channel."""
    return Channel(
        id=1,
        schema_id=sample_schema.id,
        topic="/test/topic",
        message_encoding=MessageEncoding.PROTOBUF,
        metadata={"key": "value"},
    )


@pytest.fixture
def sample_message_data() -> bytes:
    """Create sample message data."""
    return b"\x08\x2a"  # Protobuf encoded: value=42


@pytest.fixture
def multiple_messages() -> list[tuple[int, bytes, int]]:
    """Create multiple sample messages as (log_time, data, publish_time) tuples."""
    return [
        (1000000000, b"\x08\x01", 1000000000),  # value=1
        (2000000000, b"\x08\x02", 2000000000),  # value=2
        (3000000000, b"\x08\x03", 3000000000),  # value=3
        (4000000000, b"\x08\x04", 4000000000),  # value=4
        (5000000000, b"\x08\x05", 5000000000),  # value=5
    ]


@pytest.fixture
def reference_mcap_files():
    """Provide paths to reference MCAP files for conformance testing."""
    if not CONFORMANCE_DATA.exists():
        pytest.skip("Conformance test data not available")

    return {
        # Essential test files
        "minimal": CONFORMANCE_DATA / "NoData" / "NoData.mcap",
        "one_message": CONFORMANCE_DATA / "OneMessage" / "OneMessage.mcap",
        "schemaless": CONFORMANCE_DATA / "OneSchemalessMessage" / "OneSchemalessMessage.mcap",
        "ten_messages": CONFORMANCE_DATA / "TenMessages" / "TenMessages.mcap",
        "one_attachment": CONFORMANCE_DATA / "OneAttachment" / "OneAttachment.mcap",
        "one_metadata": CONFORMANCE_DATA / "OneMetadata" / "OneMetadata.mcap",
        # Demo file (if available)
        "demo": DEMO_MCAP if DEMO_MCAP.exists() else None,
    }


@pytest.fixture
def all_conformance_files():
    """Provide all conformance test MCAP files."""
    if not CONFORMANCE_DATA.exists():
        pytest.skip("Conformance test data not available")

    return list(CONFORMANCE_DATA.rglob("*.mcap"))


@pytest.fixture
def conformance_file_pairs():
    """Provide all conformance MCAP files paired with their JSON specs.

    Returns list of (mcap_path, json_path) tuples where both files exist.
    """
    if not CONFORMANCE_DATA.exists():
        pytest.skip("Conformance test data not available")

    pairs = []
    for mcap_file in CONFORMANCE_DATA.rglob("*.mcap"):
        json_file = mcap_file.with_suffix(".json")
        if json_file.exists():
            pairs.append((mcap_file, json_file))

    return pairs


def normalize_value(value: Any) -> Any:
    """Normalize a value for conformance comparison.

    Follows official MCAP normalization rules:
    - bytes → list of stringified ints
    - int → string
    - dict → recursively normalized dict (keys and values)
    - everything else → as-is
    """
    if isinstance(value, bytes):
        return [str(b) for b in value]
    if isinstance(value, int):
        return str(value)
    if isinstance(value, dict):
        # Stringify both keys and values recursively
        return {str(k) if isinstance(k, int) else k: normalize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    return value


def record_to_normalized_dict(record: McapRecord) -> dict[str, Any]:
    """Convert a record to normalized dictionary format.

    Returns: {"type": "RecordName", "fields": [["field1", value1], ...]}
    Fields are sorted alphabetically.
    """
    # Get all fields from the dataclass, sorted alphabetically
    fields = [(k, normalize_value(v)) for k, v in sorted(record.__dict__.items())]
    return {"type": type(record).__name__, "fields": fields}


def parse_mcap_to_normalized(mcap_path: Path) -> dict[str, Any]:
    """Parse an MCAP file and return normalized records."""
    with open(mcap_path, "rb") as f:
        records = list(stream_reader(f))

    return {"records": [record_to_normalized_dict(r) for r in records]}
