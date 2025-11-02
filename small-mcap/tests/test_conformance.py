"""Tests using official MCAP conformance test files with JSON validation.

These tests validate that small-mcap can correctly read all official MCAP
conformance test files and produce output matching the expected JSON specs.
"""

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from small_mcap import McapRecord, stream_reader


def normalize_value(value: Any) -> Any:
    """Normalize a value for conformance comparison."""
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
    """Convert a record to normalized dictionary format."""
    # Use dataclasses.fields() to get all fields (works with slots=True)
    fields = [
        (f.name, normalize_value(getattr(record, f.name)))
        for f in sorted(dataclasses.fields(record), key=lambda f: f.name)
    ]
    return {"type": type(record).__name__, "fields": fields}


def parse_mcap_to_normalized(mcap_path: Path) -> dict[str, Any]:
    """Parse an MCAP file and return normalized records."""
    with open(mcap_path, "rb") as f:
        records = list(stream_reader(f))

    return {"records": [record_to_normalized_dict(r) for r in records]}


def pytest_generate_tests(metafunc):
    """Generate parametrized tests for conformance files."""
    if "mcap_file" in metafunc.fixturenames and "json_file" in metafunc.fixturenames:
        # Load conformance data path
        conformance_data = Path(__file__).parent / "conformance_data"

        if not conformance_data.exists():
            pytest.skip("Conformance test data not available")

        # Find all MCAP+JSON pairs
        pairs = []
        for mcap_file in conformance_data.rglob("*.mcap"):
            json_file = mcap_file.with_suffix(".json")
            if json_file.exists():
                pairs.append((mcap_file, json_file))

        # Parametrize with test IDs using just the filename
        metafunc.parametrize(
            "mcap_file,json_file", pairs, ids=[f"{mcap.stem}" for mcap, _ in pairs]
        )


@pytest.mark.conformance
class TestConformanceValidation:
    """Test all conformance files against their JSON specifications."""

    def test_conformance_file_matches_json_spec(self, mcap_file: Path, json_file: Path):
        """Test that parsed MCAP matches its JSON specification exactly.

        This test validates each conformance file by:
        1. Parsing the MCAP file using small-mcap
        2. Normalizing the output (ints→strings, bytes→string arrays, sorted fields)
        3. Loading the expected JSON specification
        4. Comparing the normalized output with the expected spec

        This follows the same validation approach as the official MCAP tests.
        """
        # Parse MCAP and normalize
        actual = parse_mcap_to_normalized(mcap_file)

        # Load expected JSON
        with open(json_file) as f:
            expected = json.load(f)

        # Compare records
        actual_records = actual["records"]
        expected_records = expected["records"]

        assert len(actual_records) == len(expected_records), (
            f"Record count mismatch in {mcap_file.name}: "
            f"expected {len(expected_records)}, got {len(actual_records)}"
        )

        # Compare each record
        for i, (actual_record, expected_record) in enumerate(
            zip(actual_records, expected_records, strict=True)
        ):
            assert actual_record["type"] == expected_record["type"], (
                f"Record {i} type mismatch in {mcap_file.name}: "
                f"expected {expected_record['type']}, got {actual_record['type']}"
            )

            # Convert field lists to dicts for easier comparison
            actual_fields = dict(actual_record["fields"])
            expected_fields = dict(expected_record["fields"])

            assert actual_fields == expected_fields, (
                f"Record {i} ({expected_record['type']}) fields mismatch in {mcap_file.name}:\n"
                f"Expected: {expected_fields}\n"
                f"Got: {actual_fields}"
            )
