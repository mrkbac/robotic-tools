"""Tests for pymcap_cli.utils — utility functions edge cases."""

from __future__ import annotations

import pytest
from pymcap_cli.utils import (
    bytes_to_human,
    compile_topic_patterns,
    parse_time_arg,
    parse_timestamp_args,
)

# ---------------------------------------------------------------------------
# bytes_to_human
# ---------------------------------------------------------------------------


class TestBytesToHuman:
    def test_none(self):
        assert bytes_to_human(None) == "N/A"

    def test_zero(self):
        assert bytes_to_human(0) == "0 bytes"

    def test_bytes_range(self):
        assert bytes_to_human(512) == "512 bytes"

    def test_kilobytes(self):
        # Rich's filesize.decimal uses SI prefixes: 1.5 kB
        result = bytes_to_human(1500)
        assert "1.5" in result
        assert "kB" in result

    def test_megabytes(self):
        result = bytes_to_human(5_000_000)
        assert "5.0" in result
        assert "MB" in result

    def test_gigabytes(self):
        result = bytes_to_human(2_500_000_000)
        assert "2.5" in result
        assert "GB" in result

    def test_negative_uses_abs(self):
        assert bytes_to_human(-1000) == bytes_to_human(1000)


# ---------------------------------------------------------------------------
# parse_time_arg
# ---------------------------------------------------------------------------


class TestParseTimeArg:
    def test_empty_string(self):
        assert parse_time_arg("") == 0

    def test_integer_nanoseconds(self):
        assert parse_time_arg("1000000000") == 1_000_000_000

    def test_zero(self):
        assert parse_time_arg("0") == 0

    def test_negative(self):
        assert parse_time_arg("-100") == -100

    def test_rfc3339_known_epoch(self):
        # 2024-01-01T00:00:00Z = 1704067200 seconds = 1704067200_000_000_000 ns
        result = parse_time_arg("2024-01-01T00:00:00+00:00")
        assert result == 1_704_067_200_000_000_000

    def test_rfc3339_z_suffix(self):
        result = parse_time_arg("2024-01-01T00:00:00Z")
        assert result == 1_704_067_200_000_000_000

    def test_rfc3339_z_matches_utc(self):
        z_result = parse_time_arg("2024-01-01T00:00:00Z")
        utc_result = parse_time_arg("2024-01-01T00:00:00+00:00")
        assert z_result == utc_result

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid time format"):
            parse_time_arg("not-a-time")


# ---------------------------------------------------------------------------
# compile_topic_patterns
# ---------------------------------------------------------------------------


class TestCompileTopicPatterns:
    def test_empty_list(self):
        assert compile_topic_patterns([]) == []

    def test_single_pattern(self):
        patterns = compile_topic_patterns(["/camera/.*"])
        assert len(patterns) == 1
        assert patterns[0].match("/camera/left")

    def test_pattern_does_not_match_wrong_topic(self):
        patterns = compile_topic_patterns(["/camera/.*"])
        assert not patterns[0].match("/lidar/points")

    def test_multiple_patterns(self):
        patterns = compile_topic_patterns(["/camera/.*", "/lidar/.*"])
        assert len(patterns) == 2
        assert patterns[1].match("/lidar/points")

    def test_case_insensitive(self):
        patterns = compile_topic_patterns(["/Camera"])
        assert patterns[0].match("/camera")

    def test_invalid_regex(self):
        with pytest.raises(ValueError, match="Invalid regex"):
            compile_topic_patterns(["[invalid"])


# ---------------------------------------------------------------------------
# parse_timestamp_args
# ---------------------------------------------------------------------------


class TestParseTimestampArgs:
    def test_all_empty(self):
        assert parse_timestamp_args("", 0, 0) is None

    def test_date_or_nanos_takes_precedence(self):
        result = parse_timestamp_args("1000", 2, 3)
        assert result == 1000

    def test_seconds_over_nanoseconds(self):
        result = parse_timestamp_args("", 5, 999)
        assert result == 5_000_000_000

    def test_nanoseconds_fallback(self):
        result = parse_timestamp_args("", 0, 42)
        assert result == 42

    def test_rfc3339_date(self):
        result = parse_timestamp_args("2024-01-01T00:00:00Z", 0, 0)
        assert result == 1_704_067_200_000_000_000
