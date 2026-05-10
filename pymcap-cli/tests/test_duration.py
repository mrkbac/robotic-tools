import pytest
from pymcap_cli.constants import NS_TO_MS, NS_TO_SEC
from pymcap_cli.types.duration import parse_duration_ns


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1s", NS_TO_SEC),
        ("500ms", 500 * NS_TO_MS),
        ("1.5m", int(1.5 * 60 * NS_TO_SEC)),
        ("1h", 3600 * NS_TO_SEC),
        ("60", 60 * NS_TO_SEC),
        ("1500us", 1_500_000),
        ("250ns", 250),
    ],
)
def test_parse_duration_ns_accepts(value: str, expected: int) -> None:
    assert parse_duration_ns(value) == expected


@pytest.mark.parametrize("value", ["", "60S", "5M", "abc", "10x", "s"])
def test_parse_duration_ns_rejects(value: str) -> None:
    with pytest.raises(ValueError, match="Invalid duration"):
        parse_duration_ns(value)
