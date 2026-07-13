from __future__ import annotations

from types import SimpleNamespace

import pytest
from pymcap_cli.core.message_filter import MessageFilterOptions
from small_mcap import Channel


def _channel(topic: str) -> Channel:
    return Channel(1, 0, topic, "json", {})


def test_message_filter_topic_selectors_union_and_exclusions() -> None:
    options = MessageFilterOptions.from_args(
        topic=["/literal_1", r"/camera/.*", r"/lidar/.*"],
        exclude_topic=["/camera/rear", r".*debug", r"/lidar/raw"],
    )
    predicate = options.create_channel_predicate()

    assert predicate(_channel("/literal_1"), None)
    assert not predicate(_channel("/literal_10"), None)
    assert predicate(_channel("/camera/front"), None)
    assert not predicate(_channel("/camera/rear"), None)
    assert predicate(_channel("/lidar/points"), None)
    assert not predicate(_channel("/lidar/raw"), None)
    assert not predicate(_channel("/camera/debug"), None)


def test_message_filter_base_predicate_cannot_be_overridden() -> None:
    options = MessageFilterOptions.from_args(topic=["/camera"])
    predicate = options.create_channel_predicate(
        lambda channel, _schema: channel.topic.startswith("/camera")
    )

    assert predicate(_channel("/camera"), None)
    assert not predicate(_channel("/unsupported"), None)


def test_message_filter_invalid_regex_fails_during_construction() -> None:
    with pytest.raises(ValueError, match="Invalid topic regex"):
        MessageFilterOptions.from_args(topic=["["])


def test_message_filter_resolves_relative_bounds() -> None:
    options = MessageFilterOptions.from_args(start="@2s", end="end-1s")
    summary = SimpleNamespace(
        statistics=SimpleNamespace(
            message_start_time=10_000_000_000,
            message_end_time=20_000_000_000,
        )
    )

    resolved = options.resolve(summary)  # type: ignore[arg-type]

    assert resolved.start_time_ns == 12_000_000_000
    assert resolved.end_time_ns == 19_000_000_000


def test_message_filter_resolves_relative_shorthand() -> None:
    options = MessageFilterOptions.from_args(start="+2s", end="-1s")
    summary = SimpleNamespace(
        statistics=SimpleNamespace(
            message_start_time=10_000_000_000,
            message_end_time=20_000_000_000,
        )
    )

    resolved = options.resolve(summary)  # type: ignore[arg-type]

    assert resolved.start_time_ns == 12_000_000_000
    assert resolved.end_time_ns == 19_000_000_000


def test_message_filter_relative_bounds_require_summary() -> None:
    options = MessageFilterOptions.from_args(start="@2s")

    with pytest.raises(ValueError, match="summary statistics"):
        options.resolve(None)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"early_bail": True}, "requires --end"),
        ({"start": "10", "end": "10"}, "must be less"),
    ],
)
def test_message_filter_rejects_invalid_time_combinations(
    kwargs: dict[str, str | bool], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MessageFilterOptions.from_args(**kwargs)  # type: ignore[arg-type]
