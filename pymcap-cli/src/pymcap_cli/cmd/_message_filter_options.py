"""Reusable Cyclopts annotations for MCAP message filters."""

from typing import Annotated

from cyclopts import Group, Parameter

from pymcap_cli.core.message_filter import MessageFilterOptions

TOPIC_FILTERING_GROUP = Group("Topic Filtering")
TIME_FILTERING_GROUP = Group("Time Filtering")

TopicOption = Annotated[
    list[str] | None,
    Parameter(
        name=["-t", "--topic"],
        group=TOPIC_FILTERING_GROUP,
        help="Include a topic regex using full-match semantics (repeatable).",
    ),
]
ExcludeTopicOption = Annotated[
    list[str] | None,
    Parameter(
        name=["-x", "--exclude-topic"],
        group=TOPIC_FILTERING_GROUP,
        help="Exclude a topic regex using full-match semantics (repeatable).",
    ),
]
StartTimeOption = Annotated[
    str,
    Parameter(
        name=["-S", "--start"],
        group=TIME_FILTERING_GROUP,
        help=(
            "Inclusive log-time bound: nanoseconds, RFC3339, or a recording-relative "
            "value such as @10s/start+10s/end-30s."
        ),
    ),
]
EndTimeOption = Annotated[
    str,
    Parameter(
        name=["-E", "--end"],
        group=TIME_FILTERING_GROUP,
        help=(
            "Exclusive log-time bound: nanoseconds, RFC3339, or a recording-relative "
            "value such as @20s/start+20s/end-5s."
        ),
    ),
]
EarlyBailOption = Annotated[
    bool,
    Parameter(
        name=["--early-bail"],
        group=TIME_FILTERING_GROUP,
        help=("Assume monotonic log time and stop at the first message at or after --end."),
    ),
]


def create_message_filter(
    *,
    topic: list[str] | None,
    exclude_topic: list[str] | None,
    start: str,
    end: str,
    early_bail: bool,
) -> MessageFilterOptions:
    return MessageFilterOptions.from_args(
        topic=topic,
        exclude_topic=exclude_topic,
        start=start,
        end=end,
        early_bail=early_bail,
    )
