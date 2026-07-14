"""Build message filters from the shared CLI option values."""

from pymcap_cli.core.message_filter import MessageFilterOptions


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
