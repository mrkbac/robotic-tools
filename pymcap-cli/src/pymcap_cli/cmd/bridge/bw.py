"""`pymcap-cli bridge bw` — monitor live topic payload bandwidth."""

from typing import Annotated

from cyclopts import Parameter

from pymcap_cli.cmd._cli_options import (
    TOPIC_FILTERING_GROUP,
    AllTopicsOption,
    BridgeTarget,
    ConnectTimeoutOption,
    ExcludeTopicOption,
    JsonOutputOption,
    LiveDurationOption,
    MonitorWindowOption,
    RefreshIntervalOption,
    TopicOption,
)
from pymcap_cli.cmd.bridge._topic_monitor import (
    TOPIC_SELECTION_CONSTRAINT,
    TopicMonitorView,
    run_topic_monitor,
)


def bw(
    target: BridgeTarget,
    *,
    topic: Annotated[
        TopicOption, Parameter(group=[TOPIC_FILTERING_GROUP, TOPIC_SELECTION_CONSTRAINT])
    ] = None,
    all_topics: Annotated[
        AllTopicsOption, Parameter(group=[TOPIC_FILTERING_GROUP, TOPIC_SELECTION_CONSTRAINT])
    ] = False,
    exclude_topic: ExcludeTopicOption = None,
    window: MonitorWindowOption = 10.0,
    interval: RefreshIntervalOption = 1.0,
    duration: LiveDurationOption = None,
    json_output: JsonOutputOption = False,
    connect_timeout: ConnectTimeoutOption = 5.0,
) -> int:
    """Monitor received payload bandwidth for live bridge topics.

    Bandwidth counts Foxglove message payload bytes and excludes WebSocket, TLS,
    and transport framing overhead. Runs until Ctrl+C unless ``--duration`` is
    supplied.

    Examples
    --------
    ```
    pymcap-cli bridge bw robot:8765 -t /camera/image
    pymcap-cli bridge bw robot:8765 --all -x '/camera/.*'
    ```
    """
    return run_topic_monitor(
        target,
        topic=topic,
        all_topics=all_topics,
        exclude_topic=exclude_topic,
        window=window,
        interval=interval,
        duration=duration,
        json_output=json_output,
        connect_timeout=connect_timeout,
        view=TopicMonitorView.BW,
    )
