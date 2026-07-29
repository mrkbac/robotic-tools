"""`pymcap-cli bridge stats` — combined live topic health overview."""

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


def stats(
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
    """Monitor frequency, payload bandwidth, and corrected message delay together.

    Message delay subtracts the measured bridge clock offset from local receive
    time minus the bridge message timestamp. It is unavailable when no bridge
    time samples arrive. Use ``bridge delay`` for detailed bridge-clock and
    decoded ROS ``header.stamp`` measurements.

    Examples
    --------
    ```
    pymcap-cli bridge stats robot:8765 --all
    pymcap-cli bridge stats robot:8765 -t '/imu/.*' --duration 10
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
        view=TopicMonitorView.STATS,
    )
