"""`pymcap-cli bridge hz` — monitor live topic receive rates."""

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


def hz(
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
    """Monitor receiving frequency for live bridge topics.

    The bridge streams selected messages continuously after one subscription.
    The heading is printed once, then one compact timestamped row per topic is
    appended every ``--interval`` seconds so changes remain in terminal
    scrollback. Frequency uses the rolling, time-based ``--window``. Runs until
    Ctrl+C unless ``--duration`` is supplied.

    Examples
    --------
    ```
    pymcap-cli bridge hz robot:8765 -t /imu/data
    pymcap-cli bridge hz robot:8765 -t /imu/data -t '/camera/.*'
    pymcap-cli bridge hz robot:8765 --all --window 30
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
        view=TopicMonitorView.HZ,
    )
