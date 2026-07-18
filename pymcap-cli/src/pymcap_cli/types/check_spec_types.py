# ruff: noqa: I001, E501
# AUTO-GENERATED from pymcap-cli/schemas/ - DO NOT EDIT


from typing import Literal, TypedDict, Union
from typing_extensions import Required


class CheckSpecInput(TypedDict, total=False):
    r"""
    CheckSpecInput.

    Version 1 recording contract for `pymcap-cli check` and `pymcap-cli bridge check`. Structural
    rules only; regex validity, MessagePath validity, and cross-field constraints are enforced by the
    parser.
    """

    version: Required[Literal[1]]
    r"""
    Spec format version; must be 1.

    Required property
    """

    topics: Required[dict[str, "TopicRuleSpec"]]
    r"""
    Topic rules keyed by a human-readable rule name.

    propertyNames:
      __type__: string
      minLength: 1

    Required property
    """

    live: "LiveRootSpec"
    r"""
    LiveRootSpec.

    Live-only constraints; only evaluated by `bridge check`.
    """


ComparableValue = bool | int | float | str
r"""
ComparableValue.

Scalar compared against evaluated values.
"""


Duration = str | int | float
r"""
Duration.

Positive duration such as '500ms', '2s', or '1.5m'; a bare number means seconds.

pattern: ^\s*(?=[0-9.]*[1-9])[0-9]*\.?[0-9]+\s*(ns|us|ms|s|m|h)?\s*$
exclusiveMinimum: 0
"""


class EndpointRuleSpec(TypedDict, total=False):
    r"""
    EndpointRuleSpec.

    Publisher/subscriber count bounds and node identity; at least one key must be set.

    minProperties: 1
    """

    min: int
    r"""
    Minimum endpoint count.

    minimum: 0
    """

    max: int
    r"""
    Maximum endpoint count.

    minimum: 0
    """

    node: str
    r"""
    Case-insensitive regular expression matched against endpoint node names.

    minLength: 1
    """


class FrequencyRuleSpec(TypedDict, total=False):
    r"""
    FrequencyRuleSpec.

    Sliding-window message rate bounds; at least one of min or max must be set.

    anyOf:
      - required:
        - min
      - required:
        - max
    """

    min: int | float
    r"""
    Minimum frequency in Hz.

    minimum: 0
    """

    max: int | float
    r"""
    Maximum frequency in Hz.

    minimum: 0
    """

    tolerance: int | float
    r"""
    Relative tolerance applied to min and max, e.g. 0.05 for 5%.

    minimum: 0
    exclusiveMaximum: 1
    default: 0
    """

    window: Required["Duration"]
    r"""
    Duration.

    Positive duration such as '500ms', '2s', or '1.5m'; a bare number means seconds.

    pattern: ^\s*(?=[0-9.]*[1-9])[0-9]*\.?[0-9]+\s*(ns|us|ms|s|m|h)?\s*$
    exclusiveMinimum: 0

    Required property
    """


class LiveNodeRuleSpec(TypedDict, total=False):
    r"""
    LiveNodeRuleSpec.

    Presence check for a node in the live connection graph.
    """

    node: Required[str]
    r"""
    Case-insensitive regular expression matched against node names.

    minLength: 1

    Required property
    """

    expected: bool
    r"""
    Whether the node must be present (true) or absent (false).

    default: True
    """

    severity: "Severity"
    r"""
    Severity.

    Violation severity; warnings do not cause a non-zero exit.

    default: error
    """


class LiveRootSpec(TypedDict, total=False):
    r"""
    LiveRootSpec.

    Live-only constraints; only evaluated by `bridge check`.
    """

    nodes: Required[dict[str, "LiveNodeRuleSpec"]]
    r"""
    Live node rules keyed by a human-readable rule name.

    minProperties: 1
    propertyNames:
      __type__: string
      minLength: 1

    Required property
    """


class LiveTopicRuleSpec(TypedDict, total=False):
    r"""
    LiveTopicRuleSpec.

    Live graph constraints for a topic; at least one of publishers or subscribers must be set. Only
    evaluated by `bridge check`.

    minProperties: 1
    """

    publishers: "EndpointRuleSpec"
    r"""
    EndpointRuleSpec.

    Publisher/subscriber count bounds and node identity; at least one key must be set.

    minProperties: 1
    """

    subscribers: "EndpointRuleSpec"
    r"""
    EndpointRuleSpec.

    Publisher/subscriber count bounds and node identity; at least one key must be set.

    minProperties: 1
    """


MessagePathSpec = str
r"""
MessagePathSpec.

Relative MessagePath starting with '.' or '{' and ending in a predicate filter, e.g.
'.linear_acceleration.@norm{<=30}'.

pattern: ^[.{].*\}\s*$
"""


SEVERITY_DEFAULT = "error"
r""" Default value of the field path 'severity' """


class SchemaRuleSpec(TypedDict, total=False):
    r"""
    SchemaRuleSpec.

    Required schema name and/or encoding; at least one must be set.

    minProperties: 1
    """

    name: str
    r"""
    Required schema name, e.g. 'sensor_msgs/msg/Imu'.

    minLength: 1
    """

    encoding: str
    r"""
    Required schema encoding, e.g. 'ros2msg'.

    minLength: 1
    """


Severity = Literal["warn"] | Literal["error"]
r"""
Severity.

Violation severity; warnings do not cause a non-zero exit.

default: error
"""
SEVERITY_WARN: Literal["warn"] = "warn"
r"""The values for the 'Severity' enum"""
SEVERITY_ERROR: Literal["error"] = "error"
r"""The values for the 'Severity' enum"""


class TopicRuleSpec(TypedDict, total=False):
    r"""
    TopicRuleSpec.

    Checks applied to every topic matching the selector. A rule with `expected: false` forbids the
    topic and must not define other checks.

    allOf:
      - if:
          properties:
            expected:
              const: false
          required:
          - expected
        then:
          propertyNames:
            enum:
            - topic
            - expected
            - severity
    """

    topic: Required[str]
    r"""
    Case-insensitive regular expression matched against the whole topic name.

    minLength: 1

    Required property
    """

    expected: bool
    r"""
    Whether the topic must be present (true) or absent (false).

    default: True
    """

    severity: "Severity"
    r"""
    Severity.

    Violation severity; warnings do not cause a non-zero exit.

    default: error
    """

    schema: "SchemaRuleSpec"
    r"""
    SchemaRuleSpec.

    Required schema name and/or encoding; at least one must be set.

    minProperties: 1
    """

    message_encoding: str
    r"""
    Required channel message encoding, e.g. 'cdr'.

    minLength: 1
    """

    frequency: "FrequencyRuleSpec"
    r"""
    FrequencyRuleSpec.

    Sliding-window message rate bounds; at least one of min or max must be set.

    anyOf:
      - required:
        - min
      - required:
        - max
    """

    timeout: "Duration"
    r"""
    Duration.

    Positive duration such as '500ms', '2s', or '1.5m'; a bare number means seconds.

    pattern: ^\s*(?=[0-9.]*[1-9])[0-9]*\.?[0-9]+\s*(ns|us|ms|s|m|h)?\s*$
    exclusiveMinimum: 0
    """

    values: list["ValueRuleSpec"]
    r"""
    Value checks evaluated on every decoded message.

    minItems: 1
    """

    live: "LiveTopicRuleSpec"
    r"""
    LiveTopicRuleSpec.

    Live graph constraints for a topic; at least one of publishers or subscribers must be set. Only evaluated by `bridge check`.

    minProperties: 1
    """


class ValueRuleMappingSpec(TypedDict, total=False):
    r"""
    ValueRuleMappingSpec.

    Value check with explicit comparators; numeric bounds cannot be combined with equals or one_of.

    allOf:
      - not:
          required:
          - min
          - equals
      - not:
          required:
          - max
          - equals
      - not:
          required:
          - min
          - one_of
      - not:
          required:
          - max
          - one_of
      - not:
          required:
          - equals
          - one_of
      - if:
          not:
            anyOf:
            - required:
              - min
            - required:
              - max
            - required:
              - equals
            - required:
              - one_of
        then:
          properties:
            path:
              pattern: \}\s*$
    """

    path: Required[str]
    r"""
    Relative MessagePath starting with '.' or '{'.

    pattern: ^[.{]

    Required property
    """

    min: int | float
    r""" Inclusive minimum for numeric results. """

    max: int | float
    r""" Inclusive maximum for numeric results. """

    equals: "ComparableValue"
    r"""
    ComparableValue.

    Scalar compared against evaluated values.
    """

    one_of: list["ComparableValue"]
    r"""
    Result must equal one of these values.

    minItems: 1
    """


ValueRuleSpec = Union["MessagePathSpec", "ValueRuleMappingSpec"]
r"""
ValueRuleSpec.

A relative MessagePath ending in a predicate filter, or a mapping with explicit bounds.

Aggregation type: oneOf
"""


_FREQUENCYRULESPEC_TOLERANCE_DEFAULT = 0
r""" Default value of the field path 'FrequencyRuleSpec tolerance' """


_LIVENODERULESPEC_EXPECTED_DEFAULT = True
r""" Default value of the field path 'LiveNodeRuleSpec expected' """


_TOPICRULESPEC_EXPECTED_DEFAULT = True
r""" Default value of the field path 'TopicRuleSpec expected' """
