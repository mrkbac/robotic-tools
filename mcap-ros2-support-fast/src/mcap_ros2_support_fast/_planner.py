"""ROS2 message plan generation for parser-agnostic deserialization.

This module generates execution plans for ROS2 messages without depending
on any specific parser implementation. Plans can be consumed by different
backends (interpreted, compiled, code-generated, etc.).
"""

from collections.abc import Callable
from dataclasses import make_dataclass
from typing import Any

from ros_parser import (
    MessageDefinition,
)
from ros_parser.message_definition import BUILTIN_TYPES, for_each_msgdef

from mcap_ros2_support_fast._dynamic_decoder import create_decoder
from mcap_ros2_support_fast._dynamic_encoder import create_encoder
from mcap_ros2_support_fast._plans import (
    STRING_TO_TYPE_ID,
    ActionType,
    ComplexAction,
    ComplexArrayAction,
    DecoderFunction,
    EncoderFunction,
    PlanActions,
    PlanList,
    PrimitiveAction,
    PrimitiveArrayAction,
    PrimitiveGroupAction,
    TypeId,
)

# Type size information for alignment-aware grouping
_TYPE_SIZES = {
    TypeId.FLOAT64: 8,
    TypeId.INT64: 8,
    TypeId.UINT64: 8,
    TypeId.FLOAT32: 4,
    TypeId.INT32: 4,
    TypeId.UINT32: 4,
    TypeId.INT16: 2,
    TypeId.UINT16: 2,
    TypeId.BOOL: 1,
    TypeId.BYTE: 1,
    TypeId.CHAR: 1,
    TypeId.INT8: 1,
    TypeId.UINT8: 1,
    TypeId.PADDING: 1,
}


def _generate_plan(
    msgdef: MessageDefinition,
    msgdefs: dict[str, MessageDefinition],
) -> PlanList:
    """Generate a pre-computed parsing plan for a message definition."""
    plan: PlanActions = []

    # Create dataclass fields - using Any type for simplicity as requested
    fields = [(field.name, Any) for field in msgdef.fields]

    # Create a valid Python identifier from the message name
    # e.g., "std_msgs/msg/String" -> "std_msgs_msg_String"
    class_name = msgdef.name.replace("/", "_") if msgdef.name else "Message"

    msg_class = make_dataclass(
        class_name,
        fields,
        namespace={
            "_type": msgdef.name,
            "_full_text": str(msgdef),
        },
        slots=True,  # Use slots for better performance
        eq=True,
    )

    for field in msgdef.fields:
        field_type = field.type
        field_name = field.name

        if field_type.package_name is not None:
            # Complex type - generate nested plan
            type_path = f"{field_type.package_name}/{field_type.type_name}"
            nested_msgdef = msgdefs.get(type_path)
            if nested_msgdef is None:
                raise ValueError(f"Message definition not found for {type_path}")

            if field_type.is_array:
                # Array of complex types
                nested_plan = _generate_plan(nested_msgdef, msgdefs)
                plan.append(ComplexArrayAction(field_name, nested_plan, field_type.array_size))
            else:
                # Single complex type
                plan.append(ComplexAction(field_name, _generate_plan(nested_msgdef, msgdefs)))
        else:
            # Primitive type
            type_name = field_type.type_name
            type_id = STRING_TO_TYPE_ID.get(type_name)
            if type_id is None:
                raise ValueError(f"Unknown primitive type: {type_name}")

            if field_type.is_array:
                plan.append(PrimitiveArrayAction(field_name, type_id, field_type.array_size))
            else:
                plan.append(PrimitiveAction(field_name, type_id))

    return msg_class, plan


def generate_plans(schema_name: str, schema_text: str) -> PlanList:
    """Generate execution plan for the primary ROS2 message schema.

    This function is parser-agnostic and only generates plan data structures.
    The plan can be consumed by any parser implementation.

    :param schema_name: The name of the schema defined in `schema_text`.
    :param schema_text: The schema text to use for generating plans.
    :return: The execution plan for the primary schema type.
    """
    msgdefs: dict[str, MessageDefinition] = {
        **BUILTIN_TYPES,  # Include built-in types
    }

    def handle_msgdef(cur_schema_name: str, short_name: str, msgdef: MessageDefinition) -> None:
        # Add the message definition to the dictionary
        msgdefs[cur_schema_name] = msgdef
        msgdefs[short_name] = msgdef

    for_each_msgdef(schema_name, schema_text, handle_msgdef)

    # Generate plan for the primary message type
    primary_msgdef = msgdefs.get(schema_name)
    if primary_msgdef is None:
        raise ValueError(f"Primary message definition not found for {schema_name}")

    return _generate_plan(primary_msgdef, msgdefs)


def generate_dynamic(
    schema_name: str,
    schema_text: str,
    parser: Callable[[PlanList], DecoderFunction] = create_decoder,
) -> DecoderFunction:
    """Convert a ROS2 concatenated message definition into a message parser.

    Modeled after the `generate_dynamic` function in ROS1 `genpy.dynamic`.

    :param schema_name: The name of the schema defined in `schema_text`.
    :param schema_text: The schema text to use for deserializing the message payload.
    :param parser: Parser implementation function.
    :return: A decoder function for the primary schema type.
    """
    plan = generate_plans(schema_name, schema_text)
    optimized_plan = optimize_plan(plan)

    # Create decoder using the specified parser implementation
    return parser(optimized_plan)


def _find_groupable_primitives(steps: PlanActions) -> list[tuple[int, int]]:
    """Find ranges of consecutive primitive fields that can be grouped by alignment rules.

    Returns list of (start_idx, end_idx) tuples for groupable ranges.
    """
    if not steps:
        return []

    groupable_ranges = []
    start_idx = 0

    while start_idx < len(steps):
        # Find start of a primitive sequence
        step = steps[start_idx]
        if step.type != ActionType.PRIMITIVE:
            start_idx += 1
            continue

        # Skip strings as they can't be grouped with primitives
        if step.data in (TypeId.STRING, TypeId.WSTRING):
            start_idx += 1
            continue

        # Find end of groupable sequence with alignment-aware logic
        end_idx = start_idx
        max_alignment = _TYPE_SIZES.get(step.data, 0)
        cumulative_offset = max_alignment

        for i in range(start_idx + 1, len(steps)):
            look_step = steps[i]
            if look_step.type != ActionType.PRIMITIVE:
                break

            current_type = look_step.data

            # Skip strings
            if current_type in (TypeId.STRING, TypeId.WSTRING):
                break

            current_size = _TYPE_SIZES.get(current_type, 0)

            # Check if current type can be aligned within the group's alignment constraint
            if current_size > max_alignment:
                break

            # Check if we can align the current type at the current offset
            alignment_needed = current_size
            if cumulative_offset % alignment_needed != 0:
                # Calculate padding needed for alignment
                aligned_offset = (
                    (cumulative_offset + alignment_needed - 1) // alignment_needed
                ) * alignment_needed
                cumulative_offset = aligned_offset

            end_idx = i
            cumulative_offset += current_size

        # Only group if we have more than one field
        if end_idx > start_idx:
            groupable_ranges.append((start_idx, end_idx))

        start_idx = end_idx + 1

    return groupable_ranges


def _create_primitive_groups(steps: PlanActions) -> PlanActions:
    """Create optimized primitive groups from consecutive compatible primitives."""
    groupable_ranges = _find_groupable_primitives(steps)

    if not groupable_ranges:
        return steps

    optimized_steps: PlanActions = []
    current_idx = 0

    for start_idx, end_idx in groupable_ranges:
        # Add any non-groupable steps before this group
        while current_idx < start_idx:
            optimized_steps.append(steps[current_idx])
            current_idx += 1

        targets: list[tuple[str, TypeId]] = []
        cumulative_offset = 0
        max_alignment = 0

        # First pass: determine maximum alignment
        for i in range(start_idx, end_idx + 1):
            step = steps[i]
            if step.type == ActionType.PRIMITIVE:
                type_size = _TYPE_SIZES.get(step.data, 0)
                max_alignment = max(max_alignment, type_size)

        # Second pass: build targets with padding insertion
        for i in range(start_idx, end_idx + 1):
            step = steps[i]
            if step.type == ActionType.PRIMITIVE:
                type_size = _TYPE_SIZES.get(step.data, 0)

                # Insert padding if needed for alignment
                if cumulative_offset % type_size != 0:
                    aligned_offset = ((cumulative_offset + type_size - 1) // type_size) * type_size
                    padding_needed = aligned_offset - cumulative_offset

                    # Add padding bytes
                    for _ in range(padding_needed):
                        targets.append((f"__padding_{len(targets)}__", TypeId.PADDING))

                    cumulative_offset = aligned_offset

                targets.append((step.target, step.data))
                cumulative_offset += type_size
            else:
                raise ValueError(f"Unexpected action type: {step.type}")

        optimized_steps.append(PrimitiveGroupAction(targets=targets))
        current_idx = end_idx + 1

    # Add any remaining steps
    while current_idx < len(steps):
        optimized_steps.append(steps[current_idx])
        current_idx += 1

    return optimized_steps


def optimize_plan(plan: PlanList) -> PlanList:
    """Optimize a plan by grouping compatible primitive fields.

    Also recursively optimize nested plans.
    """

    target_type, steps = plan

    # Handle empty plans
    if not steps:
        return plan

    # Apply primitive grouping optimization
    optimized_steps = _create_primitive_groups(steps)

    # Process each step recursively for complex types
    final_steps: PlanActions = []
    for action in optimized_steps:
        if action.type in (
            ActionType.PRIMITIVE,
            ActionType.PRIMITIVE_ARRAY,
            ActionType.PRIMITIVE_GROUP,
        ):
            final_steps.append(action)
        elif action.type == ActionType.COMPLEX:
            final_steps.append(ComplexAction(action.target, optimize_plan(action.plan)))
        elif action.type == ActionType.COMPLEX_ARRAY:
            final_steps.append(
                ComplexArrayAction(action.target, optimize_plan(action.plan), action.size)
            )

    return target_type, final_steps


def serialize_dynamic(schema_name: str, schema_text: str) -> EncoderFunction:
    """Convert a ROS2 concatenated message definition into message encoders.

    This function generates encoder functions that can serialize message objects
    to CDR-encoded bytes.

    :param schema_name: The name of the schema defined in `schema_text`.
    :param schema_text: The schema text to use for serializing the message payload.
    :return: A dictionary mapping schema names to encoder functions.
    """

    # First collect all message definitions
    msgdefs: dict[str, MessageDefinition] = {
        **BUILTIN_TYPES,  # Include built-in types
    }

    def collect_msgdef(cur_schema_name: str, short_name: str, msgdef: MessageDefinition) -> None:
        # Add the message definition to the dictionary
        msgdefs[cur_schema_name] = msgdef
        msgdefs[short_name] = msgdef

    for_each_msgdef(schema_name, schema_text, collect_msgdef)

    plan = _generate_plan(msgdefs[schema_name], msgdefs)
    optimized_plan = optimize_plan(plan)
    return create_encoder(optimized_plan)
