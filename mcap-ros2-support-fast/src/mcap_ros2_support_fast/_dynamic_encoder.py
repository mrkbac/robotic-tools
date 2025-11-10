import codecs
import dataclasses
import struct
from typing import Any, cast

from mcap_ros2_support_fast.code_writer import CodeWriter

from ._plans import (
    TYPE_INFO,
    ActionType,
    EncoderFunction,
    PlanAction,
    PlanList,
    TypeId,
)


@dataclasses.dataclass
class FieldPath:
    """Represents a field access path for clean code generation."""

    base_expr: str
    path: list[str] = dataclasses.field(default_factory=list)

    def extend(self, field_name: str) -> "FieldPath":
        """Create a new FieldPath with an additional field."""
        return FieldPath(self.base_expr, [*self.path, field_name])

    def to_code(self) -> str:
        """Generate the field access code."""
        if not self.path:
            return self.base_expr
        path_args = ", ".join(f"'{field}'" for field in self.path)
        return f"_get_field({self.base_expr}, {path_args})"


def _get_field(obj: Any, *field_path: str) -> Any:
    """Get nested field using field path (works for both dict and object attributes)."""
    for f in field_path:
        obj = obj[f] if isinstance(obj, dict) else getattr(obj, f)
    return obj


class EncoderGeneratorFactory:
    """Factory class for generating encoder code with managed state."""

    def __init__(self, plan: PlanList, *, comments: bool = True) -> None:
        self.plan = plan
        self.name_counter = 0
        self.struct_patterns: dict[str, str] = {}
        self.code = CodeWriter(comments=comments)
        # Collect all required types and classes during generation
        self.message_classes: set[type] = set()
        self.current_alignment = 8  # perfect alignment at the start
        self.alias: dict[str, str] = {}

    def generate_var_name(self) -> str:
        """Generate a unique variable name."""
        self.name_counter += 1
        return f"_v{self.name_counter}"

    def generate_alias(self, base_name: str) -> str:
        """Generate a unique alias name based on the base name."""
        if base_name not in self.alias:
            self.alias[base_name] = self.generate_var_name()
        return self.alias[base_name]

    def get_struct_pattern_var_name(self, pattern: str) -> str:
        """Get or create a variable name for a struct pattern."""
        if pattern not in self.struct_patterns:
            safe_name = (
                pattern.replace("<", "le_").replace(">", "be_").replace(" ", "_").replace("?", "b")
            )
            self.struct_patterns[pattern] = f"_{safe_name}"
        return self.struct_patterns[pattern]

    def generate_alignment(self, size: int) -> None:
        """Generate optimized alignment code for a given size requirement."""
        if self.current_alignment >= size:
            self.current_alignment = size
            return
        self.current_alignment = size
        if size > 1 and size in (2, 4, 8):
            mask = size - 1
            self.code.append(f"_pad = ((_offset + {mask}) & ~{mask}) - _offset")
            self.code.append("if _pad:")
            with self.code.indent(None):
                self.code.append("_buffer.extend(b'\\x00' * _pad)")
                self.code.append("_offset += _pad")

    def reset_alignment(self, initial: int = 0) -> None:
        """Reset the current alignment to zero."""
        self.current_alignment = initial

    def generate_primitive_writer(self, value_expr: str, type_id: TypeId) -> None:
        """Generate the writer code for a given TypeId."""
        if type_id in {TypeId.UINT8, TypeId.BYTE, TypeId.CHAR}:
            self.generate_alignment(1)
            self.code.append(f"_buffer.append({value_expr})")
            self.code.append("_offset += 1")

        elif type_id == TypeId.STRING:
            str_fnc = self.generate_alias("_encode_utf8")
            self.code.append(f"_str_bytes = {str_fnc}({value_expr})[0]")
            self.code.append("_str_size = len(_str_bytes) + 1")  # +1 for null terminator
            self.generate_primitive_writer("_str_size", TypeId.UINT32)
            self.code.append("_buffer.extend(_str_bytes)")
            self.code.append("_buffer.append(0)")  # null terminator
            self.code.append("_offset += len(_str_bytes) + 1")
            self.reset_alignment()  # After string unknown position readjustment
        elif type_id == TypeId.WSTRING:
            self.code.append("raise NotImplementedError('wstring not implemented')")
        # Standard struct-based types
        elif struct_name := TYPE_INFO.get(type_id):
            struct_size = struct.calcsize(struct_name)
            self.generate_alignment(struct_size)

            pattern = f"<{struct_name}"
            pattern_var = self.get_struct_pattern_var_name(pattern)
            self.code.append(f"_buffer.extend({pattern_var}({value_expr}))")
            self.code.append(f"_offset += {struct_size}")
        else:
            raise NotImplementedError(f"Unsupported type: {type_id}")

    def generate_primitive_array_writer(
        self, value_expr: str, type_id: TypeId, array_size: int | None
    ) -> None:
        """Generate code for primitive array fields."""
        if array_size is None:  # dynamic array
            self.generate_primitive_writer(f"len({value_expr})", TypeId.UINT32)

        if type_id == TypeId.STRING:
            str_fnc = self.generate_alias("_encode_utf8")
            random_i = self.generate_var_name()

            self.reset_alignment()  # After string unknown position readjustment
            with self.code.indent(f"for {random_i} in {value_expr}:"):
                self.code.append(f"_str_bytes = {str_fnc}({random_i})[0]")
                self.code.append("_str_size = len(_str_bytes) + 1")  # +1 for null terminator
                self.generate_primitive_writer("_str_size", TypeId.UINT32)
                self.code.append("_buffer.extend(_str_bytes)")
                self.code.append("_buffer.append(0)")  # null terminator
                self.code.append("_offset += len(_str_bytes) + 1")
        elif type_id == TypeId.WSTRING:
            self.code.append("raise NotImplementedError('wstring not implemented')")

        elif type_id in {TypeId.UINT8, TypeId.BYTE, TypeId.CHAR}:
            # Special case for byte arrays
            self.code.append(f"_buffer.extend({value_expr})")
            self.code.append(f"_offset += len({value_expr})")
            self.reset_alignment()  # After string unknown position readjustment
        else:
            # Regular primitive arrays
            struct_name = TYPE_INFO[type_id]
            struct_size = struct.calcsize(struct_name)
            self.generate_alignment(struct_size)

            if array_size is not None:
                pattern = f"<{array_size}{struct_name}"
                pattern_var = self.get_struct_pattern_var_name(pattern)
                self.code.append(f"_buffer.extend({pattern_var}(*{value_expr}))")
                self.code.append(f"_offset += {array_size * struct_size}")
            else:
                # Dynamic array
                random_i = self.generate_var_name()
                pattern = f"<{struct_name}"
                pattern_var = self.get_struct_pattern_var_name(pattern)
                with self.code.indent(f"for {random_i} in {value_expr}:"):
                    self.code.append(f"_buffer.extend({pattern_var}({random_i}))")
                    self.code.append(f"_offset += {struct_size}")

    def generate_complex_array_writer(
        self, field_path: FieldPath, plan: PlanList, array_size: int | None
    ) -> None:
        """Generate code for complex array fields."""
        value_expr = field_path.to_code()
        if array_size is None:
            self.generate_primitive_writer(f"len({value_expr})", TypeId.UINT32)

        random_i = self.generate_var_name()
        self.reset_alignment()  # Need to reset alignment at the start of loops
        with self.code.indent(f"for {random_i} in {value_expr}:"):
            # Create a new FieldPath for the array element
            element_path = FieldPath(random_i)
            self.generate_plan_writer(element_path, plan)

    def generate_primitive_group_writer(
        self, field_path: FieldPath, targets: list[tuple[str, TypeId]]
    ) -> None:
        """Generate code for a group of primitive fields."""
        struct_format = "".join(TYPE_INFO[field_type] for _, field_type in targets)
        struct_size = struct.calcsize(f"<{struct_format}")
        pattern = f"<{struct_format}"

        # Align to the first type
        first_type_size = struct.calcsize(f"<{TYPE_INFO[targets[0][1]]}")
        self.generate_alignment(first_type_size)

        field_values = []
        for name, typeid in targets:
            if typeid != TypeId.PADDING:
                target_path = self._get_field_access(field_path, name)
                field_values.append(target_path.to_code())

        struct_var = self.get_struct_pattern_var_name(pattern)
        self.code.append(f"_buffer.extend({struct_var}({', '.join(field_values)}))")
        self.code.append(f"_offset += {struct_size}")

        last_size = struct.calcsize(f"<{TYPE_INFO[targets[-1][1]]}")
        self.reset_alignment(last_size)

    def generate_type_writer(self, field_path: FieldPath, step: PlanAction) -> None:
        """Generate code for a single plan action."""
        if step.type == ActionType.PRIMITIVE:
            target_path = self._get_field_access(field_path, step.target)
            self.generate_primitive_writer(target_path.to_code(), step.data)
        elif step.type == ActionType.PRIMITIVE_ARRAY:
            target_path = self._get_field_access(field_path, step.target)
            self.generate_primitive_array_writer(target_path.to_code(), step.data, step.size)
        elif step.type == ActionType.PRIMITIVE_GROUP:
            self.generate_primitive_group_writer(field_path, step.targets)
        elif step.type == ActionType.COMPLEX:
            target_path = self._get_field_access(field_path, step.target)
            self.generate_plan_writer(target_path, step.plan)
        elif step.type == ActionType.COMPLEX_ARRAY:
            target_path = self._get_field_access(field_path, step.target)
            self.generate_complex_array_writer(target_path, step.plan, step.size)
        else:
            raise ValueError(f"Unknown action type: {step}")

    def _get_field_access(self, field_path: FieldPath, field_name: str) -> FieldPath:
        """Create a new field path by extending the current one."""
        return field_path.extend(field_name)

    def generate_plan_writer(self, field_path: FieldPath, plan: PlanList) -> None:
        """Generate code for a complete plan."""
        target_type, fields = plan
        self.message_classes.add(target_type)

        # Handle empty message case (ROS 2 structure_needs_at_least_one_member)
        if not fields:
            self.code.append("_buffer.append(0)  # structure_needs_at_least_one_member")
            self.code.append("_offset += 1")
            self.reset_alignment()
            return

        for field in fields:
            self.generate_type_writer(field_path, field)

    def generate_encoder_code(self, func_name: str) -> str:
        """Generate Python source code for an encoder function"""
        with self.code.indent(f"def {func_name}(message):"):
            self.code.append("_buffer = bytearray()")
            self.code.append("_offset = 0")

            # Add CDR header (4 bytes: endianness + padding)
            self.code.append("_buffer.extend(b'\\x00\\x01\\x00\\x00')  # CDR header")
            self.code.append("_offset += 4")

            # Generate the main encoding code first to collect all types
            message_path = FieldPath("message")
            self.generate_plan_writer(message_path, self.plan)

            # Add struct pattern variables to prolog
            for var in self.struct_patterns.values():
                self.code.prolog(f"{var} = {var}g")
            for f, t in self.alias.items():
                self.code.prolog(f"{t} = {f}")

            self.code.append("return bytes(_buffer)")

        return str(self.code)

    def create_namespace(self) -> dict[str, Any]:
        """Create the execution namespace with all required functions and classes."""

        namespace: dict[str, Any] = {
            "_encode_utf8": codecs.utf_8_encode,
            "_get_field": _get_field,
            # Limit builtins for security
            "__builtins__": {
                "bytearray": bytearray,
                "bytes": bytes,
                "len": len,
                "range": range,
                "NotImplementedError": NotImplementedError,
            },
        }

        # Add struct pattern variables
        for pattern, var_name in self.struct_patterns.items():
            namespace[f"{var_name}g"] = struct.Struct(pattern).pack

        # Add message classes
        for msg_class in self.message_classes:
            namespace[msg_class.__name__] = msg_class

        return namespace


def create_encoder(plan: PlanList, *, comments: bool = True) -> EncoderFunction:
    """Create an encoder function from an execution plan using code generation.

    This generates optimized Python code for the specific plan, eliminating
    all dispatch overhead and function call overhead.
    """
    factory = EncoderGeneratorFactory(plan, comments=comments)
    target_type_name = f"encoder_{plan[0].__name__}_main"
    code = factory.generate_encoder_code(target_type_name)
    namespace = factory.create_namespace()

    exec(code, namespace)  # noqa: S102
    return cast("EncoderFunction", namespace[target_type_name])
