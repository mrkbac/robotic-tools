"""Code generation backend for ROS2 message deserialization.

This module generates optimized decoder functions at runtime by creating
specialized Python code for each message type, eliminating all dispatch
overhead and function call overhead from the interpreted approach.
"""

import codecs
import struct
from dataclasses import dataclass, field
from typing import Any

from mcap_ros2_support_fast.code_writer import CodeWriter

from ._plans import (
    ActionType,
    DecoderFunction,
    EncoderFunction,
    PlanAction,
    PlanList,
    TypeId,
)


@dataclass
class FieldPath:
    """Represents a field access path for clean code generation."""

    base_expr: str
    path: list[str] = field(default_factory=list)

    def extend(self, field_name: str) -> "FieldPath":
        """Create a new FieldPath with an additional field."""
        return FieldPath(self.base_expr, [*self.path, field_name])

    def to_code(self) -> str:
        """Generate the field access code."""
        if not self.path:
            return self.base_expr
        path_args = ", ".join(f"'{field}'" for field in self.path)
        return f"_get_field({self.base_expr}, {path_args})"


# Type metadata for code generation
_TYPE_INFO: dict[TypeId, str] = {
    TypeId.BOOL: "?",
    TypeId.BYTE: "B",
    TypeId.CHAR: "B",
    TypeId.FLOAT32: "f",
    TypeId.FLOAT64: "d",
    TypeId.INT8: "b",
    TypeId.UINT8: "B",
    TypeId.INT16: "h",
    TypeId.UINT16: "H",
    TypeId.INT32: "i",
    TypeId.UINT32: "I",
    TypeId.INT64: "q",
    TypeId.UINT64: "Q",
    TypeId.PADDING: "x",  # Padding bytes, no size
}
UTF8_FUNC_NAME = "_decode_utf8"


class CodeGeneratorFactory:
    """Factory class for generating decoder code with managed state."""

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
            self.code.append(f"_offset = (_offset + {mask}) & ~{mask}")

    def reset_alignment(self, initial: int = 0) -> None:
        """Reset the current alignment to zero."""
        self.current_alignment = initial

    def generate_primitive_reader(self, target: str, type_id: TypeId) -> None:
        """Generate the reader code for a given TypeId."""
        if type_id in {TypeId.UINT8, TypeId.BYTE, TypeId.CHAR}:
            self.generate_alignment(1)
            self.code.append(f"{target} = _data[_offset]")
            self.code.append("_offset += 1")

        elif type_id == TypeId.STRING:
            str_fnc = self.generate_alias(UTF8_FUNC_NAME)
            self.generate_primitive_reader("_str_size", TypeId.UINT32)
            self.code.append(
                f'{target} = {str_fnc}(_data[_offset:_offset + _str_size - 1], "strict", True)[0] '
                f'if _str_size > 1 else ""'
            )
            self.code.append("_offset += _str_size")
            self.reset_alignment()  # After string unknown position readjustment
        elif type_id == TypeId.WSTRING:
            self.code.append("raise NotImplementedError('wstring not implemented')")
        # Standard struct-based types
        elif struct_name := _TYPE_INFO.get(type_id):
            struct_size = struct.calcsize(struct_name)
            self.generate_alignment(struct_size)

            pattern = f"<{struct_name}"
            pattern_var = self.get_struct_pattern_var_name(pattern)
            self.code.append(f"{target} = {pattern_var}(_data, _offset)[0]")
            self.code.append(f"_offset += {struct_size}")
        else:
            raise NotImplementedError(f"Unsupported type: {type_id}")

    def generate_primitive_array(
        self, target: str, type_id: TypeId, array_size: int | None
    ) -> None:
        """Generate code for primitive array fields."""
        if array_size is None:  # dynamic array
            self.generate_primitive_reader("_array_size", TypeId.UINT32)

        if type_id == TypeId.STRING:
            str_fnc = self.generate_alias(UTF8_FUNC_NAME)
            if array_size is None:
                self.code.append(f"{target} = [''] * _array_size")
            else:
                self.code.append(f"{target} = [''] * {array_size}")

            random_i = self.generate_var_name()

            self.reset_alignment()  # After string unknown position readjustment
            range_expr = "_array_size" if array_size is None else str(array_size)
            with self.code.indent(f"for {random_i} in range({range_expr}):"):
                self.generate_primitive_reader("_array_size", TypeId.UINT32)
                with self.code.indent("if _array_size > 1:"):
                    self.code.append(
                        f"{target}[{random_i}] = {str_fnc}("
                        f"_data[_offset : _offset + _array_size - 1], 'strict', True)[0]"
                    )
                self.code.append("_offset += _array_size")
        elif type_id == TypeId.WSTRING:
            self.code.append("raise NotImplementedError('wstring not implemented')")

        elif type_id in {TypeId.UINT8, TypeId.BYTE, TypeId.CHAR}:
            # Special case for byte arrays
            if array_size is None:
                self.code.append(f"{target} = _data[_offset : _offset + _array_size]")
                self.code.append("_offset += _array_size")
            else:
                self.code.append(f"{target} = _data[_offset : _offset + {array_size}]")
                self.code.append(f"_offset += {array_size}")
            self.reset_alignment()  # After string unknown position readjustment
        elif array_size is None:  # dynamic array
            struct_name = _TYPE_INFO[type_id]
            struct_size = struct.calcsize(struct_name)
            self.code.append(
                f"{target} = list(_data[_offset : _offset + _array_size * {struct_size}]"
                f".cast('{struct_name}'))"
            )
            self.code.append(f"_offset += _array_size * {struct_size}")
        else:
            # Fixed-size array
            struct_name = _TYPE_INFO[type_id]
            struct_size = struct.calcsize(struct_name)
            self.generate_alignment(struct_size)

            self.code.append(
                f"{target} = list(_data[_offset : _offset + {array_size * struct_size}]"
                f".cast('{struct_name}'))"
            )
            self.code.append(f"_offset += {array_size * struct_size}")

    def generate_complex_array(self, target: str, plan: PlanList, array_size: int | None) -> None:
        """Generate code for complex array fields."""
        random_i = self.generate_var_name()
        if array_size is None:
            self.generate_primitive_reader("_array_size", TypeId.UINT32)
            self.code.append(f"{target} = [None] * _array_size")
            self.code.append(f"for {random_i} in range(_array_size):")
        else:
            self.code.append(f"{target} = [None] * {array_size}")
            self.code.append(f"for {random_i} in range({array_size}):")

        self.reset_alignment()  # Need to reset alignment at the start of loops
        with self.code:
            self.generate_plan(f"{target}[{random_i}]", plan)

    def generate_primitive_group(self, targets: list[tuple[str, TypeId]]) -> None:
        """Generate code for a group of primitive fields."""
        struct_format = "".join(_TYPE_INFO[field_type] for _, field_type in targets)
        struct_size = struct.calcsize(f"<{struct_format}")
        pattern = f"<{struct_format}"

        # Align to the first type
        first_type_size = struct.calcsize(f"<{_TYPE_INFO[targets[0][1]]}")
        self.generate_alignment(first_type_size)

        target_str = ", ".join(name for name, typeid in targets if typeid != TypeId.PADDING)
        struct_var = self.get_struct_pattern_var_name(pattern)
        self.code.append(f"{target_str} = {struct_var}(_data, _offset)")
        self.code.append(f"_offset += {struct_size}")

        last_size = struct.calcsize(f"<{_TYPE_INFO[targets[-1][1]]}")
        # self.reset_alignment()
        self.reset_alignment(last_size)

    def generate_type(self, step: PlanAction) -> list[str]:
        """Generate code for a single plan action."""
        if step.type == ActionType.PRIMITIVE:
            rvar = self.generate_var_name()
            self.generate_primitive_reader(rvar, step.data)
            return [rvar]
        if step.type == ActionType.PRIMITIVE_ARRAY:
            type_id = step.data
            array_size = step.size
            rvar = self.generate_var_name()
            self.generate_primitive_array(rvar, type_id, array_size)
            return [rvar]
        if step.type == ActionType.PRIMITIVE_GROUP:
            rfields = [(self.generate_var_name(), tid) for _, tid in step.targets]
            self.generate_primitive_group(rfields)
            return [name for name, tid in rfields if tid != TypeId.PADDING]
        if step.type == ActionType.COMPLEX:
            rvar = self.generate_var_name()
            self.generate_plan(rvar, step.plan)
            return [rvar]
        if step.type == ActionType.COMPLEX_ARRAY:
            plan = step.plan
            array_size = step.size
            rvar = self.generate_var_name()
            self.generate_complex_array(rvar, plan, array_size)
            return [rvar]
        raise ValueError(f"Unknown action type: {step}")

    def generate_plan(self, plan_target: str, plan: PlanList) -> None:
        """Generate code for a complete plan."""
        target_type, fields = plan
        self.message_classes.add(target_type)
        target_alias = self.generate_alias(target_type.__name__)

        # Handle empty message case (ROS 2 structure_needs_at_least_one_member)
        if not fields:
            self.code.append("_offset += 1  # structure_needs_at_least_one_member")
            self.code.append(f"{plan_target} = {target_alias}()")
            self.reset_alignment()
            return

        # Check if this type only has a single primitive group (optimization case)
        # This should work better with dataclasses that support positional args by default
        if (
            len(fields) == 1
            and fields[0].type == ActionType.PRIMITIVE_GROUP
            and all(typeid != TypeId.PADDING for _, typeid in fields[0].targets)
        ):
            # Get the struct pattern for direct unpacking
            targets = fields[0].targets
            struct_format = "".join(_TYPE_INFO[field_type] for _, field_type in targets)
            pattern = f"<{struct_format}"
            struct_var = self.get_struct_pattern_var_name(pattern)
            struct_size = struct.calcsize(pattern)

            # Generate alignment for the first type
            first_type_size = struct.calcsize(f"<{_TYPE_INFO[targets[0][1]]}")
            self.generate_alignment(first_type_size)

            # Generate optimized constructor call with argument unpacking
            self.code.append(f"{plan_target} = {target_alias}(*{struct_var}(_data, _offset))")

            # Update offset and alignment
            self.code.append(f"_offset += {struct_size}")
            last_size = struct.calcsize(f"<{_TYPE_INFO[targets[-1][1]]}")
            self.reset_alignment(last_size)

            return

        targets = []
        for field in fields:
            targets.extend(self.generate_type(field))

        # Create instance
        self.code.append(f"{plan_target} = {target_alias}({', '.join(targets)})")

    def generate_decoder_code(self, func_name: str) -> str:
        """Generate Python source code for a decoder function"""
        with self.code.indent(f"def {func_name}(_raw):"):
            self.code.append("_data = memoryview(_raw)[4:]")
            self.code.append("_offset = 0")

            # Generate the main parsing code first to collect all types
            ret_var = self.generate_var_name()
            self.generate_plan(ret_var, self.plan)
            for var in self.struct_patterns.values():
                self.code.prolog(f"{var} = {var}g")
            for f, t in self.alias.items():
                self.code.prolog(f"{t} = {f}")
            self.code.append(f"return {ret_var}")

        return str(self.code)

    def create_namespace(self) -> dict[str, Any]:
        """Create the execution namespace with all required functions and classes."""
        namespace: dict[str, Any] = {
            UTF8_FUNC_NAME: codecs.utf_8_decode,
            # Limit builtins for security
            "__builtins__": {
                "memoryview": memoryview,
                "list": list,
                "range": range,
                "NotImplementedError": NotImplementedError,
            },
        }

        # Add struct pattern variables
        for pattern, var_name in self.struct_patterns.items():
            namespace[f"{var_name}g"] = struct.Struct(pattern).unpack_from

        # Add message classes
        for msg_class in self.message_classes:
            namespace[msg_class.__name__] = msg_class

        return namespace


def create_decoder(plan: PlanList, *, comments: bool = True) -> DecoderFunction:
    """Create a decoder function from an execution plan using code generation.

    This generates optimized Python code for the specific plan, eliminating
    all dispatch overhead and function call overhead.
    """
    factory = CodeGeneratorFactory(plan, comments=comments)
    target_type_name = f"decoder_{plan[0].__name__}_main"
    code = factory.generate_decoder_code(target_type_name)
    namespace = factory.create_namespace()

    # print(code)  # Debug: show generated code

    exec(code, namespace)  # noqa: S102
    return namespace[target_type_name]


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
                self.code.append(f"_buffer.extend(b'\\x00' * _pad)")
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
        elif struct_name := _TYPE_INFO.get(type_id):
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
            struct_name = _TYPE_INFO[type_id]
            struct_size = struct.calcsize(struct_name)
            self.generate_alignment(struct_size)

            pattern = f"<{len(value_expr) if array_size else ''}{''.join([struct_name] * (array_size or 1))}"
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
        struct_format = "".join(
            _TYPE_INFO[field_type] for _, field_type in targets if field_type != TypeId.PADDING
        )
        struct_size = struct.calcsize(f"<{struct_format}")
        pattern = f"<{struct_format}"

        # Align to the first type
        first_type_size = struct.calcsize(f"<{_TYPE_INFO[targets[0][1]]}")
        self.generate_alignment(first_type_size)

        field_values = []
        for name, typeid in targets:
            if typeid != TypeId.PADDING:
                target_path = self._get_field_access(field_path, name)
                field_values.append(target_path.to_code())

        struct_var = self.get_struct_pattern_var_name(pattern)
        self.code.append(f"_buffer.extend({struct_var}({', '.join(field_values)}))")
        self.code.append(f"_offset += {struct_size}")

        last_size = struct.calcsize(f"<{_TYPE_INFO[targets[-1][1]]}")
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

        def _get_field(obj: Any, *field_path: str) -> Any:
            """Get nested field using field path (works for both dict and object attributes)."""
            for field in field_path:
                if isinstance(obj, dict):
                    obj = obj[field]
                else:
                    obj = getattr(obj, field)
            return obj

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

    # print(code)  # Debug: show generated code

    exec(code, namespace)  # noqa: S102
    return namespace[target_type_name]
