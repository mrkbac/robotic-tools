import codecs
import struct
from typing import Any, cast

from mcap_ros2_support_fast.code_writer import CodeWriter

from ._plans import (
    TYPE_INFO,
    UTF8_FUNC_NAME,
    ActionType,
    DecoderFunction,
    PlanAction,
    PlanList,
    TypeId,
)


class DecoderGeneratorFactory:
    """Factory class for generating decoder code with managed state."""

    def __init__(self, plan: PlanList, *, comments: bool = True, endianness: str = "<") -> None:
        self.plan = plan
        self.endianness = endianness  # '<' for little-endian, '>' for big-endian
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
            # Add decoder prefix to prevent conflicts with encoder patterns
            self.struct_patterns[pattern] = f"_d_{safe_name}"
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
        elif struct_name := TYPE_INFO.get(type_id):
            struct_size = struct.calcsize(struct_name)
            self.generate_alignment(struct_size)

            pattern = f"{self.endianness}{struct_name}"
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
            self.reset_alignment()  # After string unknown position readjustment
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
            struct_name = TYPE_INFO[type_id]
            struct_size = struct.calcsize(struct_name)
            self.generate_alignment(struct_size)
            self.code.append(
                f"{target} = list(_data[_offset : _offset + _array_size * {struct_size}]"
                f".cast('{struct_name}'))"
            )
            self.code.append(f"_offset += _array_size * {struct_size}")
        else:
            # Fixed-size array
            struct_name = TYPE_INFO[type_id]
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
        struct_format = "".join(TYPE_INFO[field_type] for _, field_type in targets)
        struct_size = struct.calcsize(f"{self.endianness}{struct_format}")
        pattern = f"{self.endianness}{struct_format}"

        # Align to the first type
        first_type_size = struct.calcsize(f"{self.endianness}{TYPE_INFO[targets[0][1]]}")
        self.generate_alignment(first_type_size)

        target_str = ", ".join(name for name, typeid in targets if typeid != TypeId.PADDING)
        struct_var = self.get_struct_pattern_var_name(pattern)
        self.code.append(f"{target_str} = {struct_var}(_data, _offset)")
        self.code.append(f"_offset += {struct_size}")

        last_size = struct.calcsize(f"<{TYPE_INFO[targets[-1][1]]}")
        self.reset_alignment(last_size)

    def generate_type(self, step: PlanAction) -> list[str]:
        """Generate code for a single plan action."""
        if step.type == ActionType.PRIMITIVE:
            rvar = self.generate_var_name()
            self.generate_primitive_reader(rvar, step.data)
            return [rvar]
        if step.type == ActionType.PRIMITIVE_ARRAY:
            type_id = step.data
            # For bounded arrays, treat as dynamic (they have length prefix)
            array_size = None if step.is_upper_bound else step.size
            rvar = self.generate_var_name()
            self.generate_primitive_array(rvar, type_id, array_size)
            return [rvar]
        if step.type == ActionType.PRIMITIVE_GROUP:
            rfields = [(self.generate_var_name(), tid) for _, tid, _ in step.targets]
            self.generate_primitive_group(rfields)
            return [name for name, tid in rfields if tid != TypeId.PADDING]
        if step.type == ActionType.COMPLEX:
            rvar = self.generate_var_name()
            self.generate_plan(rvar, step.plan)
            return [rvar]
        if step.type == ActionType.COMPLEX_ARRAY:
            plan = step.plan
            # For bounded arrays, treat as dynamic (they have length prefix)
            array_size = None if step.is_upper_bound else step.size
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
            and all(typeid != TypeId.PADDING for _, typeid, _ in fields[0].targets)
        ):
            # Get the struct pattern for direct unpacking
            # Convert 3-tuple targets to 2-tuple for unpacking
            targets = [(name, typeid) for name, typeid, _ in fields[0].targets]
            struct_format = "".join(TYPE_INFO[field_type] for _, field_type in targets)
            pattern = f"{self.endianness}{struct_format}"
            struct_var = self.get_struct_pattern_var_name(pattern)
            struct_size = struct.calcsize(pattern)

            # Generate alignment for the first type
            first_type_size = struct.calcsize(f"{self.endianness}{TYPE_INFO[targets[0][1]]}")
            self.generate_alignment(first_type_size)

            # Generate optimized constructor call with argument unpacking
            self.code.append(f"{plan_target} = {target_alias}(*{struct_var}(_data, _offset))")

            # Update offset and alignment
            self.code.append(f"_offset += {struct_size}")
            last_size = struct.calcsize(f"<{TYPE_INFO[targets[-1][1]]}")
            self.reset_alignment(last_size)

            return

        field_vars: list[str] = []
        for field in fields:
            field_vars.extend(self.generate_type(field))

        # Create instance
        self.code.append(f"{plan_target} = {target_alias}({', '.join(field_vars)})")

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
    factory = DecoderGeneratorFactory(plan, comments=comments)
    target_type_name = f"decoder_{plan[0].__name__}_main"
    code = factory.generate_decoder_code(target_type_name)
    namespace = factory.create_namespace()

    # print(code)  # Debug: show generated code

    exec(code, namespace)  # noqa: S102
    return cast("DecoderFunction", namespace[target_type_name])
