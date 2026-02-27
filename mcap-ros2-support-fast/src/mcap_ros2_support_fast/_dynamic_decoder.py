import array
import struct
import sys
from typing import Any, Literal, cast

from mcap_ros2_support_fast.code_writer import CodeWriter

from ._cdr import CDR_BIG_ENDIAN, CDR_HEADER_SIZE
from ._plans import (
    TYPE_INFO,
    ActionType,
    DecoderFunction,
    PlanAction,
    PlanActions,
    PlanList,
    PrimitiveGroupAction,
    TypeId,
)


class DecoderGeneratorFactory:
    """Factory class for generating decoder code with managed state."""

    def __init__(
        self, plan: PlanList, *, comments: bool = True, endianness: Literal["<", ">"] = "<"
    ) -> None:
        self.plan = plan
        self.endianness = endianness  # '<' for little-endian, '>' for big-endian
        self.name_counter = 0
        self.struct_patterns: dict[str, str] = {}
        self.code = CodeWriter(comments=comments)
        # Collect all required types and classes during generation
        self.message_classes: set[type] = set()
        self.current_alignment = 8  # perfect alignment at the start
        # Determine if byteswap is needed based on system vs data endianness
        is_system_le = sys.byteorder == "little"
        is_data_le = endianness == "<"
        self.needs_byteswap = is_system_le != is_data_le
        # Static offset tracking: when not None, we know the exact byte position
        # at compile time and can emit absolute offsets instead of _offset variable
        self.static_offset: int | None = None
        # Whether memoryview is needed for array .cast() operations
        self.needs_memoryview: bool = False

    def generate_var_name(self) -> str:
        """Generate a unique variable name."""
        self.name_counter += 1
        return f"_v{self.name_counter}"

    def get_struct_pattern_var_name(self, pattern: str) -> str:
        """Get or create a variable name for a struct pattern."""
        if pattern not in self.struct_patterns:
            safe_name = (
                pattern.replace("<", "le_")
                .replace(">", "be_")
                .replace(" ", "_")
                .replace("?", "bool")
            )
            # Add decoder prefix to prevent conflicts with encoder patterns
            self.struct_patterns[pattern] = f"_d_{safe_name}"
        return self.struct_patterns[pattern]

    def _plan_needs_memoryview(self, plan: PlanList) -> bool:
        """Pre-scan plan to determine if memoryview is needed for array .cast() operations."""
        _, fields = plan
        for field in fields:
            if field.type == ActionType.PRIMITIVE_ARRAY:
                type_id = field.data
                if type_id not in {
                    TypeId.STRING,
                    TypeId.WSTRING,
                    TypeId.UINT8,
                    TypeId.BYTE,
                    TypeId.CHAR,
                }:
                    struct_name = TYPE_INFO[type_id]
                    struct_size = struct.calcsize(struct_name)
                    # .cast() is used when no byteswap needed (or struct_size == 1)
                    if not self.needs_byteswap or struct_size == 1:
                        return True
            elif field.type in {ActionType.COMPLEX, ActionType.COMPLEX_ARRAY}:
                if self._plan_needs_memoryview(field.plan):
                    return True
        return False

    def _emit_assign(self, target: str | None, expr: str) -> None:
        """Emit assignment or return statement (Opt 5: return directly)."""
        if target is None:
            self.code.append(f"return {expr}")
        else:
            self.code.append(f"{target} = {expr}")

    def _ensure_dynamic(self) -> None:
        """Transition from static to dynamic offset tracking if needed."""
        if self.static_offset is not None:
            self.code.append(f"_offset = {self.static_offset}")
            self.static_offset = None

    def generate_alignment(self, size: int) -> None:
        """Generate optimized alignment code for a given size requirement."""
        if self.current_alignment >= size:
            self.current_alignment = size
            return
        self.current_alignment = size
        if size > 1 and size in (2, 4, 8):
            mask = size - 1
            if self.static_offset is not None:
                # Compile-time alignment: pure arithmetic, no code emitted
                self.static_offset = (self.static_offset + mask) & ~mask
            else:
                self.code.append(f"_offset = (_offset + {mask}) & ~{mask}")

    def reset_alignment(self, initial: int = 0) -> None:
        """Reset the current alignment to zero."""
        self.current_alignment = initial

    def generate_primitive_reader(self, target: str, type_id: TypeId) -> None:
        """Generate the reader code for a given TypeId."""
        if type_id in {TypeId.UINT8, TypeId.BYTE, TypeId.CHAR}:
            self.generate_alignment(1)
            if self.static_offset is not None:
                self.code.append(f"{target} = _data[{self.static_offset}]")
                self.static_offset += 1
            else:
                self.code.append(f"{target} = _data[_offset]")
                self.code.append("_offset += 1")

        elif type_id == TypeId.STRING:
            # Read string length (4 bytes, UINT32)
            self.generate_primitive_reader("_str_size", TypeId.UINT32)
            if self.static_offset is not None:
                # String data starts at the current static offset
                data_start = self.static_offset
                with self.code.indent("if _str_size > 1:"):
                    self.code.append(
                        f'{target} = str(_data[{data_start}:{data_start} + _str_size - 1], "utf-8")'
                    )
                with self.code.indent("else:"):
                    self.code.append(f'{target} = ""')
                # Transition to dynamic mode
                self.code.append(f"_offset = {data_start} + _str_size")
                self.static_offset = None
            else:
                with self.code.indent("if _str_size > 1:"):
                    self.code.append(
                        f'{target} = str(_data[_offset:'
                        f'_offset + _str_size - 1], "utf-8")'
                    )
                with self.code.indent("else:"):
                    self.code.append(f'{target} = ""')
                self.code.append("_offset += _str_size")
            self.reset_alignment()  # After string unknown position readjustment
        elif type_id == TypeId.WSTRING:
            self._ensure_dynamic()
            self.code.append("raise NotImplementedError('wstring not implemented')")
        # Standard struct-based types
        elif struct_name := TYPE_INFO.get(type_id):
            struct_size = struct.calcsize(struct_name)
            self.generate_alignment(struct_size)

            pattern = f"{self.endianness}{struct_name}"
            pattern_var = self.get_struct_pattern_var_name(pattern)
            if self.static_offset is not None:
                self.code.append(
                    f"{target}, = {pattern_var}(_data, {self.static_offset})"
                )
                self.static_offset += struct_size
            else:
                self.code.append(f"{target}, = {pattern_var}(_data, _offset)")
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
            # String arrays always need dynamic offset tracking
            self._ensure_dynamic()
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
                        f"{target}[{random_i}] = str(_data[_offset"
                        f" : _offset + _array_size - 1], 'utf-8')"
                    )
                self.code.append("_offset += _array_size")
            self.reset_alignment()  # After string unknown position readjustment
        elif type_id == TypeId.WSTRING:
            self._ensure_dynamic()
            self.code.append("raise NotImplementedError('wstring not implemented')")

        elif type_id in {TypeId.UINT8, TypeId.BYTE, TypeId.CHAR}:
            # Special case for byte arrays — uses _raw directly
            if array_size is None:
                self._ensure_dynamic()
                self.code.append(
                    f"{target} = _data[_offset"
                    f" : _offset + _array_size]"
                )
                self.code.append("_offset += _array_size")
            elif self.static_offset is not None:
                self.code.append(
                    f"{target} = _data[{self.static_offset} : {self.static_offset + array_size}]"
                )
                self.static_offset += array_size
            else:
                self.code.append(
                    f"{target} = _data[_offset"
                    f" : _offset + {array_size}]"
                )
                self.code.append(f"_offset += {array_size}")
            self.reset_alignment()  # After string unknown position readjustment
        elif array_size is None:  # dynamic array
            self._ensure_dynamic()
            struct_name = TYPE_INFO[type_id]
            struct_size = struct.calcsize(struct_name)
            self.generate_alignment(struct_size)
            # Optimize: use fast .cast() when no byteswap needed
            if not self.needs_byteswap or struct_size == 1:
                # Fast path: matching endianness, use memoryview.cast() via _data
                self.code.append(
                    f"{target} = _data[_offset : _offset + _array_size * {struct_size}]"
                    f".cast('{struct_name}')"
                )
            else:
                # Slow path: need byteswap, use array.array
                self.code.append(f"{target} = array.array('{struct_name}')")
                self.code.append(
                    f"{target}.frombytes(_data[_offset"
                    f" : _offset + _array_size * {struct_size}])"
                )
                self.code.append(f"{target}.byteswap()")
            self.code.append(f"_offset += _array_size * {struct_size}")
        else:
            # Fixed-size array
            struct_name = TYPE_INFO[type_id]
            struct_size = struct.calcsize(struct_name)
            self.generate_alignment(struct_size)
            total_bytes = array_size * struct_size
            if self.static_offset is not None:
                if not self.needs_byteswap or struct_size == 1:
                    # Fast path: .cast() via _data with payload-relative offsets
                    self.code.append(
                        f"{target} = _data[{self.static_offset}"
                        f" : {self.static_offset + total_bytes}].cast('{struct_name}')"
                    )
                else:
                    self.code.append(f"{target} = array.array('{struct_name}')")
                    self.code.append(
                        f"{target}.frombytes(_data[{self.static_offset}"
                        f" : {self.static_offset + total_bytes}])"
                    )
                    self.code.append(f"{target}.byteswap()")
                self.static_offset += total_bytes
            else:
                # Optimize: use fast .cast() when no byteswap needed
                if not self.needs_byteswap or struct_size == 1:
                    # Fast path: matching endianness, use memoryview.cast() via _data
                    self.code.append(
                        f"{target} = _data[_offset : _offset + {total_bytes}].cast('{struct_name}')"
                    )
                else:
                    # Slow path: need byteswap, use array.array
                    self.code.append(f"{target} = array.array('{struct_name}')")
                    self.code.append(
                        f"{target}.frombytes(_data[_offset"
                        f" : _offset + {total_bytes}])"
                    )
                    self.code.append(f"{target}.byteswap()")
                self.code.append(f"_offset += {total_bytes}")

    def generate_complex_array(self, target: str, plan: PlanList, array_size: int | None) -> None:
        """Generate code for complex array fields."""
        random_i = self.generate_var_name()
        temp_var = self.generate_var_name()
        if array_size is None:
            self.generate_primitive_reader("_array_size", TypeId.UINT32)
        # Complex arrays always need dynamic offset tracking
        self._ensure_dynamic()

        if array_size is None:
            self.code.append(f"{target} = [None] * _array_size")
            self.code.append(f"for {random_i} in range(_array_size):")
        else:
            self.code.append(f"{target} = [None] * {array_size}")
            self.code.append(f"for {random_i} in range({array_size}):")

        self.reset_alignment()  # Need to reset alignment at the start of loops
        with self.code:
            self.generate_plan(temp_var, plan)
            self.code.append(f"{target}[{random_i}] = {temp_var}")

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
        if self.static_offset is not None:
            self.code.append(
                f"{target_str} = {struct_var}(_data, {self.static_offset})"
            )
            self.static_offset += struct_size
        else:
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

    @staticmethod
    def _compute_fixed_tail(fields: PlanActions) -> int | None:
        """Compute tail size for tail-from-end optimization.

        V1: only applies when the last field is a single PrimitiveGroupAction.
        Returns the tail byte size, or None if the optimization doesn't apply.
        """
        if len(fields) < 2:
            return None
        last = fields[-1]
        if last.type != ActionType.PRIMITIVE_GROUP:
            return None
        struct_format = "".join(TYPE_INFO[tid] for _, tid, _ in last.targets)
        return struct.calcsize(struct_format)

    def _generate_tail_group(self, step: PrimitiveGroupAction) -> list[str]:
        """Generate code for the last PrimitiveGroupAction, skipping dead _offset advance."""
        targets = [(self.generate_var_name(), tid) for _, tid, _ in step.targets]
        struct_format = "".join(TYPE_INFO[field_type] for _, field_type in targets)
        struct_size = struct.calcsize(f"{self.endianness}{struct_format}")
        pattern = f"{self.endianness}{struct_format}"

        # Align to the first type
        first_type_size = struct.calcsize(f"{self.endianness}{TYPE_INFO[targets[0][1]]}")
        self.generate_alignment(first_type_size)

        target_str = ", ".join(name for name, typeid in targets if typeid != TypeId.PADDING)
        struct_var = self.get_struct_pattern_var_name(pattern)
        if self.static_offset is not None:
            self.code.append(
                f"{target_str} = {struct_var}(_data, {self.static_offset})"
            )
            self.static_offset += struct_size
        else:
            self.code.append(f"{target_str} = {struct_var}(_data, _offset)")
        # No _offset advance — this is the last field in the root plan

        last_size = struct.calcsize(f"<{TYPE_INFO[targets[-1][1]]}")
        self.reset_alignment(last_size)
        return [name for name, tid in targets if tid != TypeId.PADDING]

    def generate_plan(
        self, plan_target: str | None, plan: PlanList, *, is_root: bool = False
    ) -> None:
        """Generate code for a complete plan.

        When plan_target is None (root), emits 'return ...' directly (Opt 5).
        """
        target_type, fields = plan
        self.message_classes.add(target_type)

        # Handle empty message case (ROS 2 structure_needs_at_least_one_member)
        if not fields:
            if self.static_offset is not None:
                self.static_offset += 1
            else:
                self.code.append("_offset += 1  # structure_needs_at_least_one_member")
            self._emit_assign(plan_target, f"{target_type.__name__}()")
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
            if self.static_offset is not None:
                self._emit_assign(
                    plan_target,
                    f"{target_type.__name__}"
                    f"(*{struct_var}(_data, {self.static_offset}))",
                )
                self.static_offset += struct_size
            else:
                self._emit_assign(
                    plan_target,
                    f"{target_type.__name__}(*{struct_var}(_data, _offset))",
                )
                # Update offset and alignment
                self.code.append(f"_offset += {struct_size}")

            last_size = struct.calcsize(f"<{TYPE_INFO[targets[-1][1]]}")
            self.reset_alignment(last_size)

            return

        # Compute tail optimization for root-level plans
        tail_size = self._compute_fixed_tail(fields) if is_root else None

        field_vars: list[str] = []
        for i, field in enumerate(fields):
            # Apply tail-from-end optimization: use len(_data) - tail_size
            # instead of _offset for the last PrimitiveGroupAction when in dynamic mode
            if tail_size is not None and i == len(fields) - 1:
                field_vars.extend(self._generate_tail_group(field))
            else:
                field_vars.extend(self.generate_type(field))

        # Create instance
        self._emit_assign(plan_target, f"{target_type.__name__}({', '.join(field_vars)})")

    def generate_decoder_code(
        self,
        func_name: str,
        *,
        be_fallback: str | None = None,
        validate_endianness: int | None = None,
    ) -> str:
        """Generate Python source code for a decoder function.

        Args:
            be_fallback: If set, add LE guard that dispatches to this BE function
                for non-LE data (Opt 1: inline LE dispatcher).
            validate_endianness: If set, add a check that _raw[0] matches this value.
        """
        # Pre-scan to determine if memoryview is needed
        self.needs_memoryview = self._plan_needs_memoryview(self.plan)

        with self.code.indent(f"def {func_name}(_raw):"):
            # Opt 1: Validity check (for BE decoder)
            if validate_endianness is not None:
                with self.code.indent(f"if _raw[0] != {validate_endianness}:"):
                    self.code.append('raise ValueError(f"Invalid CDR header: {_raw[0]:#x}")')

            # Opt 1: LE decoder with BE fallback guard
            if be_fallback:
                with self.code.indent("if _raw[0]:"):
                    self.code.append(f"return {be_fallback}(_raw)")

            # Opt 2: bytes slice for speed; memoryview only when .cast() needed
            if self.needs_memoryview:
                self.code.append(f"_data = memoryview(_raw)[{CDR_HEADER_SIZE}:]")
            else:
                self.code.append(f"_data = _raw[{CDR_HEADER_SIZE}:]")

            # Start in static offset mode - _offset variable is only emitted
            # when we encounter a variable-length field
            self.static_offset = 0

            # Generate the main parsing code first to collect all types
            # Opt 5: pass None as target to emit 'return ...' directly
            self.generate_plan(None, self.plan, is_root=True)

        return str(self.code)


def create_decoder(plan: PlanList, *, comments: bool = True) -> DecoderFunction:
    """Create a decoder function from an execution plan using code generation.

    This generates optimized Python code for the specific plan, eliminating
    all dispatch overhead and function call overhead.

    The returned decoder automatically detects endianness from the CDR header
    and dispatches to the appropriate decoder implementation.

    Optimizations applied:
    - Opt 1: LE decoder inlined as main function, BE stays as fallback
    - Opt 2: bytes slice (_data = _raw[4:]) instead of memoryview, unless .cast() needed
    - Opt 5: Returns constructor result directly (no temp variable)
    """
    # Generate BE decoder with validity check (separate function, cold path)
    factory_be = DecoderGeneratorFactory(plan, comments=comments, endianness=">")
    decoder_be_name = f"decoder_{plan[0].__name__}_be"
    code_be = factory_be.generate_decoder_code(decoder_be_name, validate_endianness=CDR_BIG_ENDIAN)

    # Generate LE decoder inlined as main with BE fallback guard (hot path)
    factory_le = DecoderGeneratorFactory(plan, comments=comments, endianness="<")
    main_name = f"decoder_{plan[0].__name__}_main"
    code_le = factory_le.generate_decoder_code(main_name, be_fallback=decoder_be_name)

    # Create combined namespace with both decoders
    namespace: dict[str, Any] = {
        "array": array,
        "__builtins__": {
            "memoryview": memoryview,
            "list": list,
            "range": range,
            "str": str,
            "ValueError": ValueError,
            "NotImplementedError": NotImplementedError,
        },
    }

    # Add struct patterns from both factories
    for pattern, var_name in factory_le.struct_patterns.items():
        namespace[var_name] = struct.Struct(pattern).unpack_from
    for pattern, var_name in factory_be.struct_patterns.items():
        namespace[var_name] = struct.Struct(pattern).unpack_from

    # Add message classes (same for both)
    for msg_class in factory_le.message_classes:
        namespace[msg_class.__name__] = msg_class

    # Execute BE first (referenced by main), then main
    exec(code_be, namespace)  # noqa: S102
    exec(code_le, namespace)  # noqa: S102

    return cast("DecoderFunction", namespace[main_name])
