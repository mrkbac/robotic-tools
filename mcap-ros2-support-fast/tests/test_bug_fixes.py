"""Tests for bug fixes comparing against reference implementation."""

from mcap_ros2_support_fast._planner import create_decoder_function, create_encoder_function

PARAMETER_EVENT_SCHEMA = """\
builtin_interfaces/Time stamp
string node
Parameter[] new_parameters
Parameter[] changed_parameters
Parameter[] deleted_parameters
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: rcl_interfaces/Parameter
string name
ParameterValue value
================================================================================
MSG: rcl_interfaces/ParameterValue
uint8 type
bool bool_value
int64 integer_value
float64 double_value
string string_value
byte[] byte_array_value
bool[] bool_array_value
int64[] integer_array_value
float64[] double_array_value
string[] string_array_value
"""


def test_parameter_event_empty_primitive_arrays_preserve_sequence_alignment() -> None:
    payload = bytes.fromhex(
        "00010000e0944a68612ef025100000002f7265636f72645f6e6f705f62696700"
        "010000000d0000007573655f73696d5f74696d65000100000000000000000000"
        "0000000000000000000000010000000000000000000000000000000000000000"
        "00000000000000000000000000000000"
    )
    decoder = create_decoder_function("rcl_interfaces/msg/ParameterEvent", PARAMETER_EVENT_SCHEMA)

    event = decoder(payload)

    assert event.node == "/record_nop_big"
    assert len(event.new_parameters) == 1
    parameter = event.new_parameters[0]
    assert parameter.name == "use_sim_time"
    assert parameter.value.type == 1
    assert list(parameter.value.integer_array_value) == []
    assert list(parameter.value.double_array_value) == []
    assert parameter.value.string_array_value == []
    assert event.changed_parameters == []
    assert event.deleted_parameters == []


class TestCharTypeBugFix:
    """Test that char type is correctly mapped to signed byte."""

    def test_char_type_negative_values(self):
        """Test char fields with negative values."""
        schema = """
int8 field1
char field2
"""
        decoder = create_decoder_function("my_msgs/TestChar", schema)
        encoder = create_encoder_function("my_msgs/TestChar", schema)

        # Create a message with a negative char value
        msg = type("TestChar", (), {})()
        msg.field1 = -5
        msg.field2 = -10  # Negative char value

        # Encode
        encoded = encoder(msg)

        # Decode
        decoded = decoder(encoded)

        # Verify char field preserved negative value
        assert decoded.field1 == -5
        assert decoded.field2 == -10, "Char should preserve negative values (signed byte)"

    def test_char_array_negative_values(self):
        """Test char arrays with negative values."""
        schema = """
char[3] field
"""
        decoder = create_decoder_function("my_msgs/TestCharArray", schema)
        encoder = create_encoder_function("my_msgs/TestCharArray", schema)

        msg = type("TestCharArray", (), {})()
        # Encode negative values as unsigned bytes (twos complement)
        msg.field = bytes([246, 236, 226])  # These are -10, -20, -30 as unsigned bytes

        encoded = encoder(msg)
        decoded = decoder(encoded)

        # Convert to signed integers for comparison
        result = [b if b < 128 else b - 256 for b in decoded.field]
        assert result == [-10, -20, -30], "Char array should preserve negative values"


class TestBoundedArrays:
    """Test bounded array support (<=N syntax)."""

    def test_bounded_primitive_array_within_bounds(self):
        """Test bounded array with data within bounds."""
        schema = """
int32[<=5] numbers
"""
        decoder = create_decoder_function("my_msgs/TestBounded", schema)
        encoder = create_encoder_function("my_msgs/TestBounded", schema)

        msg = type("TestBounded", (), {})()
        msg.numbers = [1, 2, 3]  # Within bounds

        encoded = encoder(msg)
        decoded = decoder(encoded)

        assert list(decoded.numbers) == [1, 2, 3]

    def test_bounded_primitive_array_exceeds_bounds(self):
        """Test bounded array with data exceeding bounds gets truncated."""
        schema = """
int32[<=5] numbers
"""
        encoder = create_encoder_function("my_msgs/TestBounded", schema)
        decoder = create_decoder_function("my_msgs/TestBounded", schema)

        msg = type("TestBounded", (), {})()
        msg.numbers = [1, 2, 3, 4, 5, 6, 7, 8]  # Exceeds bounds

        encoded = encoder(msg)
        decoded = decoder(encoded)

        # Should be truncated to first 5 elements
        assert list(decoded.numbers) == [1, 2, 3, 4, 5]

    def test_bounded_complex_array(self):
        """Test bounded arrays of complex types."""
        schema = """
NestedMsg[<=3] items
============
MSG: my_msgs/NestedMsg
int32 value
"""
        encoder = create_encoder_function("my_msgs/TestBoundedComplex", schema)
        decoder = create_decoder_function("my_msgs/TestBoundedComplex", schema)

        nested_msg = type("NestedMsg", (), {})
        msg = type("TestBoundedComplex", (), {})()

        # Create 5 items (exceeds bound of 3)
        items = []
        for i in range(5):
            item = nested_msg()
            item.value = i * 10
            items.append(item)

        msg.items = items

        encoded = encoder(msg)
        decoded = decoder(encoded)

        # Should be truncated to first 3 elements
        assert len(decoded.items) == 3
        assert [item.value for item in decoded.items] == [0, 10, 20]

    def test_bounded_vs_fixed_vs_dynamic_arrays(self):
        """Test that bounded, fixed, and dynamic arrays are handled correctly."""
        schema = """
int32[5] fixed
int32[] dynamic
int32[<=5] bounded
"""
        encoder = create_encoder_function("my_msgs/TestArrayTypes", schema)
        decoder = create_decoder_function("my_msgs/TestArrayTypes", schema)

        msg = type("TestArrayTypes", (), {})()
        msg.fixed = [1, 2, 3, 4, 5]  # Must be exactly 5
        msg.dynamic = [10, 20, 30, 40, 50, 60, 70]  # Can be any length
        msg.bounded = [100, 200, 300, 400, 500, 600, 700]  # Will be truncated to 5

        encoded = encoder(msg)
        decoded = decoder(encoded)

        assert list(decoded.fixed) == [1, 2, 3, 4, 5]
        assert list(decoded.dynamic) == [10, 20, 30, 40, 50, 60, 70]
        assert list(decoded.bounded) == [100, 200, 300, 400, 500]  # Truncated


class TestDefaultValues:
    """Test default value application during serialization."""

    def test_primitive_default_values(self):
        """Test that default values are applied for primitive types."""
        schema = """
int32 count 42
string name "default_name"
float32 value 3.14
bool flag true
"""
        encoder = create_encoder_function("my_msgs/TestDefaults", schema)
        decoder = create_decoder_function("my_msgs/TestDefaults", schema)

        # Create message with None values
        msg = type("TestDefaults", (), {})()
        msg.count = None
        msg.name = None
        msg.value = None
        msg.flag = None

        encoded = encoder(msg)
        decoded = decoder(encoded)

        # Should use default values
        assert decoded.count == 42
        assert decoded.name == "default_name"
        assert abs(decoded.value - 3.14) < 0.01
        assert decoded.flag is True

    def test_fallback_defaults_for_primitives(self):
        """Test fallback defaults when no default is specified."""
        schema = """
int32 count
string name
float32 value
bool flag
"""
        encoder = create_encoder_function("my_msgs/TestFallbacks", schema)
        decoder = create_decoder_function("my_msgs/TestFallbacks", schema)

        msg = type("TestFallbacks", (), {})()
        msg.count = None
        msg.name = None
        msg.value = None
        msg.flag = None

        encoded = encoder(msg)
        decoded = decoder(encoded)

        # Should use type-specific fallbacks
        assert decoded.count == 0
        assert decoded.name == ""
        assert decoded.value == 0.0
        assert decoded.flag is False

    def test_explicit_values_override_defaults(self):
        """Test that explicit values override defaults."""
        schema = """
int32 count 42
string name "default_name"
"""
        encoder = create_encoder_function("my_msgs/TestOverride", schema)
        decoder = create_decoder_function("my_msgs/TestOverride", schema)

        msg = type("TestOverride", (), {})()
        msg.count = 100
        msg.name = "custom_name"

        encoded = encoder(msg)
        decoded = decoder(encoded)

        # Should use explicit values, not defaults
        assert decoded.count == 100
        assert decoded.name == "custom_name"

    def test_array_default_values(self):
        """Test default values for array fields."""
        schema = """
int32[] numbers [1, 2, 3]
"""
        encoder = create_encoder_function("my_msgs/TestArrayDefaults", schema)
        decoder = create_decoder_function("my_msgs/TestArrayDefaults", schema)

        msg = type("TestArrayDefaults", (), {})()
        msg.numbers = None

        encoded = encoder(msg)
        decoded = decoder(encoded)

        assert list(decoded.numbers) == [1, 2, 3]
