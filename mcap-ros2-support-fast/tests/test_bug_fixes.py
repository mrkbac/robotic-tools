"""Tests for bug fixes comparing against reference implementation."""

from mcap_ros2_support_fast._planner import generate_dynamic, serialize_dynamic


class TestCharTypeBugFix:
    """Test that char type is correctly mapped to signed byte."""

    def test_char_type_negative_values(self):
        """Test char fields with negative values."""
        schema = """
int8 field1
char field2
"""
        decoder = generate_dynamic("my_msgs/TestChar", schema)
        encoder = serialize_dynamic("my_msgs/TestChar", schema)

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
        decoder = generate_dynamic("my_msgs/TestCharArray", schema)
        encoder = serialize_dynamic("my_msgs/TestCharArray", schema)

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
        decoder = generate_dynamic("my_msgs/TestBounded", schema)
        encoder = serialize_dynamic("my_msgs/TestBounded", schema)

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
        encoder = serialize_dynamic("my_msgs/TestBounded", schema)
        decoder = generate_dynamic("my_msgs/TestBounded", schema)

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
        encoder = serialize_dynamic("my_msgs/TestBoundedComplex", schema)
        decoder = generate_dynamic("my_msgs/TestBoundedComplex", schema)

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
        encoder = serialize_dynamic("my_msgs/TestArrayTypes", schema)
        decoder = generate_dynamic("my_msgs/TestArrayTypes", schema)

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
        encoder = serialize_dynamic("my_msgs/TestDefaults", schema)
        decoder = generate_dynamic("my_msgs/TestDefaults", schema)

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
        encoder = serialize_dynamic("my_msgs/TestFallbacks", schema)
        decoder = generate_dynamic("my_msgs/TestFallbacks", schema)

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
        encoder = serialize_dynamic("my_msgs/TestOverride", schema)
        decoder = generate_dynamic("my_msgs/TestOverride", schema)

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
        encoder = serialize_dynamic("my_msgs/TestArrayDefaults", schema)
        decoder = generate_dynamic("my_msgs/TestArrayDefaults", schema)

        msg = type("TestArrayDefaults", (), {})()
        msg.numbers = None

        encoded = encoder(msg)
        decoded = decoder(encoded)

        assert list(decoded.numbers) == [1, 2, 3]
