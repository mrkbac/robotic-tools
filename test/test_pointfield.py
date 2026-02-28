"""Tests for PointField class."""

from pointcloud2 import PointField


class TestPointField:
    """Test the PointField class."""

    def test_pointfield_creation(self):
        """Test creating a PointField instance."""
        field = PointField('x', 0, PointField.FLOAT32)
        assert field.name == 'x'
        assert field.offset == 0
        assert field.datatype == PointField.FLOAT32
        assert field.count == 1

    def test_pointfield_with_count(self):
        """Test creating a PointField with count > 1."""
        field = PointField('rgb', 12, PointField.UINT8, 3)
        assert field.name == 'rgb'
        assert field.offset == 12
        assert field.datatype == PointField.UINT8
        assert field.count == 3

    def test_pointfield_constants(self):
        """Test PointField datatype constants."""
        assert PointField.INT8 == 1
        assert PointField.UINT8 == 2
        assert PointField.INT16 == 3
        assert PointField.UINT16 == 4
        assert PointField.INT32 == 5
        assert PointField.UINT32 == 6
        assert PointField.FLOAT32 == 7
        assert PointField.FLOAT64 == 8

    def test_pointfield_repr(self):
        """Test PointField string representation."""
        field = PointField('y', 4, PointField.FLOAT32)
        expected = "PointField(name='y', offset=4, datatype=7, count=1)"
        assert repr(field) == expected

    def test_pointfield_equality(self):
        """Test PointField equality comparison."""
        field1 = PointField('x', 0, PointField.FLOAT32)
        field2 = PointField('x', 0, PointField.FLOAT32)
        field3 = PointField('y', 0, PointField.FLOAT32)

        assert field1 == field2
        assert field1 != field3

    def test_pointfield_different_datatypes(self):
        """Test PointField with different datatypes."""
        fields = [
            PointField('int8_field', 0, PointField.INT8),
            PointField('uint8_field', 1, PointField.UINT8),
            PointField('int16_field', 2, PointField.INT16),
            PointField('uint16_field', 4, PointField.UINT16),
            PointField('int32_field', 6, PointField.INT32),
            PointField('uint32_field', 10, PointField.UINT32),
            PointField('float32_field', 14, PointField.FLOAT32),
            PointField('float64_field', 18, PointField.FLOAT64),
        ]

        expected_datatypes = [1, 2, 3, 4, 5, 6, 7, 8]
        for field, expected_datatype in zip(fields, expected_datatypes):
            assert field.datatype == expected_datatype
