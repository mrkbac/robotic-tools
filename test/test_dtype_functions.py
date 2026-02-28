"""Tests for dtype conversion functions."""

import numpy as np
import pytest

from pointcloud2 import PointField, dtype_from_fields, fields_from_dtype


class TestDtypeFromFields:
    """Test the dtype_from_fields function."""

    def test_single_field_float32(self):
        """Test converting single FLOAT32 field to dtype."""
        fields = [PointField('x', 0, PointField.FLOAT32)]
        dtype = dtype_from_fields(fields)

        assert dtype.names == ('x',)
        assert dtype['x'] == np.float32
        assert dtype.itemsize == 4

    def test_multiple_fields(self):
        """Test converting multiple fields to dtype."""
        fields = [
            PointField('x', 0, PointField.FLOAT32),
            PointField('y', 4, PointField.FLOAT32),
            PointField('z', 8, PointField.FLOAT32),
        ]
        dtype = dtype_from_fields(fields)

        assert dtype.names == ('x', 'y', 'z')
        assert dtype['x'] == np.float32
        assert dtype['y'] == np.float32
        assert dtype['z'] == np.float32

    def test_field_with_count_greater_than_one(self):
        """Test converting field with count > 1 to dtype."""
        fields = [PointField('rgb', 0, PointField.UINT8, 3)]
        dtype = dtype_from_fields(fields)

        assert dtype.names == ('rgb_0', 'rgb_1', 'rgb_2')
        assert dtype['rgb_0'] == np.uint8
        assert dtype['rgb_1'] == np.uint8
        assert dtype['rgb_2'] == np.uint8

    def test_mixed_datatypes(self):
        """Test converting fields with different datatypes."""
        fields = [
            PointField('x', 0, PointField.FLOAT32),
            PointField('intensity', 4, PointField.UINT8),
            PointField('ring', 5, PointField.INT16),
        ]
        dtype = dtype_from_fields(fields)

        assert dtype.names == ('x', 'intensity', 'ring')
        assert dtype['x'] == np.float32
        assert dtype['intensity'] == np.uint8
        assert dtype['ring'] == np.int16

    def test_empty_field_name(self):
        """Test converting field with empty name."""
        fields = [PointField('', 0, PointField.FLOAT32)]
        dtype = dtype_from_fields(fields)

        assert dtype.names == ('unnamed_field_0',)
        assert dtype['unnamed_field_0'] == np.float32

    def test_with_point_step(self):
        """Test dtype_from_fields with explicit point_step."""
        fields = [PointField('x', 0, PointField.FLOAT32)]
        dtype = dtype_from_fields(fields, point_step=16)

        assert dtype.itemsize == 16

    def test_field_count_zero_raises_error(self):
        """Test that field with count=0 raises assertion error."""
        fields = [PointField('x', 0, PointField.FLOAT32, 0)]
        with pytest.raises(AssertionError, match="Can't process fields with count = 0"):
            dtype_from_fields(fields)

    def test_duplicate_field_names_raises_error(self):
        """Test that duplicate field names raise assertion error."""
        fields = [
            PointField('x', 0, PointField.FLOAT32),
            PointField('x', 4, PointField.FLOAT32),
        ]
        with pytest.raises(AssertionError, match='Duplicate field names are not allowed'):
            dtype_from_fields(fields)

    def test_all_datatype_constants(self):
        """Test all PointField datatype constants."""
        fields = [
            PointField('int8', 0, PointField.INT8),
            PointField('uint8', 1, PointField.UINT8),
            PointField('int16', 2, PointField.INT16),
            PointField('uint16', 4, PointField.UINT16),
            PointField('int32', 6, PointField.INT32),
            PointField('uint32', 10, PointField.UINT32),
            PointField('float32', 14, PointField.FLOAT32),
            PointField('float64', 18, PointField.FLOAT64),
        ]
        dtype = dtype_from_fields(fields)

        expected_types = [
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.float32,
            np.float64,
        ]

        for field, expected_type in zip(fields, expected_types):
            assert dtype[field.name] == expected_type


class TestFieldsFromDtype:
    """Test the fields_from_dtype function."""

    def test_single_field_float32(self):
        """Test converting single float32 dtype to fields."""
        dtype = np.dtype([('x', '<f4')])
        fields = fields_from_dtype(dtype)

        assert len(fields) == 1
        assert fields[0].name == 'x'
        assert fields[0].offset == 0
        assert fields[0].datatype == PointField.FLOAT32
        assert fields[0].count == 1

    def test_multiple_fields(self):
        """Test converting multiple field dtype to fields."""
        dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        fields = fields_from_dtype(dtype)

        assert len(fields) == 3
        assert fields[0].name == 'x'
        assert fields[1].name == 'y'
        assert fields[2].name == 'z'
        assert all(f.datatype == PointField.FLOAT32 for f in fields)

    def test_mixed_datatypes(self):
        """Test converting dtype with mixed datatypes to fields."""
        dtype = np.dtype([('x', '<f4'), ('intensity', 'u1'), ('ring', '<i2')])
        fields = fields_from_dtype(dtype)

        assert len(fields) == 3
        assert fields[0].name == 'x'
        assert fields[0].datatype == PointField.FLOAT32
        assert fields[1].name == 'intensity'
        assert fields[1].datatype == PointField.UINT8
        assert fields[2].name == 'ring'
        assert fields[2].datatype == PointField.INT16

    def test_all_supported_dtypes(self):
        """Test all supported numpy dtypes."""
        dtype = np.dtype(
            [
                ('int8', 'i1'),
                ('uint8', 'u1'),
                ('int16', '<i2'),
                ('uint16', '<u2'),
                ('int32', '<i4'),
                ('uint32', '<u4'),
                ('float32', '<f4'),
                ('float64', '<f8'),
            ],
        )
        fields = fields_from_dtype(dtype)

        expected_datatypes = [
            PointField.INT8,
            PointField.UINT8,
            PointField.INT16,
            PointField.UINT16,
            PointField.INT32,
            PointField.UINT32,
            PointField.FLOAT32,
            PointField.FLOAT64,
        ]

        assert len(fields) == len(expected_datatypes)
        for field, expected_datatype in zip(fields, expected_datatypes):
            assert field.datatype == expected_datatype

    def test_roundtrip_conversion(self):
        """Test that dtype->fields->dtype conversion is consistent."""
        original_fields = [
            PointField('x', 0, PointField.FLOAT32),
            PointField('y', 4, PointField.FLOAT32),
            PointField('z', 8, PointField.FLOAT32),
            PointField('intensity', 12, PointField.UINT8),
        ]

        # Convert to dtype and back
        dtype = dtype_from_fields(original_fields)
        converted_fields = fields_from_dtype(dtype)

        # Check that we get the same field information
        assert len(converted_fields) == len(original_fields)
        for original, converted in zip(original_fields, converted_fields):
            assert original.name == converted.name
            assert original.datatype == converted.datatype
            # Note: offsets might differ due to dtype structure, but datatypes should match
