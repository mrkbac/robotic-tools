"""Tests for point cloud processing functions."""

import numpy as np
import pytest

from pointcloud2 import PointField, create_cloud, read_points


class MockPointCloud2:
    """Mock PointCloud2 message for testing."""

    def __init__(
        self,
        header,
        height,
        width,
        fields,
        is_bigendian,
        point_step,
        row_step,
        data,
        is_dense,
    ):
        self.header = header
        self.height = height
        self.width = width
        self.fields = fields
        self.is_bigendian = is_bigendian
        self.point_step = point_step
        self.row_step = row_step
        self.data = data
        self.is_dense = is_dense


class TestReadPoints:
    """Test the read_points function."""

    def create_test_cloud(self):
        """Create a test point cloud with x, y, z coordinates."""
        points = np.array(
            [
                (1.0, 2.0, 3.0),
                (4.0, 5.0, 6.0),
                (7.0, 8.0, 9.0),
            ],
            dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')],
        )

        fields = [
            PointField('x', 0, PointField.FLOAT32),
            PointField('y', 4, PointField.FLOAT32),
            PointField('z', 8, PointField.FLOAT32),
        ]

        return MockPointCloud2(
            header=None,
            height=1,
            width=3,
            fields=fields,
            is_bigendian=False,
            point_step=12,
            row_step=36,
            data=points.tobytes(),
            is_dense=True,
        )

    def test_read_all_points(self):
        """Test reading all points from a cloud."""
        cloud = self.create_test_cloud()
        points = read_points(cloud)

        assert len(points) == 3
        assert points.dtype.names == ('x', 'y', 'z')
        np.testing.assert_array_equal(points['x'], [1.0, 4.0, 7.0])
        np.testing.assert_array_equal(points['y'], [2.0, 5.0, 8.0])
        np.testing.assert_array_equal(points['z'], [3.0, 6.0, 9.0])

    def test_read_specific_fields(self):
        """Test reading specific fields from a cloud."""
        cloud = self.create_test_cloud()
        points = read_points(cloud, field_names=['x', 'z'])

        assert len(points) == 3
        assert points.dtype.names == ('x', 'z')
        np.testing.assert_array_equal(points['x'], [1.0, 4.0, 7.0])
        np.testing.assert_array_equal(points['z'], [3.0, 6.0, 9.0])

    def test_read_single_field(self):
        """Test reading a single field from a cloud."""
        cloud = self.create_test_cloud()
        points = read_points(cloud, field_names=['y'])

        assert len(points) == 3
        assert points.dtype.names == ('y',)
        np.testing.assert_array_equal(points['y'], [2.0, 5.0, 8.0])

    def test_read_nonexistent_field_raises_error(self):
        """Test that reading non-existent field raises assertion error."""
        cloud = self.create_test_cloud()
        with pytest.raises(AssertionError, match='Requests field is not in the fields'):
            read_points(cloud, field_names=['nonexistent'])

    def test_read_with_uvs(self):
        """Test reading specific points by indices."""
        cloud = self.create_test_cloud()
        points = read_points(cloud, uvs=[0, 2])

        assert len(points) == 2
        np.testing.assert_array_equal(points['x'], [1.0, 7.0])
        np.testing.assert_array_equal(points['y'], [2.0, 8.0])
        np.testing.assert_array_equal(points['z'], [3.0, 9.0])

    def test_read_with_numpy_uvs(self):
        """Test reading specific points with numpy array indices."""
        cloud = self.create_test_cloud()
        uvs = np.array([1, 2])
        points = read_points(cloud, uvs=uvs)

        assert len(points) == 2
        np.testing.assert_array_equal(points['x'], [4.0, 7.0])
        np.testing.assert_array_equal(points['y'], [5.0, 8.0])
        np.testing.assert_array_equal(points['z'], [6.0, 9.0])

    def create_test_cloud_with_nans(self):
        """Create a test point cloud with NaN values."""
        points = np.array(
            [
                (1.0, 2.0, 3.0),
                (np.nan, 5.0, 6.0),
                (7.0, np.nan, 9.0),
            ],
            dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')],
        )

        fields = [
            PointField('x', 0, PointField.FLOAT32),
            PointField('y', 4, PointField.FLOAT32),
            PointField('z', 8, PointField.FLOAT32),
        ]

        return MockPointCloud2(
            header=None,
            height=1,
            width=3,
            fields=fields,
            is_bigendian=False,
            point_step=12,
            row_step=36,
            data=points.tobytes(),
            is_dense=False,
        )

    def test_skip_nans(self):
        """Test skipping NaN values."""
        cloud = self.create_test_cloud_with_nans()
        points = read_points(cloud, skip_nans=True)

        assert len(points) == 1
        np.testing.assert_array_equal(points['x'], [1.0])
        np.testing.assert_array_equal(points['y'], [2.0])
        np.testing.assert_array_equal(points['z'], [3.0])

    def test_keep_nans(self):
        """Test keeping NaN values."""
        cloud = self.create_test_cloud_with_nans()
        points = read_points(cloud, skip_nans=False)

        assert len(points) == 3
        assert np.isnan(points['x'][1])
        assert np.isnan(points['y'][2])

    def create_organized_cloud(self):
        """Create a 2D organized point cloud."""
        points = np.array(
            [
                (1.0, 2.0, 3.0),
                (4.0, 5.0, 6.0),
                (7.0, 8.0, 9.0),
                (10.0, 11.0, 12.0),
            ],
            dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')],
        )

        fields = [
            PointField('x', 0, PointField.FLOAT32),
            PointField('y', 4, PointField.FLOAT32),
            PointField('z', 8, PointField.FLOAT32),
        ]

        return MockPointCloud2(
            header=None,
            height=2,
            width=2,
            fields=fields,
            is_bigendian=False,
            point_step=12,
            row_step=24,
            data=points.tobytes(),
            is_dense=True,
        )

    def test_reshape_organized_cloud(self):
        """Test reshaping organized cloud to 2D."""
        cloud = self.create_organized_cloud()
        points = read_points(cloud, reshape_organized_cloud=True)

        assert points.shape == (2, 2)
        assert points.dtype.names == ('x', 'y', 'z')


class TestCreateCloud:
    """Test the create_cloud function."""

    def test_create_cloud_from_numpy_array(self):
        """Test creating cloud from numpy array."""
        points = np.array(
            [
                (1.0, 2.0, 3.0),
                (4.0, 5.0, 6.0),
            ],
            dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')],
        )

        fields = [
            PointField('x', 0, PointField.FLOAT32),
            PointField('y', 4, PointField.FLOAT32),
            PointField('z', 8, PointField.FLOAT32),
        ]

        cloud = create_cloud(header=None, fields=fields, points=points)

        assert cloud.width == 2
        assert cloud.height == 1
        assert cloud.fields == fields
        assert cloud.point_step == 12
        assert cloud.row_step == 24
        assert len(cloud.data) == 24

    def test_create_cloud_from_unstructured_array(self):
        """Test creating cloud from unstructured numpy array."""
        points = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
        )

        fields = [
            PointField('x', 0, PointField.FLOAT32),
            PointField('y', 4, PointField.FLOAT32),
            PointField('z', 8, PointField.FLOAT32),
        ]

        cloud = create_cloud(header=None, fields=fields, points=points)

        assert cloud.width == 2
        assert cloud.height == 1
        assert cloud.fields == fields

    def test_create_cloud_from_list(self):
        """Test creating cloud from list of points."""
        points = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]

        fields = [
            PointField('x', 0, PointField.FLOAT32),
            PointField('y', 4, PointField.FLOAT32),
            PointField('z', 8, PointField.FLOAT32),
        ]

        cloud = create_cloud(header=None, fields=fields, points=points)

        assert cloud.width == 2
        assert cloud.height == 1
        assert cloud.fields == fields

    def test_create_organized_cloud(self):
        """Test creating organized (2D) cloud."""
        points = np.array(
            [
                [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
                [(7.0, 8.0, 9.0), (10.0, 11.0, 12.0)],
            ],
            dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')],
        )

        fields = [
            PointField('x', 0, PointField.FLOAT32),
            PointField('y', 4, PointField.FLOAT32),
            PointField('z', 8, PointField.FLOAT32),
        ]

        cloud = create_cloud(header=None, fields=fields, points=points)

        assert cloud.width == 2
        assert cloud.height == 2
        assert cloud.fields == fields

    def test_create_cloud_with_custom_step(self):
        """Test creating cloud with custom point step."""
        # Create points with unstructured array to test custom step
        points = np.array(
            [
                [1.0, 2.0, 3.0],
            ],
        )

        fields = [
            PointField('x', 0, PointField.FLOAT32),
            PointField('y', 4, PointField.FLOAT32),
            PointField('z', 8, PointField.FLOAT32),
        ]

        cloud = create_cloud(header=None, fields=fields, points=points, step=16)

        assert cloud.point_step == 16
        assert cloud.row_step == 16

    def test_create_cloud_too_many_dimensions_raises_error(self):
        """Test that creating cloud with too many dimensions raises error."""
        points = np.array([[[(1.0, 2.0, 3.0)]]], dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])

        fields = [
            PointField('x', 0, PointField.FLOAT32),
            PointField('y', 4, PointField.FLOAT32),
            PointField('z', 8, PointField.FLOAT32),
        ]

        with pytest.raises(AssertionError, match='Too many dimensions'):
            create_cloud(header=None, fields=fields, points=points)

    def test_mismatched_dtype_raises_error(self):
        """Test that mismatched dtype raises assertion error."""
        points = np.array(
            [
                (1.0, 2.0, 3.0),
            ],
            dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')],
        )

        fields = [
            PointField('a', 0, PointField.FLOAT32),  # Different field name
            PointField('b', 4, PointField.FLOAT32),
            PointField('c', 8, PointField.FLOAT32),
        ]

        with pytest.raises(
            AssertionError,
            match='PointFields and structured NumPy array dtype do not match',
        ):
            create_cloud(header=None, fields=fields, points=points)

    def test_roundtrip_create_read(self):
        """Test creating a cloud and reading it back."""
        original_points = np.array(
            [
                (1.0, 2.0, 3.0),
                (4.0, 5.0, 6.0),
            ],
            dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')],
        )

        fields = [
            PointField('x', 0, PointField.FLOAT32),
            PointField('y', 4, PointField.FLOAT32),
            PointField('z', 8, PointField.FLOAT32),
        ]

        cloud = create_cloud(header=None, fields=fields, points=original_points)
        read_back_points = read_points(cloud)

        assert len(read_back_points) == len(original_points)
        np.testing.assert_array_equal(read_back_points['x'], original_points['x'])
        np.testing.assert_array_equal(read_back_points['y'], original_points['y'])
        np.testing.assert_array_equal(read_back_points['z'], original_points['z'])

    def test_create_cloud_from_dict(self):
        """Test creating cloud from a dictionary of fields."""
        points = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
        )

        fields = [
            {'name': 'x', 'datatype': PointField.FLOAT32},
            {'name': 'y', 'datatype': PointField.FLOAT32},
            {'name': 'z', 'datatype': PointField.FLOAT32},
        ]

        cloud = create_cloud(header=None, fields=fields, points=points)

        assert cloud.width == 2
        assert cloud.height == 1
        assert len(cloud.fields) == 3
        assert cloud.fields[0].name == 'x'
        assert cloud.fields[1].name == 'y'
        assert cloud.fields[2].name == 'z'
        assert cloud.point_step == 12
        assert cloud.row_step == 24
        assert len(cloud.data) == 24
