# PointCloud2

PointCloud2 lib for non ROS environment.

[![PyPI version](https://img.shields.io/pypi/v/pointcloud2.svg)](https://pypi.python.org/pypi/pointcloud2/)
[![PyPI license](https://img.shields.io/pypi/l/pointcloud2.svg)](https://pypi.python.org/pypi/pointcloud2/)
[![PyPI download month](https://img.shields.io/pypi/dm/pointcloud2.svg)](https://pypi.python.org/pypi/pointcloud2/)

## Usage

```python

>>> import numpy as np
>>> from pointcloud2 import create_cloud, read_points, PointField
>>> fields = [
...     PointField('x', 0, PointField.FLOAT32, 1),
...     PointField('y', 4, PointField.FLOAT32, 1),
...     PointField('z', 8, PointField.FLOAT32, 1),
...     PointField('rgb', 12, PointField.UINT32, 1),
... ]
>>> points = np.array([
...     [1, 0, 0, 255],
...     [0, 1, 0, 255],
...     [0, 0, 1, 255],
... ], dtype=np.float32)
>>> cloud = create_cloud(header=None, fields=fields, points=points)
>>> cloud.height, cloud.width, cloud.point_step, cloud.row_step, len(cloud.data)
(1, 3, 16, 48, 48)
>>> read_points(cloud)
array([(1., 0., 0., 255), (0., 1., 0., 255), (0., 0., 1., 255)],
      dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')])

```
