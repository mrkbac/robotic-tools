"""ROS1 bag format v2.0 reader.

A lightweight, pure-Python reader for ROS1 bag files that requires
no ROS installation. Designed for use with MCAP conversion tools.
"""

from pymcap_cli.rosbag_reader._reader import read_bag_info, read_bag_messages
from pymcap_cli.rosbag_reader._types import BagConnection, BagInfo, BagMessage

__all__ = [
    "BagConnection",
    "BagInfo",
    "BagMessage",
    "read_bag_info",
    "read_bag_messages",
]
