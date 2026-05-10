"""Shared TypedDict shapes for decoded ROS messages."""

from __future__ import annotations

from typing import TypedDict


class Stamp(TypedDict):
    sec: int
    nanosec: int


class Header(TypedDict):
    stamp: Stamp
    frame_id: str
