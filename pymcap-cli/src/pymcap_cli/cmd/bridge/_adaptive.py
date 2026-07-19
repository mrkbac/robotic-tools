"""Transport-agnostic adaptive video-quality control shared by bridge commands.

Both ``bridge serve`` (file playback) and ``bridge proxy`` (live relay) encode
video on the fly and must shed bitrate when a client's send queue backs up. The
policy is identical for both: keep the requested quality constant, first cap the
frame rate, then reduce resolution, and recover after a sustained clean period.

This module owns that policy so neither command re-implements it. It knows
nothing about websockets, MCAP, or the playback engine — callers supply a
per-channel congestion boolean and a monotonic ``now`` and apply the returned
rung to their own encoder.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Generic, TypeVar

_ADAPTIVE_WINDOW_SECONDS = 2.0
_ADAPTIVE_CONGESTION_RATIO = 0.25
_ADAPTIVE_HOLD_SECONDS = 3.0
_ADAPTIVE_RECOVERY_SECONDS = 30.0
MAX_VIDEO_FPS = 30.0
_ADAPTIVE_SCALE_FACTORS = (0.75, 0.5, 0.375)
_ADAPTIVE_FRAME_RATES = (20.0, 10.0, 5.0, 2.0)

K = TypeVar("K", bound=Hashable)


@dataclass(frozen=True, slots=True)
class AdaptiveVideoRung:
    """One quality level: constant ``quality``, a resolution ``scale_factor``, and a cap."""

    quality: int
    scale_factor: float
    max_fps: float | None = None


@dataclass(frozen=True, slots=True)
class RungTransition:
    """A change in the active rung for one channel."""

    previous: AdaptiveVideoRung
    current: AdaptiveVideoRung


@dataclass(slots=True)
class AdaptiveVideoController:
    """Hysteresis state machine mapping a congestion signal to a rung index.

    Steps the rung up when at least :data:`_ADAPTIVE_CONGESTION_RATIO` of a
    :data:`_ADAPTIVE_WINDOW_SECONDS` window was congested (respecting a hold),
    and steps back down after :data:`_ADAPTIVE_RECOVERY_SECONDS` of clean time.
    """

    max_rung: int = 3
    rung: int = 0
    _window_started: float | None = None
    _observations: int = 0
    _congested_observations: int = 0
    _last_change: float = -math.inf
    _clean_since: float | None = None

    def observe(self, *, is_congested: bool, now: float) -> int:
        if is_congested:
            self._clean_since = None
        elif self._clean_since is None:
            self._clean_since = now

        if (
            self.rung > 0
            and self._clean_since is not None
            and now - self._clean_since >= _ADAPTIVE_RECOVERY_SECONDS
            and now - self._last_change >= _ADAPTIVE_HOLD_SECONDS
        ):
            self.rung -= 1
            self._last_change = now
            self._clean_since = now
            self._reset_window(now, is_congested)
            return self.rung

        if self._window_started is None:
            self._reset_window(now, is_congested)
            return self.rung

        self._observations += 1
        self._congested_observations += int(is_congested)
        if now - self._window_started < _ADAPTIVE_WINDOW_SECONDS:
            return self.rung

        congestion_ratio = self._congested_observations / self._observations
        # Only degrade while congestion is still present: a stale window whose
        # single congested sample is seconds old must not keep stepping quality
        # down during an otherwise clean period.
        if (
            is_congested
            and congestion_ratio >= _ADAPTIVE_CONGESTION_RATIO
            and self.rung < self.max_rung
        ):
            if now - self._last_change >= _ADAPTIVE_HOLD_SECONDS:
                self.rung += 1
                self._last_change = now
                self._reset_window(now, is_congested)
            return self.rung

        self._reset_window(now, is_congested)
        return self.rung

    def _reset_window(self, now: float, is_congested: bool) -> None:
        self._window_started = now
        self._observations = 1
        self._congested_observations = int(is_congested)


def adaptive_video_rungs(quality: int) -> tuple[AdaptiveVideoRung, ...]:
    """Build the rung ladder: cap frame rate first, then reduce resolution."""
    frame_rate_rungs = tuple(
        AdaptiveVideoRung(quality, 1.0, max_fps)
        for max_fps in (MAX_VIDEO_FPS, *_ADAPTIVE_FRAME_RATES)
    )
    minimum_fps = _ADAPTIVE_FRAME_RATES[-1]
    resolution_rungs = tuple(
        AdaptiveVideoRung(quality, scale_factor, minimum_fps)
        for scale_factor in _ADAPTIVE_SCALE_FACTORS
    )
    return frame_rate_rungs + resolution_rungs


def adaptive_max_dimension(
    configured_scale: int | None,
    *,
    width: int,
    height: int,
    scale_factor: float,
) -> int | None:
    """Resolve a rung's ``scale_factor`` to a max output dimension for the encoder."""
    if scale_factor == 1.0:
        return configured_scale
    source_max_dimension = max(width, height)
    base_max_dimension = (
        source_max_dimension
        if configured_scale is None
        else min(source_max_dimension, configured_scale)
    )
    return max(2, round(base_max_dimension * scale_factor))


class AdaptiveQualityGovernor(Generic[K]):
    """Per-channel adaptive rung state for a set of video channels.

    Callers register each channel with its rung ladder, then feed a congestion
    boolean per observation. :meth:`observe` returns a :class:`RungTransition`
    only when the active rung changes so the caller can rebuild its encoder;
    :meth:`should_drop_frame` enforces the active rung's frame-rate cap.
    """

    def __init__(self) -> None:
        self._rungs: dict[K, tuple[AdaptiveVideoRung, ...]] = {}
        self._controllers: dict[K, AdaptiveVideoController] = {}
        self._active: dict[K, int] = {}
        self._last_frame: dict[K, float] = {}

    def register(self, key: K, rungs: tuple[AdaptiveVideoRung, ...]) -> None:
        self._rungs[key] = rungs
        self._active[key] = 0
        if len(rungs) > 1:
            self._controllers[key] = AdaptiveVideoController(max_rung=len(rungs) - 1)

    def observe(self, key: K, *, is_congested: bool, now: float) -> RungTransition | None:
        controller = self._controllers.get(key)
        if controller is None:
            return None
        previous_index = self._active[key]
        new_index = controller.observe(is_congested=is_congested, now=now)
        if new_index == previous_index:
            return None
        self._active[key] = new_index
        rungs = self._rungs[key]
        return RungTransition(rungs[previous_index], rungs[new_index])

    def active_index(self, key: K) -> int:
        return self._active.get(key, 0)

    def active_rung(self, key: K) -> AdaptiveVideoRung:
        return self._rungs[key][self._active[key]]

    def should_drop_frame(self, key: K, *, now: float) -> bool:
        rungs = self._rungs.get(key)
        if not rungs:
            return False
        max_fps = rungs[self._active[key]].max_fps
        if max_fps is None:
            return False
        last_frame = self._last_frame.get(key)
        if last_frame is not None and now - last_frame < 1 / max_fps:
            return True
        self._last_frame[key] = now
        return False

    def clear_frame_timing(self, key: K) -> None:
        self._last_frame.pop(key, None)

    def reset_all_frame_timing(self) -> None:
        self._last_frame.clear()

    def unregister(self, key: K) -> None:
        self._rungs.pop(key, None)
        self._controllers.pop(key, None)
        self._active.pop(key, None)
        self._last_frame.pop(key, None)

    def downgraded_count(self) -> int:
        """Number of registered channels currently below their top rung."""
        return sum(1 for index in self._active.values() if index > 0)
