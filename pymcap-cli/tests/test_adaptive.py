"""Tests for the shared adaptive video-quality controller and governor."""

from __future__ import annotations

from pymcap_cli.cmd.bridge._adaptive import (
    AdaptiveQualityGovernor,
    AdaptiveVideoController,
    AdaptiveVideoRung,
    adaptive_max_dimension,
    adaptive_video_rungs,
)


def test_adaptive_video_controller_degrades_quickly_and_recovers_slowly() -> None:
    controller = AdaptiveVideoController()

    assert controller.observe(is_congested=False, now=0.0) == 0
    assert controller.observe(is_congested=True, now=0.5) == 0
    assert controller.observe(is_congested=True, now=1.0) == 0
    assert controller.observe(is_congested=False, now=1.5) == 0
    assert controller.observe(is_congested=True, now=2.0) == 1

    assert controller.observe(is_congested=True, now=4.0) == 1
    assert controller.observe(is_congested=True, now=5.0) == 2
    assert controller.observe(is_congested=False, now=5.1) == 2
    assert controller.observe(is_congested=False, now=35.0) == 2
    assert controller.observe(is_congested=False, now=35.1) == 1


def test_adaptive_video_controller_reaches_top_rung_under_sustained_congestion() -> None:
    controller = AdaptiveVideoController()
    rung = 0
    for step in range(40):
        rung = controller.observe(is_congested=True, now=float(step))
    assert rung == 3
    assert controller.max_rung == 3


def test_adaptive_video_controller_holds_rung_while_clean() -> None:
    controller = AdaptiveVideoController()
    # Degrade once, then a long clean spell that is not yet long enough to
    # recover must not degrade further just because a stale window elapses.
    assert controller.observe(is_congested=True, now=0.0) == 0
    assert controller.observe(is_congested=True, now=2.0) == 1
    for step in range(3, 25):
        assert controller.observe(is_congested=False, now=float(step)) == 1


def test_adaptive_video_rungs_reduce_fps_then_resolution_without_reducing_quality() -> None:
    assert adaptive_video_rungs(28) == (
        AdaptiveVideoRung(28, 1.0, 30.0),
        AdaptiveVideoRung(28, 1.0, 20.0),
        AdaptiveVideoRung(28, 1.0, 10.0),
        AdaptiveVideoRung(28, 1.0, 5.0),
        AdaptiveVideoRung(28, 1.0, 2.0),
        AdaptiveVideoRung(28, 0.75, 2.0),
        AdaptiveVideoRung(28, 0.5, 2.0),
        AdaptiveVideoRung(28, 0.375, 2.0),
    )


def test_adaptive_resolution_is_relative_to_source_or_explicit_scale_ceiling() -> None:
    assert adaptive_max_dimension(None, width=1920, height=1080, scale_factor=0.75) == 1440
    assert adaptive_max_dimension(960, width=1920, height=1080, scale_factor=0.75) == 720
    assert adaptive_max_dimension(960, width=640, height=480, scale_factor=0.75) == 480
    assert adaptive_max_dimension(480, width=1920, height=1080, scale_factor=1.0) == 480
    assert adaptive_max_dimension(None, width=1920, height=1080, scale_factor=1.0) is None


def test_governor_observe_returns_transition_only_on_rung_change() -> None:
    governor: AdaptiveQualityGovernor[str] = AdaptiveQualityGovernor()
    governor.register("cam", adaptive_video_rungs(28))

    assert governor.observe("cam", is_congested=False, now=0.0) is None
    assert governor.observe("cam", is_congested=True, now=0.5) is None
    transition = governor.observe("cam", is_congested=True, now=2.0)
    assert transition is not None
    assert transition.previous == AdaptiveVideoRung(28, 1.0, 30.0)
    assert transition.current == AdaptiveVideoRung(28, 1.0, 20.0)
    assert governor.active_index("cam") == 1
    assert governor.downgraded_count() == 1


def test_governor_single_rung_channel_never_transitions() -> None:
    governor: AdaptiveQualityGovernor[str] = AdaptiveQualityGovernor()
    governor.register("cam", (AdaptiveVideoRung(28, 1.0, 30.0),))
    for step in range(40):
        assert governor.observe("cam", is_congested=True, now=float(step)) is None
    assert governor.active_index("cam") == 0
    assert governor.downgraded_count() == 0


def test_governor_should_drop_frame_enforces_active_rung_fps_cap() -> None:
    governor: AdaptiveQualityGovernor[str] = AdaptiveQualityGovernor()
    governor.register("cam", (AdaptiveVideoRung(28, 1.0, None), AdaptiveVideoRung(28, 1.0, 5.0)))

    # Top rung has no cap.
    assert not governor.should_drop_frame("cam", now=0.0)
    assert not governor.should_drop_frame("cam", now=0.001)

    # Step to the capped rung.
    governor.observe("cam", is_congested=True, now=0.0)
    governor.observe("cam", is_congested=True, now=2.0)
    assert governor.active_index("cam") == 1

    assert not governor.should_drop_frame("cam", now=2.0)
    assert governor.should_drop_frame("cam", now=2.1)
    assert not governor.should_drop_frame("cam", now=2.2)


def test_governor_unregister_forgets_channel() -> None:
    governor: AdaptiveQualityGovernor[str] = AdaptiveQualityGovernor()
    governor.register("cam", adaptive_video_rungs(28))
    governor.observe("cam", is_congested=True, now=0.0)
    governor.observe("cam", is_congested=True, now=2.0)
    assert governor.downgraded_count() == 1
    governor.unregister("cam")
    assert governor.active_index("cam") == 0
    assert governor.downgraded_count() == 0
