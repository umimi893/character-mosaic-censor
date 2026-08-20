from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from character_mosaic.anatomy_filter import (
    AnatomyFilterConfig,
    _assess_candidate,
    apply_anatomy_filter,
)
from character_mosaic.types import Detection


@dataclass
class FakePose:
    body: list[list[float]]


def _pose(*, missing_hips: bool = False) -> FakePose:
    body = [[-1.0, -1.0, 0.0] for _ in range(18)]
    body[2] = [70.0, 80.0, 0.95]   # right shoulder
    body[5] = [130.0, 80.0, 0.95]  # left shoulder
    body[8] = [80.0, 220.0, 0.95]  # right hip
    body[11] = [120.0, 220.0, 0.95]  # left hip
    body[9] = [85.0, 320.0, 0.95]   # right knee
    body[12] = [115.0, 320.0, 0.95]  # left knee
    if missing_hips:
        body[11][2] = 0.1
    return FakePose(body)


def test_pelvis_candidate_is_always_kept():
    detection = Detection((90, 215, 110, 245), "pussy", 0.55)
    assessment = _assess_candidate(detection, _pose(), AnatomyFilterConfig())
    assert assessment.usable is True
    assert assessment.pelvis_near is True
    assert assessment.reason is None


def test_knee_candidate_is_rejected_when_far_from_pelvis():
    detection = Detection((78, 310, 95, 330), "pussy", 0.55)
    assessment = _assess_candidate(detection, _pose(), AnatomyFilterConfig())
    assert assessment.usable is True
    assert assessment.pelvis_near is False
    assert assessment.reason == "near_right_knee"


def test_armpit_candidate_is_rejected_when_far_from_pelvis():
    # Right armpit proxy is slightly below the right shoulder along the torso.
    detection = Detection((64, 96, 82, 116), "pussy", 0.55)
    assessment = _assess_candidate(detection, _pose(), AnatomyFilterConfig())
    assert assessment.usable is True
    assert assessment.pelvis_near is False
    assert assessment.reason == "near_right_armpit"


def test_mid_torso_candidate_stays_when_not_a_known_hard_negative():
    detection = Detection((90, 145, 110, 165), "pussy", 0.55)
    assessment = _assess_candidate(detection, _pose(), AnatomyFilterConfig())
    assert assessment.usable is True
    assert assessment.reason is None


def test_missing_hip_keypoint_fails_open():
    detection = Detection((78, 310, 95, 330), "pussy", 0.55)
    assessment = _assess_candidate(detection, _pose(missing_hips=True), AnatomyFilterConfig())
    assert assessment.usable is False
    assert assessment.reason is None


def test_full_filter_suppresses_knee_candidate():
    image = Image.new("RGB", (200, 400), "white")
    detection = Detection((78, 310, 95, 330), "pussy", 0.55)

    def fake_person_detector(_image, **_kwargs):
        return [((0, 0, 200, 400), "person", 0.99)]

    def fake_pose_estimator(_image, **_kwargs):
        return [_pose()]

    result = apply_anatomy_filter(
        image,
        [detection],
        person_detector=fake_person_detector,
        pose_estimator=fake_pose_estimator,
    )
    assert result.status == "applied"
    assert result.kept == tuple()
    assert len(result.suppressed) == 1
    assert result.suppressed[0].reason == "near_right_knee"


def test_full_filter_keeps_candidate_if_pose_model_fails():
    image = Image.new("RGB", (200, 400), "white")
    detection = Detection((78, 310, 95, 330), "pussy", 0.55)

    def fake_person_detector(_image, **_kwargs):
        return [((0, 0, 200, 400), "person", 0.99)]

    def broken_pose_estimator(_image, **_kwargs):
        raise RuntimeError("model unavailable")

    result = apply_anatomy_filter(
        image,
        [detection],
        person_detector=fake_person_detector,
        pose_estimator=broken_pose_estimator,
    )
    assert result.kept == (detection,)
    assert result.suppressed == tuple()
    assert result.status.startswith("failed:")


def test_overlapping_people_with_one_unusable_pose_fail_open():
    image = Image.new("RGB", (200, 400), "white")
    detection = Detection((78, 310, 95, 330), "pussy", 0.55)

    def fake_person_detector(_image, **_kwargs):
        return [
            ((0, 0, 200, 400), "person", 0.99),
            ((20, 0, 180, 400), "person", 0.95),
        ]

    def fake_pose_estimator(_image, **_kwargs):
        return [_pose(), _pose(missing_hips=True)]

    result = apply_anatomy_filter(
        image,
        [detection],
        person_detector=fake_person_detector,
        pose_estimator=fake_pose_estimator,
    )
    assert result.kept == (detection,)
    assert result.suppressed == tuple()
