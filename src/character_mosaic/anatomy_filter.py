from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from math import hypot
from statistics import median
from typing import Any, Callable, Iterable, Sequence

from PIL import Image

from .types import Detection


# OpenPose 18 body indices used by imgutils.pose.dwpose.
_RIGHT_SHOULDER = 2
_RIGHT_HIP = 8
_RIGHT_KNEE = 9
_LEFT_SHOULDER = 5
_LEFT_HIP = 11
_LEFT_KNEE = 12


@dataclass(frozen=True)
class AnatomyFilterConfig:
    """Conservative thresholds for anatomy-aware false-positive suppression.

    Ratios are normalized by a robust limb/torso segment length estimated from
    the pose. The filter is deliberately fail-open: insufficient or ambiguous
    pose evidence always keeps the detector candidate.
    """

    enabled: bool = True
    min_keypoint_score: float = 0.45
    person_conf_threshold: float = 0.25
    pelvis_keep_ratio: float = 0.25
    pelvis_far_ratio: float = 0.55
    knee_reject_ratio: float = 0.18
    armpit_reject_ratio: float = 0.16
    person_bbox_expand_ratio: float = 0.06


@dataclass(frozen=True)
class AnatomySuppression:
    detection: Detection
    reason: str
    person_index: int
    joint_distance_ratio: float
    pelvis_distance_ratio: float

    @property
    def log_reason(self) -> str:
        return (
            f"{self.reason};person={self.person_index};"
            f"joint_ratio={self.joint_distance_ratio:.3f};"
            f"pelvis_ratio={self.pelvis_distance_ratio:.3f}"
        )


@dataclass(frozen=True)
class AnatomyFilterResult:
    kept: tuple[Detection, ...]
    suppressed: tuple[AnatomySuppression, ...] = tuple()
    status: str = "not_run"


@dataclass(frozen=True)
class _PoseAssessment:
    usable: bool
    pelvis_near: bool = False
    reason: str | None = None
    joint_ratio: float | None = None
    pelvis_ratio: float | None = None


class AnatomyAwareDetector:
    """Wrap another detector and conservatively remove obvious body-part FPs.

    The wrapped detector remains the source of truth for recall. DWPose is used
    only as a negative sanity check after the normal detector has finished. If
    the pose stack is unavailable, fails, or is ambiguous, all candidates are
    returned unchanged.
    """

    def __init__(self, detector, config: AnatomyFilterConfig | None = None):
        self.detector = detector
        self.config = config or AnatomyFilterConfig()
        self._disabled_for_run = False
        self.reset_filter_state()

    def reset_filter_state(self) -> None:
        self.last_filter_result = AnatomyFilterResult(tuple(), status="not_run")

    def detect(self, image: Image.Image, progress=None, stop_requested=None) -> list[Detection]:
        self.reset_filter_state()
        detections = list(self._run_base_detector(image, progress, stop_requested))
        if stop_requested and stop_requested():
            self.last_filter_result = AnatomyFilterResult(tuple(detections), status="stopped")
            return detections
        if self._disabled_for_run:
            self.last_filter_result = AnatomyFilterResult(tuple(detections), status="disabled_after_failure")
            return detections

        result = apply_anatomy_filter(image, detections, self.config)
        self.last_filter_result = result
        if result.status.startswith(("failed:", "unavailable:")):
            # Missing helper models/dependencies are normally run-wide problems.
            # Do not retry downloads/imports for every image in a large batch.
            self._disabled_for_run = True
        return list(result.kept)

    def _run_base_detector(self, image: Image.Image, progress, stop_requested):
        method = self.detector.detect
        parameters = inspect.signature(method).parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        kwargs = {}
        if "progress" in parameters or accepts_kwargs:
            kwargs["progress"] = progress
        if "stop_requested" in parameters or accepts_kwargs:
            kwargs["stop_requested"] = stop_requested
        return method(image, **kwargs)

    def __getattr__(self, name: str):
        # Preserve access to detector-specific properties for diagnostics.
        return getattr(self.detector, name)


def anatomy_filter_enabled(config: AnatomyFilterConfig) -> bool:
    """Resolve the default-on filter with an emergency environment override."""

    raw = os.environ.get("CMC_ANATOMY_FILTER")
    if raw is None:
        return bool(config.enabled)
    return raw.strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}


def apply_anatomy_filter(
    image: Image.Image,
    detections: Iterable[Detection],
    config: AnatomyFilterConfig | None = None,
    *,
    person_detector: Callable[..., Any] | None = None,
    pose_estimator: Callable[..., Any] | None = None,
) -> AnatomyFilterResult:
    """Suppress candidates only when reliable pose evidence says knee/armpit.

    Dependency injection hooks make the policy testable without downloading
    inference models.
    """

    cfg = config or AnatomyFilterConfig()
    items = tuple(detections)
    if not items:
        return AnatomyFilterResult(items, status="no_candidates")
    if not anatomy_filter_enabled(cfg):
        return AnatomyFilterResult(items, status="disabled")

    try:
        if person_detector is None:
            from imgutils.detect import detect_person as person_detector
        if pose_estimator is None:
            from imgutils.pose import dwpose_estimate as pose_estimator
    except Exception as exc:
        return AnatomyFilterResult(items, status=f"unavailable:{type(exc).__name__}")

    try:
        person_results = person_detector(
            image,
            level="s",
            version="v1.1",
            conf_threshold=cfg.person_conf_threshold,
            iou_threshold=0.5,
        )
        person_boxes = [tuple(int(v) for v in result[0]) for result in person_results]
        if not person_boxes:
            return AnatomyFilterResult(items, status="no_person")
        poses = list(pose_estimator(image, auto_detect=False, out_bboxes=person_boxes))
    except Exception as exc:
        # A helper model must never make the censor detector unusable.
        return AnatomyFilterResult(items, status=f"failed:{type(exc).__name__}")

    if not poses:
        return AnatomyFilterResult(items, status="no_pose")

    paired = list(zip(person_boxes, poses))
    kept: list[Detection] = []
    suppressed: list[AnatomySuppression] = []

    for detection in items:
        matched: list[tuple[int, _PoseAssessment]] = []
        for person_index, (person_box, pose) in enumerate(paired):
            if not _candidate_matches_person(detection.box, person_box, cfg.person_bbox_expand_ratio):
                continue
            assessment = _assess_candidate(detection, pose, cfg)
            matched.append((person_index, assessment))

        # No body assignment, any unusable body assignment, a plausible pelvis
        # match, or any ambiguous matched person all mean "keep". This is the
        # core fail-open rule that protects recall in crops and crowded scenes.
        if not matched:
            kept.append(detection)
            continue
        if any(not assessment.usable for _, assessment in matched):
            kept.append(detection)
            continue
        if any(assessment.pelvis_near for _, assessment in matched):
            kept.append(detection)
            continue
        if any(assessment.reason is None for _, assessment in matched):
            kept.append(detection)
            continue

        person_index, best = min(
            matched,
            key=lambda item: float(item[1].joint_ratio if item[1].joint_ratio is not None else 999.0),
        )
        suppressed.append(
            AnatomySuppression(
                detection=detection,
                reason=str(best.reason),
                person_index=person_index,
                joint_distance_ratio=float(best.joint_ratio),
                pelvis_distance_ratio=float(best.pelvis_ratio),
            )
        )

    return AnatomyFilterResult(tuple(kept), tuple(suppressed), status="applied")


def _assess_candidate(
    detection: Detection,
    pose: Any,
    config: AnatomyFilterConfig,
) -> _PoseAssessment:
    body = getattr(pose, "body", None)
    if body is None:
        return _PoseAssessment(False)

    right_shoulder = _keypoint(body, _RIGHT_SHOULDER, config.min_keypoint_score)
    left_shoulder = _keypoint(body, _LEFT_SHOULDER, config.min_keypoint_score)
    right_hip = _keypoint(body, _RIGHT_HIP, config.min_keypoint_score)
    left_hip = _keypoint(body, _LEFT_HIP, config.min_keypoint_score)
    right_knee = _keypoint(body, _RIGHT_KNEE, config.min_keypoint_score)
    left_knee = _keypoint(body, _LEFT_KNEE, config.min_keypoint_score)

    # Both hips are mandatory. With only one hip, the pelvis location is too
    # uncertain to safely remove a candidate.
    if right_hip is None or left_hip is None:
        return _PoseAssessment(False)

    unit = _body_unit(
        right_shoulder,
        left_shoulder,
        right_hip,
        left_hip,
        right_knee,
        left_knee,
    )
    if unit is None:
        return _PoseAssessment(False)

    pelvis = _midpoint(right_hip, left_hip)
    pelvis_ratio = _point_to_box_distance(pelvis, detection.box) / unit
    if pelvis_ratio <= config.pelvis_keep_ratio:
        return _PoseAssessment(True, pelvis_near=True, pelvis_ratio=pelvis_ratio)
    if pelvis_ratio < config.pelvis_far_ratio:
        return _PoseAssessment(True, pelvis_ratio=pelvis_ratio)

    hard_negative: list[tuple[str, float]] = []

    for side, knee in (("right", right_knee), ("left", left_knee)):
        if knee is None:
            continue
        ratio = _point_to_box_distance(knee, detection.box) / unit
        if ratio <= config.knee_reject_ratio:
            hard_negative.append((f"near_{side}_knee", ratio))

    # OpenPose has no explicit armpit keypoint. Use a conservative point just
    # below each shoulder on the shoulder->hip torso line.
    for side, shoulder, hip in (
        ("right", right_shoulder, right_hip),
        ("left", left_shoulder, left_hip),
    ):
        if shoulder is None or hip is None:
            continue
        armpit = _lerp(shoulder, hip, 0.18)
        ratio = _point_to_box_distance(armpit, detection.box) / unit
        if ratio <= config.armpit_reject_ratio:
            hard_negative.append((f"near_{side}_armpit", ratio))

    if not hard_negative:
        return _PoseAssessment(True, pelvis_ratio=pelvis_ratio)

    reason, joint_ratio = min(hard_negative, key=lambda item: item[1])
    return _PoseAssessment(
        True,
        reason=reason,
        joint_ratio=joint_ratio,
        pelvis_ratio=pelvis_ratio,
    )


def _keypoint(body: Any, index: int, min_score: float) -> tuple[float, float] | None:
    try:
        row = body[index]
        x, y, score = float(row[0]), float(row[1]), float(row[2])
    except (IndexError, TypeError, ValueError):
        return None
    if score < min_score or x < 0.0 or y < 0.0:
        return None
    return x, y


def _body_unit(*points: tuple[float, float] | None) -> float | None:
    rs, ls, rh, lh, rk, lk = points
    segments: list[float] = []
    for a, b in ((rs, rh), (ls, lh), (rh, rk), (lh, lk)):
        if a is None or b is None:
            continue
        length = _distance(a, b)
        if length >= 4.0:
            segments.append(length)
    if len(segments) < 2:
        return None
    value = float(median(segments))
    return value if value >= 4.0 else None


def _candidate_matches_person(
    candidate_box: Sequence[int],
    person_box: Sequence[int],
    expand_ratio: float,
) -> bool:
    cx = (float(candidate_box[0]) + float(candidate_box[2])) / 2.0
    cy = (float(candidate_box[1]) + float(candidate_box[3])) / 2.0
    x0, y0, x1, y1 = (float(v) for v in person_box)
    pad_x = max(2.0, (x1 - x0) * expand_ratio)
    pad_y = max(2.0, (y1 - y0) * expand_ratio)
    return (x0 - pad_x) <= cx <= (x1 + pad_x) and (y0 - pad_y) <= cy <= (y1 + pad_y)


def _point_to_box_distance(point: tuple[float, float], box: Sequence[int]) -> float:
    x, y = point
    x0, y0, x1, y1 = (float(v) for v in box)
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return hypot(dx, dy)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def _lerp(a: tuple[float, float], b: tuple[float, float], amount: float) -> tuple[float, float]:
    return a[0] + (b[0] - a[0]) * amount, a[1] + (b[1] - a[1]) * amount
