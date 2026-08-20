from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from math import hypot
from statistics import median
from typing import Any, Callable, Iterable, Sequence

from PIL import Image

from .types import BodyRegion, CandidateEvidence, Detection, PoseEdge, PosePoint


_RIGHT_SHOULDER = 2
_RIGHT_ELBOW = 3
_RIGHT_WRIST = 4
_LEFT_SHOULDER = 5
_LEFT_ELBOW = 6
_LEFT_WRIST = 7
_RIGHT_HIP = 8
_RIGHT_KNEE = 9
_RIGHT_ANKLE = 10
_LEFT_HIP = 11
_LEFT_KNEE = 12
_LEFT_ANKLE = 13

_BODY_LABELS = {
    0: "nose", 1: "neck", 2: "right_shoulder", 3: "right_elbow", 4: "right_wrist",
    5: "left_shoulder", 6: "left_elbow", 7: "left_wrist", 8: "right_hip",
    9: "right_knee", 10: "right_ankle", 11: "left_hip", 12: "left_knee",
    13: "left_ankle", 14: "right_eye", 15: "left_eye", 16: "right_ear", 17: "left_ear",
}

_SKELETON_EDGES = (
    (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (8, 11),
    (0, 1), (0, 14), (14, 16), (0, 15), (15, 17),
)


@dataclass(frozen=True)
class AnatomyFilterConfig:
    enabled: bool = True
    min_keypoint_score: float = 0.45
    person_conf_threshold: float = 0.25
    pelvis_keep_ratio: float = 0.42
    pelvis_far_ratio: float = 0.68
    knee_reject_ratio: float = 0.18
    armpit_reject_ratio: float = 0.16
    person_bbox_expand_ratio: float = 0.06
    head_conf_threshold: float = 0.40
    face_conf_threshold: float = 0.25
    eye_conf_threshold: float = 0.30
    head_review_overlap: float = 0.80
    face_review_overlap: float = 0.65
    eye_suppress_overlap: float = 0.55
    eye_confirm_face_overlap: float = 0.65
    eye_confirm_head_overlap: float = 0.65


@dataclass(frozen=True)
class AnatomySuppression:
    detection: Detection
    reason: str
    person_index: int
    joint_distance_ratio: float = 0.0
    pelvis_distance_ratio: float = 999.0

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
    body_regions: tuple[BodyRegion, ...] = tuple()
    pose_points: tuple[PosePoint, ...] = tuple()
    pose_edges: tuple[PoseEdge, ...] = tuple()
    evidence: tuple[CandidateEvidence, ...] = tuple()
    failed_components: tuple[str, ...] = tuple()

    @property
    def requires_review(self) -> bool:
        return any(item.decision == "review" for item in self.evidence)


@dataclass(frozen=True)
class _PoseAssessment:
    usable: bool
    pelvis_near: bool = False
    reason: str | None = None
    joint_ratio: float | None = None
    pelvis_ratio: float | None = None


class AnatomyAwareDetector:
    """Add a fail-open body-region reasoning layer after the base detector."""

    def __init__(self, detector, config: AnatomyFilterConfig | None = None):
        self.detector = detector
        self.config = config or AnatomyFilterConfig()
        self._disabled_for_run = False
        self._disabled_components: set[str] = set()
        self.reset_filter_state()

    def reset_filter_state(self) -> None:
        self.last_filter_result = AnatomyFilterResult(tuple(), status="not_run")

    @property
    def requires_review(self) -> bool:
        return bool(self.last_filter_result.requires_review)

    def detect(self, image: Image.Image, progress=None, stop_requested=None) -> list[Detection]:
        self.reset_filter_state()
        detections = list(self._run_base_detector(image, progress, stop_requested))
        if stop_requested and stop_requested():
            self.last_filter_result = AnatomyFilterResult(tuple(detections), status="stopped")
            return detections
        if self._disabled_for_run:
            self.last_filter_result = AnatomyFilterResult(tuple(detections), status="disabled_after_failure")
            return detections

        filter_method = apply_anatomy_filter
        filter_params = inspect.signature(filter_method).parameters
        filter_accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in filter_params.values())
        filter_kwargs = {}
        if "disabled_components" in filter_params or filter_accepts_kwargs:
            filter_kwargs["disabled_components"] = frozenset(self._disabled_components)
        result = filter_method(image, detections, self.config, **filter_kwargs)
        self.last_filter_result = result
        if result.status.startswith(("unavailable:", "failed:")):
            self._disabled_for_run = True
        self._disabled_components.update(result.failed_components)
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
        return getattr(self.detector, name)


def anatomy_filter_enabled(config: AnatomyFilterConfig) -> bool:
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
    head_detector: Callable[..., Any] | None = None,
    face_detector: Callable[..., Any] | None = None,
    eye_detector: Callable[..., Any] | None = None,
    disabled_components: frozenset[str] = frozenset(),
) -> AnatomyFilterResult:
    cfg = config or AnatomyFilterConfig()
    items = tuple(detections)
    if not items:
        return AnatomyFilterResult(items, status="no_candidates")
    if not anatomy_filter_enabled(cfg):
        evidence = tuple(CandidateEvidence(detection=d) for d in items)
        return AnatomyFilterResult(items, status="disabled", evidence=evidence)

    injection_mode = any(
        helper is not None
        for helper in (person_detector, pose_estimator, head_detector, face_detector, eye_detector)
    )
    try:
        if not injection_mode:
            if "person" not in disabled_components:
                from imgutils.detect import detect_person as person_detector
            if "pose" not in disabled_components:
                from imgutils.pose import dwpose_estimate as pose_estimator
            if "head" not in disabled_components:
                from imgutils.detect import detect_heads as head_detector
            if "face" not in disabled_components:
                from imgutils.detect import detect_faces as face_detector
            if "eye" not in disabled_components:
                from imgutils.detect import detect_eyes as eye_detector
    except Exception as exc:
        evidence = tuple(CandidateEvidence(detection=d) for d in items)
        return AnatomyFilterResult(items, status=f"unavailable:{type(exc).__name__}", evidence=evidence)

    failed: list[str] = []
    body_regions: list[BodyRegion] = []
    person_boxes: list[tuple[int, int, int, int]] = []
    poses: list[Any] = []

    if person_detector is not None and "person" not in disabled_components:
        try:
            person_results = person_detector(
                image, level="m", version="v1.1",
                conf_threshold=cfg.person_conf_threshold, iou_threshold=0.5,
            )
            for idx, result in enumerate(person_results):
                box = tuple(int(v) for v in result[0])
                person_boxes.append(box)
                body_regions.append(BodyRegion(box, "person", float(result[2]), idx, "detect_person"))
        except Exception:
            failed.append("person")

    if pose_estimator is not None and person_boxes and "pose" not in disabled_components:
        try:
            poses = list(pose_estimator(image, auto_detect=False, out_bboxes=person_boxes))
        except Exception:
            failed.append("pose")

    region_specs = (
        ("head", head_detector, {"conf_threshold": cfg.head_conf_threshold, "iou_threshold": 0.7}),
        ("face", face_detector, {"level": "s", "version": "v1.4", "conf_threshold": cfg.face_conf_threshold, "iou_threshold": 0.7}),
        ("eye", eye_detector, {"level": "s", "version": "v1.0", "conf_threshold": cfg.eye_conf_threshold, "iou_threshold": 0.3}),
    )
    for kind, detector, kwargs in region_specs:
        if detector is None or kind in disabled_components:
            continue
        try:
            for result in detector(image, **kwargs):
                box = tuple(int(v) for v in result[0])
                body_regions.append(
                    BodyRegion(
                        box=box,
                        kind=kind,
                        score=float(result[2]),
                        person_index=_assign_person_index(box, person_boxes),
                        source=f"detect_{kind}",
                    )
                )
        except Exception:
            failed.append(kind)

    pose_points, pose_edges, derived_regions = _build_pose_map(poses, cfg, image.size)
    body_regions.extend(derived_regions)

    kept: list[Detection] = []
    suppressed: list[AnatomySuppression] = []
    evidence_items: list[CandidateEvidence] = []
    paired = list(zip(person_boxes, poses))
    for detection in items:
        evidence, suppression = _reason_candidate(detection, paired, tuple(body_regions), cfg)
        evidence_items.append(evidence)
        if evidence.decision == "suppress" and suppression is not None:
            suppressed.append(suppression)
        else:
            kept.append(detection)

    if injection_mode and failed == ["pose"] and pose_estimator is not None and head_detector is None and face_detector is None and eye_detector is None:
        status = "failed:pose"
    elif failed:
        status = "partial:" + ",".join(sorted(set(failed)))
    elif poses:
        status = "applied"
    elif body_regions:
        status = "regions_only"
    else:
        status = "no_body_map"

    return AnatomyFilterResult(
        tuple(kept), tuple(suppressed), status=status,
        body_regions=tuple(body_regions), pose_points=tuple(pose_points), pose_edges=tuple(pose_edges),
        evidence=tuple(evidence_items), failed_components=tuple(sorted(set(failed))),
    )


def _reason_candidate(
    detection: Detection,
    paired: Sequence[tuple[tuple[int, int, int, int], Any]],
    body_regions: tuple[BodyRegion, ...],
    config: AnatomyFilterConfig,
) -> tuple[CandidateEvidence, AnatomySuppression | None]:
    positive: list[str] = [f"detector:{detection.score:.3f}"]
    negative: list[str] = []
    matched: list[tuple[int, _PoseAssessment]] = []
    all_assessments: list[tuple[int, _PoseAssessment]] = []

    for person_index, (person_box, pose) in enumerate(paired):
        assessment = _assess_candidate(detection, pose, config)
        all_assessments.append((person_index, assessment))
        if _candidate_matches_person(detection.box, person_box, config.person_bbox_expand_ratio):
            matched.append((person_index, assessment))

    usable_global = [(idx, a) for idx, a in all_assessments if a.usable and a.pelvis_ratio is not None]
    pelvis_distance = min((a.pelvis_ratio for _, a in usable_global), default=None)
    pelvis_near = [(idx, a) for idx, a in usable_global if a.pelvis_near]
    if pelvis_near:
        idx, assessment = min(pelvis_near, key=lambda item: float(item[1].pelvis_ratio or 999.0))
        positive.append(f"near_pelvis:p{idx}:{float(assessment.pelvis_ratio):.3f}")

    overlaps = _region_overlaps(detection.box, body_regions)
    head_overlap, head_person = overlaps.get("head", (0.0, -1))
    face_overlap, face_person = overlaps.get("face", (0.0, -1))
    eye_overlap, eye_person = overlaps.get("eye", (0.0, -1))
    if head_overlap >= config.head_review_overlap:
        negative.append(f"inside_head:p{head_person}:{head_overlap:.3f}")
    if face_overlap >= config.face_review_overlap:
        negative.append(f"inside_face:p{face_person}:{face_overlap:.3f}")
    if eye_overlap >= config.eye_suppress_overlap:
        negative.append(f"inside_eye:p{eye_person}:{eye_overlap:.3f}")

    matched_indices = tuple(idx for idx, _ in matched)

    # Any reliable pelvis from any person wins. This protects oral/close-contact
    # scenes where one person's target overlaps another person's face/head.
    if pelvis_near:
        return CandidateEvidence(
            detection, "keep", tuple(positive), tuple(negative), matched_indices, pelvis_distance
        ), None

    # Eye overlap alone is not enough: require face and head confirmation too.
    if (
        eye_overlap >= config.eye_suppress_overlap
        and face_overlap >= config.eye_confirm_face_overlap
        and head_overlap >= config.eye_confirm_head_overlap
    ):
        evidence = CandidateEvidence(
            detection, "suppress", tuple(positive), tuple(negative), matched_indices, pelvis_distance
        )
        return evidence, AnatomySuppression(
            detection, "inside_eye_face_head", eye_person, 0.0,
            float(pelvis_distance if pelvis_distance is not None else 999.0),
        )

    # Face/head overlap is ambiguous by design. Oral and close-contact images can
    # legitimately place a target here, and tight crops can drift pose points.
    if face_overlap >= config.face_review_overlap or head_overlap >= config.head_review_overlap:
        return CandidateEvidence(
            detection, "review", tuple(positive), tuple(negative), matched_indices, pelvis_distance
        ), None

    # Knee/armpit suppression is allowed only when every matched usable person
    # agrees that the candidate is a hard negative and no pelvis protected it.
    if matched and all(a.usable for _, a in matched) and all(a.reason is not None for _, a in matched):
        person_index, best = min(
            matched,
            key=lambda item: float(item[1].joint_ratio if item[1].joint_ratio is not None else 999.0),
        )
        negative.append(f"{best.reason}:p{person_index}:{float(best.joint_ratio or 0.0):.3f}")
        evidence = CandidateEvidence(
            detection, "suppress", tuple(positive), tuple(negative), matched_indices, pelvis_distance
        )
        return evidence, AnatomySuppression(
            detection, str(best.reason), person_index, float(best.joint_ratio or 0.0),
            float(best.pelvis_ratio if best.pelvis_ratio is not None else 999.0),
        )

    return CandidateEvidence(
        detection, "keep", tuple(positive), tuple(negative), matched_indices, pelvis_distance
    ), None


def _build_pose_map(
    poses: Sequence[Any], config: AnatomyFilterConfig, image_size: tuple[int, int]
) -> tuple[list[PosePoint], list[PoseEdge], list[BodyRegion]]:
    points: list[PosePoint] = []
    edges: list[PoseEdge] = []
    regions: list[BodyRegion] = []
    for person_index, pose in enumerate(poses):
        body = getattr(pose, "body", None)
        if body is None:
            continue
        valid: dict[int, tuple[float, float, float]] = {}
        for index, label in _BODY_LABELS.items():
            try:
                x, y, score = float(body[index][0]), float(body[index][1]), float(body[index][2])
            except (IndexError, TypeError, ValueError):
                continue
            if score < config.min_keypoint_score or x < 0 or y < 0:
                continue
            valid[index] = (x, y, score)
            points.append(PosePoint(x, y, score, label, person_index))

        for a, b in _SKELETON_EDGES:
            if a in valid and b in valid:
                edges.append(PoseEdge(
                    (valid[a][0], valid[a][1]), (valid[b][0], valid[b][1]), person_index,
                    f"{_BODY_LABELS[a]}->{_BODY_LABELS[b]}",
                ))

        right_hip, left_hip = _xy(valid.get(_RIGHT_HIP)), _xy(valid.get(_LEFT_HIP))
        if right_hip is None or left_hip is None:
            continue
        unit = _body_unit(
            _xy(valid.get(_RIGHT_SHOULDER)), _xy(valid.get(_LEFT_SHOULDER)), right_hip, left_hip,
            _xy(valid.get(_RIGHT_KNEE)), _xy(valid.get(_LEFT_KNEE)),
        )
        if unit is None:
            continue
        pelvis = _midpoint(right_hip, left_hip)
        regions.append(BodyRegion(
            _square_box(pelvis, unit * config.pelvis_keep_ratio, image_size),
            "pelvis_safe", 1.0, person_index, "pose",
        ))
        for side, index in (("right", _RIGHT_KNEE), ("left", _LEFT_KNEE)):
            knee = _xy(valid.get(index))
            if knee is not None:
                regions.append(BodyRegion(
                    _square_box(knee, unit * config.knee_reject_ratio, image_size),
                    f"{side}_knee_zone", 1.0, person_index, "pose",
                ))
        for side, shoulder_idx, hip in (("right", _RIGHT_SHOULDER, right_hip), ("left", _LEFT_SHOULDER, left_hip)):
            shoulder = _xy(valid.get(shoulder_idx))
            if shoulder is not None:
                armpit = _lerp(shoulder, hip, 0.18)
                regions.append(BodyRegion(
                    _square_box(armpit, unit * config.armpit_reject_ratio, image_size),
                    f"{side}_armpit_zone", 1.0, person_index, "pose",
                ))
    return points, edges, regions


def _assess_candidate(detection: Detection, pose: Any, config: AnatomyFilterConfig) -> _PoseAssessment:
    body = getattr(pose, "body", None)
    if body is None:
        return _PoseAssessment(False)
    right_shoulder = _keypoint(body, _RIGHT_SHOULDER, config.min_keypoint_score)
    left_shoulder = _keypoint(body, _LEFT_SHOULDER, config.min_keypoint_score)
    right_hip = _keypoint(body, _RIGHT_HIP, config.min_keypoint_score)
    left_hip = _keypoint(body, _LEFT_HIP, config.min_keypoint_score)
    right_knee = _keypoint(body, _RIGHT_KNEE, config.min_keypoint_score)
    left_knee = _keypoint(body, _LEFT_KNEE, config.min_keypoint_score)
    if right_hip is None or left_hip is None:
        return _PoseAssessment(False)
    unit = _body_unit(right_shoulder, left_shoulder, right_hip, left_hip, right_knee, left_knee)
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
        if knee is not None:
            ratio = _point_to_box_distance(knee, detection.box) / unit
            if ratio <= config.knee_reject_ratio:
                hard_negative.append((f"near_{side}_knee", ratio))
    for side, shoulder, hip in (("right", right_shoulder, right_hip), ("left", left_shoulder, left_hip)):
        if shoulder is not None and hip is not None:
            armpit = _lerp(shoulder, hip, 0.18)
            ratio = _point_to_box_distance(armpit, detection.box) / unit
            if ratio <= config.armpit_reject_ratio:
                hard_negative.append((f"near_{side}_armpit", ratio))
    if not hard_negative:
        return _PoseAssessment(True, pelvis_ratio=pelvis_ratio)
    reason, joint_ratio = min(hard_negative, key=lambda item: item[1])
    return _PoseAssessment(True, reason=reason, joint_ratio=joint_ratio, pelvis_ratio=pelvis_ratio)


def _region_overlaps(candidate_box: Sequence[int], regions: Sequence[BodyRegion]) -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    for region in regions:
        if region.kind not in {"head", "face", "eye"}:
            continue
        ratio = _intersection_over_candidate(candidate_box, region.box)
        current = out.get(region.kind, (0.0, -1))
        if ratio > current[0]:
            out[region.kind] = (ratio, region.person_index)
    return out


def _intersection_over_candidate(a: Sequence[int], b: Sequence[int]) -> float:
    ax0, ay0, ax1, ay1 = (float(v) for v in a)
    bx0, by0, bx1, by1 = (float(v) for v in b)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    return (iw * ih) / area


def _assign_person_index(box: Sequence[int], person_boxes: Sequence[Sequence[int]]) -> int:
    if not person_boxes:
        return -1
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    matches: list[tuple[float, int]] = []
    for index, person in enumerate(person_boxes):
        x0, y0, x1, y1 = (float(v) for v in person)
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            matches.append((max(1.0, (x1 - x0) * (y1 - y0)), index))
    return min(matches)[1] if matches else -1


def _keypoint(body: Any, index: int, min_score: float) -> tuple[float, float] | None:
    try:
        row = body[index]
        x, y, score = float(row[0]), float(row[1]), float(row[2])
    except (IndexError, TypeError, ValueError):
        return None
    if score < min_score or x < 0.0 or y < 0.0:
        return None
    return x, y


def _xy(value: tuple[float, float, float] | None) -> tuple[float, float] | None:
    return None if value is None else (value[0], value[1])


def _body_unit(*points: tuple[float, float] | None) -> float | None:
    rs, ls, rh, lh, rk, lk = points
    segments: list[float] = []
    for a, b in ((rs, rh), (ls, lh), (rh, rk), (lh, lk)):
        if a is not None and b is not None:
            length = _distance(a, b)
            if length >= 4.0:
                segments.append(length)
    if len(segments) < 2:
        return None
    value = float(median(segments))
    return value if value >= 4.0 else None


def _candidate_matches_person(candidate_box: Sequence[int], person_box: Sequence[int], expand_ratio: float) -> bool:
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


def _square_box(center: tuple[float, float], radius: float, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = image_size
    x, y = center
    return (
        max(0, int(round(x - radius))), max(0, int(round(y - radius))),
        min(w, int(round(x + radius))), min(h, int(round(y + radius))),
    )
