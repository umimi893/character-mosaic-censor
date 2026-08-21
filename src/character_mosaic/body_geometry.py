from __future__ import annotations

from dataclasses import replace
from math import hypot
from typing import Iterable, Sequence

from .anatomy_filter import AnatomyFilterResult, AnatomySuppression
from .types import BodyRegion, CandidateEvidence, Detection, PosePoint


# Geometry v2 deliberately uses only pose landmarks that are already available
# from DWPose. No new model dependency is introduced here.
_MIN_SCORE = 0.45
_TORSO_SAMPLE_THRESHOLD = 0.67
_LIMB_TUBE_RATIO = 0.16
_ARMPIT_RADIUS_RATIO = 0.17
_GROIN_WIDTH_RATIO = 0.32
_GROIN_DEPTH_RATIO = 0.36

# These are pose/body-location vetoes that a later, stronger groin/pelvis
# protection is allowed to rescue. Semantic face/head suppressions, product
# review policy, and Negative Memory are intentionally not included.
_RESCUABLE_BODY_SUPPRESSIONS = frozenset({
    "inside_torso_back",
    "near_right_knee",
    "near_left_knee",
    "near_right_armpit",
    "near_left_armpit",
    "inside_upper_back",
    "inside_torso",
    "near_right_armpit_v2",
    "near_left_armpit_v2",
    "on_right_thigh",
    "on_left_thigh",
    "on_right_lower_leg",
    "on_left_lower_leg",
})


class GeometryV2Detector:
    """Apply pose-derived hard-negative geometry after the normal body reasoner."""

    def __init__(self, detector):
        self.detector = detector
        self.last_filter_result = getattr(detector, "last_filter_result", AnatomyFilterResult(tuple()))

    @property
    def requires_review(self) -> bool:
        return bool(getattr(self.last_filter_result, "requires_review", False))

    def reset_filter_state(self) -> None:
        reset = getattr(self.detector, "reset_filter_state", None)
        if callable(reset):
            reset()
        self.last_filter_result = getattr(self.detector, "last_filter_result", AnatomyFilterResult(tuple()))

    def detect(self, image, progress=None, stop_requested=None):
        detections = list(self.detector.detect(image, progress=progress, stop_requested=stop_requested))
        result = getattr(self.detector, "last_filter_result", None)
        if result is None or not getattr(result, "evidence", None):
            self.last_filter_result = result or AnatomyFilterResult(tuple(detections), status="not_run")
            return detections
        final = apply_body_geometry_v2(result, image.size)
        self.last_filter_result = final
        return list(final.kept)

    def __getattr__(self, name: str):
        return getattr(self.detector, name)


def apply_body_geometry_v2(
    result: AnatomyFilterResult,
    image_size: tuple[int, int],
) -> AnatomyFilterResult:
    """Suppress obvious torso/armpit/leg false positives using pose geometry.

    Safety rules:
    * a candidate inside any reliable directional groin zone is protected;
    * pelvis evidence from another person always wins for close-contact scenes;
    * those positive protections may rescue an earlier pose/body suppression,
      but never semantic face/head, review-policy, or memory suppressions;
    * unmatched candidates fail open for new Geometry-v2 vetoes;
    * overlapping-person scenes fail open unless every matched usable person
      independently classifies the candidate as a hard-negative body region;
    * missing shoulders/hips/knees simply removes that geometry signal.
    """

    people = _people_from_pose_points(result.pose_points)
    if not people:
        return result

    derived_regions = _visual_regions(people, image_size)
    existing_suppressed = {item.detection: item for item in result.suppressed}
    kept: list[Detection] = []
    suppressed: list[AnatomySuppression] = []
    evidence_out: list[CandidateEvidence] = []

    for evidence in result.evidence:
        detection = evidence.detection
        previous = existing_suppressed.get(detection)
        matched = tuple(idx for idx in evidence.matched_persons if idx in people)

        pelvis_people = _signal_people(evidence.positive_signals, "near_pelvis")
        protected_by_other_person = bool(matched) and any(
            person not in matched for person in pelvis_people
        )

        groin_hits = [
            person_index
            for person_index in matched
            if _candidate_in_groin(detection.box, people[person_index])
        ]

        if previous is not None:
            # BodyReasoning runs before Geometry v2. Let the stronger v1.4
            # positive protections rescue only earlier pose/body-location vetoes
            # so the documented precedence is true across detector layers.
            if (
                previous.reason in _RESCUABLE_BODY_SUPPRESSIONS
                and (protected_by_other_person or groin_hits)
            ):
                final = _protected_evidence(evidence, groin_hits)
                evidence_out.append(final)
                kept.append(detection)
                continue

            suppressed.append(previous)
            evidence_out.append(evidence)
            continue

        # No reliable person association means Geometry v2 is not entitled to
        # create a new veto over the base detector.
        if not matched:
            evidence_out.append(evidence)
            kept.append(detection)
            continue

        if protected_by_other_person:
            # Preserve the important oral/close-contact case where a candidate
            # overlaps one person's body but is plausibly near another person's
            # pelvis.
            evidence_out.append(
                evidence if evidence.decision == "keep" else replace(evidence, decision="keep")
            )
            kept.append(detection)
            continue

        candidate_people = list(matched)

        # A plausible groin location from any matched person wins before
        # hard-negative geometry. The zone is directional (below the hip line),
        # unlike the older circular pelvis distance check, so lower-back points
        # are not accidentally protected merely because they are near the hips.
        if groin_hits:
            final = _protected_evidence(evidence, groin_hits)
            evidence_out.append(final)
            kept.append(detection)
            continue

        assessments: list[tuple[int, str | None, float]] = []
        for person_index in candidate_people:
            reason, strength = _classify_hard_negative(detection.box, people[person_index])
            assessments.append((person_index, reason, strength))

        usable = [item for item in assessments if item[1] is not None]
        # For a multi-person candidate, require agreement from every matched
        # person. One unknown person means fail-open.
        if len(usable) != len(candidate_people):
            evidence_out.append(evidence)
            kept.append(detection)
            continue

        if usable:
            person_index, reason, strength = max(usable, key=lambda item: item[2])
            negative = tuple(evidence.negative_signals) + (
                f"{reason}:p{person_index}:{strength:.3f}",
            )
            final = replace(evidence, decision="suppress", negative_signals=negative)
            evidence_out.append(final)
            suppressed.append(
                AnatomySuppression(
                    detection=detection,
                    reason=str(reason),
                    person_index=person_index,
                    joint_distance_ratio=max(0.0, 1.0 - strength),
                    pelvis_distance_ratio=float(
                        evidence.pelvis_distance_ratio
                        if evidence.pelvis_distance_ratio is not None
                        else 999.0
                    ),
                )
            )
            continue

        evidence_out.append(evidence)
        kept.append(detection)

    return replace(
        result,
        kept=tuple(kept),
        suppressed=tuple(suppressed),
        evidence=tuple(evidence_out),
        body_regions=tuple(result.body_regions) + tuple(derived_regions),
    )


def _protected_evidence(
    evidence: CandidateEvidence,
    groin_hits: Sequence[int],
) -> CandidateEvidence:
    positive = list(evidence.positive_signals)
    for person_index in groin_hits:
        signal = f"inside_groin_zone:p{person_index}"
        if signal not in positive:
            positive.append(signal)
    return replace(evidence, decision="keep", positive_signals=tuple(positive))


def _people_from_pose_points(points: Iterable[PosePoint]) -> dict[int, dict[str, tuple[float, float]]]:
    people: dict[int, dict[str, tuple[float, float]]] = {}
    for point in points:
        if point.score < _MIN_SCORE:
            continue
        people.setdefault(point.person_index, {})[point.label] = (float(point.x), float(point.y))
    return people


def _classify_hard_negative(
    box: Sequence[int],
    points: dict[str, tuple[float, float]],
) -> tuple[str | None, float]:
    center = _box_center(box)
    scale = _body_scale(points)
    if scale is None:
        return None, 0.0

    rs, ls = points.get("right_shoulder"), points.get("left_shoulder")
    rh, lh = points.get("right_hip"), points.get("left_hip")
    if rs and ls and rh and lh:
        torso = (rs, ls, lh, rh)
        fraction = _box_sample_fraction_in_polygon(box, torso)
        if fraction >= _TORSO_SAMPLE_THRESHOLD:
            # Upper torso is where shoulder-blade/back false positives are most
            # common. Lower torso is still a hard-negative as long as the
            # directional groin zone did not protect it above.
            shoulder_y = (rs[1] + ls[1]) / 2.0
            hip_y = (rh[1] + lh[1]) / 2.0
            if abs(hip_y - shoulder_y) >= 8.0:
                vertical = (center[1] - shoulder_y) / (hip_y - shoulder_y)
            else:
                vertical = 0.5
            reason = "inside_upper_back" if vertical <= 0.62 else "inside_torso"
            return reason, min(1.0, fraction)

    # Armpit is estimated from both shoulder->hip and shoulder->elbow context.
    for side in ("right", "left"):
        shoulder = points.get(f"{side}_shoulder")
        hip = points.get(f"{side}_hip")
        elbow = points.get(f"{side}_elbow")
        if shoulder and hip:
            armpit = _lerp(shoulder, hip, 0.16)
            radius = scale * _ARMPIT_RADIUS_RATIO
            d = _point_box_distance(armpit, box)
            if d <= radius:
                # If an elbow is available, require the arm to actually leave
                # the shoulder by a meaningful distance. This reduces phantom
                # armpit zones on broken poses.
                if elbow is None or _distance(shoulder, elbow) >= scale * 0.28:
                    return f"near_{side}_armpit_v2", max(0.0, 1.0 - d / max(1.0, radius))

    # Legs are tubes around the pose bones rather than axis-aligned boxes, so
    # diagonal and bent legs remain suppressible.
    for side in ("right", "left"):
        hip = points.get(f"{side}_hip")
        knee = points.get(f"{side}_knee")
        ankle = points.get(f"{side}_ankle")
        if hip and knee:
            # Ignore the very top of the thigh close to the groin by starting
            # the negative tube 18% down the femur.
            thigh_start = _lerp(hip, knee, 0.18)
            distance = _point_segment_distance(center, thigh_start, knee)
            radius = scale * _LIMB_TUBE_RATIO
            if distance <= radius:
                return f"on_{side}_thigh", max(0.0, 1.0 - distance / max(1.0, radius))
        if knee and ankle:
            distance = _point_segment_distance(center, knee, ankle)
            radius = scale * _LIMB_TUBE_RATIO
            if distance <= radius:
                return f"on_{side}_lower_leg", max(0.0, 1.0 - distance / max(1.0, radius))

    return None, 0.0


def _candidate_in_groin(box: Sequence[int], points: dict[str, tuple[float, float]]) -> bool:
    rh, lh = points.get("right_hip"), points.get("left_hip")
    if not rh or not lh:
        return False
    scale = _body_scale(points)
    if scale is None:
        return False
    center = _box_center(box)
    hip_mid = _midpoint(rh, lh)
    shoulder_points = [p for p in (points.get("right_shoulder"), points.get("left_shoulder")) if p]
    if len(shoulder_points) == 2:
        shoulder_mid = _midpoint(shoulder_points[0], shoulder_points[1])
        axis = (hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1])
    else:
        axis = (0.0, 1.0)
    axis_len = hypot(axis[0], axis[1])
    if axis_len < 1.0:
        axis = (0.0, 1.0)
        axis_len = 1.0
    ux, uy = axis[0] / axis_len, axis[1] / axis_len
    vx, vy = -uy, ux
    rel = (center[0] - hip_mid[0], center[1] - hip_mid[1])
    along = rel[0] * ux + rel[1] * uy
    across = abs(rel[0] * vx + rel[1] * vy)
    return (
        -scale * 0.04 <= along <= scale * _GROIN_DEPTH_RATIO
        and across <= scale * _GROIN_WIDTH_RATIO
    )


def _visual_regions(
    people: dict[int, dict[str, tuple[float, float]]],
    image_size: tuple[int, int],
) -> list[BodyRegion]:
    regions: list[BodyRegion] = []
    for person_index, points in people.items():
        rs, ls = points.get("right_shoulder"), points.get("left_shoulder")
        rh, lh = points.get("right_hip"), points.get("left_hip")
        if rs and ls and rh and lh:
            regions.append(BodyRegion(
                _bounds((rs, ls, rh, lh), image_size),
                "torso_geometry_v2", 1.0, person_index, "geometry_v2",
            ))
        for side in ("right", "left"):
            shoulder, hip = points.get(f"{side}_shoulder"), points.get(f"{side}_hip")
            if shoulder and hip:
                regions.append(BodyRegion(
                    _square(_lerp(shoulder, hip, 0.16), (_body_scale(points) or 20.0) * _ARMPIT_RADIUS_RATIO, image_size),
                    f"{side}_armpit_geometry_v2", 1.0, person_index, "geometry_v2",
                ))
            h, k, a = points.get(f"{side}_hip"), points.get(f"{side}_knee"), points.get(f"{side}_ankle")
            if h and k:
                regions.append(BodyRegion(
                    _segment_bounds(_lerp(h, k, 0.18), k, (_body_scale(points) or 20.0) * _LIMB_TUBE_RATIO, image_size),
                    f"{side}_thigh_geometry_v2", 1.0, person_index, "geometry_v2",
                ))
            if k and a:
                regions.append(BodyRegion(
                    _segment_bounds(k, a, (_body_scale(points) or 20.0) * _LIMB_TUBE_RATIO, image_size),
                    f"{side}_lower_leg_geometry_v2", 1.0, person_index, "geometry_v2",
                ))
    return regions


def _body_scale(points: dict[str, tuple[float, float]]) -> float | None:
    segments: list[float] = []
    for side in ("right", "left"):
        shoulder, hip = points.get(f"{side}_shoulder"), points.get(f"{side}_hip")
        hip2, knee = points.get(f"{side}_hip"), points.get(f"{side}_knee")
        if shoulder and hip:
            segments.append(_distance(shoulder, hip))
        if hip2 and knee:
            segments.append(_distance(hip2, knee))
    segments = [value for value in segments if value >= 8.0]
    if len(segments) < 2:
        return None
    segments.sort()
    mid = len(segments) // 2
    return segments[mid] if len(segments) % 2 else (segments[mid - 1] + segments[mid]) / 2.0


def _box_sample_fraction_in_polygon(box: Sequence[int], polygon: Sequence[tuple[float, float]]) -> float:
    x0, y0, x1, y1 = (float(v) for v in box)
    samples = []
    for fy in (0.2, 0.5, 0.8):
        for fx in (0.2, 0.5, 0.8):
            samples.append((x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy))
    inside = sum(_point_in_polygon(point, polygon) for point in samples)
    return inside / len(samples)


def _point_in_polygon(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            cross_x = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x < cross_x:
                inside = not inside
        j = i
    return inside


def _signal_people(signals: Sequence[str], prefix: str) -> set[int]:
    people: set[int] = set()
    needle = prefix + ":p"
    for signal in signals:
        if not signal.startswith(needle):
            continue
        person_text = signal[len(needle):].split(":", 1)[0]
        try:
            people.add(int(person_text))
        except ValueError:
            continue
    return people


def _point_segment_distance(point, start, end) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    qx, qy = ax + t * dx, ay + t * dy
    return hypot(px - qx, py - qy)


def _point_box_distance(point, box) -> float:
    x, y = point
    x0, y0, x1, y1 = (float(v) for v in box)
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return hypot(dx, dy)


def _box_center(box) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def _distance(a, b) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _midpoint(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _lerp(a, b, amount: float):
    return (a[0] + (b[0] - a[0]) * amount, a[1] + (b[1] - a[1]) * amount)


def _bounds(points, image_size):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return _clamp((min(xs), min(ys), max(xs), max(ys)), image_size)


def _square(center, radius, image_size):
    return _clamp((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), image_size)


def _segment_bounds(start, end, radius, image_size):
    return _clamp((min(start[0], end[0]) - radius, min(start[1], end[1]) - radius,
                   max(start[0], end[0]) + radius, max(start[1], end[1]) + radius), image_size)


def _clamp(box, image_size):
    width, height = image_size
    x0, y0, x1, y1 = box
    return (
        max(0, min(width, int(round(x0)))), max(0, min(height, int(round(y0)))),
        max(0, min(width, int(round(x1)))), max(0, min(height, int(round(y1)))),
    )
