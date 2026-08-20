from __future__ import annotations

import inspect
from dataclasses import replace
from math import hypot
from typing import Iterable, Sequence

from PIL import Image

from .anatomy_filter import (
    AnatomyFilterConfig,
    AnatomyFilterResult,
    AnatomySuppression,
    apply_anatomy_filter,
)
from .types import BodyRegion, CandidateEvidence, Detection, PosePoint


# Final-policy thresholds.  The first anatomy pass remains recall-oriented;
# this layer uses only high-confidence spatial evidence to remove obvious
# false positives that survived it.
_TORSO_MIN_OVERLAP = 0.72
_TORSO_MIN_VERTICALITY = 0.55
_TORSO_INSET_X = 0.10
_TORSO_INSET_TOP = 0.06
_TORSO_INSET_BOTTOM = 0.02


class BodyReasoningDetector:
    """Base censor detector + anatomy map + final false-positive policy.

    The upstream detector is always the source of candidate boxes.  The
    existing anatomy pass builds person/head/face/eye/pose evidence, then this
    final layer applies two product-level rules learned from real use:

    * an anatomy ``review`` candidate without pelvis evidence is treated as a
      false positive instead of being sent through as a censor box;
    * a candidate strongly contained by the same person's shoulder-to-hip
      torso/back core is suppressed.

    A reliable pelvis from another person always wins.  This keeps close-contact
    and oral scenes safe when one person's legitimate target overlaps another
    person's head or body.
    """

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
        # Final body reasoning intentionally does not create body-analysis
        # Review items.  General low-confidence Review remains handled by the
        # normal pipeline threshold.
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

        disabled_before = frozenset(self._disabled_components)
        filter_method = apply_anatomy_filter
        params = inspect.signature(filter_method).parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        kwargs = {}
        if "disabled_components" in params or accepts_kwargs:
            kwargs["disabled_components"] = disabled_before

        result = filter_method(image, detections, self.config, **kwargs)
        result = enhance_anatomy_result(result, image.size)

        if result.status.startswith(("unavailable:", "failed:")):
            self._disabled_for_run = True
        self._disabled_components.update(result.failed_components)

        # A helper that failed on an earlier image is intentionally not retried
        # for every file.  Keep that degraded state visible instead of silently
        # reporting a later image as fully applied.
        disabled_now = sorted(self._disabled_components)
        if disabled_now and not result.status.startswith((
            "unavailable:", "failed:", "partial:", "disabled_after_failure", "disabled",
        )):
            result = replace(result, status="partial_disabled:" + ",".join(disabled_now))

        self.last_filter_result = result
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


def enhance_anatomy_result(
    result: AnatomyFilterResult,
    image_size: tuple[int, int],
) -> AnatomyFilterResult:
    """Apply final torso/back and no-body-review policy to an anatomy result."""

    torso_regions = tuple(_derive_torso_regions(result.pose_points, image_size))
    body_regions = tuple(result.body_regions) + torso_regions
    if not result.evidence:
        return replace(result, body_regions=body_regions)

    existing_suppressed = {item.detection: item for item in result.suppressed}
    kept: list[Detection] = []
    suppressed: list[AnatomySuppression] = []
    evidence_out: list[CandidateEvidence] = []

    for evidence in result.evidence:
        detection = evidence.detection
        previous = existing_suppressed.get(detection)
        if previous is not None:
            suppressed.append(previous)
            evidence_out.append(evidence if evidence.decision == "suppress" else replace(evidence, decision="suppress"))
            continue

        pelvis_people = _signal_people(evidence.positive_signals, "near_pelvis")
        torso_hit = _best_torso_overlap(detection.box, torso_regions, evidence.matched_persons)

        # Torso/back evidence can override a same-person pelvis proximity result
        # because the torso core is explicitly above the hip line.  A pelvis
        # from a *different* person is protected: this is the important
        # close-contact/oral-scene case.
        if torso_hit is not None:
            overlap, torso_person = torso_hit
            protected_by_other_person = any(person != torso_person for person in pelvis_people)
            if not protected_by_other_person and overlap >= _TORSO_MIN_OVERLAP:
                negative = tuple(evidence.negative_signals) + (
                    f"inside_torso_back:p{torso_person}:{overlap:.3f}",
                )
                final = replace(evidence, decision="suppress", negative_signals=negative)
                evidence_out.append(final)
                suppressed.append(
                    AnatomySuppression(
                        detection=detection,
                        reason="inside_torso_back",
                        person_index=torso_person,
                        joint_distance_ratio=1.0 - overlap,
                        pelvis_distance_ratio=float(
                            evidence.pelvis_distance_ratio
                            if evidence.pelvis_distance_ratio is not None
                            else 999.0
                        ),
                    )
                )
                continue

        # Any pelvis evidence that was not invalidated by the same person's
        # torso core remains a strong KEEP signal.
        if pelvis_people:
            final = evidence if evidence.decision == "keep" else replace(evidence, decision="keep")
            evidence_out.append(final)
            kept.append(detection)
            continue

        # Real-use feedback showed that final body-analysis REVIEW boxes were
        # consistently false positives.  The raw anatomy layer is still
        # conservative, but the product-level policy now removes them.  This
        # happens only after the pelvis protection above.
        if evidence.decision == "review":
            person_index = _best_person_for_evidence(evidence)
            negative = tuple(evidence.negative_signals) + ("review_without_pelvis",)
            final = replace(evidence, decision="suppress", negative_signals=negative)
            evidence_out.append(final)
            suppressed.append(
                AnatomySuppression(
                    detection=detection,
                    reason="review_without_pelvis",
                    person_index=person_index,
                    joint_distance_ratio=0.0,
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
        body_regions=body_regions,
        evidence=tuple(evidence_out),
    )


def _derive_torso_regions(
    pose_points: Iterable[PosePoint],
    image_size: tuple[int, int],
) -> list[BodyRegion]:
    by_person: dict[int, dict[str, PosePoint]] = {}
    for point in pose_points:
        by_person.setdefault(point.person_index, {})[point.label] = point

    regions: list[BodyRegion] = []
    for person_index, points in by_person.items():
        needed = ("right_shoulder", "left_shoulder", "right_hip", "left_hip")
        if any(label not in points for label in needed):
            continue

        rs, ls = points["right_shoulder"], points["left_shoulder"]
        rh, lh = points["right_hip"], points["left_hip"]
        shoulder_mid = ((rs.x + ls.x) / 2.0, (rs.y + ls.y) / 2.0)
        hip_mid = ((rh.x + lh.x) / 2.0, (rh.y + lh.y) / 2.0)
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        length = hypot(dx, dy)
        if length < 8.0 or abs(dy) / length < _TORSO_MIN_VERTICALITY:
            # Axis-aligned torso BBoxes are unsafe for nearly-horizontal bodies.
            continue

        x0 = min(rs.x, ls.x, rh.x, lh.x)
        x1 = max(rs.x, ls.x, rh.x, lh.x)
        y0 = min(rs.y, ls.y, rh.y, lh.y)
        y1 = max(rs.y, ls.y, rh.y, lh.y)
        width, height = x1 - x0, y1 - y0
        if width < 8.0 or height < 8.0:
            continue

        x0 += width * _TORSO_INSET_X
        x1 -= width * _TORSO_INSET_X
        y0 += height * _TORSO_INSET_TOP
        y1 -= height * _TORSO_INSET_BOTTOM
        box = _clamp_box((x0, y0, x1, y1), image_size)
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            continue
        regions.append(BodyRegion(box, "torso_back_zone", 1.0, person_index, "pose"))
    return regions


def _best_torso_overlap(
    candidate_box: Sequence[int],
    torso_regions: Sequence[BodyRegion],
    matched_persons: Sequence[int],
) -> tuple[float, int] | None:
    allowed = set(matched_persons)
    best: tuple[float, int] | None = None
    for region in torso_regions:
        if allowed and region.person_index not in allowed:
            continue
        overlap = _intersection_over_candidate(candidate_box, region.box)
        if best is None or overlap > best[0]:
            best = (overlap, region.person_index)
    return best


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


def _best_person_for_evidence(evidence: CandidateEvidence) -> int:
    if evidence.matched_persons:
        return int(evidence.matched_persons[0])
    for prefix in ("inside_eye", "inside_face", "inside_head"):
        people = _signal_people(evidence.negative_signals, prefix)
        if people:
            return min(people)
    return -1


def _intersection_over_candidate(a: Sequence[int], b: Sequence[int]) -> float:
    ax0, ay0, ax1, ay1 = (float(v) for v in a)
    bx0, by0, bx1, by1 = (float(v) for v in b)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    return (iw * ih) / area


def _clamp_box(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = image_size
    x0, y0, x1, y1 = box
    return (
        max(0, min(width, int(round(x0)))),
        max(0, min(height, int(round(y0)))),
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
    )
