from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from .anatomy_filter import AnatomyFilterResult, AnatomySuppression
from .semantic_probe import candidate_context_crop
from .types import CandidateEvidence, Detection
from .verifier_model import (
    VerifierKnnModel,
    default_verifier_model_path,
    default_verifier_report_path,
    normalize_embedding,
)


_VALID_MODES = frozenset({"off", "shadow", "active", "auto"})
_REASON = "learned_verifier_negative"
_PROTECTED_SIGNAL_PREFIXES = (
    "inside_groin_zone:",
    "strong_full_same_person_pelvis_safety:",
)
_NEAR_PELVIS_PROTECT_RATIO = 0.20


@dataclass(frozen=True)
class LearnedVerifierDiagnostic:
    detection: Detection
    mode: str
    positive_score: float
    max_similarity: float
    neighbors: int
    would_suppress: bool
    protected: bool
    positive_similarity: float = -1.0
    negative_similarity: float = -1.0
    positive_neighbors: int = 0
    negative_neighbors: int = 0
    positive_support: float = -1.0
    negative_support: float = -1.0
    margin: float = 0.0
    error: str | None = None


class LearnedVerifierDetector:
    """Human-labelled WD14-embedding verifier layered over the legacy policy.

    Modes:
    * off: no model load and no runtime cost;
    * shadow: score candidates and annotate diagnostics, never change decisions;
    * active: allow conservative learned vetoes;
    * auto: become active only when training_report.json explicitly recommends it.

    Any model/load/inference error fails open. Strong groin/pelvis safety signals also
    fail open so the learned veto cannot undo the recall protections added earlier.
    """

    def __init__(
        self,
        detector,
        *,
        mode: str = "auto",
        model_path: Path | None = None,
        report_path: Path | None = None,
        model: VerifierKnnModel | None = None,
        embedder: Callable | None = None,
    ):
        mode = str(mode).strip().lower()
        if mode not in _VALID_MODES:
            raise ValueError(f"unknown verifier mode: {mode}")
        self.detector = detector
        self.requested_mode = mode
        self.model_path = Path(model_path or default_verifier_model_path())
        self.report_path = Path(report_path or default_verifier_report_path())
        self._model = model
        self._embedder = embedder
        self._resolved_mode: str | None = None
        self._load_attempted = model is not None
        self.last_verifier_diagnostics: tuple[LearnedVerifierDiagnostic, ...] = tuple()
        self.last_filter_result = getattr(detector, "last_filter_result", AnatomyFilterResult(tuple()))

    @property
    def requires_review(self) -> bool:
        return bool(getattr(self.last_filter_result, "requires_review", False))

    @property
    def resolved_mode(self) -> str:
        self._ensure_model()
        return self._resolved_mode or "off"

    def reset_filter_state(self) -> None:
        reset = getattr(self.detector, "reset_filter_state", None)
        if callable(reset):
            reset()
        self.last_verifier_diagnostics = tuple()
        self.last_filter_result = getattr(self.detector, "last_filter_result", AnatomyFilterResult(tuple()))

    def detect(self, image, progress=None, stop_requested=None):
        detections = list(self.detector.detect(image, progress=progress, stop_requested=stop_requested))
        result = getattr(self.detector, "last_filter_result", None)
        if result is None or not getattr(result, "evidence", None):
            self.last_filter_result = result or AnatomyFilterResult(tuple(detections), status="not_run")
            self.last_verifier_diagnostics = tuple()
            return detections

        self._ensure_model()
        mode = self._resolved_mode or "off"
        if mode == "off" or self._model is None:
            self.last_filter_result = result
            self.last_verifier_diagnostics = tuple()
            return list(result.kept)

        final, diagnostics = apply_learned_verifier(
            result,
            image,
            self._model,
            mode=mode,
            embedder=self._embedder,
        )
        self.last_filter_result = final
        self.last_verifier_diagnostics = diagnostics
        return list(final.kept)

    def _ensure_model(self) -> None:
        if self._resolved_mode is not None:
            return
        if self.requested_mode == "off":
            self._resolved_mode = "off"
            return

        if self._model is None and not self._load_attempted:
            self._load_attempted = True
            try:
                if self.model_path.is_file():
                    self._model = VerifierKnnModel.load(self.model_path)
            except Exception:
                self._model = None

        if self._model is None:
            self._resolved_mode = "off"
            return

        if self.requested_mode in {"shadow", "active"}:
            self._resolved_mode = self.requested_mode
            return

        try:
            report = json.loads(self.report_path.read_text(encoding="utf-8"))
            self._resolved_mode = "active" if bool(report.get("activation_recommended")) else "off"
        except Exception:
            self._resolved_mode = "off"

    def __getattr__(self, name: str):
        return getattr(self.detector, name)


def apply_learned_verifier(
    result: AnatomyFilterResult,
    image,
    model: VerifierKnnModel,
    *,
    mode: str = "active",
    embedder: Callable | None = None,
) -> tuple[AnatomyFilterResult, tuple[LearnedVerifierDiagnostic, ...]]:
    mode = str(mode).strip().lower()
    if mode not in {"shadow", "active"}:
        return result, tuple()

    kept_set = set(result.kept)
    decisions: dict[Detection, AnatomySuppression] = {}
    annotations: dict[Detection, str] = {}
    diagnostics: list[LearnedVerifierDiagnostic] = []

    if embedder is None:
        from imgutils.tagging import get_wd14_tags

        embedder = get_wd14_tags

    for evidence in result.evidence:
        if evidence.decision != "keep" or evidence.detection not in kept_set:
            continue

        protected = _is_recall_protected(evidence)
        try:
            crop, _ = candidate_context_crop(
                image,
                evidence.detection,
                scale=model.crop_scale,
                min_side=model.min_crop_side,
            )
            embedding = normalize_embedding(
                embedder(crop, model_name=model.model_name, fmt="embedding")
            )
            would_suppress, score = model.should_suppress(embedding)
            signal = (
                f"verifier_{mode}:p={score.positive_score:.3f}:sim={score.max_similarity:.3f}:"
                f"psim={score.positive_similarity:.3f}:nsim={score.negative_similarity:.3f}:"
                f"psup={score.positive_support:.3f}:nsup={score.negative_support:.3f}:"
                f"margin={score.margin:+.3f}:pn={score.positive_neighbors}:nn={score.negative_neighbors}"
            )
            annotations[evidence.detection] = signal
            diagnostics.append(
                LearnedVerifierDiagnostic(
                    detection=evidence.detection,
                    mode=mode,
                    positive_score=score.positive_score,
                    max_similarity=score.max_similarity,
                    neighbors=score.neighbors,
                    would_suppress=bool(would_suppress),
                    protected=protected,
                    positive_similarity=score.positive_similarity,
                    negative_similarity=score.negative_similarity,
                    positive_neighbors=score.positive_neighbors,
                    negative_neighbors=score.negative_neighbors,
                    positive_support=score.positive_support,
                    negative_support=score.negative_support,
                    margin=score.margin,
                )
            )
            if mode != "active" or not would_suppress or protected:
                continue

            person_index = int(evidence.matched_persons[0]) if len(evidence.matched_persons) == 1 else -1
            decisions[evidence.detection] = AnatomySuppression(
                detection=evidence.detection,
                reason=_REASON,
                person_index=person_index,
                joint_distance_ratio=max(0.0, min(1.0, score.positive_score)),
                pelvis_distance_ratio=float(
                    evidence.pelvis_distance_ratio
                    if evidence.pelvis_distance_ratio is not None
                    else 999.0
                ),
            )
        except Exception as exc:
            diagnostics.append(
                LearnedVerifierDiagnostic(
                    detection=evidence.detection,
                    mode=mode,
                    positive_score=0.5,
                    max_similarity=-1.0,
                    neighbors=0,
                    would_suppress=False,
                    protected=protected,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    if not annotations and not decisions:
        return result, tuple(diagnostics)

    diagnostic_by_detection = {item.detection: item for item in diagnostics}
    evidence_out: list[CandidateEvidence] = []
    new_suppressions: list[AnatomySuppression] = []
    for evidence in result.evidence:
        signal = annotations.get(evidence.detection)
        suppression = decisions.get(evidence.detection)
        if signal is None:
            evidence_out.append(evidence)
            continue

        positive = tuple(evidence.positive_signals) + (signal,)
        if suppression is None:
            evidence_out.append(replace(evidence, positive_signals=positive))
            continue

        diagnostic = diagnostic_by_detection[evidence.detection]
        negative_signal = (
            f"{_REASON}:margin={diagnostic.margin:+.3f}:"
            f"nsim={diagnostic.negative_similarity:.3f}"
        )
        evidence_out.append(
            replace(
                evidence,
                decision="suppress",
                positive_signals=positive,
                negative_signals=tuple(evidence.negative_signals) + (negative_signal,),
            )
        )
        new_suppressions.append(suppression)

    final = replace(
        result,
        kept=tuple(detection for detection in result.kept if detection not in decisions),
        suppressed=tuple(result.suppressed) + tuple(new_suppressions),
        evidence=tuple(evidence_out),
    )
    return final, tuple(diagnostics)


def _is_recall_protected(evidence: CandidateEvidence) -> bool:
    if len(evidence.matched_persons) > 1:
        return True
    if any(signal.startswith(_PROTECTED_SIGNAL_PREFIXES) for signal in evidence.positive_signals):
        return True
    if (
        evidence.pelvis_distance_ratio is not None
        and evidence.pelvis_distance_ratio <= _NEAR_PELVIS_PROTECT_RATIO
        and any(signal.startswith("near_pelvis:") for signal in evidence.positive_signals)
    ):
        return True
    matched = set(int(value) for value in evidence.matched_persons)
    pelvis_people = _signal_people(evidence.positive_signals, "near_pelvis")
    return bool(pelvis_people - matched)


def _signal_people(signals: tuple[str, ...], prefix: str) -> set[int]:
    people: set[int] = set()
    needle = prefix + ":p"
    for signal in signals:
        if not signal.startswith(needle):
            continue
        text = signal[len(needle):].split(":", 1)[0]
        try:
            people.add(int(text))
        except ValueError:
            pass
    return people
