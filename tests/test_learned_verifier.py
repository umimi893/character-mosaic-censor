from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from character_mosaic.anatomy_filter import AnatomyFilterResult
from character_mosaic.learned_verifier import LearnedVerifierDetector, apply_learned_verifier
from character_mosaic.pseudo_labels import classify_pseudo_label
from character_mosaic.types import CandidateEvidence, Detection


class _AlwaysNegativeModel:
    crop_scale = 3.8
    min_crop_side = 32
    model_name = "fake"

    def should_suppress(self, embedding):
        return True, SimpleNamespace(
            positive_score=0.04,
            max_similarity=0.93,
            neighbors=18,
            positive_similarity=0.86,
            negative_similarity=0.93,
            positive_neighbors=9,
            negative_neighbors=9,
            positive_support=0.84,
            negative_support=0.92,
            margin=0.08,
        )


class _Inner:
    def __init__(self, result):
        self.last_filter_result = result


def _embedder(image, *, model_name, fmt):
    assert model_name == "fake"
    assert fmt == "embedding"
    return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)


def _result(*, protected: bool = False):
    detection = Detection((40, 40, 70, 80), "pussy", 0.36, "tile_2x2_1of4")
    positive = ["detector:0.360"]
    pelvis = None
    if protected:
        positive.append("near_pelvis:p0:0.110")
        pelvis = 0.11
    evidence = CandidateEvidence(
        detection=detection,
        decision="keep",
        positive_signals=tuple(positive),
        negative_signals=tuple(),
        matched_persons=(0,),
        pelvis_distance_ratio=pelvis,
    )
    return AnatomyFilterResult(kept=(detection,), evidence=(evidence,), status="applied")


def test_active_learned_verifier_can_veto_unprotected_known_negative():
    original = _result()
    final, diagnostics = apply_learned_verifier(
        original,
        Image.new("RGB", (128, 128), "white"),
        _AlwaysNegativeModel(),
        mode="active",
        embedder=_embedder,
    )
    assert final.kept == tuple()
    assert len(final.suppressed) == 1
    assert final.suppressed[0].reason == "learned_verifier_negative"
    assert final.evidence[0].decision == "suppress"
    assert diagnostics[0].would_suppress is True
    assert diagnostics[0].protected is False
    assert diagnostics[0].margin > 0.0
    assert "margin=+0.080" in final.evidence[0].negative_signals[-1]


def test_learned_verifier_never_vetoes_close_same_person_pelvis_signal():
    original = _result(protected=True)
    final, diagnostics = apply_learned_verifier(
        original,
        Image.new("RGB", (128, 128), "white"),
        _AlwaysNegativeModel(),
        mode="active",
        embedder=_embedder,
    )
    assert final.kept == original.kept
    assert final.suppressed == tuple()
    assert final.evidence[0].decision == "keep"
    assert diagnostics[0].would_suppress is True
    assert diagnostics[0].protected is True


def test_shadow_verifier_records_margin_without_changing_decision():
    original = _result()
    final, diagnostics = apply_learned_verifier(
        original,
        Image.new("RGB", (128, 128), "white"),
        _AlwaysNegativeModel(),
        mode="shadow",
        embedder=_embedder,
    )
    assert final.kept == original.kept
    assert final.suppressed == tuple()
    assert final.evidence[0].decision == "keep"
    signal = next(signal for signal in final.evidence[0].positive_signals if signal.startswith("verifier_shadow:"))
    assert "psup=0.840" in signal
    assert "nsup=0.920" in signal
    assert "margin=+0.080" in signal
    assert diagnostics[0].would_suppress is True


def test_auto_mode_requires_training_report_recommendation(tmp_path):
    result = _result()
    report = tmp_path / "training_report.json"
    report.write_text('{"activation_recommended": false}', encoding="utf-8")
    detector = LearnedVerifierDetector(
        _Inner(result),
        mode="auto",
        model=_AlwaysNegativeModel(),
        report_path=report,
        embedder=_embedder,
    )
    assert detector.resolved_mode == "off"

    report.write_text('{"activation_recommended": true}', encoding="utf-8")
    detector = LearnedVerifierDetector(
        _Inner(result),
        mode="auto",
        model=_AlwaysNegativeModel(),
        report_path=report,
        embedder=_embedder,
    )
    assert detector.resolved_mode == "active"


def test_learned_verifier_suppression_is_never_gold_pseudo_label():
    original = _result()
    final, _ = apply_learned_verifier(
        original,
        Image.new("RGB", (128, 128), "white"),
        _AlwaysNegativeModel(),
        mode="active",
        embedder=_embedder,
    )
    assert classify_pseudo_label(
        final.evidence[0], "learned_verifier_negative"
    ) == ("negative", "silver", "other")
