from __future__ import annotations

import numpy as np

from character_mosaic.verifier_store import VerifierLabelSample
from character_mosaic.verifier_trainer import (
    _select_training_samples,
    choose_conservative_policy,
    cross_validated_scores,
    stable_source_id,
)


def _label_sample(fingerprint: str, label: str) -> VerifierLabelSample:
    return VerifierLabelSample(
        fingerprint=fingerprint,
        label=label,
        source_path=f"F:/clean/{fingerprint}.png",
        crop_path=None,
        box=(1, 2, 10, 20),
        detector_score=0.3,
        detector_source="full",
        final_decision="keep",
        positive_signals="[]",
        negative_signals="[]",
        pelvis_distance=None,
        suppression_reason=None,
    )


class _FakeStore:
    def __init__(self, samples):
        self.samples = list(samples)

    def labeled_samples(self, *, labels, limit=None, exclude_derived=True):
        result = [sample for sample in self.samples if sample.label in set(labels)]
        return result if limit is None else result[: int(limit)]


def test_curated_selection_uses_latest_label_window_before_dropping_uncertain():
    samples = [
        _label_sample("old-positive", "positive"),
        _label_sample("old-negative", "negative"),
        _label_sample("new-negative", "negative"),
        _label_sample("new-uncertain", "uncertain"),
        _label_sample("new-positive", "positive"),
    ]

    selected, info = _select_training_samples(_FakeStore(samples), 3)

    assert [sample.fingerprint for sample in selected] == ["new-negative", "new-positive"]
    assert info["window_total"] == 3
    assert info["window_positive"] == 1
    assert info["window_negative"] == 1
    assert info["window_uncertain"] == 1
    assert info["training_rows_before_io"] == 2


def test_conservative_policy_preserves_all_observed_positives():
    scores = np.asarray([0.82, 0.76, 0.71, 0.02, 0.08, 0.18], dtype=np.float32)
    negative_similarities = np.asarray([0.71, 0.69, 0.68, 0.94, 0.90, 0.86], dtype=np.float32)
    labels = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int8)
    policy = choose_conservative_policy(
        scores,
        negative_similarities,
        labels,
        max_suppress_threshold=0.35,
    )
    assert policy["suppress_threshold"] == 0.35
    assert policy["positive_false_suppressed"] == 0
    assert policy["positive_recall"] == 1.0
    assert policy["negative_suppression_rate"] > 0.0


def test_negative_similarity_floor_rejects_out_of_distribution_negative_vote():
    scores = np.asarray([0.90, 0.80, 0.10, 0.12], dtype=np.float32)
    negative_similarities = np.asarray([0.70, 0.72, 0.95, 0.50], dtype=np.float32)
    labels = np.asarray([1, 1, 0, 0], dtype=np.int8)

    policy = choose_conservative_policy(
        scores,
        negative_similarities,
        labels,
        max_suppress_threshold=0.35,
    )

    assert policy["positive_false_suppressed"] == 0
    assert policy["negative_suppressed"] == 1
    assert policy["negative_precision_among_suppressed"] == 1.0


def test_cross_validation_never_votes_from_same_source():
    embeddings = np.asarray([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.1, 0.9],
    ], dtype=np.float32)
    labels = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.int8)
    source_ids = np.asarray([10, 10, 11, 20, 20, 21], dtype=np.int64)
    scores, negative_similarities, neighbors = cross_validated_scores(
        embeddings,
        labels,
        source_ids,
        k=3,
        temperature=0.05,
    )
    assert np.all(neighbors > 0)
    assert np.all(np.isfinite(scores))
    assert np.all(negative_similarities <= 1.00001)


def test_stable_source_id_is_repeatable_and_path_sensitive():
    assert stable_source_id("A:/one.png") == stable_source_id("A:/one.png")
    assert stable_source_id("A:/one.png") != stable_source_id("A:/two.png")
