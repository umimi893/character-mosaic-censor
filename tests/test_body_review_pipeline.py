from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from character_mosaic.pipeline import BatchProcessor, PipelineConfig
from character_mosaic.types import BodyRegion, CandidateEvidence, Detection, PosePoint


class BodyReviewDetector:
    def __init__(self, *, disabled_components=()):
        self.detection = Detection((40, 40, 60, 60), "pussy", 0.90)
        self._disabled_components = set(disabled_components)
        self.last_filter_result = SimpleNamespace(
            suppressed=tuple(),
            status="applied",
            body_regions=(BodyRegion((10, 10, 90, 90), "face", 0.9, 0, "test"),),
            pose_points=(PosePoint(50.0, 70.0, 0.9, "neck", 0),),
            pose_edges=tuple(),
            evidence=(CandidateEvidence(self.detection, "review", ("detector:0.900",), ("inside_face:p0:0.900",), (0,), 1.0),),
        )

    @property
    def requires_review(self):
        return True

    def detect(self, _image, progress=None, stop_requested=None):
        if progress:
            progress("full", (self.detection,))
        return [self.detection]


def _process(tmp_path: Path, detector: BodyReviewDetector):
    source = tmp_path / "in.png"
    output = tmp_path / "out.png"
    review = tmp_path / "review.png"
    Image.new("RGB", (100, 100), "white").save(source)
    result = BatchProcessor(PipelineConfig(auto_threshold=0.30), detector=detector).process_file(source, output, review)
    return result, review


def test_body_review_decision_enters_normal_review_workflow(tmp_path: Path):
    result, review = _process(tmp_path, BodyReviewDetector())

    assert result.error is None
    assert result.review_required is True
    assert result.review_path == review
    assert review.exists()
    assert result.candidate_evidence[0].decision == "review"
    assert result.body_regions[0].kind == "face"


def test_disabled_helper_remains_visible_in_result_status(tmp_path: Path):
    result, _ = _process(tmp_path, BodyReviewDetector(disabled_components={"head"}))

    assert result.error is None
    assert result.anatomy_filter_status == "partial_disabled:head"
