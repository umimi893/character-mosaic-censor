from __future__ import annotations

from pathlib import Path

from PIL import Image

from character_mosaic.pipeline import BatchProcessor, PipelineConfig
from character_mosaic.types import Detection


class FakeDetector:
    def __init__(self, detections):
        self.detections = list(detections)

    def detect(self, _image, progress=None, stop_requested=None):
        if progress:
            progress("full", tuple(self.detections))
        return list(self.detections)


def test_overwrite_rerun_refreshes_output_and_removes_stale_review(tmp_path: Path):
    source = tmp_path / "in.png"
    output = tmp_path / "out.png"
    review = tmp_path / "review.png"
    Image.new("RGB", (100, 100), "white").save(source)

    low = Detection((40, 40, 60, 60), "pussy", 0.20)
    first = BatchProcessor(
        PipelineConfig(overwrite=True, auto_threshold=0.30),
        detector=FakeDetector([low]),
    ).process_file(source, output, review)

    assert first.skipped is False
    assert review.exists()

    high = Detection((40, 40, 60, 60), "pussy", 0.90)
    second = BatchProcessor(
        PipelineConfig(overwrite=True, auto_threshold=0.30),
        detector=FakeDetector([high]),
    ).process_file(source, output, review)

    assert second.skipped is False
    assert second.review_required is False
    assert output.exists()
    assert not review.exists()


def test_overwrite_rerun_removes_stale_manual_review_bundle(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    source = input_dir / "a.png"
    Image.new("RGB", (100, 100), "white").save(source)

    two = [
        Detection((10, 10, 30, 30), "pussy", 0.9),
        Detection((60, 60, 80, 80), "pussy", 0.9),
    ]
    cfg = PipelineConfig(
        overwrite=True,
        expected_person_count=1,
        review_only_over_count=True,
    )
    first = BatchProcessor(cfg, detector=FakeDetector(two)).process_folder(input_dir, output_dir)[0]

    edit = output_dir / "_manual_review" / "edit" / "a.png"
    reference = output_dir / "_manual_review" / "reference_bbox" / "a.png"
    auto = output_dir / "_manual_review" / "auto_censored" / "a.png"
    assert first.count_mismatch is True
    assert edit.exists()
    assert reference.exists()
    assert auto.exists()

    one = [Detection((40, 40, 60, 60), "pussy", 0.9)]
    second = BatchProcessor(cfg, detector=FakeDetector(one)).process_folder(input_dir, output_dir)[0]

    assert second.skipped is False
    assert second.count_mismatch is False
    assert not edit.exists()
    assert not reference.exists()
    assert not auto.exists()
