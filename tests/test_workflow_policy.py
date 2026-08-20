import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QEvent, QMimeData, QUrl
from PySide6.QtWidgets import QApplication

from character_mosaic.pipeline import BatchProcessor, PipelineConfig
from character_mosaic.types import Detection
from character_mosaic.ui.settings_safety import EnhancedControlPanel, _PathDropGuard


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image, progress=None):
        if progress:
            progress("full", tuple(self.detections))
        return list(self.detections)


def _app():
    return QApplication.instance() or QApplication([])


def test_gui_config_uses_over_detection_only_policy():
    _app()
    panel = EnhancedControlPanel("ja")
    config = panel.config()

    assert config.review_only_over_count is True
    assert panel.person_count_label.text() == "画像内の最大人数"
    assert panel.input_open_button.text() == "開く"
    assert "ドラッグ&ドロップ" in panel.input_edit.toolTip()


def test_no_target_is_normal_in_over_detection_mode(tmp_path: Path):
    src = tmp_path / "input.png"
    out = tmp_path / "output.png"
    manual = tmp_path / "manual" / "original" / "input.png"
    annotated = tmp_path / "manual" / "annotated" / "input.png"
    Image.new("RGB", (64, 64), "white").save(src)
    cfg = PipelineConfig(expected_person_count=1, review_only_over_count=True)

    result = BatchProcessor(cfg, detector=FakeDetector([])).process_file(
        src,
        out,
        manual_review_copy=manual,
        manual_review_annotated=annotated,
    )

    assert result.error is None
    assert result.detections == tuple()
    assert result.count_mismatch is False
    assert result.manual_review_path is None
    assert result.review_required is False
    assert not manual.exists()
    assert not annotated.exists()
    assert out.read_bytes() == src.read_bytes()


def test_no_target_can_still_be_opted_into_review(tmp_path: Path):
    src = tmp_path / "input.png"
    out = tmp_path / "output.png"
    review = tmp_path / "review.png"
    Image.new("RGB", (64, 64), "white").save(src)
    cfg = PipelineConfig(
        expected_person_count=1,
        review_only_over_count=True,
        copy_no_detection_to_review=True,
    )

    result = BatchProcessor(cfg, detector=FakeDetector([])).process_file(src, out, review)

    assert result.count_mismatch is False
    assert result.review_required is True
    assert review.exists()


def test_excess_detections_are_quarantined(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    Image.new("RGB", (100, 100), "white").save(input_dir / "a.png")
    detections = [
        Detection((10, 10, 30, 30), "pussy", 0.9),
        Detection((60, 60, 80, 80), "pussy", 0.8),
    ]
    cfg = PipelineConfig(expected_person_count=1, review_only_over_count=True)

    result = BatchProcessor(cfg, detector=FakeDetector(detections)).process_folder(input_dir, output_dir)[0]

    assert result.count_mismatch is True
    assert result.manual_review_path == output_dir / "_manual_review" / "original" / "a.png"
    assert result.manual_review_path.exists()


class _FakeDropEvent:
    def __init__(self, path: Path):
        self._mime = QMimeData()
        self._mime.setUrls([QUrl.fromLocalFile(str(path))])

    def mimeData(self):
        return self._mime

    def type(self):
        return QEvent.Type.Drop


def test_dropping_image_uses_its_parent_folder(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = image_dir / "sample.png"
    image.write_bytes(b"x")

    assert _PathDropGuard.path_from_event(_FakeDropEvent(image)) == str(image_dir)
