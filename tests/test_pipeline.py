from pathlib import Path

from PIL import Image

from character_mosaic.pipeline import BatchProcessor, PipelineConfig, write_review_html
from character_mosaic.types import Detection, ProcessResult


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image, progress=None):
        if progress:
            progress("full", tuple(self.detections))
        return list(self.detections)


class StoppingDetector:
    def __init__(self, stop_flag):
        self.stop_flag = stop_flag

    def detect(self, image, progress=None, stop_requested=None):
        detections = [Detection((40, 40, 60, 60), "pussy", 0.8)]
        if progress:
            progress("full", tuple(detections))
        self.stop_flag[0] = True
        return detections


def test_low_confidence_is_censored_and_reviewed(tmp_path: Path):
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    review = tmp_path / "review.png"
    Image.new("RGB", (100, 100), "white").save(src)
    det = Detection((40, 40, 60, 60), "pussy", 0.2)
    p = BatchProcessor(PipelineConfig(detection_threshold=0.1, auto_threshold=0.3), detector=FakeDetector([det]))
    result = p.process_file(src, out, review)
    assert result.error is None
    assert result.review_required is True
    assert result.review_path == review
    assert out.exists()
    assert review.exists()


def test_high_confidence_is_not_put_in_review(tmp_path: Path):
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    review = tmp_path / "review.png"
    Image.new("RGB", (100, 100), "white").save(src)
    det = Detection((40, 40, 60, 60), "pussy", 0.9)
    p = BatchProcessor(PipelineConfig(auto_threshold=0.3), detector=FakeDetector([det]))
    result = p.process_file(src, out, review)
    assert result.review_required is False
    assert not review.exists()


def test_preview_emits_original_detection_and_censored_stages(tmp_path: Path):
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    Image.new("RGB", (100, 100), "white").save(src)
    det = Detection((40, 40, 60, 60), "pussy", 0.8)
    p = BatchProcessor(PipelineConfig(), detector=FakeDetector([det]))
    frames = []
    p.process_file(src, out, preview=frames.append)
    stages = [f.stage for f in frames]
    assert stages[0] == "original"
    assert "detecting" in stages
    assert "detected" in stages
    assert stages[-1] == "censored"


def test_output_subfolder_is_not_reprocessed(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = input_dir / "_censored"
    review_dir = input_dir / "_review"
    input_dir.mkdir()
    Image.new("RGB", (20, 20), "white").save(input_dir / "a.png")
    output_dir.mkdir()
    Image.new("RGB", (20, 20), "black").save(output_dir / "old.png")
    review_dir.mkdir()
    Image.new("RGB", (20, 20), "black").save(review_dir / "old_review.png")

    p = BatchProcessor(PipelineConfig(), detector=FakeDetector([]))
    results = p.process_folder(input_dir, output_dir, review_dir)
    assert [r.source.name for r in results] == ["a.png"]


def test_subfolder_structure_is_preserved(tmp_path: Path):
    input_dir = tmp_path / "input"
    nested = input_dir / "a" / "b"
    nested.mkdir(parents=True)
    Image.new("RGB", (20, 20), "white").save(nested / "x.png")
    output_dir = tmp_path / "output"
    p = BatchProcessor(PipelineConfig(), detector=FakeDetector([]))
    p.process_folder(input_dir, output_dir)
    assert (output_dir / "a" / "b" / "x.png").exists()


def test_review_html_lists_review_images(tmp_path: Path):
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    image = review_dir / "x.png"
    Image.new("RGB", (20, 20), "white").save(image)
    result = ProcessResult(
        source=tmp_path / "input" / "x.png",
        output=tmp_path / "output" / "x.png",
        detections=(Detection((1, 2, 3, 4), "pussy", 0.21),),
        review_required=True,
        review_path=image,
    )
    index = write_review_html([result], review_dir, input_dir=tmp_path / "input")
    text = index.read_text(encoding="utf-8")
    assert "x.png" in text
    assert "pussy 0.210" in text


def test_no_detection_can_be_forced_into_review(tmp_path: Path):
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    review = tmp_path / "review.png"
    Image.new("RGB", (100, 100), "white").save(src)
    cfg = PipelineConfig(copy_no_detection_to_review=True)
    result = BatchProcessor(cfg, detector=FakeDetector([])).process_file(src, out, review)
    assert result.review_required is True
    assert review.exists()


def test_cancelled_image_is_not_written(tmp_path: Path):
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    review = tmp_path / "review.png"
    Image.new("RGB", (100, 100), "white").save(src)
    stop_flag = [False]
    processor = BatchProcessor(PipelineConfig(), detector=StoppingDetector(stop_flag))
    frames = []
    result = processor.process_file(
        src,
        out,
        review,
        preview=frames.append,
        stop_requested=lambda: stop_flag[0],
    )
    assert result.cancelled is True
    assert result.output is None
    assert not out.exists()
    assert not review.exists()
    assert frames[-1].status == "停止: この画像は保存しません"


def test_output_parent_of_input_is_rejected(tmp_path: Path):
    from character_mosaic.pipeline import validate_processing_paths

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    try:
        validate_processing_paths(input_dir, tmp_path, None)
    except ValueError as exc:
        assert "親" in str(exc)
    else:
        raise AssertionError("output ancestor must be rejected")


def test_preview_is_downscaled_but_keeps_full_coordinate_size(tmp_path: Path):
    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    Image.new("RGB", (3000, 2000), "white").save(src)
    det = Detection((1000, 600, 1200, 900), "pussy", 0.8)
    p = BatchProcessor(PipelineConfig(preview_max_side=1000), detector=FakeDetector([det]))
    frames = []
    result = p.process_file(src, out, preview=frames.append)
    assert result.error is None
    assert frames[0].coordinate_size == (3000, 2000)
    assert max(frames[0].image.size) == 1000
    assert frames[-1].coordinate_size == (3000, 2000)


def test_review_manifest_survives_resume_and_skips(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    review_dir = tmp_path / "review"
    input_dir.mkdir()
    Image.new("RGB", (100, 100), "white").save(input_dir / "a.png")
    low = Detection((40, 40, 60, 60), "pussy", 0.2)
    p = BatchProcessor(PipelineConfig(), detector=FakeDetector([low]))
    first = p.process_folder(input_dir, output_dir, review_dir)
    assert first[0].review_path is not None
    assert "a.png" in (review_dir / "index.html").read_text(encoding="utf-8")

    # Second run skips existing output; persistent manifest must keep old card.
    second = p.process_folder(input_dir, output_dir, review_dir)
    assert second[0].skipped is True
    assert "a.png" in (review_dir / "index.html").read_text(encoding="utf-8")


def test_successful_high_confidence_rerun_removes_stale_review(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    review_dir = tmp_path / "review"
    input_dir.mkdir()
    Image.new("RGB", (100, 100), "white").save(input_dir / "a.png")

    low = Detection((40, 40, 60, 60), "pussy", 0.2)
    BatchProcessor(PipelineConfig(), detector=FakeDetector([low])).process_folder(input_dir, output_dir, review_dir)
    assert (review_dir / "a.png").exists()

    high = Detection((40, 40, 60, 60), "pussy", 0.9)
    cfg = PipelineConfig(overwrite=True)
    BatchProcessor(cfg, detector=FakeDetector([high])).process_folder(input_dir, output_dir, review_dir)
    assert not (review_dir / "a.png").exists()
    text = (review_dir / "index.html").read_text(encoding="utf-8")
    assert "Review images — 0件" in text


class FatalDetector:
    def detect(self, image, progress=None, stop_requested=None):
        raise RuntimeError("model load failed")


def test_fatal_detector_error_stops_batch_after_first_image(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        Image.new("RGB", (20, 20), "white").save(input_dir / name)
    results = BatchProcessor(PipelineConfig(), detector=FatalDetector()).process_folder(input_dir, output_dir)
    assert len(results) == 1
    assert results[0].fatal_error is True
    assert "model load failed" in (results[0].error or "")


def test_jsonl_log_contains_run_boundaries_and_censor_boxes(tmp_path: Path):
    import json
    from character_mosaic.pipeline import JsonlRunLogger

    path = tmp_path / "run.jsonl"
    cfg = PipelineConfig()
    result = ProcessResult(
        source=tmp_path / "in.png",
        output=tmp_path / "out.png",
        detections=(Detection((10, 10, 20, 20), "pussy", 0.5),),
        review_required=False,
        censor_boxes=((5, 5, 25, 25),),
    )
    logger = JsonlRunLogger(path, cfg).open(total_images=1)
    logger.log_result(result)
    logger.finish([result], stopped=False)
    logger.close()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["type"] for row in rows] == ["run_start", "image", "run_end"]
    assert rows[1]["censor_boxes"] == [[5, 5, 25, 25]]
    assert rows[2]["processed"] == 1


def test_invalid_threshold_order_is_rejected():
    cfg = PipelineConfig(detection_threshold=0.4, auto_threshold=0.3)
    try:
        cfg.validate()
    except ValueError as exc:
        assert "Review threshold" in str(exc)
    else:
        raise AssertionError("invalid threshold order must fail")


def test_corrupt_image_is_logged_as_nonfatal_and_batch_continues(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "a.png").write_bytes(b"not-a-real-png")
    Image.new("RGB", (30, 30), "white").save(input_dir / "b.png")

    results = BatchProcessor(PipelineConfig(), detector=FakeDetector([])).process_folder(input_dir, output_dir)

    assert len(results) == 2
    assert results[0].error is not None
    assert results[0].fatal_error is False
    assert results[1].error is None
    assert (output_dir / "b.png").exists()


def test_no_detection_copy_preserves_original_bytes(tmp_path: Path):
    src = tmp_path / "in.jpg"
    out = tmp_path / "out.jpg"
    Image.new("RGB", (64, 48), (120, 80, 40)).save(src, quality=87, comment=b"preserve-me")
    original = src.read_bytes()

    result = BatchProcessor(PipelineConfig(), detector=FakeDetector([])).process_file(src, out)

    assert result.error is None
    assert out.read_bytes() == original


def test_detected_png_preserves_generation_text_metadata(tmp_path: Path):
    from PIL import PngImagePlugin

    src = tmp_path / "in.png"
    out = tmp_path / "out.png"
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", "prompt: test, steps: 30")
    pnginfo.add_text("custom_note", "keep-this")
    Image.new("RGBA", (80, 60), (10, 20, 30, 180)).save(src, pnginfo=pnginfo, dpi=(144, 144))
    det = Detection((20, 20, 40, 40), "pussy", 0.8)

    result = BatchProcessor(PipelineConfig(), detector=FakeDetector([det])).process_file(src, out)

    assert result.error is None
    with Image.open(out) as saved:
        assert saved.info.get("parameters") == "prompt: test, steps: 30"
        assert saved.info.get("custom_note") == "keep-this"
        assert saved.info.get("dpi") is not None


def test_detected_jpeg_keeps_exif_but_removes_applied_orientation(tmp_path: Path):
    src = tmp_path / "in.jpg"
    out = tmp_path / "out.jpg"
    image = Image.new("RGB", (80, 50), (50, 100, 150))
    exif = image.getexif()
    exif[274] = 6  # rotate 90 degrees clockwise on display
    exif[315] = "Character Mosaic Test"  # Artist
    image.save(src, quality=90, exif=exif)
    det = Detection((10, 10, 30, 30), "pussy", 0.8)

    result = BatchProcessor(PipelineConfig(), detector=FakeDetector([det])).process_file(src, out)

    assert result.error is None
    with Image.open(out) as saved:
        assert saved.size == (50, 80)  # pixels were physically transposed
        saved_exif = saved.getexif()
        assert saved_exif.get(315) == "Character Mosaic Test"
        assert saved_exif.get(274) in (None, 1)


def test_review_html_url_encodes_special_filename_characters(tmp_path: Path):
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    image = review_dir / "a #1.png"
    Image.new("RGB", (20, 20), "white").save(image)
    result = ProcessResult(
        source=tmp_path / "input" / "a #1.png",
        output=tmp_path / "output" / "a #1.png",
        detections=(Detection((1, 2, 3, 4), "pussy", 0.2),),
        review_required=True,
        review_path=image,
    )
    index = write_review_html([result], review_dir, input_dir=tmp_path / "input")
    text = index.read_text(encoding="utf-8")
    assert "a%20%231.png" in text
    assert "href='a #1.png'" not in text
