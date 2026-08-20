from __future__ import annotations

import json
from pathlib import Path

from character_mosaic.pipeline import JsonlRunLogger, PipelineConfig
from character_mosaic.types import Detection, ProcessResult


def test_anatomy_filter_is_enabled_by_default():
    assert PipelineConfig().anatomy_filter is True


def test_jsonl_contains_anatomy_suppression_diagnostics(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    suppressed = Detection((10, 20, 30, 40), "pussy", 0.42, "full")
    result = ProcessResult(
        source=tmp_path / "in.png",
        output=tmp_path / "out.png",
        detections=tuple(),
        review_required=False,
        anatomy_suppressed=(suppressed,),
        anatomy_suppression_reasons=(
            "near_right_knee;person=0;joint_ratio=0.050;pelvis_ratio=0.900",
        ),
        anatomy_filter_status="applied",
    )

    logger = JsonlRunLogger(path, PipelineConfig()).open(total_images=1)
    logger.log_result(result)
    logger.finish([result], stopped=False)
    logger.close()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    image_row = rows[1]
    end_row = rows[2]

    assert image_row["anatomy_filter_status"] == "applied"
    assert image_row["anatomy_suppressed"][0]["box"] == [10, 20, 30, 40]
    assert image_row["anatomy_suppression_reasons"][0].startswith("near_right_knee")
    assert end_row["anatomy_suppressed"] == 1
