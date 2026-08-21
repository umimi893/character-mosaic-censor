from __future__ import annotations

import json
from pathlib import Path

from character_mosaic.pipeline import JsonlRunLogger, PipelineConfig
from character_mosaic.types import CandidateEvidence, Detection, ProcessResult


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
    assert end_row["verifier_scored_candidates"] == 0
    assert end_row["verifier_suppressed_candidates"] == 0


def test_jsonl_run_end_summarizes_learned_verifier_activity(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    vetoed = Detection((100, 120, 160, 190), "pussy", 0.34, "tile_2x2_2of4")
    kept = Detection((400, 700, 500, 820), "pussy", 0.72, "full+tile_2x2_4of4")
    result = ProcessResult(
        source=tmp_path / "in.png",
        output=tmp_path / "out.png",
        detections=(kept,),
        review_required=False,
        anatomy_suppressed=(vetoed,),
        anatomy_suppression_reasons=(
            "learned_verifier_negative;person=0;joint_ratio=0.040;pelvis_ratio=999.000",
        ),
        anatomy_filter_status="applied",
        candidate_evidence=(
            CandidateEvidence(
                detection=vetoed,
                decision="suppress",
                positive_signals=("detector:0.340", "verifier_active:p=0.040:sim=0.930:n=9"),
                negative_signals=("learned_verifier_negative:p=0.040:sim=0.930",),
                matched_persons=(0,),
            ),
            CandidateEvidence(
                detection=kept,
                decision="keep",
                positive_signals=("detector:0.720", "verifier_active:p=0.980:sim=0.910:n=9"),
                matched_persons=(0,),
            ),
        ),
    )

    logger = JsonlRunLogger(path, PipelineConfig()).open(total_images=1)
    logger.log_result(result)
    logger.finish([result], stopped=False)
    logger.close()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    end_row = rows[-1]
    assert end_row["verifier_scored_candidates"] == 2
    assert end_row["verifier_suppressed_candidates"] == 1
