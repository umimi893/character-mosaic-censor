from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from . import __version__
from .pipeline import BatchProcessor, JsonlRunLogger, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="アニメ/CG画像のセンシティブ領域を検出してローカル処理します。")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("input", type=Path, help="入力フォルダ")
    p.add_argument("output", type=Path, nargs="?", default=None, help="出力フォルダ（省略時は入力内の _censored）")
    p.add_argument("--review", type=Path, default=None, help="低信頼度画像の確認用フォルダ")
    p.add_argument("--people", type=int, default=1, help="画像1枚あたりの想定人数")
    p.add_argument("--detect-threshold", type=float, default=0.12)
    p.add_argument("--auto-threshold", type=float, default=0.30)
    p.add_argument("--padding", type=int, default=12)
    p.add_argument("--padding-ratio", type=float, default=0.15)
    p.add_argument("--block-size", type=int, default=20)
    p.add_argument("--mode", choices=["mosaic", "blur", "black"], default="mosaic")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--no-tiles", action="store_true")
    p.add_argument("--no-flip-tta", action="store_true")
    p.add_argument("--tile-trigger", type=int, default=1200)
    p.add_argument("--tile-3x3-trigger", type=int, default=3000)
    p.add_argument("--tile-overlap", type=float, default=0.16)
    p.add_argument("--merge-iou", type=float, default=0.45)
    p.add_argument("--merge-ios", type=float, default=0.70)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--include-male", action="store_true")
    p.add_argument("--review-no-detection", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output or (args.input / "_censored")
    cfg = PipelineConfig(
        expected_person_count=args.people,
        detection_threshold=args.detect_threshold,
        auto_threshold=args.auto_threshold,
        padding_px=args.padding,
        padding_ratio=args.padding_ratio,
        block_size=args.block_size,
        mode=args.mode,
        recursive=not args.no_recursive,
        overwrite=args.overwrite,
        tile_large_images=not args.no_tiles,
        tile_trigger_px=args.tile_trigger,
        tile_grid_3_trigger_px=args.tile_3x3_trigger,
        tile_overlap=args.tile_overlap,
        merge_iou_threshold=args.merge_iou,
        merge_ios_threshold=args.merge_ios,
        flip_tta=not args.no_flip_tta,
        female_only=not args.include_male,
        review_enabled=args.review is not None,
        copy_no_detection_to_review=args.review_no_detection,
    )
    cfg.validate()
    processor = BatchProcessor(cfg)
    images = processor.discover_images(args.input, output, args.review)

    log = output.resolve().parent / "logs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jsonl"
    logger = JsonlRunLogger(log, cfg).open(total_images=len(images))

    def progress(i, total, src, result):
        if result and result.error:
            state = f"ERROR {result.error}"
        elif result and result.cancelled:
            state = "CANCEL"
        elif result and result.skipped:
            state = "SKIP"
        else:
            n = len(result.detections) if result else 0
            review = " REVIEW" if result and result.review_required else ""
            state = f"{n} detection(s){review}"
        print(f"[{i}/{total}] {src.name}: {state}")

    try:
        results = processor.process_folder(
            args.input,
            output,
            args.review,
            progress=progress,
            images=images,
            result_callback=logger.log_result,
        )
        logger.finish(results, stopped=any(r.cancelled for r in results))
    except Exception as exc:
        logger.log_event("run_error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        logger.close()

    errors = sum(1 for r in results if r.error)
    reviews = sum(1 for r in results if r.review_required)
    print(f"完了: {len(results)}件 / 要確認 {reviews}件 / エラー {errors}件")
    print(f"ログ: {log}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
