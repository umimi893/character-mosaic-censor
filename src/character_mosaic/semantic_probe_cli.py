from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .image_ops import normalize_image
from .pipeline import BatchProcessor, PipelineConfig
from .semantic_probe import probe_evidence


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "現行のKEEP/SUPPRESSを変更せず、候補cropへWD14を当てて"
            "semantic scoreをJSONL出力します。"
        )
    )
    parser.add_argument("input", type=Path, help="画像ファイルまたは画像フォルダ")
    parser.add_argument("--output", type=Path, default=None, help="JSONL保存先（省略時は標準出力のみ）")
    parser.add_argument("--max-images", type=int, default=None, help="最大解析画像数")
    parser.add_argument("--detect-threshold", type=float, default=0.25)
    parser.add_argument("--crop-scale", type=float, default=3.5)
    parser.add_argument("--min-crop-side", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-recursive", action="store_true")
    return parser


def _iter_images(root: Path, recursive: bool):
    root = root.expanduser().resolve()
    if root.is_file():
        if root.suffix.lower() in _IMAGE_SUFFIXES:
            yield root
        return
    if not root.is_dir():
        raise ValueError(f"入力が見つかりません: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in sorted(iterator):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            yield path


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images は1以上にしてください。")

    cfg = PipelineConfig(
        detection_threshold=args.detect_threshold,
        auto_threshold=max(0.30, args.detect_threshold),
        anatomy_filter=True,
        learning_enabled=False,
        review_only_over_count=True,
    )
    cfg.validate()
    processor = BatchProcessor(cfg)

    output_handle = None
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w", encoding="utf-8")

    processed = 0
    candidates = 0
    errors = 0
    try:
        for path in _iter_images(args.input, recursive=not args.no_recursive):
            if args.max_images is not None and processed >= args.max_images:
                break
            processed += 1
            try:
                with Image.open(path) as opened:
                    image = normalize_image(opened.copy())
                processor.detector.reset_filter_state()
                processor.detector.detect(image)
                result = getattr(processor.detector, "last_filter_result", None)
                evidence_items = tuple(getattr(result, "evidence", tuple()) or tuple())
                for evidence in evidence_items:
                    probe = probe_evidence(
                        path,
                        image,
                        evidence,
                        crop_scale=args.crop_scale,
                        min_crop_side=args.min_crop_side,
                        top_k=args.top_k,
                    )
                    line = json.dumps(probe.as_dict(), ensure_ascii=False)
                    print(line)
                    if output_handle is not None:
                        output_handle.write(line + "\n")
                    candidates += 1
            except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as exc:
                errors += 1
                event = {
                    "type": "error",
                    "source": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                line = json.dumps(event, ensure_ascii=False)
                print(line)
                if output_handle is not None:
                    output_handle.write(line + "\n")

        summary = {
            "type": "summary",
            "processed_images": processed,
            "candidate_rows": candidates,
            "errors": errors,
        }
        line = json.dumps(summary, ensure_ascii=False)
        print(line)
        if output_handle is not None:
            output_handle.write(line + "\n")
    finally:
        if output_handle is not None:
            output_handle.close()

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
