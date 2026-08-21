from __future__ import annotations

import argparse
import json
from pathlib import Path

from .verifier_trainer import train_verifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verifier Labの人間ラベルからWD14 embeddingベースの候補Verifierを学習します。"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default="SwinV2_v3")
    parser.add_argument("--crop-scale", type=float, default=3.8)
    parser.add_argument("--min-crop-side", type=int, default=320)
    parser.add_argument("--k", type=int, default=9)
    parser.add_argument("--temperature", type=float, default=0.06)
    parser.add_argument("--max-suppress-threshold", type=float, default=0.35)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    def progress(index, total, sample, state):
        print(
            f"[{index:>4}/{total}] {sample.label:<8} "
            f"score={sample.detector_score:.3f} {state}  {Path(sample.source_path).name}"
        )

    result = train_verifier(
        output_dir=args.output_dir,
        model_name=args.model_name,
        crop_scale=args.crop_scale,
        min_crop_side=args.min_crop_side,
        k=args.k,
        temperature=args.temperature,
        max_suppress_threshold=args.max_suppress_threshold,
        max_samples=args.max_samples,
        rebuild_cache=args.rebuild_cache,
        progress=progress,
    )
    report = result.report
    coverage = report["coverage"]
    labels = report["manual_labels"]
    usable = report["usable_training_rows"]
    cv = report["cross_validation"]

    print("\n=== Verifier training summary ===")
    print(
        f"Candidate coverage: {coverage['labeled']}/{coverage['candidates']} labelled "
        f"(unlabelled {coverage['unlabeled']})"
    )
    print(
        f"Manual labels: positive={labels['positive']} negative={labels['negative']} "
        f"uncertain={labels['uncertain']}"
    )
    print(
        f"Usable images: positive={usable['positive']} negative={usable['negative']} "
        f"failures={len(report['embedding_failures'])}"
    )
    print(
        f"Leave-source-out: positive recall={cv['positive_recall']:.2%} / "
        f"negative suppression={cv['negative_suppression_rate']:.2%} / "
        f"negative precision={cv['negative_precision_among_suppressed']:.2%}"
    )
    print(
        f"Policy: positive_score < {cv['suppress_threshold']:.4f} and "
        f"similarity >= {cv['similarity_floor']:.4f}"
    )
    print(f"Activation recommended: {'YES' if report['activation_recommended'] else 'NO'}")
    print(f"Model: {result.model_path}")
    print(f"Report: {result.report_path}")
    print("\nREPORT_JSON=" + json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
