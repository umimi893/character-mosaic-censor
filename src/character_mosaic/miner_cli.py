from __future__ import annotations

import argparse
from pathlib import Path

from .corpus_miner import CorpusMiner, CorpusMinerConfig
from .experience_store import ExperienceStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="character-mosaic-mine",
        description="Mine noisy legacy image folders/ZIPs for hard-negative experience.",
    )
    parser.add_argument("roots", nargs="+", type=Path, help="Folders containing legacy images or ZIP archives")
    parser.add_argument("--no-zip", action="store_true", help="Do not inspect ZIP archives")
    parser.add_argument("--no-idle-wait", action="store_true", help="Run even while the GPU is busy")
    parser.add_argument("--max-gpu-util", type=int, default=30, help="Idle threshold in percent (default: 30)")
    parser.add_argument("--max-images", type=int, default=0, help="Maximum images per root, 0 = unlimited")
    parser.add_argument("--no-crops", action="store_true", help="Do not store compact high-confidence candidate crops")
    parser.add_argument("--db", type=Path, default=None, help="Optional custom SQLite experience database")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    store = ExperienceStore(args.db) if args.db else ExperienceStore()
    config = CorpusMinerConfig(
        include_zip=not args.no_zip,
        idle_gpu_only=not args.no_idle_wait,
        max_gpu_utilization=max(5, min(95, args.max_gpu_util)),
        max_images=args.max_images or None,
        save_crops=not args.no_crops,
    )
    miner = CorpusMiner(config, store=store)
    exit_code = 0
    for root in args.roots:
        try:
            stats = miner.mine(
                root,
                progress=lambda s, msg: print(
                    f"[{s.processed:,}] candidates={s.candidates:,} gold={s.gold_negative:,} "
                    f"dup={s.duplicates:,} skip={s.skipped:,} :: {msg}",
                    flush=True,
                ),
            )
            print(
                f"DONE {root}: processed={stats.processed:,}, candidates={stats.candidates:,}, "
                f"gold_negative={stats.gold_negative:,}, duplicates={stats.duplicates:,}, "
                f"skipped={stats.skipped:,}, errors={stats.errors:,}",
                flush=True,
            )
            if stats.errors:
                exit_code = 1
        except Exception as exc:
            print(f"ERROR {root}: {type(exc).__name__}: {exc}", flush=True)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
