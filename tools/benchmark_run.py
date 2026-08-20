from __future__ import annotations

import argparse
import html
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from character_mosaic.detector import get_runtime_info  # noqa: E402
from character_mosaic.pipeline import BatchProcessor, PipelineConfig  # noqa: E402


def _nearest_rank(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * p) - 1))
    return ordered[index]


def _safe_runtime(info) -> dict:
    if is_dataclass(info):
        raw = asdict(info)
    else:
        raw = {name: getattr(info, name) for name in dir(info) if not name.startswith("_")}
    out = {}
    for key, value in raw.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = list(value)
    return out


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return path.name


def build_report(*, input_dir: Path, results, runtime, started: float, started_at: datetime, stopped: bool) -> dict:
    wall = max(0.0, time.perf_counter() - started)
    durations = [float(getattr(r, "elapsed_seconds", 0.0)) for r in results if not getattr(r, "skipped", False)]
    durations = [v for v in durations if v > 0]
    path_counts: Counter[str] = Counter()
    pass_counts: Counter[str] = Counter()
    stage_values: dict[str, list[float]] = defaultdict(list)
    images = []

    for result in results:
        passes = tuple(getattr(result, "analysis_passes", ()) or ())
        if passes:
            path_counts[" -> ".join(passes)] += 1
            pass_counts.update(passes)
        timings = tuple(getattr(result, "timings", ()) or ())
        for name, seconds in timings:
            if seconds >= 0:
                stage_values[str(name)].append(float(seconds))
        images.append(
            {
                "source": _relative(Path(result.source), input_dir),
                "elapsed_ms": round(float(getattr(result, "elapsed_seconds", 0.0)) * 1000, 3),
                "detections": len(getattr(result, "detections", ()) or ()),
                "review": bool(getattr(result, "review_required", False) or getattr(result, "count_mismatch", False)),
                "skipped": bool(getattr(result, "skipped", False)),
                "error": getattr(result, "error", None),
                "analysis_mode": getattr(result, "analysis_mode", None),
                "analysis_passes": list(passes),
                "analysis_reasons": list(getattr(result, "analysis_reasons", ()) or ()),
                "timings_ms": {str(name): round(float(seconds) * 1000, 3) for name, seconds in timings},
            }
        )

    processed = len(results)
    review_count = sum(1 for r in results if getattr(r, "review_required", False) or getattr(r, "count_mismatch", False))
    summary = {
        "processed": processed,
        "wall_seconds": round(wall, 3),
        "images_per_second": round(processed / wall, 4) if wall > 0 else 0.0,
        "average_ms": round((sum(durations) / len(durations)) * 1000, 3) if durations else 0.0,
        "median_ms": round(median(durations) * 1000, 3) if durations else 0.0,
        "p90_ms": round(_nearest_rank(durations, 0.90) * 1000, 3) if durations else 0.0,
        "p95_ms": round(_nearest_rank(durations, 0.95) * 1000, 3) if durations else 0.0,
        "p99_ms": round(_nearest_rank(durations, 0.99) * 1000, 3) if durations else 0.0,
        "review_count": review_count,
        "review_rate": round(review_count / processed, 6) if processed else 0.0,
        "error_count": sum(1 for r in results if getattr(r, "error", None)),
        "skipped_count": sum(1 for r in results if getattr(r, "skipped", False)),
        "adaptive_full_only": sum(1 for r in results if getattr(r, "analysis_mode", None) == "adaptive" and tuple(getattr(r, "analysis_passes", ()) or ()) == ("full",)),
        "adaptive_escalated": sum(1 for r in results if getattr(r, "analysis_mode", None) == "adaptive" and len(tuple(getattr(r, "analysis_passes", ()) or ())) > 1),
    }
    stage_summary = {
        name: {
            "samples": len(values),
            "average_ms": round(sum(values) / len(values) * 1000, 3),
            "p95_ms": round(_nearest_rank(values, 0.95) * 1000, 3),
        }
        for name, values in sorted(stage_values.items()) if values
    }
    slowest = sorted(images, key=lambda item: item["elapsed_ms"], reverse=True)[:25]
    return {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(),
        "started_at": started_at.isoformat(),
        "stopped": bool(stopped),
        "runtime": _safe_runtime(runtime),
        "summary": summary,
        "stage_timings": stage_summary,
        "analysis_paths": dict(path_counts.most_common()),
        "pass_counts": dict(pass_counts.most_common()),
        "slowest_images": slowest,
        "images": images,
    }


def render_html(data: dict) -> str:
    s = data["summary"]
    runtime = data.get("runtime", {})
    stages = data.get("stage_timings", {})
    paths = data.get("analysis_paths", {})
    slow = data.get("slowest_images", [])
    stage_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{v['average_ms']:.1f} ms</td><td>{v['p95_ms']:.1f} ms</td><td>{v['samples']}</td></tr>"
        for name, v in stages.items()
    ) or '<tr><td colspan="4">このビルドでは工程別計測なし</td></tr>'
    path_rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>" for name, count in paths.items()) or '<tr><td colspan="2">このビルドではPass内訳なし</td></tr>'
    slow_rows = "".join(
        f"<tr><td>{html.escape(i['source'])}</td><td>{i['elapsed_ms']:.1f} ms</td><td>{html.escape(' -> '.join(i['analysis_passes']))}</td></tr>"
        for i in slow
    ) or '<tr><td colspan="3">データなし</td></tr>'
    return f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>CMC Benchmark</title>
<style>body{{font-family:Segoe UI,Meiryo,sans-serif;background:#10151c;color:#e5edf5;margin:32px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}.card{{background:#171f29;border:1px solid #303c49;border-radius:10px;padding:16px}}.big{{font-size:27px;font-weight:700}}table{{width:100%;border-collapse:collapse;background:#151c25;margin:18px 0 28px}}th,td{{padding:9px 11px;border-bottom:1px solid #2d3742;text-align:left}}th{{color:#9db0c2}}small{{color:#91a1b1}}</style></head><body>
<h1>Character Mosaic Censor - Development Benchmark</h1><small>{html.escape(data['generated_at'])}</small>
<div class="grid"><div class="card">処理枚数<div class="big">{s['processed']}</div></div><div class="card">速度<div class="big">{s['images_per_second']:.2f} img/s</div></div><div class="card">中央値<div class="big">{s['median_ms']:.0f} ms</div></div><div class="card">P95<div class="big">{s['p95_ms']:.0f} ms</div></div><div class="card">Review率<div class="big">{s['review_rate']*100:.1f}%</div></div><div class="card">Error<div class="big">{s['error_count']}</div></div></div>
<h2>Runtime</h2><table><tr><th>GPU / Runtime</th><td>{html.escape(str(runtime.get('display_text','')))}</td></tr><tr><th>ONNX Runtime</th><td>{html.escape(str(runtime.get('onnxruntime_version','')))}</td></tr><tr><th>Python</th><td>{html.escape(str(runtime.get('python_version','')))}</td></tr><tr><th>CUDA selected</th><td>{html.escape(str(runtime.get('using_cuda','')))}</td></tr></table>
<h2>工程別</h2><table><tr><th>Stage</th><th>平均</th><th>P95</th><th>件数</th></tr>{stage_rows}</table>
<h2>Detection Path</h2><table><tr><th>Path</th><th>枚数</th></tr>{path_rows}</table>
<h2>遅かった画像 Top 25</h2><table><tr><th>画像</th><th>総時間</th><th>Path</th></tr>{slow_rows}</table>
<p><small>同名JSONが共有・解析用です。絶対パスは記録していません。</small></p></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Development-only benchmark for Character Mosaic Censor")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500, help="0 = all images")
    parser.add_argument("--keep-output", action="store_true")
    args = parser.parse_args()

    input_dir = args.input.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_root = input_dir / f"_cmc_benchmark_{stamp}"
    output_dir = work_root / "output"
    review_dir = work_root / "review"
    report_dir = ROOT / "benchmark_results"
    report_dir.mkdir(exist_ok=True)

    config = PipelineConfig()
    # Benchmark should measure real processing, not skip old outputs.
    config.overwrite = True
    processor = BatchProcessor(config)
    images = processor.discover_images(input_dir, output_dir, review_dir)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit("No supported images found.")

    print(f"Images: {len(images)}")
    print("Checking runtime...")
    runtime = get_runtime_info()
    print(getattr(runtime, "display_text", runtime))
    print(f"Temporary output: {work_root}")

    started_at = datetime.now()
    started = time.perf_counter()
    results = []

    def on_progress(index, total, source, result):
        results.append(result)
        if index == 1 or index == total or index % 25 == 0:
            elapsed = max(0.001, time.perf_counter() - started)
            rate = index / elapsed
            eta = (total - index) / rate if rate > 0 else 0
            print(f"[{index:>5}/{total}] {rate:6.2f} img/s  ETA {eta:7.1f}s  {Path(source).name}")

    processor.process_folder(
        input_dir,
        output_dir,
        review_dir,
        progress=on_progress,
        images=images,
    )
    data = build_report(input_dir=input_dir, results=results, runtime=runtime, started=started, started_at=started_at, stopped=False)
    json_path = report_dir / f"benchmark_{stamp}.json"
    html_path = report_dir / f"benchmark_{stamp}.html"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html(data), encoding="utf-8")

    s = data["summary"]
    print("\n=== RESULT ===")
    print(f"Processed : {s['processed']}")
    print(f"Speed     : {s['images_per_second']:.2f} img/s")
    print(f"Median    : {s['median_ms']:.1f} ms")
    print(f"P95       : {s['p95_ms']:.1f} ms")
    print(f"Review    : {s['review_count']} ({s['review_rate']*100:.1f}%)")
    print(f"Errors    : {s['error_count']}")
    print(f"JSON      : {json_path}")
    print(f"HTML      : {html_path}")

    if not args.keep_output:
        shutil.rmtree(work_root, ignore_errors=True)
    if os.name == "nt":
        try:
            os.startfile(html_path)  # type: ignore[attr-defined]
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
