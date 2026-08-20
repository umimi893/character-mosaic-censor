from __future__ import annotations

import html
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

from .pipeline_storage import _write_text_atomic
from .types import ProcessResult

def write_review_html(results: Iterable[ProcessResult], review_dir: Path, input_dir: Path | None = None) -> Path:
    """Render a persistent Review index.

    The manifest survives stop/resume and later runs, so rerunning a half-finished
    batch does not make earlier Review cards disappear from index.html.
    """
    review_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_review_manifest(review_dir)
    if input_dir is not None:
        input_dir = input_dir.resolve()
    for result in results:
        _update_review_manifest_for_result(manifest, result, review_dir, input_dir)
    _write_review_manifest(review_dir, manifest)

    cards: list[str] = []
    items = manifest.get("items", {})
    for key in sorted(items):
        item = items[key]
        image_rel = str(item.get("review_path", ""))
        if not image_rel:
            continue
        image_path = review_dir / Path(image_rel)
        if not image_path.exists():
            continue
        scores = ", ".join(
            f"{d.get('label', '?')} {float(d.get('score', 0.0)):.3f}"
            for d in item.get("detections", [])
        ) or "検出なし"
        image_url = html.escape(quote(image_rel, safe="/"), quote=True)
        cards.append(
            "<article class='card'>"
            f"<a href='{image_url}'><img loading='lazy' src='{image_url}'></a>"
            f"<div class='meta'><strong>{html.escape(str(item.get('source', key)))}</strong>"
            f"<span>{html.escape(scores)}</span></div></article>"
        )

    body = "\n".join(cards) if cards else "<p class='empty'>Review対象はありません。</p>"
    doc = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Character Mosaic Review</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
body {{ margin: 0; background: #11151b; color: #e8edf3; }}
header {{ position: sticky; top: 0; padding: 16px 22px; background: #171d25ee; backdrop-filter: blur(8px); z-index: 1; }}
h1 {{ margin: 0; font-size: 20px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; padding: 18px; }}
.card {{ background: #1b222c; border: 1px solid #303a47; border-radius: 12px; overflow: hidden; }}
.card img {{ width: 100%; height: 240px; object-fit: contain; background: #090b0f; display: block; }}
.meta {{ padding: 10px 12px; display: grid; gap: 6px; font-size: 13px; overflow-wrap: anywhere; }}
.meta span {{ color: #aeb9c6; }}
.empty {{ padding: 24px; }}
</style>
</head>
<body>
<header><h1>Review images — {len(cards)}件</h1></header>
<main class="grid">{body}</main>
</body>
</html>
"""
    index = review_dir / "index.html"
    _write_text_atomic(index, doc)
    return index
def _load_review_manifest(review_dir: Path | None) -> dict:
    if review_dir is None:
        return {"version": 1, "items": {}}
    path = review_dir / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("items"), dict):
            return data
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return {"version": 1, "items": {}}
def _write_review_manifest(review_dir: Path, manifest: dict) -> None:
    _write_text_atomic(review_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
def _update_review_manifest_for_result(
    manifest: dict,
    result: ProcessResult,
    review_dir: Path,
    input_dir: Path | None,
) -> bool:
    items = manifest.setdefault("items", {})
    try:
        source_key = result.source.resolve().relative_to(input_dir.resolve()).as_posix() if input_dir else result.source.name
    except ValueError:
        source_key = result.source.name

    # Skips/cancel/errors must not erase a known-good older Review record.
    if result.skipped or result.cancelled or result.error:
        return False

    if result.review_path and result.review_path.exists():
        try:
            review_rel = result.review_path.resolve().relative_to(review_dir.resolve()).as_posix()
        except ValueError:
            review_rel = result.review_path.name
        value = {
            "source": source_key,
            "review_path": review_rel,
            "detections": [asdict(d) for d in result.detections],
            "censor_boxes": [list(box) for box in result.censor_boxes],
            "updated_at": datetime.now().isoformat(),
        }
        if items.get(source_key) != value:
            items[source_key] = value
            return True
        return False

    if source_key in items:
        del items[source_key]
        return True
    return False
