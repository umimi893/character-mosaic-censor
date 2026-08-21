from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from PIL import Image

from .types import CandidateEvidence, Detection, ProcessResult


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_learning_root() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "CharacterMosaicCensor" / "learning"
    return Path.home() / ".character_mosaic_censor" / "learning"


def default_experience_db() -> Path:
    return default_learning_root() / "experience.sqlite3"


@dataclass(frozen=True)
class MiningStats:
    discovered: int = 0
    processed: int = 0
    duplicates: int = 0
    skipped: int = 0
    errors: int = 0
    candidates: int = 0
    gold_negative: int = 0
    silver: int = 0
    quarantine: int = 0


class ExperienceStore:
    """SQLite-backed long-term memory for detector candidates.

    The database stores metadata and compact candidate crops only. Original
    source images are never copied into the learning store.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path or default_experience_db())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    id INTEGER PRIMARY KEY,
                    source_key TEXT NOT NULL UNIQUE,
                    container_path TEXT NOT NULL,
                    member_path TEXT,
                    signature TEXT,
                    sha256 TEXT,
                    size_bytes INTEGER,
                    width INTEGER,
                    height INTEGER,
                    status TEXT NOT NULL DEFAULT 'new',
                    skip_reason TEXT,
                    duplicate_of INTEGER REFERENCES sources(id),
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sources_sha256 ON sources(sha256);
                CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(status);

                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY,
                    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                    x0 INTEGER NOT NULL,
                    y0 INTEGER NOT NULL,
                    x1 INTEGER NOT NULL,
                    y1 INTEGER NOT NULL,
                    detector_label TEXT NOT NULL,
                    detector_score REAL NOT NULL,
                    detector_source TEXT,
                    final_decision TEXT NOT NULL,
                    pseudo_label TEXT NOT NULL,
                    quality_tier TEXT NOT NULL,
                    negative_kind TEXT,
                    positive_signals TEXT NOT NULL,
                    negative_signals TEXT NOT NULL,
                    pelvis_distance REAL,
                    suppression_reason TEXT,
                    fingerprint TEXT,
                    fingerprint_prefix TEXT,
                    crop_path TEXT,
                    app_version TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(source_id, x0, y0, x1, y1, detector_label, detector_source)
                );
                CREATE INDEX IF NOT EXISTS idx_candidates_pseudo ON candidates(pseudo_label, quality_tier);
                CREATE INDEX IF NOT EXISTS idx_candidates_fp ON candidates(fingerprint_prefix, pseudo_label);
                CREATE INDEX IF NOT EXISTS idx_candidates_negative_kind ON candidates(negative_kind);

                CREATE TABLE IF NOT EXISTS mining_runs (
                    id INTEGER PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    root TEXT,
                    include_zip INTEGER NOT NULL,
                    stopped INTEGER NOT NULL DEFAULT 0,
                    stats_json TEXT
                );
                """
            )

    def source_seen(self, source_key: str, signature: str) -> bool:
        with self._lock, self.connect() as db:
            row = db.execute(
                "SELECT signature, status FROM sources WHERE source_key=?",
                (source_key,),
            ).fetchone()
            return bool(row and row["signature"] == signature and row["status"] in {"processed", "duplicate", "skipped"})

    def upsert_source(
        self,
        *,
        source_key: str,
        container_path: str,
        member_path: str | None,
        signature: str,
        sha256: str | None,
        size_bytes: int | None,
        width: int | None,
        height: int | None,
        status: str,
        skip_reason: str | None = None,
        duplicate_of: int | None = None,
    ) -> int:
        now = utc_now()
        with self._lock, self.connect() as db:
            db.execute(
                """
                INSERT INTO sources(
                    source_key, container_path, member_path, signature, sha256,
                    size_bytes, width, height, status, skip_reason, duplicate_of,
                    first_seen, last_seen
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(source_key) DO UPDATE SET
                    container_path=excluded.container_path,
                    member_path=excluded.member_path,
                    signature=excluded.signature,
                    sha256=excluded.sha256,
                    size_bytes=excluded.size_bytes,
                    width=excluded.width,
                    height=excluded.height,
                    status=excluded.status,
                    skip_reason=excluded.skip_reason,
                    duplicate_of=excluded.duplicate_of,
                    last_seen=excluded.last_seen
                """,
                (
                    source_key, container_path, member_path, signature, sha256,
                    size_bytes, width, height, status, skip_reason, duplicate_of,
                    now, now,
                ),
            )
            row = db.execute("SELECT id FROM sources WHERE source_key=?", (source_key,)).fetchone()
            return int(row[0])

    def duplicate_source_id(self, sha256: str, excluding_key: str | None = None) -> int | None:
        if not sha256:
            return None
        with self._lock, self.connect() as db:
            if excluding_key:
                row = db.execute(
                    "SELECT id FROM sources WHERE sha256=? AND source_key<>? AND status='processed' LIMIT 1",
                    (sha256, excluding_key),
                ).fetchone()
            else:
                row = db.execute(
                    "SELECT id FROM sources WHERE sha256=? AND status='processed' LIMIT 1",
                    (sha256,),
                ).fetchone()
            return int(row[0]) if row else None

    def record_candidate(
        self,
        source_id: int,
        evidence: CandidateEvidence,
        *,
        pseudo_label: str,
        quality_tier: str,
        negative_kind: str | None,
        suppression_reason: str | None,
        fingerprint: str | None,
        crop_path: str | None,
        app_version: str,
    ) -> None:
        detection = evidence.detection
        now = utc_now()
        prefix = fingerprint[:4] if fingerprint else None
        with self._lock, self.connect() as db:
            db.execute(
                """
                INSERT INTO candidates(
                    source_id,x0,y0,x1,y1,detector_label,detector_score,detector_source,
                    final_decision,pseudo_label,quality_tier,negative_kind,
                    positive_signals,negative_signals,pelvis_distance,suppression_reason,
                    fingerprint,fingerprint_prefix,crop_path,app_version,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(source_id,x0,y0,x1,y1,detector_label,detector_source) DO UPDATE SET
                    final_decision=excluded.final_decision,
                    pseudo_label=excluded.pseudo_label,
                    quality_tier=excluded.quality_tier,
                    negative_kind=excluded.negative_kind,
                    positive_signals=excluded.positive_signals,
                    negative_signals=excluded.negative_signals,
                    pelvis_distance=excluded.pelvis_distance,
                    suppression_reason=excluded.suppression_reason,
                    fingerprint=excluded.fingerprint,
                    fingerprint_prefix=excluded.fingerprint_prefix,
                    crop_path=COALESCE(excluded.crop_path,candidates.crop_path),
                    app_version=excluded.app_version,
                    created_at=excluded.created_at
                """,
                (
                    source_id, *detection.box, detection.label, float(detection.score), detection.source,
                    evidence.decision, pseudo_label, quality_tier, negative_kind,
                    json.dumps(list(evidence.positive_signals), ensure_ascii=False),
                    json.dumps(list(evidence.negative_signals), ensure_ascii=False),
                    evidence.pelvis_distance_ratio, suppression_reason,
                    fingerprint, prefix, crop_path, app_version, now,
                ),
            )

    def start_mining_run(self, root: Path, include_zip: bool) -> int:
        with self._lock, self.connect() as db:
            cur = db.execute(
                "INSERT INTO mining_runs(started_at,root,include_zip) VALUES(?,?,?)",
                (utc_now(), str(root), int(include_zip)),
            )
            return int(cur.lastrowid)

    def finish_mining_run(self, run_id: int, stats: MiningStats, stopped: bool) -> None:
        with self._lock, self.connect() as db:
            db.execute(
                "UPDATE mining_runs SET finished_at=?,stopped=?,stats_json=? WHERE id=?",
                (utc_now(), int(stopped), json.dumps(asdict(stats), ensure_ascii=False), run_id),
            )

    def stats(self) -> dict[str, int]:
        with self._lock, self.connect() as db:
            source_rows = db.execute("SELECT status,COUNT(*) n FROM sources GROUP BY status").fetchall()
            candidate_rows = db.execute(
                "SELECT pseudo_label,quality_tier,COUNT(*) n FROM candidates GROUP BY pseudo_label,quality_tier"
            ).fetchall()
        out: dict[str, int] = {"sources": 0, "candidates": 0}
        for row in source_rows:
            count = int(row["n"])
            out["sources"] += count
            out[f"source_{row['status']}"] = count
        for row in candidate_rows:
            count = int(row["n"])
            out["candidates"] += count
            out[f"candidate_{row['pseudo_label']}_{row['quality_tier']}"] = count
        return out

    def close_negative_matches(self, fingerprint: str, *, max_hamming: int = 4, limit: int = 100) -> int:
        if not fingerprint:
            return 0
        prefix = fingerprint[:4]
        with self._lock, self.connect() as db:
            rows = db.execute(
                """
                SELECT fingerprint FROM candidates
                WHERE fingerprint_prefix=? AND pseudo_label='negative' AND quality_tier='gold'
                LIMIT ?
                """,
                (prefix, int(limit)),
            ).fetchall()
        return sum(
            1
            for row in rows
            if row["fingerprint"] and fingerprint_hamming(fingerprint, str(row["fingerprint"])) <= max_hamming
        )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def candidate_crop(image: Image.Image, detection: Detection, *, context_ratio: float = 1.8) -> Image.Image:
    x0, y0, x1, y1 = detection.box
    width, height = max(2, x1 - x0), max(2, y1 - y0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_w = width * context_ratio / 2.0
    half_h = height * context_ratio / 2.0
    crop_box = (
        max(0, int(round(cx - half_w))),
        max(0, int(round(cy - half_h))),
        min(image.width, int(round(cx + half_w))),
        min(image.height, int(round(cy + half_h))),
    )
    crop = image.crop(crop_box).convert("RGB")
    crop.thumbnail((256, 256), Image.Resampling.LANCZOS)
    return crop


def candidate_fingerprint(image: Image.Image) -> str:
    """Return a compact 128-bit perceptual fingerprint using aHash + dHash."""
    gray = image.convert("L")
    ah = gray.resize((8, 8), Image.Resampling.LANCZOS)
    values = list(ah.getdata())
    mean = sum(values) / max(1, len(values))
    a_bits = 0
    for value in values:
        a_bits = (a_bits << 1) | int(value >= mean)

    dh = gray.resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(dh.getdata())
    d_bits = 0
    for y in range(8):
        row = pixels[y * 9:(y + 1) * 9]
        for x in range(8):
            d_bits = (d_bits << 1) | int(row[x] > row[x + 1])
    return f"{a_bits:016x}{d_bits:016x}"


def fingerprint_hamming(left: str, right: str) -> int:
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except (TypeError, ValueError):
        return 999


def classify_pseudo_label(
    evidence: CandidateEvidence,
    suppression_reason: str | None,
) -> tuple[str, str, str | None]:
    """Backward-compatible proxy to the canonical pseudo-label policy.

    Keeping the implementation in ``pseudo_labels.py`` prevents runtime
    learning, corpus mining, and compatibility imports from drifting into
    different GOLD/SILVER policies.
    """
    from .pseudo_labels import classify_pseudo_label as canonical_classify

    return canonical_classify(evidence, suppression_reason)


def suppression_reason_map(result: ProcessResult | object) -> dict[Detection, str]:
    out: dict[Detection, str] = {}
    analysis_reasons = getattr(result, "anatomy_suppression_reasons", ())
    analysis_detections = getattr(result, "anatomy_suppressed", ())
    for detection, reason in zip(analysis_detections, analysis_reasons):
        out[detection] = str(reason).split(";", 1)[0]
    return out
