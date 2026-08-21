from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .experience_store import default_experience_db, utc_now


_VALID_LABELS = frozenset({"positive", "negative", "uncertain"})


@dataclass(frozen=True)
class VerifierCandidate:
    candidate_id: int
    fingerprint: str
    source_path: str
    crop_path: str | None
    box: tuple[int, int, int, int]
    detector_score: float
    detector_source: str
    final_decision: str
    pseudo_label: str
    quality_tier: str
    positive_signals: str
    negative_signals: str
    pelvis_distance: float | None
    suppression_reason: str | None
    manual_label: str | None = None


@dataclass(frozen=True)
class VerifierLabelSample:
    """One human-labelled candidate used by the learned verifier."""

    fingerprint: str
    label: str
    source_path: str
    crop_path: str | None
    box: tuple[int, int, int, int]
    detector_score: float
    detector_source: str
    final_decision: str
    positive_signals: str
    negative_signals: str
    pelvis_distance: float | None
    suppression_reason: str | None


class VerifierStore:
    """Human ground-truth labels layered on top of Experience Store.

    Labels are keyed by candidate fingerprint rather than candidate row id so
    normal-processing reruns can replace stale candidate rows without deleting
    the user's ground truth. Source images remain in place and are never copied.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path or default_experience_db())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30.0)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS verifier_labels (
                    fingerprint TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    source_path TEXT,
                    crop_path TEXT,
                    x0 INTEGER,
                    y0 INTEGER,
                    x1 INTEGER,
                    y1 INTEGER,
                    detector_score REAL,
                    detector_source TEXT,
                    final_decision TEXT,
                    positive_signals TEXT,
                    negative_signals TEXT,
                    pelvis_distance REAL,
                    suppression_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_verifier_labels_label
                    ON verifier_labels(label);
                """
            )

    def candidates(
        self,
        *,
        decision: str = "all",
        only_unlabeled: bool = True,
        limit: int = 5000,
    ) -> list[VerifierCandidate]:
        where = ["c.fingerprint IS NOT NULL"]
        params: list[object] = []
        if decision in {"keep", "suppress", "review"}:
            where.append("c.final_decision=?")
            params.append(decision)
        if only_unlabeled:
            where.append("vl.fingerprint IS NULL")
        params.append(max(1, int(limit)))

        sql = f"""
            SELECT
                c.id AS candidate_id,
                c.fingerprint,
                s.container_path AS source_path,
                c.crop_path,
                c.x0,c.y0,c.x1,c.y1,
                c.detector_score,c.detector_source,c.final_decision,
                c.pseudo_label,c.quality_tier,c.positive_signals,c.negative_signals,
                c.pelvis_distance,c.suppression_reason,
                vl.label AS manual_label
            FROM candidates c
            JOIN sources s ON s.id=c.source_id
            LEFT JOIN verifier_labels vl ON vl.fingerprint=c.fingerprint
            WHERE {' AND '.join(where)}
            ORDER BY
                CASE c.final_decision WHEN 'keep' THEN 0 WHEN 'suppress' THEN 1 ELSE 2 END,
                CASE WHEN c.detector_source LIKE '%full%' THEN 1 ELSE 0 END,
                ABS(c.detector_score - 0.38),
                c.created_at DESC
            LIMIT ?
        """
        with self.connect() as db:
            rows = db.execute(sql, params).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    def set_label(self, candidate: VerifierCandidate, label: str) -> None:
        label = str(label).strip().lower()
        if label not in _VALID_LABELS:
            raise ValueError(f"unknown verifier label: {label}")
        now = utc_now()
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO verifier_labels(
                    fingerprint,label,source_path,crop_path,x0,y0,x1,y1,
                    detector_score,detector_source,final_decision,
                    positive_signals,negative_signals,pelvis_distance,
                    suppression_reason,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(fingerprint) DO UPDATE SET
                    label=excluded.label,
                    source_path=excluded.source_path,
                    crop_path=excluded.crop_path,
                    x0=excluded.x0,y0=excluded.y0,x1=excluded.x1,y1=excluded.y1,
                    detector_score=excluded.detector_score,
                    detector_source=excluded.detector_source,
                    final_decision=excluded.final_decision,
                    positive_signals=excluded.positive_signals,
                    negative_signals=excluded.negative_signals,
                    pelvis_distance=excluded.pelvis_distance,
                    suppression_reason=excluded.suppression_reason,
                    updated_at=excluded.updated_at
                """,
                (
                    candidate.fingerprint,
                    label,
                    candidate.source_path,
                    candidate.crop_path,
                    *candidate.box,
                    candidate.detector_score,
                    candidate.detector_source,
                    candidate.final_decision,
                    candidate.positive_signals,
                    candidate.negative_signals,
                    candidate.pelvis_distance,
                    candidate.suppression_reason,
                    now,
                    now,
                ),
            )

    def labeled_samples(
        self,
        *,
        labels: Sequence[str] = ("positive", "negative"),
        limit: int | None = None,
    ) -> list[VerifierLabelSample]:
        wanted = tuple(str(label).strip().lower() for label in labels)
        if not wanted or any(label not in _VALID_LABELS for label in wanted):
            raise ValueError("labels must contain positive, negative, or uncertain")
        placeholders = ",".join("?" for _ in wanted)
        params: list[object] = list(wanted)
        tail = ""
        if limit is not None:
            if int(limit) < 1:
                return []
            tail = " LIMIT ?"
            params.append(int(limit))
        sql = f"""
            SELECT
                fingerprint,label,source_path,crop_path,x0,y0,x1,y1,
                detector_score,detector_source,final_decision,
                positive_signals,negative_signals,pelvis_distance,suppression_reason
            FROM verifier_labels
            WHERE label IN ({placeholders})
            ORDER BY updated_at, fingerprint
            {tail}
        """
        with self.connect() as db:
            rows = db.execute(sql, params).fetchall()
        return [self._row_to_label_sample(row) for row in rows]

    def coverage_stats(self) -> dict[str, int]:
        """Return distinct candidate coverage by human labels."""

        with self.connect() as db:
            row = db.execute(
                """
                SELECT
                    COUNT(DISTINCT c.fingerprint) AS candidates,
                    COUNT(DISTINCT CASE WHEN vl.fingerprint IS NOT NULL THEN c.fingerprint END) AS labeled,
                    COUNT(DISTINCT CASE WHEN vl.fingerprint IS NULL THEN c.fingerprint END) AS unlabeled
                FROM candidates c
                LEFT JOIN verifier_labels vl ON vl.fingerprint=c.fingerprint
                WHERE c.fingerprint IS NOT NULL
                """
            ).fetchone()
        return {
            "candidates": int(row["candidates"] or 0),
            "labeled": int(row["labeled"] or 0),
            "unlabeled": int(row["unlabeled"] or 0),
        }

    def stats(self) -> dict[str, int]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT label, COUNT(*) AS n FROM verifier_labels GROUP BY label"
            ).fetchall()
        counts = {"positive": 0, "negative": 0, "uncertain": 0}
        for row in rows:
            counts[str(row["label"])] = int(row["n"])
        counts["total"] = sum(counts.values())
        return counts

    @staticmethod
    def _row_to_candidate(row: sqlite3.Row) -> VerifierCandidate:
        return VerifierCandidate(
            candidate_id=int(row["candidate_id"]),
            fingerprint=str(row["fingerprint"]),
            source_path=str(row["source_path"] or ""),
            crop_path=str(row["crop_path"]) if row["crop_path"] else None,
            box=(int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])),
            detector_score=float(row["detector_score"]),
            detector_source=str(row["detector_source"] or ""),
            final_decision=str(row["final_decision"]),
            pseudo_label=str(row["pseudo_label"]),
            quality_tier=str(row["quality_tier"]),
            positive_signals=str(row["positive_signals"] or "[]"),
            negative_signals=str(row["negative_signals"] or "[]"),
            pelvis_distance=(
                float(row["pelvis_distance"])
                if row["pelvis_distance"] is not None
                else None
            ),
            suppression_reason=(
                str(row["suppression_reason"])
                if row["suppression_reason"] is not None
                else None
            ),
            manual_label=(str(row["manual_label"]) if row["manual_label"] else None),
        )

    @staticmethod
    def _row_to_label_sample(row: sqlite3.Row) -> VerifierLabelSample:
        return VerifierLabelSample(
            fingerprint=str(row["fingerprint"]),
            label=str(row["label"]),
            source_path=str(row["source_path"] or ""),
            crop_path=str(row["crop_path"]) if row["crop_path"] else None,
            box=(int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])),
            detector_score=float(row["detector_score"] or 0.0),
            detector_source=str(row["detector_source"] or ""),
            final_decision=str(row["final_decision"] or ""),
            positive_signals=str(row["positive_signals"] or "[]"),
            negative_signals=str(row["negative_signals"] or "[]"),
            pelvis_distance=(
                float(row["pelvis_distance"])
                if row["pelvis_distance"] is not None
                else None
            ),
            suppression_reason=(
                str(row["suppression_reason"])
                if row["suppression_reason"] is not None
                else None
            ),
        )
