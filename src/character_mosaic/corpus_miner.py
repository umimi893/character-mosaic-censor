from __future__ import annotations

import io
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image, UnidentifiedImageError

from . import __version__
from .anatomy_filter import AnatomyFilterConfig
from .body_geometry import GeometryV2Detector
from .body_reasoning import BodyReasoningDetector
from .detector import AnimeCensorDetector, DetectorConfig
from .experience_store import (
    ExperienceStore,
    MiningStats,
    candidate_crop,
    candidate_fingerprint,
    default_learning_root,
    sha256_bytes,
    sha256_file,
)
from .image_ops import normalize_image
from .pseudo_labels import classify_pseudo_label


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class CorpusMinerConfig:
    include_zip: bool = True
    recursive: bool = True
    min_side: int = 128
    max_pixels: int = 80_000_000
    max_file_bytes: int = 256 * 1024 * 1024
    max_archive_member_bytes: int = 256 * 1024 * 1024
    save_crops: bool = True
    idle_gpu_only: bool = True
    max_gpu_utilization: int = 30
    idle_poll_seconds: float = 4.0
    max_images: int | None = None


@dataclass(frozen=True)
class CorpusEntry:
    source_key: str
    container_path: Path
    member_path: str | None
    signature: str
    size_bytes: int
    data: bytes | None = None
    skip_reason: str | None = None


ProgressCallback = Callable[[MiningStats, str], None]
StopRequested = Callable[[], bool]


class CorpusMiner:
    """Mine noisy image folders/ZIPs for reusable candidate experience.

    Legacy material is never trusted wholesale as ground truth. The miner first
    asks the current detector for candidate regions, then stores only
    conservative pseudo-label/evidence records. Corrupt, duplicate, ambiguous,
    or unreasonably large inputs do not abort the run.
    """

    def __init__(
        self,
        config: CorpusMinerConfig | None = None,
        *,
        store: ExperienceStore | None = None,
        detector=None,
    ):
        self.config = config or CorpusMinerConfig()
        self.store = store or ExperienceStore()
        self.detector = detector or self._build_detector()

    @staticmethod
    def _build_detector():
        # A zero-result corpus image contributes no candidate, so expensive
        # flip/rotation retries add little mining value. Production processing
        # keeps its normal TTA behavior; only the unattended miner disables it.
        base = AnimeCensorDetector(DetectorConfig(flip_tta=False))
        body = BodyReasoningDetector(base, AnatomyFilterConfig(enabled=True))
        return GeometryV2Detector(body)

    def mine(
        self,
        root: Path,
        *,
        progress: ProgressCallback | None = None,
        stop_requested: StopRequested | None = None,
    ) -> MiningStats:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"学習素材フォルダが見つかりません: {root}")

        run_id = self.store.start_mining_run(root, self.config.include_zip)
        counters = {
            "discovered": 0,
            "processed": 0,
            "duplicates": 0,
            "skipped": 0,
            "errors": 0,
            "candidates": 0,
            "gold_negative": 0,
            "silver": 0,
            "quarantine": 0,
        }
        stopped = False
        try:
            for entry in self._iter_entries(root, stop_requested):
                counters["discovered"] += 1
                if self.config.max_images is not None and counters["processed"] >= self.config.max_images:
                    break
                if stop_requested and stop_requested():
                    stopped = True
                    break

                if self.store.source_seen(entry.source_key, entry.signature):
                    counters["skipped"] += 1
                    self._emit(
                        progress,
                        counters,
                        f"既に解析済み: {entry.member_path or entry.container_path.name}",
                    )
                    continue

                if entry.skip_reason:
                    counters["skipped"] += 1
                    self._record_bad_entry(entry, entry.skip_reason)
                    self._emit(
                        progress,
                        counters,
                        f"SKIP: {entry.member_path or entry.container_path.name} ({entry.skip_reason})",
                    )
                    continue

                if self.config.idle_gpu_only and not self._wait_for_gpu_idle(stop_requested):
                    stopped = True
                    break

                try:
                    outcome = self._process_entry(entry)
                    counters["processed"] += 1
                    counters["duplicates"] += int(outcome["duplicate"])
                    counters["candidates"] += int(outcome["candidates"])
                    counters["gold_negative"] += int(outcome["gold_negative"])
                    counters["silver"] += int(outcome["silver"])
                    counters["quarantine"] += int(outcome["quarantine"])
                    self._emit(progress, counters, str(outcome["message"]))
                except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
                    counters["skipped"] += 1
                    self._record_bad_entry(entry, f"{type(exc).__name__}: {exc}")
                    self._emit(progress, counters, f"SKIP: {entry.member_path or entry.container_path.name}")
                except Exception as exc:
                    counters["errors"] += 1
                    self._record_bad_entry(
                        entry,
                        f"{type(exc).__name__}: {exc}",
                        status="error",
                    )
                    self._emit(
                        progress,
                        counters,
                        f"ERROR: {entry.member_path or entry.container_path.name}: {exc}",
                    )
        finally:
            stats = MiningStats(**counters)
            self.store.finish_mining_run(run_id, stats, stopped)
        return stats

    def _process_entry(self, entry: CorpusEntry) -> dict[str, int | bool | str]:
        raw = entry.data if entry.data is not None else entry.container_path.read_bytes()
        digest = sha256_bytes(raw) if entry.data is not None else sha256_file(entry.container_path)
        duplicate_of = self.store.duplicate_source_id(digest, excluding_key=entry.source_key)
        if duplicate_of is not None:
            self.store.upsert_source(
                source_key=entry.source_key,
                container_path=str(entry.container_path),
                member_path=entry.member_path,
                signature=entry.signature,
                sha256=digest,
                size_bytes=entry.size_bytes,
                width=None,
                height=None,
                status="duplicate",
                duplicate_of=duplicate_of,
            )
            return {
                "duplicate": True,
                "candidates": 0,
                "gold_negative": 0,
                "silver": 0,
                "quarantine": 0,
                "message": f"重複をスキップ: {entry.member_path or entry.container_path.name}",
            }

        image = self._decode(raw)
        if min(image.size) < self.config.min_side:
            raise ValueError(f"too_small:{image.width}x{image.height}")
        if image.width * image.height > self.config.max_pixels:
            raise ValueError(f"too_large:{image.width}x{image.height}")

        source_id = self.store.upsert_source(
            source_key=entry.source_key,
            container_path=str(entry.container_path),
            member_path=entry.member_path,
            signature=entry.signature,
            sha256=digest,
            size_bytes=entry.size_bytes,
            width=image.width,
            height=image.height,
            status="processed",
        )

        self.detector.detect(image)
        analysis = getattr(self.detector, "last_filter_result", None)
        evidence_items = tuple(getattr(analysis, "evidence", ())) if analysis is not None else tuple()
        suppressed = (
            {item.detection: item.reason for item in tuple(getattr(analysis, "suppressed", ()))}
            if analysis is not None
            else {}
        )

        gold = silver = quarantine = 0
        for evidence in evidence_items:
            reason = suppressed.get(evidence.detection)
            pseudo_label, tier, negative_kind = classify_pseudo_label(evidence, reason)
            if pseudo_label == "negative" and tier == "gold":
                gold += 1
            elif tier == "silver":
                silver += 1
            else:
                quarantine += 1

            crop = candidate_crop(image, evidence.detection)
            fingerprint = candidate_fingerprint(crop)
            crop_path: str | None = None
            if self.config.save_crops and tier in {"gold", "silver"}:
                crop_path = str(
                    self._save_crop(crop, fingerprint, pseudo_label, negative_kind)
                )
            self.store.record_candidate(
                source_id,
                evidence,
                pseudo_label=pseudo_label,
                quality_tier=tier,
                negative_kind=negative_kind,
                suppression_reason=reason,
                fingerprint=fingerprint,
                crop_path=crop_path,
                app_version=__version__,
            )

        return {
            "duplicate": False,
            "candidates": len(evidence_items),
            "gold_negative": gold,
            "silver": silver,
            "quarantine": quarantine,
            "message": (
                f"解析: {entry.member_path or entry.container_path.name} / "
                f"候補 {len(evidence_items)} / GOLD負例 {gold}"
            ),
        }

    def _decode(self, raw: bytes) -> Image.Image:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            return normalize_image(opened.copy())

    def _record_bad_entry(
        self,
        entry: CorpusEntry,
        reason: str,
        status: str = "skipped",
    ) -> None:
        self.store.upsert_source(
            source_key=entry.source_key,
            container_path=str(entry.container_path),
            member_path=entry.member_path,
            signature=entry.signature,
            sha256=None,
            size_bytes=entry.size_bytes,
            width=None,
            height=None,
            status=status,
            skip_reason=reason[:1000],
        )

    def _iter_entries(
        self,
        root: Path,
        stop_requested: StopRequested | None,
    ) -> Iterator[CorpusEntry]:
        iterator = root.rglob("*") if self.config.recursive else root.glob("*")
        for path in sorted(iterator):
            if stop_requested and stop_requested():
                return
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            stat = path.stat()
            if suffix in _IMAGE_SUFFIXES:
                too_large = stat.st_size > self.config.max_file_bytes
                yield CorpusEntry(
                    source_key=f"file://{path.resolve()}",
                    container_path=path.resolve(),
                    member_path=None,
                    signature=f"{stat.st_size}:{stat.st_mtime_ns}:v{__version__}",
                    size_bytes=stat.st_size,
                    skip_reason=(
                        f"file_too_large:{stat.st_size}"
                        if too_large
                        else None
                    ),
                )
            elif suffix == ".zip" and self.config.include_zip:
                yield from self._iter_zip(path.resolve(), stop_requested)

    def _iter_zip(
        self,
        path: Path,
        stop_requested: StopRequested | None,
    ) -> Iterator[CorpusEntry]:
        stat = path.stat()
        try:
            with zipfile.ZipFile(path, "r") as archive:
                for info in archive.infolist():
                    if stop_requested and stop_requested():
                        return
                    if info.is_dir() or Path(info.filename).suffix.lower() not in _IMAGE_SUFFIXES:
                        continue

                    source_key = f"zip://{path}#{info.filename}"
                    signature = (
                        f"{stat.st_size}:{stat.st_mtime_ns}:{info.CRC}:"
                        f"{info.file_size}:v{__version__}"
                    )
                    entry = CorpusEntry(
                        source_key,
                        path,
                        info.filename,
                        signature,
                        info.file_size,
                    )

                    if info.file_size > self.config.max_archive_member_bytes:
                        yield CorpusEntry(
                            **{
                                **entry.__dict__,
                                "skip_reason": f"zip_member_too_large:{info.file_size}",
                            }
                        )
                        continue

                    if self.store.source_seen(source_key, signature):
                        # Yield a zero-byte placeholder; mine() will recognize
                        # source_seen before decode and therefore avoid ZIP I/O.
                        yield CorpusEntry(
                            source_key,
                            path,
                            info.filename,
                            signature,
                            info.file_size,
                            b"",
                        )
                        continue

                    try:
                        data = archive.read(info)
                    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                        yield CorpusEntry(
                            source_key,
                            path,
                            info.filename,
                            signature,
                            info.file_size,
                            b"",
                            f"zip_member_read_failed:{type(exc).__name__}",
                        )
                        continue
                    yield CorpusEntry(
                        source_key,
                        path,
                        info.filename,
                        signature,
                        info.file_size,
                        data,
                    )
        except (OSError, zipfile.BadZipFile):
            # A broken ZIP is noisy corpus input, not a fatal mining error.
            return

    def _save_crop(
        self,
        image: Image.Image,
        fingerprint: str,
        pseudo_label: str,
        negative_kind: str | None,
    ) -> Path:
        root = (
            default_learning_root()
            / "crops"
            / pseudo_label
            / (negative_kind or "generic")
        )
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{fingerprint}.webp"
        if not path.exists():
            image.save(path, "WEBP", quality=86, method=4)
        return path

    def _wait_for_gpu_idle(self, stop_requested: StopRequested | None) -> bool:
        while True:
            if stop_requested and stop_requested():
                return False
            utilization = _gpu_utilization_percent()
            if utilization is None or utilization <= self.config.max_gpu_utilization:
                return True
            time.sleep(max(0.5, self.config.idle_poll_seconds))

    @staticmethod
    def _emit(
        progress: ProgressCallback | None,
        counters: dict,
        message: str,
    ) -> None:
        if progress is not None:
            progress(MiningStats(**counters), message)


def _gpu_utilization_percent() -> int | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        try:
            return int(line.strip())
        except ValueError:
            continue
    return None
