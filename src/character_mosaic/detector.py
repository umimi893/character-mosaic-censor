from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable

from PIL import Image, ImageOps

from .types import Detection

DetectorProgress = Callable[[str, tuple[Detection, ...]], None]
StopRequested = Callable[[], bool]


@dataclass
class DetectorConfig:
    detection_threshold: float = 0.12
    model_level: str = "s"
    model_version: str = "v1.0"
    iou_threshold: float = 0.7
    merge_iou_threshold: float = 0.45
    merge_ios_threshold: float = 0.70
    tile_large_images: bool = True
    tile_trigger_px: int = 1200
    tile_grid_3_trigger_px: int = 3000
    tile_overlap: float = 0.16
    flip_tta: bool = True
    female_only: bool = True

    def validate(self) -> None:
        if not 0.0 < self.detection_threshold < 1.0:
            raise ValueError("Confidence は 0 より大きく 1 未満にしてください。")
        if self.model_level not in {"s", "n"}:
            raise ValueError("モデルレベルは s または n です。")
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("Model IoU は 0 より大きく 1 以下にしてください。")
        if not 0.0 < self.merge_iou_threshold <= 1.0:
            raise ValueError("Merge IoU は 0 より大きく 1 以下にしてください。")
        if not 0.0 < self.merge_ios_threshold <= 1.0:
            raise ValueError("Merge IoS は 0 より大きく 1 以下にしてください。")
        if self.tile_trigger_px < 256:
            raise ValueError("2x2タイル開始サイズは 256px 以上にしてください。")
        if self.tile_grid_3_trigger_px < self.tile_trigger_px:
            raise ValueError("3x3タイル開始サイズは2x2開始サイズ以上にしてください。")
        if not 0.0 <= self.tile_overlap <= 0.4:
            raise ValueError("タイルOverlapは 0〜40% の範囲にしてください。")


@dataclass(frozen=True)
class RuntimeInfo:
    selected_provider: str
    available_providers: tuple[str, ...]
    onnxruntime_version: str | None = None
    gpu_name: str | None = None
    python_version: str | None = None

    @property
    def cuda_available(self) -> bool:
        return "CUDAExecutionProvider" in self.available_providers

    @property
    def using_cuda(self) -> bool:
        return self.selected_provider == "CUDAExecutionProvider"

    @property
    def display_text(self) -> str:
        if self.using_cuda:
            return f"CUDA / {self.gpu_name or 'NVIDIA GPU'}"
        if self.available_providers:
            return f"CPU / {self.selected_provider}"
        return "ONNX Runtime 未確認"


class Detector(ABC):
    """Small detector boundary so another YOLO model can be added later."""

    @abstractmethod
    def detect(
        self,
        image: Image.Image,
        progress: DetectorProgress | None = None,
        stop_requested: StopRequested | None = None,
    ) -> list[Detection]:
        raise NotImplementedError


class AnimeCensorDetector(Detector):
    """Adapter around dghs-imgutils' anime censor detector.

    Recall is intentionally favored: a full-frame pass is combined with
    overlapping tiles, then zero-result images get flip/rotation retries.
    Cross-pass detections are *union-merged* instead of simply dropping lower-scoring boxes.
    This keeps a wider alternate box from being lost, which is safer for
    censor coverage.
    """

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.config.validate()

    def detect(
        self,
        image: Image.Image,
        progress: DetectorProgress | None = None,
        stop_requested: StopRequested | None = None,
    ) -> list[Detection]:
        try:
            from imgutils.detect import detect_censors
        except ImportError as exc:
            raise RuntimeError(
                "dghs-imgutils が未インストールです。install.bat または install_gpu.bat を実行してください。"
            ) from exc

        results: list[Detection] = []

        def should_stop() -> bool:
            return bool(stop_requested and stop_requested())

        def run_pass(pass_image: Image.Image, offset: tuple[int, int], source: str) -> bool:
            if should_stop():
                return False
            results.extend(self._run_one(detect_censors, pass_image, offset=offset, source=source, flipped=False))
            self._emit_progress(progress, source, results)
            return not should_stop()

        if not run_pass(image, (0, 0), "full"):
            return self._filtered_merged(results, image.size)

        if self.config.tile_large_images and max(image.size) >= self.config.tile_trigger_px:
            grid = 3 if max(image.size) >= self.config.tile_grid_3_trigger_px else 2
            tiles = _make_grid_tiles(image, grid=grid, overlap=self.config.tile_overlap)
            total = len(tiles)
            for idx, (crop, offset) in enumerate(tiles, start=1):
                if not run_pass(crop, offset, f"tile_{grid}x{grid}_{idx}of{total}"):
                    break

        merged = self._filtered_merged(results, image.size)
        if merged or not self.config.flip_tta or should_stop():
            return merged

        # Orientation fallback is intentionally expensive, so it is only used
        # when the normal full-frame and tiled passes found no target at all.
        for transform in ("hflip", "vflip", "rot90", "rot180", "rot270"):
            if should_stop():
                break
            source = f"retry_{transform}"
            results.extend(
                self._run_one(
                    detect_censors,
                    image,
                    offset=(0, 0),
                    source=source,
                    transform=transform,
                )
            )
            self._emit_progress(progress, source, results)

        return self._filtered_merged(results, image.size)

    def _emit_progress(self, callback: DetectorProgress | None, source: str, results: list[Detection]) -> None:
        if callback is None:
            return
        # We do not know the full image size here, but raw boxes are already
        # valid detector outputs. Final clipping happens after all passes.
        callback(source, tuple(self._filtered_merged(results, None)))

    def _filtered_merged(
        self,
        results: Iterable[Detection],
        image_size: tuple[int, int] | None,
    ) -> list[Detection]:
        labels = {"pussy"} if self.config.female_only else {"pussy", "penis"}
        filtered = [d for d in results if d.label in labels]
        if image_size is not None:
            filtered = [_clip_detection(d, image_size) for d in filtered]
            filtered = [d for d in filtered if d is not None]
        return _merge_detections(
            filtered,
            iou_threshold=self.config.merge_iou_threshold,
            ios_threshold=self.config.merge_ios_threshold,
        )

    def _run_one(
        self,
        fn,
        image: Image.Image,
        offset: tuple[int, int],
        source: str,
        flipped: bool = False,
        transform: str | None = None,
    ) -> list[Detection]:
        transform = transform or ("hflip" if flipped else "identity")
        inference_image = _transform_image(image, transform)
        try:
            raw = fn(
                inference_image,
                level=self.config.model_level,
                version=self.config.model_version,
                conf_threshold=self.config.detection_threshold,
                iou_threshold=self.config.iou_threshold,
            )
        except Exception as exc:
            raise RuntimeError(f"detect_censors 推論に失敗しました ({source}): {exc}") from exc
        ox, oy = offset
        out: list[Detection] = []
        for box, label, score in raw:
            x0, y0, x1, y1 = _map_box_from_transform(
                tuple(int(v) for v in box),
                image.size,
                transform,
            )
            if x1 <= x0 or y1 <= y0:
                continue
            out.append(
                Detection(
                    box=(x0 + ox, y0 + oy, x1 + ox, y1 + oy),
                    label=str(label),
                    score=float(score),
                    source=source,
                )
            )
        return out


def _transform_image(image: Image.Image, transform: str) -> Image.Image:
    if transform == "identity":
        return image
    if transform == "hflip":
        return ImageOps.mirror(image)
    if transform == "vflip":
        return ImageOps.flip(image)
    if transform == "rot90":
        return image.transpose(Image.Transpose.ROTATE_90)
    if transform == "rot180":
        return image.transpose(Image.Transpose.ROTATE_180)
    if transform == "rot270":
        return image.transpose(Image.Transpose.ROTATE_270)
    raise ValueError(f"unknown detector transform: {transform}")


def _map_box_from_transform(
    box: tuple[int, int, int, int],
    original_size: tuple[int, int],
    transform: str,
) -> tuple[int, int, int, int]:
    """Map a box from a transformed image back to the original coordinates."""
    x0, y0, x1, y1 = box
    width, height = original_size
    if transform == "identity":
        return box
    if transform == "hflip":
        return width - x1, y0, width - x0, y1
    if transform == "vflip":
        return x0, height - y1, x1, height - y0
    if transform == "rot90":
        return width - y1, x0, width - y0, x1
    if transform == "rot180":
        return width - x1, height - y1, width - x0, height - y0
    if transform == "rot270":
        return y0, height - x1, y1, height - x0
    raise ValueError(f"unknown detector transform: {transform}")


def get_runtime_info() -> RuntimeInfo:
    """Best-effort ONNX/CUDA diagnostic used by the GUI."""
    import platform

    available: tuple[str, ...] = tuple()
    selected = "CPUExecutionProvider"
    ort_version: str | None = None
    try:
        import onnxruntime as ort

        available = tuple(ort.get_available_providers())
        ort_version = getattr(ort, "__version__", None)
    except Exception:
        pass

    try:
        from imgutils.utils.onnxruntime import get_onnx_provider

        selected = str(get_onnx_provider())
    except Exception:
        if "CUDAExecutionProvider" in available:
            selected = "CUDAExecutionProvider"
        elif available:
            selected = available[0]

    gpu_name = _query_nvidia_gpu_name() if "CUDAExecutionProvider" in available else None
    return RuntimeInfo(
        selected_provider=selected,
        available_providers=available,
        onnxruntime_version=ort_version,
        gpu_name=gpu_name,
        python_version=platform.python_version(),
    )


def _query_nvidia_gpu_name() -> str | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return names[0] if names else None


def _make_tiles(image: Image.Image, overlap: float) -> list[tuple[Image.Image, tuple[int, int]]]:
    return _make_grid_tiles(image, grid=2, overlap=overlap)


def _make_grid_tiles(
    image: Image.Image,
    grid: int,
    overlap: float,
) -> list[tuple[Image.Image, tuple[int, int]]]:
    """Create an overlapping 2x2 or 3x3 grid in deterministic order."""
    if grid not in {2, 3}:
        raise ValueError("grid must be 2 or 3")
    w, h = image.size
    overlap = max(0.0, min(0.4, float(overlap)))
    x_edges = [round(i * w / grid) for i in range(grid + 1)]
    y_edges = [round(i * h / grid) for i in range(grid + 1)]
    tiles: list[tuple[Image.Image, tuple[int, int]]] = []

    for row in range(grid):
        for col in range(grid):
            base_x0, base_x1 = x_edges[col], x_edges[col + 1]
            base_y0, base_y1 = y_edges[row], y_edges[row + 1]
            pad_x = round((base_x1 - base_x0) * overlap)
            pad_y = round((base_y1 - base_y0) * overlap)
            x0 = max(0, base_x0 - (pad_x if col > 0 else 0))
            x1 = min(w, base_x1 + (pad_x if col < grid - 1 else 0))
            y0 = max(0, base_y0 - (pad_y if row > 0 else 0))
            y1 = min(h, base_y1 + (pad_y if row < grid - 1 else 0))
            tiles.append((image.crop((x0, y0, x1, y1)), (x0, y0)))
    return tiles


def _intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int]:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    return iw * ih, max(1, (ax1 - ax0) * (ay1 - ay0)), max(1, (bx1 - bx0) * (by1 - by0))


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    inter, area_a, area_b = _intersection(a, b)
    if inter <= 0:
        return 0.0
    return inter / float(area_a + area_b - inter)


def _ios(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over the smaller box, useful for nested TTA boxes."""
    inter, area_a, area_b = _intersection(a, b)
    if inter <= 0:
        return 0.0
    return inter / float(min(area_a, area_b))


def _overlaps(a: Detection, b: Detection, iou_threshold: float, ios_threshold: float) -> bool:
    if a.label != b.label:
        return False
    return _iou(a.box, b.box) >= iou_threshold or _ios(a.box, b.box) >= ios_threshold


def _union_detection(a: Detection, b: Detection) -> Detection:
    ax0, ay0, ax1, ay1 = a.box
    bx0, by0, bx1, by1 = b.box
    sources = []
    for source in (a.source, b.source):
        for part in source.split("+"):
            if part and part not in sources:
                sources.append(part)
    return Detection(
        box=(min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)),
        label=a.label,
        score=max(a.score, b.score),
        source="+".join(sources),
    )


def _merge_detections(
    items: Iterable[Detection],
    iou_threshold: float,
    ios_threshold: float = 0.70,
) -> list[Detection]:
    """Union overlapping cross-pass boxes while preserving max confidence.

    Merging is repeated until stable because unioning two boxes can make the
    merged region overlap a third TTA/tile candidate. The result is sorted by
    score for deterministic previews/logs.
    """
    pending = sorted(items, key=lambda d: d.score, reverse=True)
    merged: list[Detection] = []
    for item in pending:
        current = item
        changed = True
        while changed:
            changed = False
            next_merged: list[Detection] = []
            for old in merged:
                if _overlaps(current, old, iou_threshold, ios_threshold):
                    current = _union_detection(current, old)
                    changed = True
                else:
                    next_merged.append(old)
            merged = next_merged
        merged.append(current)
    return sorted(merged, key=lambda d: d.score, reverse=True)


def _deduplicate(items: Iterable[Detection], iou_threshold: float) -> list[Detection]:
    """Compatibility name retained for callers/tests; now safety-union merges."""
    return _merge_detections(items, iou_threshold=iou_threshold)


def _clip_detection(detection: Detection, image_size: tuple[int, int]) -> Detection | None:
    w, h = image_size
    x0, y0, x1, y1 = detection.box
    box = (max(0, x0), max(0, y0), min(w, x1), min(h, y1))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    if box == detection.box:
        return detection
    return Detection(box=box, label=detection.label, score=detection.score, source=detection.source)
