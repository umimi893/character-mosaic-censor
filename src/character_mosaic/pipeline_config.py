from __future__ import annotations

from dataclasses import dataclass

from .detector import DetectorConfig

@dataclass
class PipelineConfig:
    language: str = "ja"
    expected_person_count: int = 1
    detection_threshold: float = 0.12
    auto_threshold: float = 0.30
    padding_px: int = 12
    padding_ratio: float = 0.15
    block_size: int = 20
    mode: str = "mosaic"
    recursive: bool = True
    overwrite: bool = False
    tile_large_images: bool = True
    tile_trigger_px: int = 1200
    tile_grid_3_trigger_px: int = 3000
    tile_overlap: float = 0.16
    flip_tta: bool = True
    female_only: bool = True
    model_level: str = "s"
    model_version: str = "v1.0"
    model_iou_threshold: float = 0.70
    merge_iou_threshold: float = 0.45
    merge_ios_threshold: float = 0.70
    review_enabled: bool = True
    copy_low_confidence_to_review: bool = True
    copy_no_detection_to_review: bool = False
    generate_review_html: bool = True
    preview_max_side: int = 1600
    jpeg_quality: int = 95

    def validate(self) -> None:
        if self.language not in {"ja", "en"}:
            raise ValueError("Language must be 'ja' or 'en'.")
        if not 1 <= self.expected_person_count <= 20:
            raise ValueError("画像内の人数は 1〜20 人で指定してください。")
        if not 0.0 < self.detection_threshold < 1.0:
            raise ValueError("Confidence は 0 より大きく 1 未満にしてください。")
        if not self.detection_threshold <= self.auto_threshold < 1.0:
            raise ValueError("Review threshold は Confidence 以上かつ 1 未満にしてください。")
        if self.padding_px < 0:
            raise ValueError("固定余白は 0px 以上にしてください。")
        if not 0.0 <= self.padding_ratio <= 1.0:
            raise ValueError("Padding は 0〜100% の範囲にしてください。")
        if self.block_size < 2:
            raise ValueError("Strength は 2 以上にしてください。")
        if self.mode not in {"mosaic", "blur", "black"}:
            raise ValueError("処理方式が不正です。")
        if self.preview_max_side < 320:
            raise ValueError("Preview最大辺は 320px 以上にしてください。")
        if not 70 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG品質は 70〜100 の範囲にしてください。")
        DetectorConfig(
            detection_threshold=self.detection_threshold,
            model_level=self.model_level,
            model_version=self.model_version,
            iou_threshold=self.model_iou_threshold,
            merge_iou_threshold=self.merge_iou_threshold,
            merge_ios_threshold=self.merge_ios_threshold,
            tile_large_images=self.tile_large_images,
            tile_trigger_px=self.tile_trigger_px,
            tile_grid_3_trigger_px=self.tile_grid_3_trigger_px,
            tile_overlap=self.tile_overlap,
            flip_tta=self.flip_tta,
            female_only=self.female_only,
        ).validate()
