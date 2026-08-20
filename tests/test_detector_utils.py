from PIL import Image

from character_mosaic.detector import AnimeCensorDetector, DetectorConfig, _deduplicate, _make_grid_tiles, _make_tiles
from character_mosaic.types import Detection


def test_make_tiles_four():
    image = Image.new("RGB", (2000, 1200), "white")
    tiles = _make_tiles(image, 0.16)
    assert len(tiles) == 4
    assert all(crop.width > 0 and crop.height > 0 for crop, _ in tiles)


def test_make_grid_tiles_nine_for_3x3():
    image = Image.new("RGB", (3600, 3000), "white")
    tiles = _make_grid_tiles(image, 3, 0.16)
    assert len(tiles) == 9
    assert tiles[0][1] == (0, 0)
    assert all(crop.width > 0 and crop.height > 0 for crop, _ in tiles)


def test_tiles_overlap_neighbors():
    image = Image.new("RGB", (1200, 1200), "white")
    tiles = _make_grid_tiles(image, 2, 0.16)
    first, second = tiles[0], tiles[1]
    first_right = first[1][0] + first[0].width
    second_left = second[1][0]
    assert second_left < first_right


def test_deduplicate_prefers_higher_score():
    d1 = Detection((10, 10, 50, 50), "pussy", 0.9)
    d2 = Detection((12, 12, 49, 49), "pussy", 0.4)
    kept = _deduplicate([d2, d1], 0.45)
    assert kept == [d1]


def test_flip_tta_box_is_mapped_back_to_original_coordinates():
    detector = AnimeCensorDetector(DetectorConfig())

    def fake_detect(_image, **_kwargs):
        # Coordinates are measured on the horizontally mirrored 100px image.
        return [((10, 20, 30, 50), "pussy", 0.8)]

    image = Image.new("RGB", (100, 80), "white")
    out = detector._run_one(fake_detect, image, offset=(5, 7), source="test_flip", flipped=True)
    assert out[0].box == (75, 27, 95, 57)
    assert out[0].source == "test_flip"


def test_merge_detections_unions_wider_lower_score_box():
    from character_mosaic.detector import _merge_detections

    high = Detection((20, 20, 40, 40), "pussy", 0.9, "full")
    wide = Detection((10, 15, 50, 45), "pussy", 0.4, "tile")
    merged = _merge_detections([high, wide], iou_threshold=0.2, ios_threshold=0.7)
    assert len(merged) == 1
    assert merged[0].box == (10, 15, 50, 45)
    assert merged[0].score == 0.9
    assert "full" in merged[0].source and "tile" in merged[0].source


def test_nested_boxes_merge_by_ios_even_when_iou_is_low():
    from character_mosaic.detector import _merge_detections

    small = Detection((40, 40, 60, 60), "pussy", 0.8)
    large = Detection((20, 20, 80, 80), "pussy", 0.5)
    merged = _merge_detections([small, large], iou_threshold=0.45, ios_threshold=0.7)
    assert len(merged) == 1
    assert merged[0].box == (20, 20, 80, 80)


def test_runtime_info_distinguishes_cuda_available_from_selected():
    from character_mosaic.detector import RuntimeInfo

    cpu_selected = RuntimeInfo("CPUExecutionProvider", ("CUDAExecutionProvider", "CPUExecutionProvider"))
    assert cpu_selected.cuda_available is True
    assert cpu_selected.using_cuda is False
    gpu_selected = RuntimeInfo("CUDAExecutionProvider", ("CUDAExecutionProvider", "CPUExecutionProvider"))
    assert gpu_selected.using_cuda is True
