from PIL import Image

from character_mosaic.image_ops import apply_mosaic, draw_review_overlay, expand_box
from character_mosaic.types import Detection


def test_expand_box_clips_to_bounds():
    assert expand_box((2, 3, 8, 9), (10, 10), padding_px=5, padding_ratio=0) == (0, 0, 10, 10)


def test_expand_box_adds_fixed_and_ratio_padding():
    assert expand_box((100, 100, 200, 200), (400, 400), padding_px=5, padding_ratio=0.2) == (75, 75, 225, 225)


def test_mosaic_changes_target_region_only():
    img = Image.new("RGB", (20, 20), "white")
    for x in range(5, 15):
        for y in range(5, 15):
            img.putpixel((x, y), (x * 10 % 255, y * 10 % 255, 80))
    out = apply_mosaic(img, [(5, 5, 15, 15)], block_size=5, mode="mosaic")
    assert out.getpixel((0, 0)) == img.getpixel((0, 0))
    assert out.crop((5, 5, 15, 15)).tobytes() != img.crop((5, 5, 15, 15)).tobytes()


def test_mosaic_uses_oval_mask_instead_of_filling_box_corners():
    img = Image.new("RGB", (30, 30), "white")
    for x in range(5, 25):
        for y in range(5, 25):
            img.putpixel((x, y), (x * 7 % 255, y * 9 % 255, 80))
    out = apply_mosaic(img, [(5, 5, 25, 25)], block_size=5, mode="mosaic")
    assert out.getpixel((5, 5)) == img.getpixel((5, 5))
    assert out.getpixel((15, 15)) != img.getpixel((15, 15))


def test_black_mode():
    img = Image.new("RGB", (10, 10), "white")
    out = apply_mosaic(img, [(2, 2, 8, 8)], mode="black")
    assert out.getpixel((4, 4)) == (0, 0, 0)


def test_review_overlay_draws_annotations():
    img = Image.new("RGB", (120, 120), "white")
    det = Detection((40, 40, 60, 70), "pussy", 0.2)
    out = draw_review_overlay(img, [det], [(35, 35, 65, 75)])
    assert out.mode == "RGBA"
    assert out.tobytes() != img.convert("RGBA").tobytes()


def test_rgba_mosaic_preserves_semitransparent_alpha():
    image = Image.new("RGBA", (12, 12), (100, 120, 140, 128))
    out = apply_mosaic(image, [(2, 2, 10, 10)], block_size=4, mode="mosaic")
    assert out.getpixel((5, 5))[3] == 128
