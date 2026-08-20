from __future__ import annotations

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .types import Detection


def normalize_image(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation while preserving alpha when present."""
    return ImageOps.exif_transpose(image)


def expand_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding_px: int = 12,
    padding_ratio: float = 0.12,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    width, height = image_size
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    # Use both a fixed safety margin and a box-relative margin. The original
    # prototype described this as "12px + ratio" but accidentally used max().
    px = max(0, int(padding_px)) + max(0, round(bw * float(padding_ratio)))
    py = max(0, int(padding_px)) + max(0, round(bh * float(padding_ratio)))
    return (
        max(0, int(x0) - px),
        max(0, int(y0) - py),
        min(width, int(x1) + px),
        min(height, int(y1) + py),
    )


def apply_mosaic(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    block_size: int = 16,
    mode: str = "mosaic",
) -> Image.Image:
    """Apply a censor effect through an oval mask inside each detector box."""
    out = image.copy()
    block_size = max(2, int(block_size))

    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(out.width, x1), min(out.height, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        region = out.crop((x0, y0, x1, y1))

        if mode == "black":
            patch = Image.new(region.mode, region.size, _black_for_mode(region.mode))
        elif mode == "blur":
            patch = region.filter(ImageFilter.GaussianBlur(radius=max(2, block_size / 2)))
        else:
            # Block size is interpreted as approximate pixel-block width.
            small_w = max(1, region.width // block_size)
            small_h = max(1, region.height // block_size)
            patch = region.resize((small_w, small_h), Image.Resampling.BILINEAR)
            patch = patch.resize(region.size, Image.Resampling.NEAREST)

        # The detector only supplies a box, not a segmentation outline. An
        # antialiased oval mask follows the target's usual compact shape and
        # avoids censoring all four corners of the expanded rectangle.
        mask = _oval_mask(region.size)
        out.paste(patch, (x0, y0), mask)
    return out


def draw_review_overlay(
    image: Image.Image,
    detections: list[Detection] | tuple[Detection, ...],
    censor_boxes: list[tuple[int, int, int, int]] | tuple[tuple[int, int, int, int], ...],
    no_detection: bool = False,
) -> Image.Image:
    """Create a review copy with detector and applied-censor bounds visible."""

    out = image.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    line_width = max(2, round(max(out.size) / 600))

    for box in censor_boxes:
        draw.ellipse(box, outline=(255, 170, 0, 220), width=line_width)

    for det in detections:
        x0, y0, x1, y1 = det.box
        draw.rectangle(det.box, outline=(20, 220, 120, 240), width=line_width)
        label = f"{det.label} {det.score:.2f}"
        text_box = draw.textbbox((0, 0), label)
        tw = text_box[2] - text_box[0]
        th = text_box[3] - text_box[1]
        ty = max(0, y0 - th - 6)
        draw.rectangle((x0, ty, x0 + tw + 8, ty + th + 6), fill=(0, 0, 0, 190))
        draw.text((x0 + 4, ty + 3), label, fill=(255, 255, 255, 255))

    if no_detection:
        label = "NO DETECTION - REVIEW"
        box = draw.textbbox((0, 0), label)
        tw = box[2] - box[0]
        th = box[3] - box[1]
        draw.rectangle((8, 8, tw + 24, th + 20), fill=(180, 40, 40, 220))
        draw.text((16, 12), label, fill=(255, 255, 255, 255))

    return out


def _oval_mask(size: tuple[int, int], scale: int = 4) -> Image.Image:
    width, height = size
    scale = max(1, int(scale))
    mask = Image.new("L", (max(1, width * scale), max(1, height * scale)), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, mask.width - 1, mask.height - 1), fill=255)
    if scale > 1:
        mask = mask.resize((width, height), Image.Resampling.LANCZOS)
        mask = mask.point(lambda value: 255 if value >= 250 else value)
    return mask


def _black_for_mode(mode: str):
    if mode == "RGBA":
        return (0, 0, 0, 255)
    if mode == "RGB":
        return (0, 0, 0)
    if mode == "L":
        return 0
    if mode == "LA":
        return (0, 255)
    return 0
