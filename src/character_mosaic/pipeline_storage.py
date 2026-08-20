from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image, PngImagePlugin

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

def _iter_images(root: Path, recursive: bool, excluded_roots: set[Path] | None = None):
    excluded_roots = {p.resolve() for p in (excluded_roots or set())}
    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in sorted(iterator):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        resolved = path.resolve()
        if any(resolved == ex or ex in resolved.parents for ex in excluded_roots):
            continue
        yield path
def _make_preview_image(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) <= max_side:
        return image.copy()
    preview = image.copy()
    preview.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return preview
def _save_image_atomic(image: Image.Image, output: Path, original_suffix: str, jpeg_quality: int = 95) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower() or original_suffix or ".png"
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=suffix, dir=output.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        _save_image(image, temp, original_suffix, jpeg_quality=jpeg_quality)
        os.replace(temp, output)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
def _copy_file_atomic(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=output.suffix, dir=output.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        shutil.copy2(source, temp)
        os.replace(temp, output)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
def _assert_directory_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        fd, temp_name = tempfile.mkstemp(prefix=".cmc_write_test_", dir=path)
        os.close(fd)
        Path(temp_name).unlink(missing_ok=True)
    except OSError as exc:
        raise PermissionError(f"書き込みできないフォルダです: {path} ({exc})") from exc
def _save_image(image: Image.Image, output: Path, original_suffix: str, jpeg_quality: int = 95) -> None:
    suffix = output.suffix.lower() or original_suffix
    save_kwargs = _metadata_save_kwargs(image, suffix)
    if suffix in {".jpg", ".jpeg"}:
        if image.mode in {"RGBA", "LA"}:
            bg = Image.new("RGB", image.size, "white")
            alpha = image.getchannel("A")
            bg.paste(image.convert("RGB"), mask=alpha)
            # Conversion creates a new image, so metadata must remain in the
            # explicit save kwargs captured above.
            image = bg
        elif image.mode != "RGB":
            image = image.convert("RGB")
        image.save(output, quality=jpeg_quality, subsampling=0, **save_kwargs)
    elif suffix == ".webp":
        image.save(output, quality=jpeg_quality, method=6, **save_kwargs)
    else:
        image.save(output, **save_kwargs)
def _metadata_save_kwargs(image: Image.Image, suffix: str) -> dict:
    """Preserve useful source metadata without re-applying EXIF rotation.

    AI-generated PNGs commonly store generation parameters in text chunks.
    Re-encoding a censored image should not silently discard those fields.
    ICC/DPI and EXIF/XMP are also retained when Pillow supports them. The EXIF
    orientation tag is removed because ``normalize_image`` already applied it.
    """
    info = dict(getattr(image, "info", {}) or {})
    kwargs: dict = {}

    icc = info.get("icc_profile")
    if isinstance(icc, (bytes, bytearray)) and icc:
        kwargs["icc_profile"] = bytes(icc)

    dpi = info.get("dpi")
    if isinstance(dpi, (tuple, list)) and len(dpi) >= 2:
        try:
            kwargs["dpi"] = (float(dpi[0]), float(dpi[1]))
        except (TypeError, ValueError):
            pass

    try:
        exif = image.getexif()
        if exif:
            # 274 = Orientation. Pixel data is already exif_transpose()'d.
            exif.pop(274, None)
            exif_bytes = exif.tobytes()
            if exif_bytes:
                kwargs["exif"] = exif_bytes
    except Exception:
        pass

    if suffix == ".webp":
        xmp = info.get("xmp")
        if isinstance(xmp, (bytes, bytearray)) and xmp:
            kwargs["xmp"] = bytes(xmp)

    if suffix == ".png":
        pnginfo = PngImagePlugin.PngInfo()
        text_count = 0
        # Pillow exposes normal tEXt/zTXt/iTXt chunks as strings in info.
        # Preserve them, including common Stable Diffusion ``parameters``.
        reserved = {"icc_profile", "dpi", "exif", "transparency", "gamma"}
        for key, value in info.items():
            if key in reserved or not isinstance(key, str) or not isinstance(value, str):
                continue
            pnginfo.add_text(key, value)
            text_count += 1
        if text_count:
            kwargs["pnginfo"] = pnginfo

    return kwargs
