from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time

from PIL import Image

from character_mosaic.detector import AnimeCensorDetector, DetectorConfig, get_runtime_info


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Character Mosaic Censor runtime diagnostic")
    parser.add_argument(
        "--model-test",
        action="store_true",
        help="モデルを実際にロードして1回推論する（初回はモデル取得でネット接続が必要な場合があります）",
    )
    args = parser.parse_args(argv)

    print("Character Mosaic Censor - Diagnostic")
    print(f"Python: {sys.version.split()[0]}")
    versions = {
        "dghs-imgutils": _version("dghs-imgutils"),
        "Pillow": _version("Pillow"),
        "PySide6": _version("PySide6"),
        "onnxruntime": _version("onnxruntime"),
        "onnxruntime-gpu": _version("onnxruntime-gpu"),
    }
    for name, version in versions.items():
        print(f"{name}: {version}")

    missing_required = [name for name in ("dghs-imgutils", "Pillow", "PySide6") if versions[name] == "NOT INSTALLED"]
    if missing_required:
        print("ERROR: 必須パッケージがありません: " + ", ".join(missing_required))

    info = get_runtime_info()
    print(f"Selected provider: {info.selected_provider}")
    print("Available providers: " + (", ".join(info.available_providers) or "none"))
    print(f"GPU: {info.gpu_name or 'not detected by app'}")
    if not info.cuda_available:
        print("WARNING: CUDAExecutionProvider がありません。推論はCPUになります。")
    elif not info.using_cuda:
        print("WARNING: CUDAExecutionProvider は利用可能ですが選択されていません。ONNX_MODE等を確認してください。")

    if args.model_test:
        print("Model test: loading censor_detect_v1.0_s ...")
        started = time.perf_counter()
        try:
            detector = AnimeCensorDetector(
                DetectorConfig(
                    detection_threshold=0.12,
                    tile_large_images=False,
                    flip_tta=False,
                )
            )
            detector.detect(Image.new("RGB", (640, 640), "white"))
        except Exception as exc:
            print(f"Model test FAILED: {type(exc).__name__}: {exc}")
            return 2
        print(f"Model test OK: {time.perf_counter() - started:.2f}s")

    if missing_required or not info.available_providers:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
