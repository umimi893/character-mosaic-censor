# Third-party attribution

Character Mosaic Censor contains original application/integration code and does **not** vendor third-party source trees or detector model weights.

## dghs-imgutils

- Project: `deepghs/imgutils`
- Upstream: https://github.com/deepghs/imgutils
- Purpose: anime/CG object detection, person detection, DWPose integration, and ONNX inference
- Version pinned by this project: `0.19.0`
- Upstream license: MIT License
- Upstream authors include narugo1992 and 7eu7d7

The upstream project metadata and GitHub repository identify `dghs-imgutils` as MIT-licensed.

## anime_censor_detection model

- Model repository: `deepghs/anime_censor_detection`
- Upstream: https://huggingface.co/deepghs/anime_censor_detection
- Default model used by this application: `censor_detect_v1.0_s`
- Upstream model-repository license: MIT

Model weights are downloaded/cached by the upstream stack and are intentionally excluded from this repository.

## Anime person detection model

- Model repository used by `imgutils.detect.detect_person`: `deepghs/anime_person_detection`
- Upstream: https://huggingface.co/deepghs/anime_person_detection
- Purpose in Character Mosaic Censor: obtain per-person bounding boxes for the optional anatomy sanity check
- Default model requested by v1.1: `person_detect_v1.1_s`

The model is downloaded/cached by `dghs-imgutils`; no person-detection model weights are committed to this repository. Redistribution must follow the upstream model repository's applicable terms.

## DWPose model

- Model integration: `imgutils.pose.dwpose_estimate`
- Model repository requested by `dghs-imgutils`: `yzd-v/DWPose`
- Default model file requested by the pinned upstream implementation: `dw-ll_ucoco_384.onnx`
- Purpose in Character Mosaic Censor: estimate shoulder, hip, and knee keypoints used only as a conservative false-positive sanity check

The DWPose model is downloaded/cached by the upstream stack and is not included in this repository. Redistribution must follow the upstream model repository's applicable terms.

## PySide6 / Qt for Python

- Project: Qt for Python / PySide6
- Upstream documentation: https://doc.qt.io/qtforpython-6/
- Purpose: Windows desktop GUI
- PySide6 is available under LGPLv3/GPLv3 and the Qt commercial license, subject to the specific Qt modules and distribution method used.

This repository does not vendor PySide6 or Qt binaries. Anyone distributing a packaged executable should review the applicable Qt/PySide6 license obligations for that distribution.

## Pillow

- Project: Pillow
- Purpose: image decoding, transforms, compositing, metadata handling, and image output
- Installed as a runtime dependency; not vendored here.

## Reference implementation review

The workflow was compared at a high level with `tnisizawa/anime-mosaic`, another batch-censoring project built around `dghs-imgutils`. No source file from that repository is vendored or copied into Character Mosaic Censor.
