# Third-party attribution

Character Mosaic Censor contains original application/integration code and does **not** vendor third-party source trees or detector model weights.

## dghs-imgutils

- Project: `deepghs/imgutils`
- Upstream: https://github.com/deepghs/imgutils
- Purpose: anime/CG censor, person, head, face, eye, pose, and ONNX inference integration
- Version pinned by this project: `0.19.0`
- Upstream license: MIT License

## anime_censor_detection model

- Model repository: `deepghs/anime_censor_detection`
- Upstream: https://huggingface.co/deepghs/anime_censor_detection
- Default model used by this application: `censor_detect_v1.0_s`

## Anime person detection model

- Model repository used by `imgutils.detect.detect_person`: `deepghs/anime_person_detection`
- Upstream: https://huggingface.co/deepghs/anime_person_detection
- Purpose: per-person BBoxes for body-region reasoning
- Default model requested by v1.2: `person_detect_v1.1_m`

## Anime head detection model

- Model repository used by `imgutils.detect.detect_heads`: `deepghs/anime_head_detection`
- Upstream: https://huggingface.co/deepghs/anime_head_detection
- Purpose: head-region evidence and visualization
- Default model requested by the pinned upstream implementation: `head_detect_v2.0_s`

## Anime face detection model

- Model repository used by `imgutils.detect.detect_faces`: `deepghs/anime_face_detection`
- Upstream: https://huggingface.co/deepghs/anime_face_detection
- Purpose: face-region evidence and visualization
- Default model requested by v1.2: `face_detect_v1.4_s`

## Anime eye detection model

- Model repository used by `imgutils.detect.detect_eyes`: `deepghs/anime_eye_detection`
- Upstream: https://huggingface.co/deepghs/anime_eye_detection
- Purpose: strong facial hard-negative confirmation and visualization
- Default model requested by v1.2: `eye_detect_v1.0_s`

The detector models above are downloaded/cached by `dghs-imgutils`; no corresponding model weights are committed to this repository. Redistribution must follow each upstream model repository's applicable terms.

## DWPose model

- Model integration: `imgutils.pose.dwpose_estimate`
- Model repository requested by `dghs-imgutils`: `yzd-v/DWPose`
- Default model file requested by the pinned upstream implementation: `dw-ll_ucoco_384.onnx`
- Purpose: visible skeleton, pelvis protection, and conservative knee/armpit evidence

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
