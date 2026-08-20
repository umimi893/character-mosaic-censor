from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from character_mosaic.anatomy_filter import AnatomyFilterConfig, _assess_candidate, apply_anatomy_filter
from character_mosaic.types import Detection


@dataclass
class FakePose:
    body: list[list[float]]


def _pose(*, missing_hips: bool = False, offset_x: float = 0.0) -> FakePose:
    body = [[-1.0, -1.0, 0.0] for _ in range(18)]
    coords = {0:(100,55),1:(100,100),2:(70,80),3:(75,130),4:(80,180),5:(130,80),6:(125,130),7:(120,180),8:(80,220),9:(85,320),10:(85,390),11:(120,220),12:(115,320),13:(115,390),14:(90,50),15:(110,50),16:(82,55),17:(118,55)}
    for index,(x,y) in coords.items(): body[index]=[x+offset_x,y,0.95]
    if missing_hips: body[11][2]=0.1
    return FakePose(body)


def _person_detector(_image, **_kwargs): return [((0,0,200,440),"person",0.99)]
def _pose_estimator(_image, **_kwargs): return [_pose()]
def _head_detector(_image, **_kwargs): return [((50,20,150,130),"head",0.95)]
def _face_detector(_image, **_kwargs): return [((65,35,135,120),"face",0.95)]
def _eye_detector(_image, **_kwargs): return [((80,45,100,65),"eye",0.95),((105,45,125,65),"eye",0.95)]


def _run(detection: Detection):
    return apply_anatomy_filter(Image.new("RGB",(400,450),"white"),[detection],person_detector=_person_detector,pose_estimator=_pose_estimator,head_detector=_head_detector,face_detector=_face_detector,eye_detector=_eye_detector)


def test_pelvis_candidate_is_kept():
    result=_run(Detection((90,215,110,245),"pussy",0.55)); assert result.evidence[0].decision=="keep"; assert result.suppressed==tuple(); assert any(s.startswith("near_pelvis:") for s in result.evidence[0].positive_signals)


def test_knee_candidate_is_suppressed():
    result=_run(Detection((78,310,95,330),"pussy",0.55)); assert result.evidence[0].decision=="suppress"; assert result.suppressed[0].reason=="near_right_knee"


def test_eye_candidate_needs_head_and_face_confirmation_then_suppresses():
    result=_run(Detection((82,47,98,63),"pussy",0.55)); assert result.evidence[0].decision=="suppress"; assert result.suppressed[0].reason=="inside_eye_face_head"


def test_face_candidate_is_review_not_suppressed_for_oral_safety():
    result=_run(Detection((90,90,112,115),"pussy",0.55)); assert result.evidence[0].decision=="review"; assert result.requires_review is True; assert result.suppressed==tuple()


def test_face_overlap_does_not_override_other_person_pelvis():
    image=Image.new("RGB",(500,450),"white"); detection=Detection((290,215,310,245),"pussy",0.65)
    def persons(_image,**_kwargs): return [((220,130,380,320),"person",0.95),((200,0,400,440),"person",0.94)]
    def poses(_image,**_kwargs): return [_pose(offset_x=200),_pose(offset_x=200)]
    def heads(_image,**_kwargs): return [((260,190,340,270),"head",0.95)]
    def faces(_image,**_kwargs): return [((270,200,330,260),"face",0.95)]
    result=apply_anatomy_filter(image,[detection],person_detector=persons,pose_estimator=poses,head_detector=heads,face_detector=faces,eye_detector=lambda *_a,**_k:[])
    assert result.evidence[0].decision=="keep"; assert any(s.startswith("near_pelvis:") for s in result.evidence[0].positive_signals)


def test_missing_hip_fails_open():
    detection=Detection((78,310,95,330),"pussy",0.55); assessment=_assess_candidate(detection,_pose(missing_hips=True),AnatomyFilterConfig()); assert assessment.usable is False; assert assessment.reason is None


def test_body_map_contains_person_head_face_eye_and_pose():
    result=_run(Detection((90,215,110,245),"pussy",0.55)); kinds={region.kind for region in result.body_regions}; assert {"person","head","face","eye","pelvis_safe"}<=kinds; assert len(result.pose_points)>=10; assert result.pose_edges


def test_failed_auxiliary_detector_is_partial_but_keeps_candidate():
    detection=Detection((180,150,195,170),"pussy",0.55)
    def broken_head(_image,**_kwargs): raise RuntimeError("head unavailable")
    result=apply_anatomy_filter(Image.new("RGB",(400,450),"white"),[detection],person_detector=_person_detector,pose_estimator=_pose_estimator,head_detector=broken_head,face_detector=lambda *_a,**_k:[],eye_detector=lambda *_a,**_k:[])
    assert result.status.startswith("partial:"); assert "head" in result.failed_components; assert result.evidence[0].decision=="keep"
