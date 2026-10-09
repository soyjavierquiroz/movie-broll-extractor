import numpy as np
import cv2
import json

from movie_broll import broll_pilot, production
from movie_broll.intrusive_text import inspect_event

from movie_broll.active_picture import crop_dimensions, detect_frames, full_frame, source_x
from movie_broll.finalization import reframe_fingerprint, render_vertical


def _frame(top=0, bottom=0):
    image=np.full((100, 160, 3), 120, dtype=np.uint8)
    if top: image[:top]=0
    if bottom: image[-bottom:]=0
    return image


def test_persistent_letterbox_uses_active_picture():
    active=detect_frames([_frame(10,10) for _ in range(8)], minimum_bar_pixels=8, edge_coverage_required=.75)
    assert (active['x'],active['y'],active['width'],active['height']) == (0,10,160,80)
    assert crop_dimensions(active) == (60,80)


def test_temporary_dark_frame_is_not_a_structural_bar():
    active=detect_frames([_frame(20,20)] + [_frame() for _ in range(7)], minimum_bar_pixels=8, edge_coverage_required=.75)
    assert not active['structural_bars']
    assert (active['x'],active['y'],active['width'],active['height']) == (0,0,160,100)


def test_no_bars_retains_frame_and_coordinate_origin_is_explicit():
    active=detect_frames([_frame() for _ in range(4)], minimum_bar_pixels=8)
    assert (active['width'],active['height']) == (160,100)
    assert source_x({'x':12}, 30) == 42


def test_active_picture_changes_vertical_only_fingerprint():
    event={'source_shot_ids':['S1'],'visual':{}}
    shots={'S1':{'start_seconds':0,'end_seconds':1}}
    a={'x':0,'y':0,'width':160,'height':100,'detection_profile':'v','structural_bars':False}
    b={'x':0,'y':10,'width':160,'height':80,'detection_profile':'v','structural_bars':True}
    assert reframe_fingerprint(event,shots,160,100,active_picture=a) != reframe_fingerprint(event,shots,160,80,active_picture=b)


def test_vertical_render_uses_active_y_origin(tmp_path):
    source=tmp_path/'source.mp4'; output=tmp_path/'vertical.mp4'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),24,(160,100))
    frame=_frame(10,10); frame[10:90]=[20,180,40]
    for _ in range(3): writer.write(frame)
    writer.release()
    event={'start_seconds':0,'end_seconds':.125,'start_frame':0,'end_frame_exclusive':3}
    plan=[{'shot_id':'S','start_seconds':0,'end_seconds':.125,'render_start_frame':0,'render_end_frame_exclusive':3,'x':50,'anchors':[{'time':0,'x':50}]}]
    render_vertical(source,output,event,plan,{'x':0,'y':10,'width':160,'height':80})
    cap=cv2.VideoCapture(str(output)); ok,image=cap.read(); cap.release()
    assert ok and image.shape[:2] == (80,60)
    assert image[0].mean() > 30  # A container-origin crop would be black here.


def test_visual_signals_use_active_pixels_for_all_statistics(monkeypatch, tmp_path):
    """Letterbox bars cannot alter luma, dark fraction, sharpness, motion, or HSV."""
    active={'x':0,'y':60,'width':1920,'height':960,'source_width':1920,
            'source_height':1080,'structural_bars':True,'detection_profile':'test'}
    content=np.full((960,1920,3), (30,130,210), dtype=np.uint8)
    content[:,::19]=(0,255,0)  # non-flat content also exercises sharpness/HSV
    frame=np.zeros((1080,1920,3),dtype=np.uint8); frame[60:1020]=content
    class Capture:
        payload=frame
        def __init__(self,*_): self.released=False
        def get(self,_): return 10
        def set(self,*_): pass
        def read(self): return True,self.payload.copy()
        def release(self): self.released=True
    monkeypatch.setattr(cv2,'VideoCapture',Capture)
    shot={'shot_id':'S','start_seconds':0,'end_seconds':1}
    cropped=broll_pilot.visual_signals(tmp_path/'source.mp4',[shot],sample_fps=1,active_picture=active)[0]
    Capture.payload=content
    direct=broll_pilot.visual_signals(tmp_path/'source.mp4',[shot],sample_fps=1,
                                      active_picture={'x':0,'y':0,'width':1920,'height':960,
                                                      'source_width':1920,'source_height':960,
                                                      'structural_bars':False,'detection_profile':'test'})[0]
    for field in ('brightness_mean','brightness_std','sharpness_score','motion_score','near_black_fraction'):
        assert cropped[field] == direct[field]
    assert np.allclose(cropped['_hist'],direct['_hist'])
    assert cropped['near_black_fraction'] == 0


def test_intrusive_text_normalizes_regions_in_active_picture_coordinates(monkeypatch, tmp_path):
    import movie_broll.intrusive_text as text
    active={'x':0,'y':10,'width':160,'height':80,'source_width':160,'source_height':100,
            'structural_bars':True,'detection_profile':'test'}
    source=np.zeros((100,160,3),dtype=np.uint8)
    source[10:90]=80
    class Capture:
        def __init__(self,*_): pass
        def set(self,*_): pass
        def read(self): return True,source.copy()
        def release(self): pass
    captured=[]
    monkeypatch.setattr(cv2,'VideoCapture',Capture)
    monkeypatch.setattr(text,'evaluate_frames',lambda frames,**_: captured.extend(frames) or {'decision':'PASS'})
    inspect_event(tmp_path/'source.mp4',{'start_seconds':0,'end_seconds':1},sample_count=2,active_picture=active)
    assert [x.shape[:2] for x in captured] == [(80,160),(80,160)]


def test_active_picture_changes_cache_fingerprint_and_invalidates_old_cache(monkeypatch, tmp_path):
    movie=tmp_path/'movie.mp4'; movie.write_bytes(b'movie'); run=tmp_path/'run'; run.mkdir()
    units=[{'analysis_unit_id':'S__VAU_001','parent_shot_id':'S','shot_id':'S','start_frame':0,
            'end_frame_exclusive':24,'start_seconds':0.,'end_seconds':1.,'duration_seconds':1.,
            'derived_from_long_shot':False}]
    active={'x':0,'y':60,'width':1920,'height':960,'source_width':1920,'source_height':1080,
            'structural_bars':True,'detection_profile':'active_picture_v1'}
    info={'movie':movie,'run':run,'movie_sha256':'a'*64,'active_picture':active}
    (run/'visual_signals_v2.json').write_text(json.dumps({'fingerprint':'old-full-frame','signals':{'S__VAU_001':{}}}))
    calls=[]
    def signals(_movie, missing, **kwargs):
        calls.append(kwargs['active_picture'])
        for unit in missing:
            kwargs['on_complete'](unit,{'status':'COMPLETE','brightness_mean':1,'brightness_std':0,
                                        'sharpness_score':1,'motion_score':0,'near_black_fraction':0,
                                        '_hist':np.zeros((16,16),dtype=np.float32)})
    monkeypatch.setattr(broll_pilot,'visual_signals',signals)
    production._cached_visual_signals(info,units,lambda _:None)
    cache=json.loads((run/'visual_signals_v2.json').read_text())
    assert calls == [active] and cache['algorithm_version'] == 'single_pass_active_picture_v3'
    assert cache['schema_version'] == 'visual_signals_v3'
    production._cached_visual_signals(info,units,lambda _:None)
    assert len(calls) == 1  # identical active geometry reuses cache
    changed={**info,'active_picture':{**active,'y':0,'height':1080,'structural_bars':False}}
    production._cached_visual_signals(changed,units,lambda _:None)
    assert len(calls) == 2


def test_vertical_detector_and_renderer_share_active_picture_coordinates(tmp_path):
    source=tmp_path/'source.mp4'; output=tmp_path/'vertical.mp4'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),24,(160,100))
    frame=_frame(10,10); frame[10:90,110:150]=[20,180,40]
    for _ in range(3): writer.write(frame)
    writer.release()
    active={'x':0,'y':10,'width':160,'height':80,'source_width':160,'source_height':100,
            'structural_bars':True,'detection_profile':'test'}
    seen=[]
    event={'start_seconds':0,'end_seconds':.125,'start_frame':0,'end_frame_exclusive':3,
           'source_shot_ids':['S'],'visual':{'shot_focus_plan':[{'shot_id':'S','focus_subject':'environment'}]}}
    plan=__import__('movie_broll.finalization',fromlist=['build_shot_crop_plan']).build_shot_crop_plan(
        source,event,{'S':{'start_seconds':0,'end_seconds':.125,'start_frame':0,'end_frame_exclusive':3}},160,80,
        detector=lambda image: seen.append(image.shape[:2]) or [],active_picture=active)
    render_vertical(source,output,event,plan,active)
    assert seen and all(shape == (80,160) for shape in seen)
    cap=cv2.VideoCapture(str(output)); ok,image=cap.read(); cap.release()
    assert ok and image.shape[:2] == (80,60) and image.mean() > 20


def test_gemini_candidate_sheet_uses_active_picture_pixels(monkeypatch, tmp_path):
    import movie_broll.finalization as finalization
    source=tmp_path/'source.mp4'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),24,(160,100))
    frame=_frame(10,10); frame[10:90]=[20,180,40]
    for _ in range(2): writer.write(frame)
    writer.release()
    monkeypatch.setattr(finalization,'person_detector_preflight',lambda: None)
    seen=[]
    monkeypatch.setattr(finalization,'detect_people',lambda image: seen.append(image.shape[:2]) or [])
    active={'x':0,'y':10,'width':160,'height':80,'source_width':160,'source_height':100,
            'structural_bars':True,'detection_profile':'test'}
    sheet=broll_pilot.candidate_contact_sheet(source,{'start_frame':0,'end_frame_exclusive':2},24,
                                              active_picture=active)
    image=cv2.imdecode(np.frombuffer(sheet,dtype=np.uint8),cv2.IMREAD_COLOR)
    assert seen == [(80,160)] * 5 and image.shape[:2] == (480,1920)
