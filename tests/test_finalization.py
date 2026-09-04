import json
from pathlib import Path
import tomllib
import cv2
import numpy as np
from movie_broll.finalization import REFRAME_ALGORITHM_VERSION, VERTICAL_VALIDATION_VERSION, _choose_target, _directive, _remove_incomplete_assets, _shot_validation, _vertical_reuse_valid, asset_identity, build_shot_crop_plan, crop_x, letterbox, person_detector_preflight, reframe_fingerprint, render_vertical, safe_cleanup, shot_crop_plan, slugify, thumbnail, unletterbox_bbox, validate_vertical

def event(position='left', people=None, interaction=None):
    return {'visual_event_id':'VE_000123','start_frame':0,'end_frame_exclusive':24,'start_seconds':0.,'end_seconds':1.,'source_shot_ids':['S1','S2'],
      'visual':{'summary_es':'Conversación en terraza','primary_subject_position':position,'actions':['gesticular'],'visible_interactions':interaction or []},
      'people':people or [{'position':position}], 'editorial':{'decision':'KEEP','status':'VALIDATED','standalone_meaning_es':'Conversación en terraza'}}

def synthetic(path:Path):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),24,(160,120))
    for i in range(24):
        frame=np.full((120,160,3),255,dtype='uint8'); cv2.rectangle(frame,(5 if i<12 else 120,25),(35 if i<12 else 150,105),(0,0,255),-1); writer.write(frame)
    writer.release()

def test_stable_registry_slug_and_flat_filenames(tmp_path):
    e=event(); aid,slug=asset_identity(tmp_path,'romper-el-circulo',e)
    assert (aid,slug)==('rc001','conversacion-en-terraza')
    assert asset_identity(tmp_path,'romper-el-circulo',e)==('rc001',slug)
    assert slugify('Mujer: reflexión sola!')=='mujer-reflexion-sola'
    registry=json.loads((tmp_path/'asset_registry.json').read_text()); assert registry['events']['VE_000123']['asset_id']=='rc001'

def test_crop_is_subject_and_shot_aware_not_fixed_center():
    e=event('left'); shots={'S1':{'start_seconds':0,'end_seconds':.5,'primary_subject_position':'left'},'S2':{'start_seconds':.5,'end_seconds':1,'primary_subject_position':'right'}}
    plan=shot_crop_plan(e,shots,160,120)
    assert plan[0]['x']==0 and plan[1]['x']==70 and crop_x(160,120,'center')==35

def test_true_vertical_render_validation_and_distinct_thumbnail_sources(tmp_path):
    horizontal=tmp_path/'h.mp4'; vertical=tmp_path/'v.mp4'; synthetic(horizontal); e=event(); plan=shot_crop_plan(e,{'S1':{'start_seconds':0,'end_seconds':.5,'primary_subject_position':'left'},'S2':{'start_seconds':.5,'end_seconds':1,'primary_subject_position':'right'}},160,120)
    render_vertical(horizontal,vertical,e,plan); result=validate_vertical(vertical,1.,plan,e)
    assert result['status']=='PASS' and (result['width'],result['height'])==(90,120) and not result['black_bars']
    hjpg,vjpg=tmp_path/'h.jpg',tmp_path/'v.jpg'; thumbnail(horizontal,hjpg); thumbnail(vertical,vjpg)
    assert hjpg.exists() and vjpg.exists() and cv2.imread(str(hjpg)).shape[1] != cv2.imread(str(vjpg)).shape[1]

def test_interaction_that_loses_opposite_subject_is_review(tmp_path):
    e=event(people=[{'position':'left'},{'position':'right'}],interaction=['conversation']); p=shot_crop_plan(e,{},160,120)
    assert validate_vertical(tmp_path/'missing.mp4',1,p,e)['status']=='REVIEW'

def test_safe_cleanup_is_owned_and_keeps_no_residue(tmp_path):
    work=tmp_path/'runs'/'film'/'.work'; work.mkdir(parents=True); (work/'old.tmp').write_bytes(b'x'); final=tmp_path/'runs'/'film'/'assets'/'rc001.mp4'; final.parent.mkdir(); final.write_bytes(b'final')
    assert safe_cleanup(work)==1 and not list(work.iterdir()) and final.read_bytes()==b'final'

def _geometry_plan(tmp_path, directive, detections, ids=['S1']):
    video=tmp_path/'h.mp4'; synthetic(video)
    e=event(); e['source_shot_ids']=ids; e['visual']['shot_focus']=directive
    shots={sid:{'shot_id':sid,'start_seconds':i*.5,'end_seconds':(i+1)*.5} for i,sid in enumerate(ids)}
    calls=iter(detections if detections and isinstance(detections[0],list) else [detections]*20)
    return build_shot_crop_plan(video,e,shots,160,120,detector=lambda _:next(calls,[]))

def test_local_geometry_focuses_left_right_and_ots_visible_face(tmp_path):
    left=_geometry_plan(tmp_path,[{'shot_id':'S1','focus_subject':'man','focus_role':'primary','focus_reason':'reaction','preserve_interaction':False,'position':'left'}],[{'bbox':{'x':10,'y':20,'width':28,'height':35},'face_visible':True}])
    assert left[0]['x']==0 and _shot_validation(left[0])['focus_subject_safe']
    right=_geometry_plan(tmp_path,[{'shot_id':'S1','focus_subject':'woman','focus_role':'primary','focus_reason':'reaction','preserve_interaction':False,'position':'right'}],[{'bbox':{'x':115,'y':20,'width':28,'height':35},'face_visible':True}])
    assert right[0]['x']>=65 and _shot_validation(right[0])['focus_subject_safe']
    ots=_geometry_plan(tmp_path,[{'shot_id':'S1','focus_subject':'woman','focus_role':'primary','focus_reason':'visible face','preserve_interaction':False}],[{'bbox':{'x':0,'y':0,'width':55,'height':100},'foreground':True,'face_visible':False},{'bbox':{'x':110,'y':15,'width':25,'height':30},'face_visible':True}])
    assert ots[0]['focus_bbox']['x']==110

def test_reverse_shot_and_interaction_decisions(tmp_path):
    directives=[{'shot_id':'S1','focus_subject':'man','focus_role':'primary','focus_reason':'reaction','preserve_interaction':False,'position':'left'},{'shot_id':'S2','focus_subject':'woman','focus_role':'primary','focus_reason':'reaction','preserve_interaction':False,'position':'right'}]
    plan=_geometry_plan(tmp_path,directives,[[{'bbox':{'x':10,'y':20,'width':25,'height':30},'face_visible':True}]]*5+[[{'bbox':{'x':120,'y':20,'width':25,'height':30},'face_visible':True}]]*5,['S1','S2'])
    assert plan[0]['x'] < plan[1]['x'] and plan[0]['anchors'][-1]['time'] < plan[1]['start_seconds'] + .5
    fit=_geometry_plan(tmp_path,[{'shot_id':'S1','focus_subject':'both','focus_role':'primary','focus_reason':'shared','preserve_interaction':True}],[{'bbox':{'x':35,'y':20,'width':20,'height':30},'face_visible':True},{'bbox':{'x':75,'y':20,'width':20,'height':30},'face_visible':True}])
    assert not fit[0]['review_required']
    focal=_geometry_plan(tmp_path,[{'shot_id':'S1','focus_subject':'man','focus_role':'primary','focus_reason':'reaction','preserve_interaction':False,'focus_position':'left'}],[{'bbox':{'x':0,'y':20,'width':25,'height':30},'face_visible':True},{'bbox':{'x':130,'y':20,'width':25,'height':30},'face_visible':True}])
    assert not focal[0]['review_required']
    required=_geometry_plan(tmp_path,[{'shot_id':'S1','focus_subject':'both','focus_role':'primary','focus_reason':'embrace','preserve_interaction':True}],[{'bbox':{'x':0,'y':20,'width':25,'height':30},'face_visible':True},{'bbox':{'x':130,'y':20,'width':25,'height':30},'face_visible':True}])
    assert required[0]['review_required'] and _shot_validation(required[0])['status']=='FAIL'

def test_track_stability_action_and_source_edge_exception(tmp_path):
    stable=_geometry_plan(tmp_path,[{'shot_id':'S1','focus_subject':'man','focus_role':'primary','focus_reason':'close','preserve_interaction':False}],[[{'bbox':{'x':40+i%2,'y':20,'width':25,'height':30},'face_visible':True}] for i in range(5)])
    assert len(stable[0]['anchors'])==1
    moving=_geometry_plan(tmp_path,[{'shot_id':'S1','focus_subject':'man','focus_role':'primary','focus_reason':'walk','preserve_interaction':False}],[[{'bbox':{'x':10+i*20,'y':20,'width':25,'height':30},'face_visible':True}] for i in range(5)])
    assert len(moving[0]['anchors'])>1 and _shot_validation(moving[0])['crop_stable']
    action=_geometry_plan(tmp_path,[{'shot_id':'S1','focus_subject':'man','focus_role':'primary','focus_reason':'hands','preserve_interaction':False,'required_action_region':[{'x':72,'y':50,'width':10,'height':20}]}],[{'bbox':{'x':25,'y':20,'width':25,'height':30},'face_visible':True}])
    assert _shot_validation(action[0])['action_preserved']
    clipped=stable[0] | {'focus_bbox':{'x':0,'y':20,'width':20,'height':30},'x':20}
    assert _shot_validation(clipped)['source_edge_exception'] and not _shot_validation(clipped)['introduced_subject_clipping']

def test_incomplete_asset_is_not_left_in_asset_hub(tmp_path):
    assets=tmp_path/'assets'; assets.mkdir(); (assets/'rc001-x.mp4').write_bytes(b'x'); (assets/'rc002-y.mp4').write_bytes(b'x')
    for name in ('vrc002-y.mp4','rc002-y.jpg','vrc002-y.jpg','rc002-y.json'): (assets/name).write_bytes(b'x')
    _remove_incomplete_assets(assets)
    assert not (assets/'rc001-x.mp4').exists() and len(list(assets.iterdir()))==5

def test_required_focus_missing_is_never_safe_or_pass():
    value=_shot_validation({'shot_id':'S1','focus_subject':'woman','required_person_focus':True,'directive_available':True,'focus_bbox':None,'subject_bboxes':[],'crop_width':90,'source_width':160,'x':0,'anchors':[],'strategy':'subject_focus','action_preserved':True})
    assert not value['focus_subject_present'] and not value['focus_subject_safe'] and value['status'] == 'FAIL'

def test_reframe_fingerprint_is_vertical_only_and_rejects_old_package(tmp_path):
    e=event(); e['visual']['shot_focus_plan']=[{'shot_id':'S1','focus_subject':'man','focus_reason':'speaker','preserve_secondary_subject':False,'interaction_requires_both':False}]
    shots={'S1':{'start_seconds':0,'end_seconds':1}}; first=reframe_fingerprint(e,shots,160,120)
    assert first == reframe_fingerprint(e,shots,160,120)
    assets=tmp_path/'assets'; assets.mkdir(); base='rc001-x'
    for n in (f'{base}.mp4',f'v{base}.mp4',f'{base}.jpg',f'v{base}.jpg'): (assets/n).write_bytes(b'x')
    (assets/f'{base}.json').write_text(json.dumps({'visual':{'final_vertical':{'reframe_algorithm_version':'old','reframe_fingerprint':'old'}}}))
    assert not _vertical_reuse_valid(assets,base,first)
    (assets/f'{base}.json').write_text(json.dumps({'visual':{'final_vertical':{'reframe_algorithm_version':REFRAME_ALGORITHM_VERSION,'vertical_validation_version':VERTICAL_VALIDATION_VERSION,'reframe_fingerprint':first}}}))
    assert _vertical_reuse_valid(assets,base,first)

def test_sequence_interaction_allows_safe_reverse_shots_and_ots():
    base={'shot_id':'S1','focus_subject':'man','required_person_focus':True,'directive_available':True,'focus_bbox':{'x':20,'y':20,'width':20,'height':30},'subject_bboxes':[],'crop_width':90,'source_width':160,'x':0,'anchors':[],'strategy':'subject_focus','action_preserved':True,'review_required':True}
    man=_shot_validation(base|{'interaction_requirement':'sequence'})
    woman=_shot_validation(base|{'shot_id':'S2','focus_subject':'woman','interaction_requirement':'sequence'})
    assert man['status'] == woman['status'] == 'PASS' and man['interaction_preserved']
    simultaneous=_shot_validation(base|{'interaction_requirement':'simultaneous'})
    assert simultaneous['status'] == 'FAIL' and not simultaneous['interaction_preserved']

def test_legacy_conversation_directive_defaults_to_sequence_not_simultaneous():
    e=event(interaction=['conversation']); e['visual']['actions']=['hablar']; e['visual']['shot_focus']=[{'shot_id':'S1','focus_subject':'man','focus_reason':'habla','preserve_interaction':True}]
    assert _directive(e,{'shot_id':'S1'})['interaction_requirement'] == 'sequence'
    e['visual']['actions']=['abrazo']; assert _directive(e,{'shot_id':'S1'})['interaction_requirement'] == 'simultaneous'

def test_validation_version_invalidates_old_vertical_package(tmp_path):
    e=event(); shots={'S1':{'start_seconds':0,'end_seconds':1}}; fp=reframe_fingerprint(e,shots,160,120); assets=tmp_path/'assets'; assets.mkdir(); base='rc001-x'
    for n in (f'{base}.mp4',f'v{base}.mp4',f'{base}.jpg',f'v{base}.jpg'): (assets/n).write_bytes(b'x')
    (assets/f'{base}.json').write_text(json.dumps({'visual':{'final_vertical':{'reframe_algorithm_version':REFRAME_ALGORITHM_VERSION,'reframe_fingerprint':fp,'vertical_validation_version':'old'}}}))
    assert not _vertical_reuse_valid(assets,base,fp)
    assert VERTICAL_VALIDATION_VERSION != 'old'

def test_detector_preflight_rejects_missing_or_incomplete_model(monkeypatch,tmp_path):
    import movie_broll.finalization as f
    monkeypatch.setattr(f,'_model_path',lambda:tmp_path/'yolo.onnx')
    import pytest
    with pytest.raises(RuntimeError,match='missing or incomplete'): person_detector_preflight(provision=False)
    (tmp_path/'yolo.onnx').write_bytes(b'bad')
    with pytest.raises(RuntimeError,match='missing or incomplete'): person_detector_preflight(provision=False)

def test_detector_preflight_loads_backend_and_reports_truth(monkeypatch,tmp_path):
    import movie_broll.finalization as f
    path=tmp_path/'yolo.onnx'; path.write_bytes(b'x'*2048); monkeypatch.setattr(f,'_model_path',lambda:path)
    calls=[]
    class Net:
        def setInput(self, value): pass
        def forward(self): return np.ones((1,1,6),dtype=np.float32)
    monkeypatch.setattr(f.cv2.dnn,'readNetFromONNX',lambda value:calls.append(value) or Net())
    value=person_detector_preflight(provision=False)
    assert calls == [str(path)] and value['loaded'] and not value['inference_executed'] and value['model_path']==str(path)

def test_yolov5n_provisioning_uses_official_weights_then_atomic_export(monkeypatch,tmp_path):
    import movie_broll.finalization as f
    path=tmp_path/'yolov5n.onnx'; monkeypatch.setattr(f,'_model_path',lambda:path); monkeypatch.setattr(f,'_missing_detector_dependencies',lambda:[]); calls=[]
    class Source:
        sent=False
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self,n=-1):
            if self.sent: return b''
            self.sent=True; return b'w'*2048
    monkeypatch.setattr(f.urllib.request,'urlopen',lambda url,timeout:calls.append(url) or Source())
    def export(weights,target): target.write_bytes(b'x'*2048)
    monkeypatch.setattr(f,'_export_yolov5n',export)
    class Net:
        def setInput(self,x): pass
        def forward(self): return np.ones((1,1,6),dtype=np.float32)
    monkeypatch.setattr(f.cv2.dnn,'readNetFromONNX',lambda _:Net())
    value=f.person_detector_preflight()
    assert calls == [f.PERSON_WEIGHTS_URL] and path.exists() and value['model_id']=='yolov5n'
    assert '.onnx' not in f.PERSON_WEIGHTS_URL

def test_preflight_names_exact_missing_export_dependencies(monkeypatch,tmp_path):
    import movie_broll.finalization as f
    monkeypatch.setattr(f,'_model_path',lambda:tmp_path/'yolov5n.onnx'); monkeypatch.setattr(f,'_missing_detector_dependencies',lambda:['torchvision','Pillow','PyYAML'])
    import pytest
    with pytest.raises(RuntimeError,match='torchvision, Pillow, PyYAML'): f.person_detector_preflight()

def test_export_failure_keeps_useful_stderr_and_cleans_source(monkeypatch,tmp_path):
    import movie_broll.finalization as f
    weights=tmp_path/'yolov5n.pt'; weights.write_bytes(b'x'); target=tmp_path/'yolov5n.onnx'
    class Result:
        returncode=1; stdout=''; stderr="Traceback\nModuleNotFoundError: No module named 'torchvision'"
    monkeypatch.setattr(f.subprocess,'run',lambda *a,**k:Result())
    import pytest
    with pytest.raises(RuntimeError,match="ModuleNotFoundError: No module named 'torchvision'"): f._export_yolov5n(weights,target)
    assert not (tmp_path/'.yolov5-export-source').exists() and not target.exists()

def test_yolov5_decoder_filters_person_class_and_nms(monkeypatch,tmp_path):
    import movie_broll.finalization as f
    path=tmp_path/'yolov5n.onnx'; path.write_bytes(b'x'*2048); monkeypatch.setattr(f,'_model_path',lambda:path)
    class Net:
        def setInput(self,x): pass
        def forward(self):
            out=np.zeros((1,2,85),dtype=np.float32); out[0,0,:6]=[320,320,100,100,.9,.9]; out[0,1,:7]=[320,320,100,100,.9,.1,.9]; return out
    monkeypatch.setattr(f.cv2.dnn,'readNetFromONNX',lambda _:Net())
    boxes=f._yolo_people(np.zeros((640,640,3),dtype=np.uint8))
    assert len(boxes)==1 and boxes[0]['detector']=='yolo_person'

def test_detector_extra_declares_pinned_exporter_imports():
    data=tomllib.loads((Path(__file__).parents[1]/'pyproject.toml').read_text())
    extra=' '.join(data['project']['optional-dependencies']['detector']).lower()
    for name in ('torchvision','pillow','pyyaml','scipy','pandas','seaborn','ipython','onnxscript','setuptools'):
        assert name in extra

def test_yolov5_letterbox_preserves_cinematic_aspect_and_reverses_coordinates():
    image=np.zeros((720,1720,3),dtype=np.uint8); boxed,transform=letterbox(image)
    assert boxed.shape == (640,640,3) and transform['gain'] == 640/1720 and transform['pad_y'] > 180 and transform['pad_x'] == 0
    original=unletterbox_bbox({'x':100,'y':transform['pad_y']+50,'width':200,'height':100},transform)
    assert 0 <= original['x'] < 1720 and 0 <= original['y'] < 720 and original['width'] > 500

def test_focus_core_allows_large_person_body_but_rejects_lost_critical_region():
    base={'shot_id':'S1','focus_subject':'woman','required_person_focus':True,'directive_available':True,'focus_bbox':{'x':100,'y':0,'width':900,'height':700},'crop_width':540,'source_width':1720,'x':300,'anchors':[],'strategy':'subject_focus','action_preserved':True,'interaction_requirement':'sequence'}
    passed=_shot_validation(base)
    assert passed['full_bbox_clipping'] and not passed['critical_focus_clipping'] and passed['status']=='PASS'
    failed=_shot_validation(base|{'x':600})
    assert failed['critical_focus_clipping'] and failed['status']=='FAIL'

def test_action_and_environment_ignore_unrelated_person_bbox_clipping():
    base={'shot_id':'S1','directive_available':True,'focus_bbox':{'x':0,'y':0,'width':900,'height':700},'crop_width':540,'source_width':1720,'x':300,'anchors':[],'strategy':'subject_focus','action_preserved':True,'interaction_requirement':'none'}
    assert _shot_validation(base|{'focus_subject':'action_region'})['status']=='PASS'
    assert _shot_validation(base|{'focus_subject':'environment'})['status']=='PASS'

def test_spatial_focus_association_beats_confidence_and_refuses_ambiguity():
    people=[{'bbox':{'x':10,'y':0,'width':100,'height':200},'confidence':.51},{'bbox':{'x':800,'y':0,'width':100,'height':200},'confidence':.99}]
    assert _choose_target(people,{'focus_position':'left'},1000)['bbox']['x'] == 10
    assert _choose_target(people,{'focus_position':'right'},1000)['bbox']['x'] == 800
    assert _choose_target(people,{'focus_position':'unclear'},1000) is None
    ots=[{'bbox':{'x':0,'y':0,'width':400,'height':600},'confidence':.99,'foreground':True},people[1]]
    assert _choose_target(ots,{'focus_position':'right'},1000)['bbox']['x'] == 800

def test_geometry_uses_source_absolute_timeline_and_stores_relative_render_boundaries(monkeypatch,tmp_path):
    import movie_broll.finalization as f
    calls=[]; frame=np.zeros((720,1720,3),dtype=np.uint8)
    monkeypatch.setattr(f,'_sample_frames',lambda source,start,end,count:(calls.append((source,start,end,count)) or [(start,frame)]))
    event={'start_seconds':2018.208333,'end_seconds':2025.0,'source_shot_ids':['S1'],'visual':{'shot_focus_plan':[{'shot_id':'S1','focus_subject':'woman','focus_position':'right','focus_reason':'speaker','interaction_requirement':'sequence'}]}}
    plan=f.build_shot_crop_plan(tmp_path/'movie.mp4',event,{'S1':{'start_seconds':2024.75,'end_seconds':2025.}},1720,720,detector=lambda _: [{'bbox':{'x':1200,'y':20,'width':100,'height':200},'confidence':.9}],sample_count=3)
    assert calls[0][1:]==(2024.75,2025.,3) and plan[0]['sampling']['timeline_basis']=='source_absolute'
    assert abs(plan[0]['render_start_seconds']-6.541667)<.001 and plan[0]['focus_position']=='right'

def test_render_uses_integer_frame_boundary_on_first_new_shot_frame(tmp_path):
    horizontal=tmp_path/'h.mp4'; vertical=tmp_path/'v.mp4'; synthetic(horizontal)
    e={'start_seconds':100.,'end_seconds':101.}; plan=[
        {'shot_id':'A','start_seconds':100.,'end_seconds':100.5,'render_start_frame':0,'render_end_frame_exclusive':12,'anchors':[{'time':100.,'x':0}],'x':0},
        {'shot_id':'B','start_seconds':100.5,'end_seconds':101.,'render_start_frame':12,'render_end_frame_exclusive':24,'anchors':[{'time':100.5,'x':70}],'x':70}]
    render_vertical(horizontal,vertical,e,plan); cap=cv2.VideoCapture(str(vertical)); cap.set(cv2.CAP_PROP_POS_FRAMES,12); ok,frame=cap.read(); cap.release()
    assert ok and np.mean(frame[:,:,2]) > 20  # frame 12 uses B/right crop, never A's stale crop
