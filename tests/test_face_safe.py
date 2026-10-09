import copy
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from movie_broll import face_safe as fs, finalization as f, vertical_repair as vr
from movie_broll.utils import sha256_file, write_json


def face(x=100, y=50, width=80, height=100):
    return dict(x=x, y=y, width=width, height=height, kind='face', detector='yunet', confidence=.95)


def test_primary_face_clipped_source_has_room():
    result = fs.assess(face(), 400, 300, 120, 200)
    assert result['decision'] == 'REPAIR'
    assert result['repairable'] and result['source_allows_better_crop'] == 'MATCH'
    assert 'face_partially_clipped_but_source_has_room' in result['reason_codes']


def test_primary_face_safe():
    result = fs.assess(face(), 400, 300, 40, 200)
    assert result['decision'] == 'PASS'
    assert result['face_fully_inside_frame'] == 'MATCH'
    assert result['both_eyes_visible'] == 'UNKNOWN'


@pytest.mark.parametrize('region', [face(x=0), face(y=0), face(width=250)])
def test_source_limits_do_not_retry(region):
    result = fs.assess(region, 400, 300, 120, 200)
    assert result['decision'] == 'SOURCE_LIMIT'
    assert not result['repairable']


@pytest.mark.parametrize('region', [None, fs.head_proxy(face())])
def test_unavailable_proxy_is_unknown_not_pass(region):
    result = fs.assess(region, 400, 300, 40, 200)
    assert result['decision'] == 'AMBIGUOUS'
    assert result['face_detected'] == result['both_eyes_visible'] == 'UNKNOWN'


def test_wrong_face_cannot_satisfy_binding():
    target = dict(x=200, y=20, width=100, height=250)
    assert fs.bind_face([face(x=50)], target) is None
    assert fs.assess(face(), 400, 300, 40, 200, binding=False)['decision'] == 'AMBIGUOUS'
    # Two overlapping person boxes cannot both claim the same face.
    assert fs.bind_face([face(x=210, width=60)], target, [dict(x=190, y=20, width=120, height=250)]) is None


def test_eye_retention_only_with_reliable_source_landmarks():
    region = face()
    region['landmarks'] = {'left_eye': [160,80], 'right_eye': [115,80]}
    result = fs.assess(region, 400, 300, 120, 200)
    assert result['both_eyes_visible'] == 'MISMATCH'
    assert 'both_eyes_not_visible_when_source_allows' in result['reason_codes']
    region['landmarks']['left_eye'] = [116,80]  # profile/inconsistent evidence
    assert fs.assess(region, 400, 300, 40, 200)['both_eyes_visible'] == 'UNKNOWN'


def test_one_reliable_bad_frame_cannot_be_averaged_away():
    good = fs.assess(face(), 400, 300, 40, 200)
    bad = fs.assess(face(), 400, 300, 120, 200)
    assert fs.aggregate([good]*4+[bad])['decision'] == 'REPAIR'


def write_video(path, width=400, height=320):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 24, (width,height))
    rng = np.random.default_rng(42)
    image = rng.integers(30,230,(height,width,3),dtype=np.uint8)
    for _ in range(24):
        writer.write(image)
    writer.release()


def test_face_priority_keeps_semantic_target_and_uses_face_center(tmp_path, monkeypatch):
    video = tmp_path/'movie.mp4'; write_video(video)
    event = dict(start_seconds=0., end_seconds=1., start_frame=0, end_frame_exclusive=24,
                 source_shot_ids=['S1'], visual={'shot_focus_plan':[dict(shot_id='S1',focus_subject='woman',target_person_ids=['P2'], target_binding_confidence='high')]})
    people = [dict(detector='yolo_person', confidence=.9, bbox=dict(x=0,y=0,width=90,height=290)),
              dict(detector='yolo_person', confidence=.9, bbox=dict(x=100,y=20,width=290,height=280))]
    monkeypatch.setattr(fs,'detect_faces',lambda frame:[face(x=120)])
    normal = f.build_shot_crop_plan(video,event,{},400,300,detector=lambda _:people)
    priority = f.build_shot_crop_plan(video,event,{},400,300,detector=lambda _:people,strategy='face_priority')
    assert normal[0]['target_person_ids'] == priority[0]['target_person_ids'] == ['P2']
    assert priority[0]['target_binding_resolved'] is True
    assert priority[0]['x'] < normal[0]['x']-50
    assert priority[0]['face_safe_samples'][0]['region']['detector'] == 'yunet'


def test_face_head_interval_projects_person_anchor_at_both_edges():
    # The old person-centre anchors cut the face-side margin.  The face/head
    # interval moves them only as far as the minimum safe correction requires.
    crop = 225
    right = face(x=350, width=40)
    right_interval = fs.crop_interval(right, 500, crop)
    old_right = f._anchor(dict(x=150, y=0, width=250, height=280), 500, crop)
    new_right = max(right_interval['min_x'], min(right_interval['max_x'], old_right))
    assert old_right < right_interval['min_x']
    assert new_right == right_interval['min_x']
    assert new_right + crop >= right['x'] + right['width'] + right_interval['margin']

    left = face(x=110, width=40)
    left_interval = fs.crop_interval(left, 500, crop)
    old_left = f._anchor(dict(x=100, y=0, width=250, height=280), 500, crop)
    new_left = max(left_interval['min_x'], min(left_interval['max_x'], old_left))
    assert old_left > left_interval['max_x']
    assert new_left == left_interval['max_x']
    assert new_left <= left['x'] - left_interval['margin']


def test_face_head_constraint_smoothing_never_leaves_each_sample_interval():
    bounds = [
        {'min_x': 20., 'max_x': 25.},
        {'min_x': 55., 'max_x': 60.},
        {'min_x': 35., 'max_x': 40.},
    ]
    anchors = f._smooth([(0., 0.), (1., 80.), (2., 0.)], 90, constraints=bounds)
    assert len(anchors) == len(bounds)
    assert all(row['min_x'] <= row['x'] <= row['max_x'] for row in anchors)


def test_missing_face_uses_bound_target_head_proxy_as_planning_constraint(tmp_path, monkeypatch):
    samples = [(time, np.zeros((120, 160, 3), dtype=np.uint8)) for time in (.12, .31, .50, .69, .88)]
    monkeypatch.setattr(f, '_sample_frames', lambda *args, **kwargs: samples)
    monkeypatch.setattr(fs, 'detect_faces', lambda _: [])
    event = {
        'visual_event_id': 'VE_PROXY', 'start_seconds': 0., 'end_seconds': 1.,
        'start_frame': 0, 'end_frame_exclusive': 24, 'source_shot_ids': ['S1'],
        'visual': {'shot_focus_plan': [{'shot_id': 'S1', 'focus_subject': 'man',
            'focus_role': 'primary', 'target_person_ids': ['P1'], 'target_binding_confidence': 'high'}]},
    }
    person = {'detector': 'yolo_person', 'confidence': .9, 'bbox': {'x': 120, 'y': 5, 'width': 30, 'height': 100}}
    rule = f.build_shot_crop_plan(tmp_path/'unused.mp4', event, {'S1': {'start_seconds': 0., 'end_seconds': 1.}}, 160, 120,
                                  detector=lambda _: [person], sample_count=5)[0]
    assert rule['face_safe_samples'][0]['region']['kind'] == 'head_proxy'
    assert rule['anchors'][0]['constrained'] is True
    interval = rule['face_safe_samples'][0]['constraint']
    assert interval['min_x'] <= rule['anchors'][0]['x'] <= interval['max_x']


def test_ambiguous_face_evidence_is_review_not_publish_ready(tmp_path, monkeypatch):
    vertical = tmp_path/'vertical.mp4'; write_video(vertical, 90, 120)
    rule = {'shot_id': 'S1', 'start_seconds': 0., 'end_seconds': 1., 'crop_width': 90,
            'source_width': 90, 'x': 0., 'anchors': [{'time': 0., 'x': 0.}],
            'focus_subject': 'environment', 'action_preserved': True}
    monkeypatch.setattr(f, 'post_render_vertical_audit', lambda *_args, **_kwargs: {
        'decision': 'AMBIGUOUS', 'face_safe_audit': {'decision': 'AMBIGUOUS'}})
    result = f.validate_vertical(vertical, 1., [rule], {})
    assert result['status'] == 'REVIEW'
    assert result['review_reason'] == 'face_safe_ambiguous'
    assert result['face_safe_outcome'] == 'REVIEW_AMBIGUOUS'


def test_actual_registered_face_clipping_is_hard_failure_not_publish_ready(tmp_path, monkeypatch):
    vertical = tmp_path/'vertical.mp4'; write_video(vertical, 90, 120)
    rule = {'shot_id': 'S1', 'start_seconds': 0., 'end_seconds': 1., 'crop_width': 90,
            'source_width': 90, 'x': 0., 'anchors': [{'time': 0., 'x': 0.}],
            'focus_subject': 'environment', 'action_preserved': True}
    monkeypatch.setattr(f, 'post_render_vertical_audit', lambda *_args, **_kwargs: {
        'decision': 'RETRY', 'face_safe_audit': {'decision': 'REPAIR', 'repairable': True}})
    result = f.validate_vertical(vertical, 1., [rule], {})
    assert result['status'] == 'REVIEW'
    assert result['face_safe_outcome'] == 'HARD_FAIL_REPAIRABLE'
    assert 'important_face_or_head_materially_clipped' in result['hard_failures']


@pytest.mark.skipif(
    not (Path('input/mi-otra-yo-s03e02/movie.mp4').is_file()
         and Path('runs/mi-otra-yo-s03e02/visual_events.json').is_file()),
    reason='local MOYS042 regression source fixture is unavailable',
)
def test_moys042_real_source_geometry_moves_unsafe_person_anchor_without_rendering():
    """Read-only regression for superseded VE_f68979082539620e (MOYS042)."""
    run = Path('runs/mi-otra-yo-s03e02')
    event = next(row for row in json.loads((run/'visual_events.json').read_text())['events']
                 if row['visual_event_id'] == 'VE_f68979082539620e')
    observation = json.loads((run/'semantic_observations/v1/events/VE_f68979082539620e.json').read_text())['observation']
    event['visual'] = {**event.get('visual', {}), 'shot_focus_plan': observation['shot_focus_plan']}
    shots = {row['shot_id']: row for row in json.loads((run/'technical_shots.json').read_text())['shots']}
    active = json.loads((run/'active_picture.json').read_text())['active_picture']

    plan = f.build_shot_crop_plan(Path('input/mi-otra-yo-s03e02/movie.mp4'), event, shots,
                                  active['width'], active['height'], active_picture=active)
    rule = next(row for row in plan if row['shot_id'] == 'FULL_SHOT_0107')
    targets = {row['time']: row['bbox'] for row in rule['target_samples']}
    samples = [row for row in rule['face_safe_samples'] if row['constraint'] and row['constraint']['feasible']]
    old_unsafe = []
    for sample in samples:
        interval = sample['constraint']
        old = f._anchor(targets[sample['time']], active['width'], rule['crop_width'])
        new = f._x_at(rule, sample['time'])
        old_unsafe.append(not (interval['min_x'] <= old <= interval['max_x']))
        assert interval['min_x'] <= new <= interval['max_x']
    assert any(old_unsafe), 'fixture must exercise a person-anchor face/head edge cut'


def test_real_mp4_registration_observes_actual_crop(tmp_path):
    source = tmp_path/'source.mp4'; vertical = tmp_path/'v.mp4'; write_video(source)
    event = dict(start_seconds=0.,end_seconds=1.,start_frame=0,end_frame_exclusive=24)
    rule = dict(shot_id='S1',render_start_frame=0,render_end_frame_exclusive=24,start_seconds=0,end_seconds=1,
                anchors=[dict(time=0,x=120)],focus_subject='woman',target_person_ids=['P1'],target_binding_resolved=True,
                face_safe_samples=[dict(time=.5,target=dict(x=90,y=20,width=250,height=280))])
    f.render_vertical(source,vertical,event,[rule])
    rule['anchors'][0]['x']=0  # misleading persisted crop cannot make the audit pass
    result=fs.audit_video(vertical,source,event,[rule],face_detector=lambda frame:[face()])
    assert result['decision']=='REPAIR'
    assert all(abs(s['observed_crop_x']-120)<4 for s in result['samples'])


def test_model_checksum_is_verified_and_no_automatic_redownload(tmp_path,monkeypatch):
    model=tmp_path/fs.MODEL_NAME; model.write_bytes(b'corrupted')
    monkeypatch.setattr(f,'_model_path',lambda:tmp_path/'yolo.onnx')
    monkeypatch.setattr(fs,'_PROVISION_RESULT',None)
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:pytest.fail('must not redownload existing corrupt model'))
    result=fs.preflight()
    assert not result['available'] and 'checksum mismatch' in result['failure_reason']
    with pytest.raises(RuntimeError,match='checksum'):
        fs.detect_faces(np.zeros((300,400,3),dtype=np.uint8))


def test_offline_missing_model_download_attempt_is_bounded(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'_model_path',lambda:tmp_path/'yolo.onnx')
    monkeypatch.setattr(fs,'_PROVISION_RESULT',None)
    calls=[]
    def offline(*a,**k):
        calls.append(1); raise OSError('offline')
    monkeypatch.setattr('urllib.request.urlopen',offline)
    assert not fs.preflight()['available']
    assert not fs.preflight()['available']
    assert len(calls)==1
    assert fs.detect_faces(np.zeros((300,400,3),dtype=np.uint8))==[]


def replacement_files(tmp_path):
    stage=tmp_path/'stage'; final=tmp_path/'assets'; stage.mkdir();final.mkdir()
    names=['vasset.mp4','vasset.jpg','asset.json']
    for name in names:
        (stage/name).write_text('new '+name);(final/name).write_text('old '+name)
    for name in ['asset.mp4','asset.jpg']:
        (final/name).write_text('horizontal '+name)
    hashes={p.name:sha256_file(p) for p in final.iterdir()}
    return [stage/n for n in names],[final/n for n in names],hashes


def test_replacement_failure_restores_previous_vertical(tmp_path,monkeypatch):
    staged,final,hashes=replacement_files(tmp_path)
    original=vr.os.replace
    def fail(source,destination):
        if Path(source)==staged[1]:raise OSError('injected replacement failure')
        return original(source,destination)
    monkeypatch.setattr(vr.os,'replace',fail)
    with pytest.raises(OSError,match='injected'):
        vr.replace_vertical(staged,final,tmp_path/'backup',hashes)
    assert {p.name:sha256_file(p) for p in final[0].parent.iterdir()}==hashes
    assert json.loads((tmp_path/'backup/transaction.json').read_text())['state']=='ROLLED_BACK'


def test_replacement_success_never_touches_horizontal(tmp_path):
    staged,final,hashes=replacement_files(tmp_path)
    h=final[0].parent/'asset.mp4'; before=h.stat().st_mtime_ns
    vr.replace_vertical(staged,final,tmp_path/'backup',hashes)
    assert h.stat().st_mtime_ns==before and sha256_file(h)==hashes[h.name]
    assert final[0].read_text()=='new vasset.mp4'
    assert (tmp_path/'backup/vasset.mp4').read_text()=='old vasset.mp4'


def test_audit_command_is_read_only(tmp_path,monkeypatch):
    assets=tmp_path/'assets';assets.mkdir(); movie=tmp_path/'movie.mp4';movie.write_bytes(b'source')
    for n in ['asset.mp4','asset.jpg','vasset.mp4','vasset.jpg']:(assets/n).write_text(n)
    write_json(assets/'asset.json',{'asset':{'id':'rc001'}})
    before={str(p):(sha256_file(p),p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.setattr(vr,'context',lambda _: (tmp_path,movie,{},{}))
    monkeypatch.setattr(vr,'preflight',lambda **k:{'available':False})
    monkeypatch.setattr(vr,'audit_asset',lambda *a:dict(producer_asset_id='rc001',decision='AMBIGUOUS'))
    vr.audit_verticals(assets)
    assert all((sha256_file(Path(p)),Path(p).stat().st_mtime_ns)==state for p,state in before.items())
    with pytest.raises(ValueError,match='outside assets'):
        vr.audit_verticals(assets,assets/'report.json')


def test_cli_dispatches_only_selected_asset(monkeypatch):
    from movie_broll.cli import main
    calls=[]
    monkeypatch.setattr(vr,'repair_verticals',lambda *args,**kwargs: calls.append(args) or {'results':[]})
    assert main(['repair-verticals','input/film','--audit','audit.json','--asset-id','rc214'])==0
    assert calls==[('input/film','audit.json','rc214')]


def test_selective_repair_same_id_horizontal_untouched(tmp_path,monkeypatch):
    assets=tmp_path/'assets';assets.mkdir(); movie=tmp_path/'movie.mp4'; write_video(movie)
    h=assets/'asset.mp4';shutil=__import__('shutil');shutil.copy2(movie,h)
    v=assets/'vasset.mp4';shutil.copy2(movie,v)
    ht=assets/'asset.jpg';vt=assets/'vasset.jpg';f.thumbnail(h,ht);f.thumbnail(v,vt)
    technical=dict(duration_seconds=1.,width=400,height=300,fps=24)
    data={'schema_version':'asset_metadata_v1','asset':{'id':'rc001','slug':'stable'},
          'source_timeline':{'visual_event_id':'E1'},'visual':{'final_vertical':{}},
          'media':{'horizontal':f._rendition(h,ht,0,technical,'landscape'),'vertical':f._rendition(v,vt,0,technical,'portrait')}}
    md=assets/'asset.json';write_json(md,data)
    row=dict(decision='REPAIR',producer_asset_id='rc001',metadata_file=md.name,
             vertical_sha256=sha256_file(v),metadata_sha256=sha256_file(md))
    audit=tmp_path/'audit.json';write_json(audit,dict(schema_version='vertical_repair_audit_v1',run=str(tmp_path),assets=[row,dict(decision='PASS',producer_asset_id='rc002')]))
    event=dict(start_frame=0,end_frame_exclusive=24,start_seconds=0.,end_seconds=1.,visual_event_id='E1')
    plan=[dict(shot_id='S1',render_start_frame=0,render_end_frame_exclusive=24,start_seconds=0,end_seconds=1,anchors=[dict(time=0,x=50)])]
    monkeypatch.setattr(vr,'context',lambda _: (tmp_path,movie,{'E1':event},{}))
    monkeypatch.setattr(vr,'preflight',lambda **k:{'available':True})
    monkeypatch.setattr(vr,'source_plan',lambda *args:copy.deepcopy(plan))
    monkeypatch.setattr(vr,'audit_asset',lambda *args:row)
    monkeypatch.setattr(f,'validate_vertical',lambda *args:dict(status='PASS',duration_seconds=1.,width=224,height=300,fps=24))
    before={p.name:(sha256_file(p),p.stat().st_mtime_ns) for p in [h,ht]}
    result=vr.repair_verticals(tmp_path,audit)
    assert len(result['results'])==1
    assert result['results'][0]['same_asset_id'] and result['results'][0]['package_valid']
    assert json.loads(md.read_text())['asset']==data['asset']
    assert all((sha256_file(assets/n),(assets/n).stat().st_mtime_ns)==state for n,state in before.items())
    assert sha256_file(v)!=row['vertical_sha256']


def test_retry_policy_is_bounded_and_source_limit_stops():
    def validation(decision):
        return {'status':'REVIEW','post_render_audit':{'face_safe_audit':{'decision':decision,'repairable':decision=='REPAIR'}}}
    assert f.next_vertical_strategy('subject_focus',validation('REPAIR'))=='face_priority'
    assert f.next_vertical_strategy('face_priority',validation('REPAIR')) is None
    for decision in ('SOURCE_LIMIT','AMBIGUOUS','ERROR'):
        assert f.next_vertical_strategy('subject_focus',validation(decision)) is None


def test_interrupted_promotion_recovers_from_backup(tmp_path):
    staged,final,hashes=replacement_files(tmp_path)
    backup=tmp_path/'backup';backup.mkdir()
    import shutil
    for p in final:shutil.copy2(p,backup/p.name)
    write_json(backup/'transaction.json',{'state':'PREPARED','destinations':[str(p) for p in final],'hashes':hashes})
    final[0].write_text('interrupted new video')
    vr.restore_transaction(backup,final[0].parent)
    assert {p.name:sha256_file(p) for p in final[0].parent.iterdir()}==hashes


def test_face_head_constraint_version_invalidates_legacy_vertical_packages(tmp_path, monkeypatch):
    data={'visual':{'final_vertical':dict(reframe_fingerprint='legacy',reframe_algorithm_version='3e.2.3.8-source-preserving-export-v7-target-binding',validation_status='PASS')}}
    write_json(tmp_path/'asset.json',data)
    monkeypatch.setattr(f, '_complete_package', lambda *_: True)
    monkeypatch.setattr(f, '_asset_metadata_contract_valid', lambda *_: True)
    assert not f._vertical_reuse_valid(tmp_path,'asset','legacy')


def test_replacement_cannot_address_horizontal_even_with_bad_metadata(tmp_path):
    staged,final,hashes=replacement_files(tmp_path)
    final[0]=final[0].parent/'asset.mp4'
    with pytest.raises(ValueError,match='canonical vertical'):
        vr.replace_vertical(staged,final,tmp_path/'backup',hashes)
    assert sha256_file(final[0])==hashes['asset.mp4']


def test_bad_backup_aborts_before_replacement(tmp_path,monkeypatch):
    staged,final,hashes=replacement_files(tmp_path)
    original=vr.shutil.copy2
    def corrupt(source,destination):
        result=original(source,destination)
        Path(destination).write_bytes(b'bad copy')
        return result
    monkeypatch.setattr(vr.shutil,'copy2',corrupt)
    with pytest.raises(RuntimeError,match='backup verification'):
        vr.replace_vertical(staged,final,tmp_path/'backup',hashes)
    assert {p.name:sha256_file(p) for p in final[0].parent.iterdir()}==hashes


def test_unregistered_render_retention_signals_are_unknown(tmp_path,monkeypatch):
    source=tmp_path/'s.mp4';vertical=tmp_path/'v.mp4';write_video(source);write_video(vertical)
    rule=dict(shot_id='S1',focus_subject='woman',target_binding_resolved=True,target_person_ids=['P1'],
              render_start_frame=0,render_end_frame_exclusive=24,
              face_safe_samples=[dict(time=.5,target=dict(x=90,y=20,width=250,height=280))])
    monkeypatch.setattr(fs,'register_crop',lambda *args:(None,.1))
    result=fs.audit_video(vertical,source,{'start_frame':0,'start_seconds':0.},[rule],face_detector=lambda frame:[face()])
    assert result['decision']=='AMBIGUOUS'
    assert all(s['face_fully_inside_frame']=='UNKNOWN' and s['face_edge_margin'] is None for s in result['samples'])
