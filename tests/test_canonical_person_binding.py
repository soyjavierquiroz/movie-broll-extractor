"""Local regressions: no providers, episode state, media export, or model loads."""
import copy
from pathlib import Path

import numpy as np
import pytest

from movie_broll import finalization as f
from movie_broll.semantic_observations import apply_policy_result, canonical_evidence_catalog

SHOT='FULL_SHOT_0190'
BOX={'x':30.,'y':10.,'width':20.,'height':80.}


def candidates():
    return [{'person_id':'P1','box':dict(BOX)}]


def test_catalog_contract_and_owner_resolution():
    catalog=canonical_evidence_catalog({'visual_event_id':'VE'}, {'technical_shots':[{'shot_id':SHOT,'candidates':[{'person_id':'P1'}]}]})
    identity=catalog['people'][0]['id']
    assert identity == f'{SHOT}:P1'
    assert f.resolve_shot_person_id(identity,SHOT,candidates())=='P1'
    with pytest.raises(ValueError,match='cross_shot'):
        f.resolve_shot_person_id(identity,'FULL_SHOT_0191',candidates())


@pytest.mark.parametrize('identity',['FULL_SHOT_0190:P0','FULL_SHOT_0190:P01','FULL_SHOT_0190:P1:extra',':P1','P1','FULL_SHOT_0190:',' FULL_SHOT_0190:P1',None])
def test_malformed(identity):
    with pytest.raises(ValueError,match='malformed'):
        f.resolve_shot_person_id(identity,SHOT,candidates())


def test_missing_ambiguous_and_unusable():
    with pytest.raises(ValueError,match='missing'):
        f.resolve_shot_person_id(f'{SHOT}:P2',SHOT,candidates())
    with pytest.raises(ValueError,match='ambiguous'):
        f.resolve_shot_person_id(f'{SHOT}:P1',SHOT,candidates()*2)
    rows=candidates(); rows[0]['box']['width']=float('nan')
    with pytest.raises(ValueError,match='unusable'):
        f.resolve_shot_person_id(f'{SHOT}:P1',SHOT,rows)


def make_plan(monkeypatch,ids,positions=(30,),subject='woman',interaction='none'):
    event={'visual_event_id':'VE','start_seconds':0.,'end_seconds':1.,'source_shot_ids':[SHOT],
           'visual':{'shot_focus_plan':[{'shot_id':SHOT,'focus_subject':subject,'focus_position':'right',
               'target_person_ids':ids,'interaction_requirement':interaction,'preserve_secondary_subject':len(ids)>1}]}}
    before=copy.deepcopy(event)
    frame=np.zeros((120,160,3),dtype=np.uint8)
    monkeypatch.setattr(f,'_sample_frames',lambda *args:[(.12,frame),(.5,frame),(.88,frame)])
    monkeypatch.setattr('movie_broll.face_safe.detect_faces',lambda _:[])
    # Any network/provider/provisioning attempt fails the test immediately.
    monkeypatch.setattr(f,'person_detector_preflight',lambda *args,**kwargs:pytest.fail('model provisioning forbidden'))
    monkeypatch.setattr(f.urllib.request,'urlopen',lambda *args,**kwargs:pytest.fail('network forbidden'))
    detector=lambda _:[{'detector':'yolo_person','bbox':dict(BOX,x=float(x))} for x in positions]
    rule=f.build_shot_crop_plan(Path('/unused/source.mp4'),event,{SHOT:{'start_seconds':0.,'end_seconds':1.}},160,120,detector=detector)[0]
    assert event==before
    return rule


def test_single_subject_shot_focus_and_no_fallback(monkeypatch):
    rule=make_plan(monkeypatch,[f'{SHOT}:P1'])
    assert rule['target_person_ids']==[f'{SHOT}:P1']
    assert rule['resolved_local_person_ids']==['P1']
    assert rule['target_binding_resolved'] is True
    assert rule['focus_bbox']==BOX
    assert rule['target_samples']
    assert rule['tracking']['mode']=='semantic_bound_geometry'
    assert rule['crop_fallback_used'] is False
    assert rule['anchors'][0]['x']==0.


@pytest.mark.parametrize('identity',['FULL_SHOT_0191:P1',f'{SHOT}:P2','garbage'])
def test_failed_binding_cannot_guess_by_position(monkeypatch,identity):
    rule=make_plan(monkeypatch,[identity])
    assert rule['target_binding_resolved'] is False
    assert rule['focus_bbox'] is None
    assert rule['review_required'] is True
    assert f._shot_validation(rule)['status']=='FAIL'


@pytest.mark.parametrize('positions,retained', [((30,60),True),((10,120),False)])
def test_interaction_tracks_every_participant_or_fails(monkeypatch,positions,retained):
    rule=make_plan(monkeypatch,[f'{SHOT}:P1',f'{SHOT}:P2'],positions,interaction='simultaneous')
    assert rule['target_binding_resolved'] is True
    assert rule['resolved_local_person_ids']==['P1','P2']
    assert rule['tracking']['mode']=='semantic_bound_multi_geometry'
    assert rule['focus_bbox']['width']==positions[1]+20-positions[0]
    check=f._shot_validation(rule)
    assert check['interaction_preserved'] is retained
    qa=f.classify_vertical_qa({'width':90,'height':120,'aspect_ratio':'3:4','shots':[check]})
    assert ('required_multi_person_interaction_lost' in qa['hard_failures']) is not retained


def test_insufficient_interaction_binding_fails(monkeypatch):
    rule=make_plan(monkeypatch,[f'{SHOT}:P1'],interaction='simultaneous')
    assert rule['target_binding_resolved'] is False
    assert rule['review_required'] is True


@pytest.mark.parametrize('validation',[
    {'status':'FAIL'}, {'status':'PASS','semantic_retained':False},
    {'status':'PASS','hard_failures':['required_multi_person_interaction_lost']},
    {'status':'PASS','shots':[{'target_binding_resolved':False}]},
    {'status':'PASS','shots':[{'focus_subject_present':False}]},
    {'status':'PASS','shots':[{'interaction_preserved':False}]},
    {'status':'REVIEW','post_render_audit':{'decision':'AMBIGUOUS'}},
])
def test_rendition_semantic_validation_uses_retention(tmp_path,validation):
    media=tmp_path/'v.mp4';media.write_bytes(b'media')
    thumb=tmp_path/'v.jpg';thumb.write_bytes(b'thumb')
    tech=dict(validation,duration_seconds=1.,width=90,height=120,fps=24.)
    assert f._rendition(media,thumb,0.,tech,'portrait')['semantic_validated'] is False
    assert f._semantic_retention_valid({'status':'PASS','semantic_retained':True}) is True


def test_observation_people_projection_preserves_canonical_ids():
    event={'visual_event_id':'VE','narrative_segment_ids':['N1']}
    body={'people_count':'2_plus','visible_person_ids':[f'{SHOT}:P1',f'{SHOT}:P2'],'shot_focus_plan':[]}
    observation={'event_id':'VE','observation':body}
    before=copy.deepcopy(observation)
    apply_policy_result(event,observation,{'policy_decision':'KEEP','policy_version':'v','policy_reasons':[]})
    assert observation==before
    assert event['semantic_people']['contains_people'] is True
    assert event['semantic_people']['people_count'] is None  # 2_plus is a lower bound
    assert event['semantic_people']['visible_person_ids']==body['visible_person_ids']
    assert event['narrative_segment_ids']==['N1']


def test_only_vertical_fingerprint_changes():
    event={'source_shot_ids':[SHOT],'visual':{'shot_focus_plan':[]}}
    assert f.reframe_fingerprint(event,{},160,120)!=f.reframe_fingerprint(event,{},160,120,algorithm_version='phase-f-face-head-constrained-reframe-v3')


def test_metadata_projects_people_narrative_and_failed_retention(tmp_path):
    event={'visual_event_id':'VE','start_seconds':0.,'end_seconds':1.,
           'source_shot_ids':[SHOT],'narrative_segment_ids':['N1'],
           'semantic_people':{'contains_people':True,'people_count':1},'editorial':{'decision':'KEEP'}}
    paths=[tmp_path/name for name in ('h.mp4','v.mp4','h.jpg','v.jpg')]
    for path in paths:
        path.write_bytes(path.name.encode())
    technical={'status':'PASS','width':90,'height':120,'fps':24.,'duration_seconds':1.}
    metadata=f._asset_metadata('fixture','a'*64,'asset','slug',event,*paths,0.,0.,technical,
        dict(technical,status='REVIEW',semantic_retained=False),'source_encode',1,[],1,'fp')
    assert metadata['visual']['people']['contains_people'] is True
    assert metadata['visual']['people']['people_count']==1
    assert metadata['narrative']=={'segment_ids':['N1']}
    assert metadata['source_timeline']['narrative_segment_ids']==['N1']
    assert metadata['media']['vertical']['semantic_validated'] is False
    assert metadata['analysis']['final_asset_semantics_validated'] is False
    assert f.is_publish_ready(metadata) is False
    assert f._asset_metadata_contract_valid(metadata,tmp_path) is True


def test_cached_policy_projection_makes_zero_provider_requests(monkeypatch,tmp_path):
    from movie_broll import semantic_observations as observations
    body={'people_count':'1','visible_person_ids':[f'{SHOT}:P1'],'shot_focus_plan':[]}
    observation={'event_id':'VE','observation':body}
    monkeypatch.setattr(observations,'effective_observation',lambda *args:observation)
    monkeypatch.setattr(observations,'validate_observation',lambda *args:[])
    monkeypatch.setattr(observations,'policy_evaluate',lambda *args,**kwargs:{'policy_decision':'KEEP','policy_version':'v','policy_reasons':[]})
    monkeypatch.setattr(observations,'observe_missing_events',lambda *args,**kwargs:pytest.fail('semantic observation forbidden'))
    result=observations.apply_cached_policy(tmp_path,[{'visual_event_id':'VE'}])
    assert result['provider_requests']==0
    assert result['api_cost_usd']==0.
