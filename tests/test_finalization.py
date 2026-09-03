import json
from pathlib import Path
import cv2
import numpy as np
from movie_broll.finalization import asset_identity, crop_x, render_vertical, safe_cleanup, shot_crop_plan, slugify, thumbnail, validate_vertical

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
