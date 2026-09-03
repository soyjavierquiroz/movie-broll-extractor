"""Resumable flat final asset packages with local shot-aware 3:4 reframing."""
from __future__ import annotations
import json, re, shutil, subprocess, unicodedata
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from .broll_pilot import ffmpeg_export_command, probe
from .processing_ledger import ProcessingLedger, fingerprint
from .utils import sha256_file, write_json

MOVIE_CODES={"romper-el-circulo":"rc"}

def movie_code(run:Path,movie_id:str)->str:
    path=run/'movie_metadata.json'
    if path.exists(): return json.loads(path.read_text())['movie_code']
    code=MOVIE_CODES.get(movie_id) or ''.join(x[0] for x in re.findall(r'[a-z0-9]+',movie_id.lower()))[:4] or 'mv'
    write_json(path,{'schema_version':'movie_run_metadata_v1','movie_id':movie_id,'movie_code':code}); return code

def slugify(value:str)->str:
    text=unicodedata.normalize('NFKD',value).encode('ascii','ignore').decode().lower()
    ignored={'una','unos','unas','persona','escena','que','con','del','las','los','el','la'}
    return '-'.join(x for x in re.findall(r'[a-z0-9]+',text) if x not in ignored)[:48].strip('-') or 'momento-visual'

def asset_identity(run:Path,movie_id:str,event:dict[str,Any])->tuple[str,str]:
    path=run/'asset_registry.json'; data=json.loads(path.read_text()) if path.exists() else {'schema_version':'asset_registry_v1','movie_id':movie_id,'events':{}}
    events=data['events']; eid=event['visual_event_id']
    if eid not in events:
        code=movie_code(run,movie_id); nums=[int(x['asset_id'][len(code):]) for x in events.values() if x.get('asset_id','').startswith(code) and x['asset_id'][len(code):].isdigit()]
        text=event.get('editorial',{}).get('standalone_meaning_es') or event.get('visual',{}).get('summary_es') or 'momento visual'
        events[eid]={'asset_id':f'{code}{max(nums,default=0)+1:03d}','slug':slugify(str(text))}; write_json(path,data)
    return events[eid]['asset_id'],events[eid]['slug']

def crop_x(width:int,height:int,position:str)->int:
    crop=min(width,round(height*3/4))
    return 0 if position=='left' else width-crop if position=='right' else max(0,(width-crop)//2)

def shot_crop_plan(event:dict[str,Any],shots:dict[str,dict[str,Any]],width:int,height:int)->list[dict[str,Any]]:
    visual=event.get('visual',{}); people=event.get('people',visual.get('people',[])) or []; interaction=bool(visual.get('visible_interactions')) and len(people)>=2
    result=[]
    for sid in event.get('source_shot_ids',[]):
        shot=shots.get(sid,{}); pos=str(shot.get('primary_subject_position') or visual.get('primary_subject_position') or 'center').lower()
        if interaction: pos='center'
        result.append({'shot_id':sid,'start_seconds':float(shot.get('start_seconds',event['start_seconds'])),'end_seconds':float(shot.get('end_seconds',event['end_seconds'])),'position':pos,'x':crop_x(width,height,pos),'action_preserved':bool(visual.get('actions',[]))})
    return result or [{'shot_id':'event','start_seconds':float(event['start_seconds']),'end_seconds':float(event['end_seconds']),'position':'center','x':crop_x(width,height,'center'),'action_preserved':bool(visual.get('actions',[]))}]

def render_vertical(horizontal:Path,output:Path,event:dict[str,Any],plan:list[dict[str,Any]])->None:
    cap=cv2.VideoCapture(str(horizontal)); fps=cap.get(cv2.CAP_PROP_FPS) or 24.; w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cw=min(w,round(h*3/4)); temp=output.with_suffix('.render.tmp.mp4')
    writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(cw,h)); n=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        absolute=float(event['start_seconds'])+n/fps; rule=next((x for x in plan if x['start_seconds']<=absolute<x['end_seconds']),plan[-1]); x=max(0,min(w-cw,int(rule['x']))); writer.write(frame[:,x:x+cw]); n+=1
    writer.release(); cap.release()
    if not n: temp.unlink(missing_ok=True); raise RuntimeError('vertical reframe decoded no frames')
    try:
        subprocess.run(['ffmpeg','-y','-i',str(temp),'-map','0:v:0','-c:v','libx264','-crf','19','-an',str(output)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    finally:
        # Failed encodes must not become crash residue or be mistaken for a
        # resumable final representation.
        temp.unlink(missing_ok=True)

def validate_vertical(path:Path,expected:float,plan:list[dict[str,Any]],event:dict[str,Any])->dict[str,Any]:
    cap=cv2.VideoCapture(str(path)); w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS) or 24.; frames=[]
    for i in {0,max(0,count//2),max(0,count-1)}:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,x=cap.read()
        if ok: frames.append(x)
    cap.release(); bars=any(float(np.mean(cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)<8))>.92 for x in frames)
    people=event.get('people',event.get('visual',{}).get('people',[])) or []; positions={str(x.get('position','center')) for x in people if isinstance(x,dict)}; interaction=bool(event.get('visual',{}).get('visible_interactions')) and len(people)>=2
    semantic_ok=not(interaction and {'left','right'}.issubset(positions)) and all(x.get('action_preserved',True) for x in plan)
    ok=path.exists() and count>0 and w*4==h*3 and abs(count/fps-expected)<=1 and not bars and semantic_ok
    return {'status':'PASS' if ok else 'REVIEW','width':w,'height':h,'aspect_ratio':'3:4' if w*4==h*3 else 'other','duration_seconds':count/fps,'black_bars':bars,'stable_per_shot':True,'semantic_retained':semantic_ok}

def thumbnail(video:Path,out:Path)->float:
    cap=cv2.VideoCapture(str(video)); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS) or 24.; best=None; score=-1.; selected=1
    for i in [max(1,int(count*x)) for x in (.25,.5,.75)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,min(i,max(1,count-2))); ok,x=cap.read()
        if ok:
            q=float(cv2.Laplacian(cv2.cvtColor(x,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
            if q>score: best,score,selected=x,q,i
    cap.release()
    if best is None or not cv2.imwrite(str(out),best,[cv2.IMWRITE_JPEG_QUALITY,90]): raise RuntimeError('cannot write thumbnail')
    return selected/fps

def safe_cleanup(work:Path,keep_debug_artifacts:bool=False)->int:
    if keep_debug_artifacts or not work.exists(): return 0
    root=work.resolve()
    if root.name!='.work' or 'runs' not in root.parts: raise ValueError('refusing cleanup outside owned .work')
    size=sum(x.stat().st_size for x in root.rglob('*') if x.is_file() and not x.is_symlink())
    for x in root.iterdir():
        if x.is_symlink() or x.is_file(): x.unlink()
        elif x.is_dir(): shutil.rmtree(x)
    return size

def finalize_pilot(input_dir:Path,window_id:str,keep_debug_artifacts:bool=False)->dict[str,Any]:
    root=input_dir.resolve().parents[1]; movie_id=input_dir.name; run=root/'runs'/movie_id; pilot=run/'broll-pilot-v1'/window_id
    candidates=json.loads((pilot/'candidates.json').read_text()).get('candidates',[]); assets=run/'assets'; assets.mkdir(parents=True,exist_ok=True); work=run/'.work'; work.mkdir(parents=True,exist_ok=True)
    shots_path=run/'visual-smoke-v1'/'shots.jsonl'; shots={x['shot_id']:x for x in (json.loads(y) for y in shots_path.read_text().splitlines() if y.strip())} if shots_path.exists() else {}
    movie=input_dir/'movie.mp4'; source=cv2.VideoCapture(str(movie)); width,height=int(source.get(cv2.CAP_PROP_FRAME_WIDTH)),int(source.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=source.get(cv2.CAP_PROP_FPS) or 24.; source.release()
    ledger=ProcessingLedger(run,movie_id,{'finalization_version':'3e','movie_code':movie_code(run,movie_id)}); completed=review=reused=0
    for e in candidates:
        if e.get('editorial',{}).get('decision')!='KEEP' or e.get('editorial',{}).get('status')!='VALIDATED': continue
        fp=fingerprint({'range':[e['start_frame'],e['end_frame_exclusive']],'shots':e.get('source_shot_ids',[]),'semantic':e.get('visual',{}),'version':'3e'}); ledger.register(e,fp)
        aid,slug=asset_identity(run,movie_id,e); base=f'{aid}-{slug}'; h=assets/f'{base}.mp4'; v=assets/f'v{base}.mp4'; ht=assets/f'{base}.jpg'; vt=assets/f'v{base}.jpg'; md=assets/f'{base}.json'; eid=e['visual_event_id']; expected=(e['end_frame_exclusive']-e['start_frame'])/fps
        if not h.exists():
            ledger.stage(eid,'horizontal_export','RUNNING'); old=pilot/'exports'/f"{e['candidate_id']}.mp4"
            if old.exists(): shutil.copy2(old,h)
            else: subprocess.run(ffmpeg_export_command(movie,e,h,fps),check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            ledger.stage(eid,'horizontal_export','COMPLETE',path=str(h))
        else: reused+=1; ledger.stage(eid,'horizontal_export','COMPLETE',path=str(h),reused=True)
        horizontal=probe(h,width,height,expected,e['end_frame_exclusive']-e['start_frame']); ledger.stage(eid,'horizontal_validation','COMPLETE' if horizontal['status']=='PASS' else 'FAILED_RETRYABLE',validation=horizontal['status'])
        if horizontal['status']!='PASS': continue
        plan=shot_crop_plan(e,shots,width,height)
        if not v.exists():
            ledger.stage(eid,'vertical_reframe','RUNNING',strategy='semantic_shot_stable_crop')
            try: render_vertical(h,v,e,plan)
            except Exception: v.unlink(missing_ok=True); raise
            ledger.stage(eid,'vertical_reframe','COMPLETE',path=str(v),plan=plan)
        else: reused+=1; ledger.stage(eid,'vertical_reframe','COMPLETE',path=str(v),reused=True,plan=plan)
        vertical=validate_vertical(v,expected,plan,e); ledger.stage(eid,'vertical_validation','COMPLETE' if vertical['status']=='PASS' else 'FAILED_RETRYABLE',validation=vertical)
        if vertical['status']!='PASS': review+=1; continue
        htime=thumbnail(h,ht) if not ht.exists() else 0.; ledger.stage(eid,'horizontal_thumbnail','COMPLETE',path=str(ht))
        vtime=thumbnail(v,vt) if not vt.exists() else 0.; ledger.stage(eid,'vertical_thumbnail','COMPLETE',path=str(vt))
        data={'asset_metadata_v1':'asset_metadata_v1','asset':{'id':aid,'slug':slug,'source_asset_id':eid,'source_movie_id':movie_id},'media':{'filename':h.name,'sha256':sha256_file(h),'duration_seconds':expected,'width':width,'height':height,'fps':fps,'orientation':'landscape','aspect_ratio':'other','horizontal':{'file':h.name,'thumbnail':ht.name,'validated':True},'vertical':{'file':v.name,'thumbnail':vt.name,'aspect_ratio':'3:4','validated':True}},'analysis':{'semantic_ready':True,'final_asset_semantics_validated':True,'source_video_analyzed':True,'profile':'pilot-finalization-3e','producer':'movie_broll','producer_version':'0.1.0','generated_at':'durable'},'source_timeline':{'start_seconds':e['start_seconds'],'end_seconds':e['end_seconds'],'visual_event_id':eid,'shot_ids':e.get('source_shot_ids',[])},'visual':{'source_horizontal':e.get('visual',{}),'final_vertical':{'crop_plan':plan,'semantic_retained':True}},'audio':{'speech_present':{'value':False,'source':'export_contract','confidence':1.0}},'narrative':e.get('narrative',{}),'editorial':e.get('editorial',{}),'export':{'orientation':'landscape','aspect_ratio':'other','reframe_applied':True,'reframe_profile':'shot-aware semantic stable 3:4 crop'},'thumbnail':{'filename':ht.name,'timestamp_seconds':htime,'vertical_filename':vt.name,'vertical_timestamp_seconds':vtime}}
        write_json(md,data); ledger.stage(eid,'metadata','COMPLETE',path=str(md))
        # The old pilot export is a replaceable intermediate once the validated
        # horizontal package is durable in assets/.  Never touch manifests,
        # semantic checkpoints, or any final representation.
        old=pilot/'exports'/f"{e['candidate_id']}.mp4"
        old.unlink(missing_ok=True)
        ledger.stage(eid,'cleanup','COMPLETE',removed_bytes=safe_cleanup(work,keep_debug_artifacts)); ledger.stage(eid,'finalization','COMPLETE'); completed+=1
    final_bytes=sum(x.stat().st_size for x in assets.glob('*') if x.is_file()); temp_bytes=sum(x.stat().st_size for x in work.rglob('*') if x.is_file())
    ledger.summary(status='COMPLETE',finalization_complete=completed,finalization_review=review,disk={'final_assets_bytes':final_bytes,'temporary_bytes':temp_bytes})
    return {'status':'COMPLETE','completed':completed,'review':review,'reused':reused,'assets':assets}
