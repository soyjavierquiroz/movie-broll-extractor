"""Resumable final packages and local shot-level 3:4 reframing."""
from __future__ import annotations
import json, re, shutil, subprocess, unicodedata
from pathlib import Path
from typing import Any, Callable
import cv2
import numpy as np
from .broll_pilot import ffmpeg_export_command, probe
from .processing_ledger import ProcessingLedger, fingerprint
from .utils import sha256_file, write_json

MOVIE_CODES={"romper-el-circulo":"rc"}; SAFE_MARGIN=.08
# This is deliberately persistent: it is part of every vertical-only reuse key.
REFRAME_ALGORITHM_VERSION="3e.2-shot-focus-face-safe-v1"
SHOT_FOCUS_SCHEMA_VERSION="shot_focus_plan_v1"
LOCAL_DETECTOR_VERSION="yolov5nu-onnx-person+haar-face-v1"
FOCUS_SUBJECTS={"woman","man","multiple_people","action_region","environment","unclear"}
def movie_code(run:Path,movie_id:str)->str:
    path=run/'movie_metadata.json'
    if path.exists(): return json.loads(path.read_text())['movie_code']
    code=MOVIE_CODES.get(movie_id) or ''.join(x[0] for x in re.findall(r'[a-z0-9]+',movie_id.lower()))[:4] or 'mv'; write_json(path,{'schema_version':'movie_run_metadata_v1','movie_id':movie_id,'movie_code':code}); return code
def slugify(value:str)->str:
    text=unicodedata.normalize('NFKD',value).encode('ascii','ignore').decode().lower(); ignored={'una','unos','unas','persona','escena','que','con','del','las','los','el','la'}
    return '-'.join(x for x in re.findall(r'[a-z0-9]+',text) if x not in ignored)[:48].strip('-') or 'momento-visual'
def asset_identity(run:Path,movie_id:str,event:dict[str,Any])->tuple[str,str]:
    path=run/'asset_registry.json'; data=json.loads(path.read_text()) if path.exists() else {'schema_version':'asset_registry_v1','movie_id':movie_id,'events':{}}; events=data['events']; eid=event['visual_event_id']
    if eid not in events:
        code=movie_code(run,movie_id); nums=[int(x['asset_id'][len(code):]) for x in events.values() if x.get('asset_id','').startswith(code) and x['asset_id'][len(code):].isdigit()]; text=event.get('editorial',{}).get('standalone_meaning_es') or event.get('visual',{}).get('summary_es') or 'momento visual'; events[eid]={'asset_id':f'{code}{max(nums,default=0)+1:03d}','slug':slugify(str(text))}; write_json(path,data)
    return events[eid]['asset_id'],events[eid]['slug']
def crop_x(width:int,height:int,position:str)->int:
    crop=min(width,round(height*3/4)); return 0 if position=='left' else width-crop if position=='right' else max(0,(width-crop)//2)
def _bbox(value:dict[str,Any])->dict[str,float]:
    b=value.get('bbox',value); x,y,w,h=(b.get(k,0.) for k in ('x','y','width','height')); return {'x':float(x),'y':float(y),'width':float(w),'height':float(h),**{k:v for k,v in value.items() if k not in {'x','y','width','height','bbox'}}}
def _model_path()->Path:
    """Standalone optional model location; never borrows another application's cache."""
    return Path(__file__).resolve().parents[2]/'cache'/'models'/'movie-broll'/'yolov5nu.onnx'
def _yolo_people(frame:np.ndarray)->list[dict[str,Any]]:
    """Best-effort CPU YOLO ONNX inference when the documented local model exists."""
    path=_model_path()
    if not path.is_file(): return []
    try:
        net=cv2.dnn.readNetFromONNX(str(path)); blob=cv2.dnn.blobFromImage(frame,1/255.,(640,640),swapRB=True); net.setInput(blob); out=np.squeeze(net.forward())
        if out.ndim==2 and out.shape[0] < out.shape[1]: out=out.T
        sx,sy=frame.shape[1]/640.,frame.shape[0]/640.; result=[]
        for row in out:
            if len(row)<5: continue
            score=float(row[4])
            if score<.35: continue
            cx,cy,w,h=(float(v) for v in row[:4]); result.append({'bbox':{'x':max(0.,(cx-w/2)*sx),'y':max(0.,(cy-h/2)*sy),'width':w*sx,'height':h*sy},'face_visible':False,'confidence':score,'detector':'yolo_person'})
        return result
    except cv2.error: return []
def detect_people(frame:np.ndarray)->list[dict[str,Any]]:
    """Local face geometry plus optional standalone YOLO person geometry; never HOG-only."""
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml'); faces=cascade.detectMultiScale(gray,1.1,4,minSize=(20,20)) if not cascade.empty() else []
    people=_yolo_people(frame)
    # Faces remain separate candidates so a focused interlocutor beats a large OTS body.
    people.extend({'bbox':{'x':float(x),'y':float(y),'width':float(w),'height':float(h)},'face_visible':True,'confidence':1.,'detector':'haar_face'} for x,y,w,h in faces)
    return people
def _directive(event:dict[str,Any],shot:dict[str,Any])->dict[str,Any]:
    visual=event.get('visual',{}); values=visual.get('shot_focus_plan',visual.get('shot_focus',event.get('shot_focus_plan',event.get('shot_focus',[])))) or []; direct=next((x for x in values if x.get('shot_id')==shot.get('shot_id')),{})
    subject=str(direct.get('focus_subject','unclear')).lower()
    return {'focus_subject':subject if subject in FOCUS_SUBJECTS else 'unclear','focus_role':direct.get('focus_role','primary'),'preserve_interaction':bool(direct.get('interaction_requires_both',direct.get('preserve_interaction',False))),'directive_available':bool(direct),**direct}
def _sample_frames(video:Path,start:float,end:float,count:int=5)->list[tuple[float,np.ndarray]]:
    cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS) or 24.; out=[]
    for t in np.linspace(start+(end-start)*.12,end-(end-start)*.12,max(1,count)):
        cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,round(t*fps))); ok,frame=cap.read()
        if ok: out.append((float(t),frame))
    cap.release(); return out
def _choose_target(found:list[dict[str,Any]],direct:dict[str,Any],width:int)->dict[str,Any]|None:
    if not found: return None
    candidates=[x for x in found if not x.get('foreground',False)] or found; faces=[x for x in candidates if x.get('face_visible',False)] or candidates; wanted=str(direct.get('focus_position',direct.get('position',''))).lower()
    def score(x):
        b=_bbox(x); c=b['x']+b['width']/2; bonus=1 if wanted and ((wanted=='left' and c<width/2) or (wanted=='right' and c>=width/2)) else 0; return bonus+float(x.get('confidence',.5)),b['width']*b['height']
    return max(faces,key=score)
def _union(boxes:list[dict[str,float]])->dict[str,float]:
    left=min(x['x'] for x in boxes); top=min(x['y'] for x in boxes); right=max(x['x']+x['width'] for x in boxes); bottom=max(x['y']+x['height'] for x in boxes); return {'x':left,'y':top,'width':right-left,'height':bottom-top}
def _anchor(box:dict[str,float],source:int,crop:int)->float: return max(0.,min(float(source-crop),box['x']+box['width']/2-crop/2))
def _smooth(values:list[tuple[float,float]],crop:int)->list[dict[str,float]]:
    if not values:return []
    raw=np.array([x[1] for x in values])
    if float(raw.max()-raw.min())<crop*.10:return [{'time':float(values[len(values)//2][0]),'x':float(np.median(raw))}]
    output=[]; old=float(raw[0])
    for t,x in values: old=max(old-crop*.18,min(old+crop*.18,float(x))); output.append({'time':float(t),'x':old})
    return output
def build_shot_crop_plan(video:Path,event:dict[str,Any],shots:dict[str,dict[str,Any]],width:int,height:int,detector:Callable[[np.ndarray],list[dict[str,Any]]]=detect_people,strategy:str='subject_focus')->list[dict[str,Any]]:
    crop=min(width,round(height*3/4)); plans=[]
    for sid in event.get('source_shot_ids',[]) or ['event']:
        shot=shots.get(sid,{}); start=float(shot.get('start_seconds',event['start_seconds'])); end=float(shot.get('end_seconds',event['end_seconds'])); direct=_directive(event,{**shot,'shot_id':sid}); samples=[]; all_boxes=[]; focus_boxes=[]; action=[]; prior=None
        for t,frame in _sample_frames(video,start,end):
            found=[_bbox(x) for x in detector(frame)]; target=_choose_target(found,direct,width)
            # Associate local detections to the prior sample, preventing a
            # different nearby person from stealing focus mid-shot.
            if prior is not None and found:
                candidates=[x for x in found if not x.get('foreground',False)] or found
                faces=[x for x in candidates if x.get('face_visible',False)] or candidates
                target=min(faces,key=lambda x:abs((x['x']+x['width']/2)-(prior['x']+prior['width']/2)))
            if target: target=_bbox(target); focus_boxes.append(target); samples.append((t,_anchor(target,width,crop)))
            if target: prior=target
            all_boxes.extend(found)
        regions=direct.get('required_action_region',[]); regions=regions if isinstance(regions,list) else [regions]; action=[_bbox(x) for x in regions if x]
        focus=_union(focus_boxes) if focus_boxes else None; preserve=bool(direct.get('preserve_interaction',False) or direct.get('required_secondary_subjects',[])); composition=[focus] if focus else []
        if (preserve or strategy=='interaction_aware') and all_boxes: composition=[_union(all_boxes)]
        composition+=action; required=_union(composition) if composition else focus; impossible=bool(required and required['width']>crop*(1-2*SAFE_MARGIN) and (preserve or action))
        if required and required['width']<=crop*(1-2*SAFE_MARGIN): anchors=_smooth([(t,_anchor(required,width,crop)) for t,_ in samples] or [(start,_anchor(required,width,crop))],crop)
        elif samples: anchors=_smooth(samples,crop)
        else:
            pos=str(shot.get('primary_subject_position',event.get('visual',{}).get('primary_subject_position','center'))).lower(); anchors=[{'time':start,'x':float(crop_x(width,height,pos))}]
        required_person=direct['focus_subject'] in {'woman','man','multiple_people'}
        unresolved=required_person and focus is None
        plans.append({'shot_id':sid,'start_seconds':start,'end_seconds':end,'focus_subject':direct['focus_subject'],'focus_role':direct['focus_role'],'focus_reason':direct.get('focus_reason','semantic shot focus plus local geometry'),'directive_available':direct['directive_available'],'required_person_focus':required_person,'preserve_interaction':preserve,'required_action_region':regions,'focus_bbox':focus,'subject_bboxes':all_boxes,'anchors':anchors,'x':float(np.median([x['x'] for x in anchors])),'crop_width':crop,'source_width':width,'strategy':strategy,'review_required':impossible or unresolved or not direct['directive_available'],'action_preserved':not action or bool(required),'detector_version':LOCAL_DETECTOR_VERSION})
    return plans
def shot_crop_plan(event:dict[str,Any],shots:dict[str,dict[str,Any]],width:int,height:int)->list[dict[str,Any]]:
    """No-video compatibility fallback; production calls build_shot_crop_plan."""
    out=[]
    for sid in event.get('source_shot_ids',[]) or ['event']:
        shot=shots.get(sid,{}); start=float(shot.get('start_seconds',event['start_seconds'])); end=float(shot.get('end_seconds',event['end_seconds']))
        pos=str(shot.get('primary_subject_position',event.get('visual',{}).get('primary_subject_position','center'))).lower(); x=float(crop_x(width,height,pos)); out.append({'shot_id':sid,'start_seconds':start,'end_seconds':end,'focus_subject':'primary','focus_role':'primary','focus_reason':'fallback semantic position','preserve_interaction':False,'focus_bbox':None,'subject_bboxes':[],'anchors':[{'time':start,'x':x}],'x':x,'crop_width':min(width,round(height*3/4)),'source_width':width,'strategy':'fallback','review_required':False,'action_preserved':True})
    return out
def _x_at(rule:dict[str,Any],time:float)->int:
    anchors=rule.get('anchors') or [{'time':rule['start_seconds'],'x':rule['x']}]
    if len(anchors)==1:return round(anchors[0]['x'])
    for a,b in zip(anchors,anchors[1:]):
        if a['time']<=time<=b['time']:return round(a['x']+(b['x']-a['x'])*(time-a['time'])/max(.001,b['time']-a['time']))
    return round(anchors[-1]['x'])
def render_vertical(horizontal:Path,output:Path,event:dict[str,Any],plan:list[dict[str,Any]])->None:
    cap=cv2.VideoCapture(str(horizontal)); fps=cap.get(cv2.CAP_PROP_FPS) or 24.; w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cw=min(w,round(h*3/4)); temp=output.with_suffix('.render.tmp.mp4'); writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(cw,h)); n=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        absolute=float(event['start_seconds'])+n/fps; rule=next((x for x in plan if x['start_seconds']<=absolute<x['end_seconds']),plan[-1]); x=max(0,min(w-cw,_x_at(rule,absolute))); writer.write(frame[:,x:x+cw]); n+=1
    writer.release(); cap.release()
    if not n:temp.unlink(missing_ok=True); raise RuntimeError('vertical reframe decoded no frames')
    try:subprocess.run(['ffmpeg','-y','-i',str(temp),'-map','0:v:0','-c:v','libx264','-crf','19','-an',str(output)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    finally:temp.unlink(missing_ok=True)
def _shot_validation(rule:dict[str,Any])->dict[str,Any]:
    crop=float(rule['crop_width']); x=float(rule['x']); margin=crop*SAFE_MARGIN; focus=rule.get('focus_bbox'); source=float(rule['source_width']); source_clip=False; introduced=False; empty=False
    required_person=bool(rule.get('required_person_focus',rule.get('focus_subject') in {'woman','man','multiple_people'}))
    if focus:
        source_clip=focus['x']<=1 or focus['x']+focus['width']>=source-1; safe=focus['x']>=x+margin and focus['x']+focus['width']<=x+crop-margin; introduced=not safe and not source_clip; empty=introduced and abs((focus['x']+focus['width']/2)-(x+crop/2))>crop*.20
    action_ok=True
    for region in rule.get('required_action_region',[]) or []:
        b=_bbox(region); action_ok=action_ok and b['x']>=x and b['x']+b['width']<=x+crop
    stable=len(rule.get('anchors',[]))<=1 or max(abs(b['x']-a['x']) for a,b in zip(rule['anchors'],rule['anchors'][1:]))<=crop*.18+1; interaction=not rule.get('review_required',False)
    present=bool(focus) if required_person else True
    safe=bool(focus) and not introduced if required_person else not introduced
    ok=bool(rule.get('directive_available',True)) and present and safe and not introduced and not empty and stable and interaction and rule.get('action_preserved',True) and action_ok
    return {'shot_id':rule['shot_id'],'focus_directive_available':bool(rule.get('directive_available',True)),'focus_subject_present':present,'focus_subject_safe':safe,'introduced_subject_clipping':introduced,'source_edge_exception':source_clip,'empty_space_while_clipped':empty,'interaction_preserved':interaction,'action_preserved':rule.get('action_preserved',True) and action_ok,'crop_stable':stable,'status':'PASS' if ok else 'FAIL'}
def validate_vertical(path:Path,expected:float,plan:list[dict[str,Any]],event:dict[str,Any])->dict[str,Any]:
    cap=cv2.VideoCapture(str(path)); w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS) or 24.; frames=[]
    for i in {0,max(0,count//2),max(0,count-1)}:cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,x=cap.read(); frames.extend([x] if ok else [])
    cap.release(); bars=any(float(np.mean(cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)<8))>.92 for x in frames); per=[_shot_validation(x) for x in plan]; ok=path.exists() and count>0 and w*4==h*3 and abs(count/fps-expected)<=1 and not bars and all(x['status']=='PASS' for x in per)
    return {'status':'PASS' if ok else 'REVIEW','review_reason':'REVIEW_VERTICAL' if not ok else None,'width':w,'height':h,'aspect_ratio':'3:4' if w*4==h*3 else 'other','duration_seconds':count/fps,'black_bars':bars,'shots':per,'stable_per_shot':all(x['crop_stable'] for x in per),'semantic_retained':all(x['interaction_preserved'] for x in per)}
def thumbnail(video:Path,out:Path)->float:
    cap=cv2.VideoCapture(str(video)); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS) or 24.; best=None; score=-1.; selected=1
    for i in [max(1,int(count*x)) for x in (.25,.5,.75)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,min(i,max(1,count-2))); ok,x=cap.read()
        if ok:
            q=float(cv2.Laplacian(cv2.cvtColor(x,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
            if q>score:best,score,selected=x,q,i
    cap.release()
    if best is None or not cv2.imwrite(str(out),best,[cv2.IMWRITE_JPEG_QUALITY,90]):raise RuntimeError('cannot write thumbnail')
    return selected/fps
def safe_cleanup(work:Path,keep_debug_artifacts:bool=False)->int:
    if keep_debug_artifacts or not work.exists():return 0
    root=work.resolve()
    if root.name!='.work' or 'runs' not in root.parts:raise ValueError('refusing cleanup outside owned .work')
    size=sum(x.stat().st_size for x in root.rglob('*') if x.is_file() and not x.is_symlink())
    for x in root.iterdir():x.unlink() if x.is_symlink() or x.is_file() else shutil.rmtree(x)
    return size
def _complete_package(assets:Path,base:str)->bool:return all((assets/f'{pre}{base}{suffix}').exists() for pre,suffix in (('', '.mp4'),('v','.mp4'),('', '.jpg'),('v','.jpg'),('', '.json')))
def reframe_fingerprint(event:dict[str,Any],shots:dict[str,dict[str,Any]],width:int,height:int)->str:
    """Vertical-only cache key.  Semantic/horizontal stages intentionally stay outside it."""
    plan=event.get('visual',{}).get('shot_focus_plan',event.get('shot_focus_plan',[]))
    shot_identity=[{'shot_id':sid,'start':shots.get(sid,{}).get('start_seconds'),'end':shots.get(sid,{}).get('end_seconds')} for sid in event.get('source_shot_ids',[])]
    return fingerprint({'reframe_algorithm_version':REFRAME_ALGORITHM_VERSION,'shot_focus_schema_version':SHOT_FOCUS_SCHEMA_VERSION,'local_detector_version':LOCAL_DETECTOR_VERSION,'crop':{'safe_margin':SAFE_MARGIN,'aspect_ratio':'3:4','width':width,'height':height},'source_shots':shot_identity,'semantic_focus_plan':plan})
def _vertical_reuse_valid(assets:Path,base:str,reframe_fp:str)->bool:
    meta=assets/f'{base}.json'
    try:
        data=json.loads(meta.read_text()); return _complete_package(assets,base) and data.get('visual',{}).get('final_vertical',{}).get('reframe_fingerprint')==reframe_fp and data.get('visual',{}).get('final_vertical',{}).get('reframe_algorithm_version')==REFRAME_ALGORITHM_VERSION
    except (OSError,json.JSONDecodeError): return False
def _retire_package(assets:Path,base:str)->None:
    """Remove a stale complete package only after its horizontal source is copied to work."""
    for pre,suffix in (('', '.mp4'),('v','.mp4'),('', '.jpg'),('v','.jpg'),('', '.json')): (assets/f'{pre}{base}{suffix}').unlink(missing_ok=True)
def _remove_incomplete_assets(assets:Path)->None:
    for file in assets.glob('*'):
        if file.is_file():
            base=(file.name[1:] if file.name.startswith('v') else file.name).rsplit('.',1)[0]
            if not _complete_package(assets,base):file.unlink()
def finalize_pilot(input_dir:Path,window_id:str,keep_debug_artifacts:bool=False)->dict[str,Any]:
    root=input_dir.resolve().parents[1]; movie_id=input_dir.name; run=root/'runs'/movie_id; pilot=run/'broll-pilot-v1'/window_id; candidates=json.loads((pilot/'candidates.json').read_text()).get('candidates',[]); assets=run/'assets'; assets.mkdir(parents=True,exist_ok=True); _remove_incomplete_assets(assets); work=run/'.work'; work.mkdir(parents=True,exist_ok=True); review_dir=run/'review'; shots_path=run/'visual-smoke-v1'/'shots.jsonl'; shots={x['shot_id']:x for x in (json.loads(y) for y in shots_path.read_text().splitlines() if y.strip())} if shots_path.exists() else {}
    movie=input_dir/'movie.mp4'; source=cv2.VideoCapture(str(movie)); width,height=int(source.get(cv2.CAP_PROP_FRAME_WIDTH)),int(source.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=source.get(cv2.CAP_PROP_FPS) or 24.; source.release(); ledger=ProcessingLedger(run,movie_id,{'finalization_version':'3e.2.1','reframe_algorithm_version':REFRAME_ALGORITHM_VERSION,'movie_code':movie_code(run,movie_id)}); completed=review=reused=review_reused=failed_retryable=failed_final=0
    for e in candidates:
        if e.get('editorial',{}).get('decision')!='KEEP' or e.get('editorial',{}).get('status')!='VALIDATED':continue
        # Do not include reframe config here: register() would incorrectly stale
        # completed horizontal/semantic work.  Vertical gets its own fingerprint.
        fp=fingerprint({'range':[e['start_frame'],e['end_frame_exclusive']],'shots':e.get('source_shot_ids',[]),'semantic':e.get('visual',{}),'version':'horizontal-v1'}); ledger.register(e,fp); aid,slug=asset_identity(run,movie_id,e); base=f'{aid}-{slug}'; final=[assets/f'{base}.mp4',assets/f'v{base}.mp4',assets/f'{base}.jpg',assets/f'v{base}.jpg',assets/f'{base}.json']; eid=e['visual_event_id']; expected=(e['end_frame_exclusive']-e['start_frame'])/fps; reframe_fp=reframe_fingerprint(e,shots,width,height)
        if _vertical_reuse_valid(assets,base,reframe_fp): reused+=1; ledger.stage(eid,'finalization','COMPLETE',decision='PASS',asset_hub_ready=True,reused=True,reframe_fingerprint=reframe_fp); continue
        prior=ledger.data['events'][eid]['stages']
        if prior.get('vertical_validation',{}).get('status') == 'COMPLETE' and prior['vertical_validation'].get('decision') == 'REVIEW_VERTICAL' and prior['vertical_validation'].get('reframe_fingerprint') == reframe_fp:
            review+=1; review_reused+=1; ledger.stage(eid,'finalization','COMPLETE',decision='REVIEW_VERTICAL',asset_hub_ready=False,reused=True,reframe_fingerprint=reframe_fp); continue
        stage=work/eid; stage.mkdir(parents=True,exist_ok=True); h,v,ht,vt,md=[stage/x.name for x in final]; old=pilot/'exports'/f"{e['candidate_id']}.mp4"; ledger.stage(eid,'horizontal_export','RUNNING')
        # A stale package can still donate its verified horizontal asset, but no
        # stale vertical member may remain in assets during replacement.
        if (assets/f'{base}.mp4').exists(): shutil.copy2(assets/f'{base}.mp4',h); _retire_package(assets,base); ledger.stage(eid,'horizontal_export','COMPLETE',reused=True)
        elif old.exists():shutil.copy2(old,h)
        else:subprocess.run(ffmpeg_export_command(movie,e,h,fps),check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        horizontal=probe(h,width,height,expected,e['end_frame_exclusive']-e['start_frame']); ledger.stage(eid,'horizontal_validation','COMPLETE' if horizontal['status']=='PASS' else 'FAILED_RETRYABLE',validation=horizontal['status'])
        if horizontal['status']!='PASS':continue
        vertical=None; plan=[]; execution_error=None
        for attempt,strategy in enumerate(('subject_focus','interaction_aware'),1):
            try:
                plan=build_shot_crop_plan(h,e,shots,width,height,strategy=strategy); v.unlink(missing_ok=True); ledger.stage(eid,'vertical_reframe','RUNNING',attempt=attempt,strategy=strategy,reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,reframe_fingerprint=reframe_fp,plan=plan); render_vertical(h,v,e,plan); vertical=validate_vertical(v,expected,plan,e)
                ledger.stage(eid,'vertical_reframe','COMPLETE',attempt=attempt,strategy=strategy,reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,reframe_fingerprint=reframe_fp)
            except (OSError,RuntimeError,cv2.error,subprocess.CalledProcessError) as error:
                execution_error=str(error); ledger.stage(eid,'vertical_reframe','FAILED_RETRYABLE',attempt=attempt,error=execution_error,reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,reframe_fingerprint=reframe_fp); break
            if vertical['status']=='PASS':break
        if execution_error:
            failed_retryable+=1; ledger.stage(eid,'vertical_validation','FAILED_RETRYABLE',reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,reframe_fingerprint=reframe_fp,error=execution_error); ledger.stage(eid,'finalization','FAILED_RETRYABLE',asset_hub_ready=False,error=execution_error); continue
        decision='PASS' if vertical and vertical['status']=='PASS' else 'REVIEW_VERTICAL'
        ledger.stage(eid,'vertical_validation','COMPLETE',decision=decision,reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,reframe_fingerprint=reframe_fp,validation=vertical)
        if decision == 'REVIEW_VERTICAL':
            review+=1; v.unlink(missing_ok=True); review_dir.mkdir(parents=True,exist_ok=True); write_json(review_dir/f'{base}.json',{'schema_version':'vertical_review_v1','asset':{'id':aid,'source_asset_id':eid},'asset_hub_ready':False,'analysis':{'semantic_ready':False,'final_asset_semantics_validated':False},'reframe_algorithm_version':REFRAME_ALGORITHM_VERSION,'reframe_fingerprint':reframe_fp,'validation':vertical}); ledger.stage(eid,'finalization','COMPLETE',decision='REVIEW_VERTICAL',asset_hub_ready=False,reframe_fingerprint=reframe_fp); safe_cleanup(work,keep_debug_artifacts); continue
        htime=thumbnail(h,ht); ledger.stage(eid,'horizontal_thumbnail','COMPLETE'); vtime=thumbnail(v,vt); ledger.stage(eid,'vertical_thumbnail','COMPLETE',reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,reframe_fingerprint=reframe_fp); data={'asset_metadata_v1':'asset_metadata_v1','asset':{'id':aid,'slug':slug,'source_asset_id':eid,'source_movie_id':movie_id},'media':{'filename':h.name,'sha256':sha256_file(h),'duration_seconds':expected,'width':width,'height':height,'fps':fps,'orientation':'landscape','aspect_ratio':'other','horizontal':{'file':h.name,'thumbnail':ht.name,'validated':True},'vertical':{'file':v.name,'thumbnail':vt.name,'sha256':sha256_file(v),'aspect_ratio':'3:4','validated':True}},'analysis':{'semantic_ready':True,'final_asset_semantics_validated':True,'source_video_analyzed':True,'profile':'pilot-finalization-3e.2','producer':'movie_broll','producer_version':'0.1.0','generated_at':'durable'},'source_timeline':{'start_seconds':e['start_seconds'],'end_seconds':e['end_seconds'],'visual_event_id':eid,'shot_ids':e.get('source_shot_ids',[])},'visual':{'source_horizontal':e.get('visual',{}),'people':e.get('people',[]),'relationships':e.get('relationships',[]),'final_vertical':{'reframe_algorithm_version':REFRAME_ALGORITHM_VERSION,'shot_focus_schema_version':SHOT_FOCUS_SCHEMA_VERSION,'local_detector_version':LOCAL_DETECTOR_VERSION,'reframe_fingerprint':reframe_fp,'reframe':{'shots':plan,'attempts':attempt},'validation':vertical}},'audio':{'speech_present':{'value':False,'source':'export_contract','confidence':1.0}},'narrative':e.get('narrative',{}),'editorial':e.get('editorial',{}),'export':{'orientation':'landscape','aspect_ratio':'other','reframe_applied':True,'reframe_profile':'local shot-aware subject track 3:4 crop'},'thumbnail':{'filename':ht.name,'timestamp_seconds':htime,'vertical_filename':vt.name,'vertical_timestamp_seconds':vtime}}; write_json(md,data)
        for src,dst in zip((h,v,ht,vt,md),final):shutil.move(str(src),dst)
        old.unlink(missing_ok=True); ledger.stage(eid,'metadata','COMPLETE',path=str(final[-1])); ledger.stage(eid,'cleanup','COMPLETE',removed_bytes=safe_cleanup(work,keep_debug_artifacts)); ledger.stage(eid,'finalization','COMPLETE',decision='PASS',asset_hub_ready=True,reframe_fingerprint=reframe_fp); completed+=1
    final_bytes=sum(x.stat().st_size for x in assets.glob('*') if x.is_file()); temp_bytes=sum(x.stat().st_size for x in work.rglob('*') if x.is_file()); status='PARTIAL' if failed_retryable or failed_final else 'COMPLETE'; ledger.summary(status=status,final_assets=completed,vertical_review=review,vertical_review_reused=review_reused,vertical_failed_retryable=failed_retryable,vertical_failed_final=failed_final,disk={'final_assets_bytes':final_bytes,'temporary_bytes':temp_bytes}); return {'status':status,'completed':completed,'review':review,'reused':reused,'review_reused':review_reused,'failed_retryable':failed_retryable,'failed_final':failed_final,'assets':assets,'review_dir':review_dir}
