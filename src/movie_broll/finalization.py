"""Resumable final packages and local shot-level 3:4 reframing."""
from __future__ import annotations
import json, re, shutil, subprocess, unicodedata, urllib.request, sys
from pathlib import Path
from typing import Any, Callable
import cv2
import numpy as np
from .broll_pilot import ffmpeg_export_command, probe
from .processing_ledger import ProcessingLedger, fingerprint
from .utils import sha256_file, write_json

MOVIE_CODES={"romper-el-circulo":"rc"}; SAFE_MARGIN=.08
# This is deliberately persistent: it is part of every vertical-only reuse key.
REFRAME_ALGORITHM_VERSION="3e.2.3.5-source-absolute-geometry-v3"
SHOT_FOCUS_SCHEMA_VERSION="shot_focus_plan_v1"
LOCAL_DETECTOR_VERSION="yolov5n-onnx-person+haar-face-v1"
PERSON_CANDIDATE_CONFIDENCE=.05; PERSON_NMS_IOU=.45
PERSON_MODEL_ID="yolov5n"; PERSON_MODEL_NAME="yolov5n.onnx"; PERSON_MODEL_VERSION="yolov5-v7.0"; PERSON_WEIGHTS_NAME="yolov5n.pt"; PERSON_WEIGHTS_URL="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt"; YOLOV5_EXPORT_REPOSITORY="https://github.com/ultralytics/yolov5.git"
VERTICAL_VALIDATION_VERSION="3e.2.2-interaction-scope-v1"
FOCUS_SUBJECTS={"woman","man","multiple_people","action_region","environment","unclear"}
INTERACTION_REQUIREMENTS={"none","sequence","simultaneous","unclear"}
_PERSON_RUNTIME:dict[str,Any]|None=None
_FACE_RUNTIME:dict[str,Any]={'available':hasattr(cv2,'CascadeClassifier'),'implementation':'opencv_haar_frontalface','inference_executed':False,'failure_reason':None if hasattr(cv2,'CascadeClassifier') else 'OpenCV Haar CascadeClassifier is unavailable'}
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
    """Project-owned cache works both from source and an installed editable wheel."""
    roots=[Path.cwd(),*Path(__file__).resolve().parents]
    root=next((x for x in roots if (x/'pyproject.toml').is_file() and (x/'src').is_dir()),Path.cwd())
    return root/'cache'/'models'/'movie-broll'/PERSON_MODEL_NAME
def _weights_path()->Path: return _model_path().with_name(PERSON_WEIGHTS_NAME)
def _missing_detector_dependencies()->list[str]:
    import importlib.util
    required={'torch':'torch','torchvision':'torchvision','onnx':'onnx','onnxscript':'onnxscript','Pillow':'PIL','PyYAML':'yaml','scipy':'scipy','pandas':'pandas','requests':'requests','tqdm':'tqdm','matplotlib':'matplotlib','seaborn':'seaborn','IPython':'IPython','setuptools':'pkg_resources'}
    return [name for name,module in required.items() if importlib.util.find_spec(module) is None]
def _command_failure(step:str,result:subprocess.CompletedProcess[str])->RuntimeError:
    detail=(result.stderr or result.stdout or '').strip().splitlines()
    tail='\n'.join(detail[-12:]) or 'no subprocess output'
    return RuntimeError(f'person detector preflight failed: {step} exited {result.returncode}: {tail}')
def _export_yolov5n(weights:Path,target:Path)->None:
    """Use the official, pinned YOLOv5 exporter; never guess an ONNX asset URL."""
    source=target.parent/'.yolov5-export-source'; output=target.with_suffix('.export.tmp.onnx')
    # An interrupted export may leave this owned scratch directory behind.
    if source.exists(): shutil.rmtree(source)
    try:
        clone=subprocess.run(['git','clone','--depth','1','--branch','v7.0',YOLOV5_EXPORT_REPOSITORY,str(source)],text=True,capture_output=True)
        if clone.returncode: raise _command_failure('official YOLOv5 v7.0 clone',clone)
        # YOLOv5 v7.0 predates Torch 2.6's secure weights-only default. The
        # checkpoint is the explicitly pinned official release acquired above.
        experimental=source/'models'/'experimental.py'; text=experimental.read_text(); old="torch.load(attempt_download(w), map_location='cpu')"
        if old not in text: raise RuntimeError('person detector preflight failed: pinned YOLOv5 exporter load hook changed unexpectedly')
        experimental.write_text(text.replace(old,"torch.load(attempt_download(w), map_location='cpu', weights_only=False)"))
        export=subprocess.run([sys.executable,str(source/'export.py'),'--weights',str(weights),'--include','onnx','--imgsz','640','640','--device','cpu'],cwd=source,text=True,capture_output=True)
        if export.returncode: raise _command_failure('YOLOv5n ONNX export',export)
        produced=next((x for x in (weights.with_suffix('.onnx'),source/f'{weights.stem}.onnx') if x.is_file()),None)
        if produced is None: raise RuntimeError('official YOLOv5 export did not produce ONNX: '+('\n'.join(((export.stderr or export.stdout or '')).splitlines()[-12:])))
        shutil.move(str(produced),output); output.replace(target)
    except OSError as error:
        output.unlink(missing_ok=True); raise RuntimeError(f'person detector preflight failed: official exporter execution failed: {error}') from error
    finally: shutil.rmtree(source,ignore_errors=True)
def person_detector_preflight(provision:bool=True)->dict[str,Any]:
    """Provision and load the project-owned ONNX model before asset processing."""
    path=_model_path(); meta=path.with_suffix('.json')
    if not path.is_file() and provision:
        missing=_missing_detector_dependencies()
        if missing: raise RuntimeError("person detector preflight failed: missing detector dependencies: "+', '.join(missing)+". Install project detector extra: pip install '.[detector]'")
        path.parent.mkdir(parents=True,exist_ok=True); weights=_weights_path(); temp=weights.with_suffix('.download.tmp')
        try:
            if not weights.is_file():
                with urllib.request.urlopen(PERSON_WEIGHTS_URL,timeout=60) as source, temp.open('wb') as target: shutil.copyfileobj(source,target)
                temp.replace(weights)
            _export_yolov5n(weights,path)
        except (OSError,urllib.error.URLError) as error:
            temp.unlink(missing_ok=True); raise RuntimeError(f'person detector preflight failed: cannot provision official {PERSON_WEIGHTS_NAME} at {weights}: {error}') from error
    if not path.is_file() or path.stat().st_size < 1024: raise RuntimeError(f'person detector preflight failed: required model is missing or incomplete: {path}')
    digest=sha256_file(path)
    try:
        net=cv2.dnn.readNetFromONNX(str(path)); net.setInput(cv2.dnn.blobFromImage(np.zeros((64,64,3),dtype=np.uint8),1/255.,(640,640),swapRB=True)); output=net.forward(); smoke_shape=list(np.asarray(output).shape)
        if np.asarray(output).size < 6: raise RuntimeError('ONNX smoke inference returned no usable detections tensor')
    except cv2.error as error: raise RuntimeError(f'person detector preflight failed: OpenCV cannot load {path}: {error}') from error
    write_json(meta,{'model_id':PERSON_MODEL_ID,'model_format':'onnx','model_version':PERSON_MODEL_VERSION,'weights_source':PERSON_WEIGHTS_URL,'sha256':digest})
    global _PERSON_RUNTIME; _PERSON_RUNTIME={'model_id':PERSON_MODEL_ID,'model_format':'onnx','model_version':PERSON_MODEL_VERSION,'model_path':str(path),'model_sha256':digest,'backend':'opencv_dnn_cpu','loaded':True,'smoke_inference_passed':True,'smoke_output_shape':smoke_shape,'inference_executed':False}
    return dict(_PERSON_RUNTIME)
def _yolo_people(frame:np.ndarray)->list[dict[str,Any]]:
    """Best-effort CPU YOLO ONNX inference when the documented local model exists."""
    path=_model_path()
    if not path.is_file(): raise RuntimeError(f'person detector unavailable: required model is missing: {path}')
    try:
        letterboxed,transform=letterbox(frame); net=cv2.dnn.readNetFromONNX(str(path)); blob=cv2.dnn.blobFromImage(letterboxed,1/255.,(640,640),swapRB=True); net.setInput(blob); out=np.squeeze(net.forward())
        if _PERSON_RUNTIME is not None: _PERSON_RUNTIME['inference_executed']=True
        if out.ndim==3: out=out[0]
        if out.ndim==2 and out.shape[1] < 6 and out.shape[0] >= 6: out=out.T
        boxes=[]; scores=[]; records=[]
        for row in out:
            if len(row)<85: continue
            objectness=float(row[4]); probs=np.asarray(row[5:],dtype=float); best=int(np.argmax(probs)); class_probability=float(probs[best]); score=objectness*class_probability
            if best != 0 or score<PERSON_CANDIDATE_CONFIDENCE: continue
            cx,cy,w,h=(float(v) for v in row[:4]); box=unletterbox_bbox({'x':cx-w/2,'y':cy-h/2,'width':w,'height':h},transform)
            if box['width']<=1 or box['height']<=1: continue
            boxes.append([int(box['x']),int(box['y']),int(box['width']),int(box['height'])]); scores.append(score); records.append({'bbox':box,'face_visible':False,'confidence':score,'detector':'yolo_person','class_id':0,'class_name':'person','objectness':objectness,'class_probability':class_probability,'preprocessing':transform})
        keep=cv2.dnn.NMSBoxes(boxes,scores,PERSON_CANDIDATE_CONFIDENCE,PERSON_NMS_IOU) if boxes else []
        return [records[int(i)] for i in np.asarray(keep).reshape(-1)]
    except cv2.error as error: raise RuntimeError(f'person detector inference failed: {error}') from error
def detect_people(frame:np.ndarray)->list[dict[str,Any]]:
    """Local face geometry plus optional standalone YOLO person geometry; never HOG-only."""
    people=_yolo_people(frame)
    faces=[]
    try:
        if not hasattr(cv2,'CascadeClassifier'): raise RuntimeError('OpenCV Haar CascadeClassifier is unavailable')
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml'); faces=cascade.detectMultiScale(gray,1.1,4,minSize=(20,20)) if not cascade.empty() else []; _FACE_RUNTIME.update(available=not cascade.empty(),inference_executed=not cascade.empty(),failure_reason=None if not cascade.empty() else 'Haar cascade unavailable')
    except (cv2.error,RuntimeError) as error: _FACE_RUNTIME.update(available=False,inference_executed=False,failure_reason=str(error))
    # Faces remain separate candidates so a focused interlocutor beats a large OTS body.
    people.extend({'bbox':{'x':float(x),'y':float(y),'width':float(w),'height':float(h)},'face_visible':True,'confidence':1.,'detector':'haar_face'} for x,y,w,h in faces)
    return people
def letterbox(frame:np.ndarray,network:int=640)->tuple[np.ndarray,dict[str,float]]:
    """YOLOv5 aspect-preserving 640-square input and reversible transform."""
    height,width=frame.shape[:2]; gain=min(network/width,network/height); resized=(round(width*gain),round(height*gain)); pad_x=(network-resized[0])/2; pad_y=(network-resized[1])/2
    image=cv2.resize(frame,resized,interpolation=cv2.INTER_LINEAR); result=cv2.copyMakeBorder(image,int(np.floor(pad_y)),int(np.ceil(pad_y)),int(np.floor(pad_x)),int(np.ceil(pad_x)),cv2.BORDER_CONSTANT,value=(114,114,114))
    return result,{'input_width':float(width),'input_height':float(height),'network_width':float(network),'network_height':float(network),'gain':gain,'pad_x':pad_x,'pad_y':pad_y}
def unletterbox_bbox(box:dict[str,float],transform:dict[str,float])->dict[str,float]:
    gain=transform['gain']; x=max(0.,min(transform['input_width'],(box['x']-transform['pad_x'])/gain)); y=max(0.,min(transform['input_height'],(box['y']-transform['pad_y'])/gain)); right=max(x,min(transform['input_width'],(box['x']+box['width']-transform['pad_x'])/gain)); bottom=max(y,min(transform['input_height'],(box['y']+box['height']-transform['pad_y'])/gain)); return {'x':x,'y':y,'width':right-x,'height':bottom-y}
def _directive(event:dict[str,Any],shot:dict[str,Any])->dict[str,Any]:
    visual=event.get('visual',{}); values=visual.get('shot_focus_plan',visual.get('shot_focus',event.get('shot_focus_plan',event.get('shot_focus',[])))) or []; direct=next((x for x in values if x.get('shot_id')==shot.get('shot_id')),{})
    subject=str(direct.get('focus_subject','unclear')).lower()
    requirement=str(direct.get('interaction_requirement','')).lower()
    if requirement not in INTERACTION_REQUIREMENTS:
        # Old focus directives only had preserve_interaction. Treat ordinary
        # dialogue/event interaction as sequence-level; reserve union framing
        # for clear physical interaction evidence.
        text=' '.join(map(str,event.get('visual',{}).get('actions',[])+event.get('visual',{}).get('visible_interactions',[])+[direct.get('focus_reason','')])).lower()
        physical=('hug','kiss','handshake','handoff','handing','fight','touch','dance','embrace','abrazo','beso','apretón','entrega','tocar')
        requirement='simultaneous' if direct.get('interaction_requires_both') or (direct.get('preserve_interaction') and any(x in text for x in physical)) else 'sequence' if direct.get('preserve_interaction') or event.get('visual',{}).get('visible_interactions') else 'none'
    return {'focus_subject':subject if subject in FOCUS_SUBJECTS else 'unclear','focus_role':direct.get('focus_role','primary'),'interaction_requirement':requirement,'preserve_interaction':requirement=='simultaneous','directive_available':bool(direct),**direct}
def _sample_frames(source_video:Path,start:float,end:float,count:int=5)->list[tuple[float,np.ndarray]]:
    """Sample source-absolute timestamps only; an empty decode is a technical error."""
    cap=cv2.VideoCapture(str(source_video)); fps=cap.get(cv2.CAP_PROP_FPS) or 24.; duration=(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)/fps; out=[]
    if start<0 or end<=start or start>=duration+.05: cap.release(); raise RuntimeError(f'source-absolute sampling interval [{start:.3f}, {end:.3f}) is outside source media duration {duration:.3f}')
    for t in np.linspace(start+(end-start)*.12,end-(end-start)*.12,max(1,count)):
        cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,round(t*fps))); ok,frame=cap.read()
        if ok: out.append((float(t),frame))
    cap.release()
    if not out: raise RuntimeError(f'source-absolute sampling decoded zero frames for [{start:.3f}, {end:.3f}) from {source_video}')
    return out
def _choose_target(found:list[dict[str,Any]],direct:dict[str,Any],width:int)->dict[str,Any]|None:
    if not found: return None
    candidates=[x for x in found if not x.get('foreground',False)] or found; faces=[x for x in candidates if x.get('face_visible',False)] or candidates; wanted=str(direct.get('focus_position',direct.get('position',''))).lower()
    if len(faces)==1: return faces[0]
    if wanted not in {'left','center','right'}: return None  # no gender/score guess for ambiguous people.
    desired={'left':width*.25,'center':width*.5,'right':width*.75}[wanted]
    # Spatial direction is the semantic bridge. Confidence only breaks nearly
    # identical spatial candidates and can never override the requested side.
    def score(x):
        b=_bbox(x); center=b['x']+b['width']/2; return (abs(center-desired)/width,-float(x.get('confidence',.5)))
    return min(faces,key=score)
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
def build_shot_crop_plan(source_video:Path,event:dict[str,Any],shots:dict[str,dict[str,Any]],width:int,height:int,detector:Callable[[np.ndarray],list[dict[str,Any]]]=detect_people,strategy:str='subject_focus',sample_count:int=5)->list[dict[str,Any]]:
    """Build geometry from source_movie on source_absolute timeline, never event clips."""
    crop=min(width,round(height*3/4)); plans=[]
    for sid in event.get('source_shot_ids',[]) or ['event']:
        shot=shots.get(sid,{}); start=float(shot.get('start_seconds',event['start_seconds'])); end=float(shot.get('end_seconds',event['end_seconds'])); direct=_directive(event,{**shot,'shot_id':sid}); samples=[]; all_boxes=[]; focus_boxes=[]; action=[]; prior=None
        sampled=_sample_frames(source_video,start,end,sample_count)
        for t,frame in sampled:
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
        focus=_union(focus_boxes) if focus_boxes else None; preserve=direct['interaction_requirement']=='simultaneous' or bool(direct.get('required_secondary_subjects',[])); composition=[focus] if focus else []
        if (preserve or strategy=='interaction_aware') and all_boxes: composition=[_union(all_boxes)]
        composition+=action; required=_union(composition) if composition else focus; impossible=bool(required and required['width']>crop*(1-2*SAFE_MARGIN) and (preserve or action))
        if required and required['width']<=crop*(1-2*SAFE_MARGIN): anchors=_smooth([(t,_anchor(required,width,crop)) for t,_ in samples] or [(start,_anchor(required,width,crop))],crop)
        elif samples: anchors=_smooth(samples,crop)
        else:
            pos=str(shot.get('primary_subject_position',event.get('visual',{}).get('primary_subject_position','center'))).lower(); anchors=[{'time':start,'x':float(crop_x(width,height,pos))}]
        required_person=direct['focus_subject'] in {'woman','man','multiple_people'}
        unresolved=required_person and focus is None
        person_count=sum(1 for x in all_boxes if x.get('detector')=='yolo_person'); face_count=sum(1 for x in all_boxes if x.get('face_visible'))
        provenance={'person_detector':dict(_PERSON_RUNTIME or {'model_id':PERSON_MODEL_ID,'loaded':False,'inference_executed':False}),'face_detector':dict(_FACE_RUNTIME)}
        plans.append({'shot_id':sid,'start_seconds':start,'end_seconds':end,'render_start_seconds':max(0.,start-float(event['start_seconds'])),'render_end_seconds':max(0.,min(float(event['end_seconds'])-float(event['start_seconds']),end-float(event['start_seconds']))),'sampling':{'media_role':'source_movie','timeline_basis':'source_absolute','requested_start':start,'requested_end':end,'sampled_frame_count':len(sampled)},'focus_subject':direct['focus_subject'],'focus_position':direct.get('focus_position','unclear'),'focus_role':direct['focus_role'],'focus_reason':direct.get('focus_reason','semantic shot focus plus local geometry'),'directive_available':direct['directive_available'],'required_person_focus':required_person,'interaction_requirement':direct['interaction_requirement'],'preserve_interaction':preserve,'required_action_region':regions,'focus_bbox':focus,'subject_bboxes':all_boxes,'person_detection_count':person_count,'face_detection_count':face_count,'geometry_resolved':bool(focus),'anchors':anchors,'x':float(np.median([x['x'] for x in anchors])),'crop_width':crop,'source_width':width,'strategy':strategy,'review_required':impossible or unresolved or not direct['directive_available'],'action_preserved':not action or bool(required),'detector_version':LOCAL_DETECTOR_VERSION if detector is detect_people and provenance['person_detector']['loaded'] else 'injected_test_detector','detector_provenance':provenance})
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
        relative=n/fps; rule=next((x for x in plan if x.get('render_start_seconds',x['start_seconds'])<=relative<x.get('render_end_seconds',x['end_seconds'])),plan[-1]); x=max(0,min(w-cw,_x_at(rule,float(event['start_seconds'])+relative))); writer.write(frame[:,x:x+cw]); n+=1
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
    stable=len(rule.get('anchors',[]))<=1 or max(abs(b['x']-a['x']) for a,b in zip(rule['anchors'],rule['anchors'][1:]))<=crop*.18+1; interaction=not rule.get('review_required',False) if rule.get('interaction_requirement')=='simultaneous' else True
    requirement='person' if required_person else 'environment' if rule.get('focus_subject')=='environment' else 'action' if rule.get('focus_subject')=='action_region' else 'none'
    present=bool(focus) if required_person else None
    safe=bool(focus) and not introduced if required_person else None
    geometry_ok=(present and safe) if required_person else True
    ok=bool(rule.get('directive_available',True)) and geometry_ok and not introduced and not empty and stable and interaction and rule.get('action_preserved',True) and action_ok
    return {'shot_id':rule['shot_id'],'focus_requirement':requirement,'focus_geometry_resolved':bool(focus),'focus_directive_available':bool(rule.get('directive_available',True)),'focus_subject_present':present,'focus_subject_safe':safe,'introduced_subject_clipping':introduced,'source_edge_exception':source_clip,'empty_space_while_clipped':empty,'interaction_preserved':interaction,'action_preserved':rule.get('action_preserved',True) and action_ok,'crop_stable':stable,'status':'PASS' if ok else 'FAIL'}
def validate_vertical(path:Path,expected:float,plan:list[dict[str,Any]],event:dict[str,Any])->dict[str,Any]:
    cap=cv2.VideoCapture(str(path)); w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS) or 24.; frames=[]
    for i in {0,max(0,count//2),max(0,count-1)}:cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,x=cap.read(); frames.extend([x] if ok else [])
    cap.release(); bars=any(float(np.mean(cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)<8))>.92 for x in frames); per=[_shot_validation(x) for x in plan]; ok=path.exists() and count>0 and w*4==h*3 and abs(count/fps-expected)<=1 and not bars and all(x['status']=='PASS' for x in per)
    sequence=[rule for rule in plan if rule.get('interaction_requirement')=='sequence']; sequence_ok=all(x['focus_subject_present'] and x['focus_subject_safe'] and x['action_preserved'] for x,rule in zip(per,plan) if rule.get('interaction_requirement')=='sequence')
    return {'status':'PASS' if ok and sequence_ok else 'REVIEW','review_reason':'REVIEW_VERTICAL' if not (ok and sequence_ok) else None,'width':w,'height':h,'aspect_ratio':'3:4' if w*4==h*3 else 'other','duration_seconds':count/fps,'black_bars':bars,'shots':per,'sequence_interaction_preserved':sequence_ok,'sequence_interaction_shots':[x['shot_id'] for x in sequence],'stable_per_shot':all(x['crop_stable'] for x in per),'semantic_retained':all(x['interaction_preserved'] for x in per)}
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
    return fingerprint({'reframe_algorithm_version':REFRAME_ALGORITHM_VERSION,'vertical_validation_version':VERTICAL_VALIDATION_VERSION,'shot_focus_schema_version':SHOT_FOCUS_SCHEMA_VERSION,'local_detector_version':LOCAL_DETECTOR_VERSION,'crop':{'safe_margin':SAFE_MARGIN,'aspect_ratio':'3:4','width':width,'height':height},'source_shots':shot_identity,'semantic_focus_plan':plan})
def _vertical_reuse_valid(assets:Path,base:str,reframe_fp:str)->bool:
    meta=assets/f'{base}.json'
    try:
        data=json.loads(meta.read_text()); final=data.get('visual',{}).get('final_vertical',{}); return _complete_package(assets,base) and final.get('reframe_fingerprint')==reframe_fp and final.get('reframe_algorithm_version')==REFRAME_ALGORITHM_VERSION and final.get('vertical_validation_version')==VERTICAL_VALIDATION_VERSION
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
    root=input_dir.resolve().parents[1]; movie_id=input_dir.name; run=root/'runs'/movie_id; pilot=run/'broll-pilot-v1'/window_id; candidates_path=pilot/'candidates.json'; candidates=json.loads(candidates_path.read_text()).get('candidates',[]); assets=run/'assets'; assets.mkdir(parents=True,exist_ok=True); _remove_incomplete_assets(assets); work=run/'.work'; work.mkdir(parents=True,exist_ok=True); review_dir=run/'review'; shots_path=run/'visual-smoke-v1'/'shots.jsonl'; shots={x['shot_id']:x for x in (json.loads(y) for y in shots_path.read_text().splitlines() if y.strip())} if shots_path.exists() else {}
    # Finalize is self-healing: stale event semantics are refreshed through the
    # existing bounded event request, never by silently using unclear geometry.
    from .broll_pilot import semantic_validate, shot_focus_compatible
    for candidate in candidates:
        candidate.setdefault('technical_shots',[{'shot_id':sid,'start_seconds':shots.get(sid,{}).get('start_seconds'),'end_seconds':shots.get(sid,{}).get('end_seconds'),'representative_image_index':i} for i,sid in enumerate(candidate.get('source_shot_ids',[]))])
    incompatible=[x for x in candidates if x.get('editorial',{}).get('decision')=='KEEP' and not shot_focus_compatible({'visual':x.get('visual',{})},x)]
    semantic_reused=0
    if incompatible:
        srt=next((input_dir/x for x in ('subtitles.srt',f'{movie_id}.srt') if (input_dir/x).is_file()),None); narrative=run/'narrative-v2'/'narrative_map.json'
        if srt is None or not narrative.is_file(): raise RuntimeError('shot-focus semantic refresh requires canonical SRT and narrative map')
        semantic_reused=semantic_validate(incompatible,input_dir/'movie.mp4',srt,narrative,pilot/'semantic_checkpoints',24.,window_id).get('reused',0)
        write_json(candidates_path,{'schema_version':'broll_pilot_candidates_v4','semantic_schema_version':'broll_semantics_v6','semantic_prompt_version':'broll_semantic_prompt_v6','window_id':window_id,'candidates':candidates})
    movie=input_dir/'movie.mp4'; source=cv2.VideoCapture(str(movie)); width,height=int(source.get(cv2.CAP_PROP_FRAME_WIDTH)),int(source.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=source.get(cv2.CAP_PROP_FPS) or 24.; source.release(); ledger=ProcessingLedger(run,movie_id,{'finalization_version':'3e.2.3.4','reframe_algorithm_version':REFRAME_ALGORITHM_VERSION,'vertical_validation_version':VERTICAL_VALIDATION_VERSION,'movie_code':movie_code(run,movie_id)}); completed=review=reused=review_reused=horizontal_reused=failed_retryable=failed_final=0
    if any(e.get('editorial',{}).get('decision')=='KEEP' and e.get('editorial',{}).get('status')=='VALIDATED' for e in candidates): person_detector_preflight()
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
        if (assets/f'{base}.mp4').exists(): shutil.copy2(assets/f'{base}.mp4',h); _retire_package(assets,base); horizontal_reused+=1; ledger.stage(eid,'horizontal_export','COMPLETE',reused=True)
        elif old.exists():shutil.copy2(old,h)
        else:subprocess.run(ffmpeg_export_command(movie,e,h,fps),check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        horizontal=probe(h,width,height,expected,e['end_frame_exclusive']-e['start_frame']); ledger.stage(eid,'horizontal_validation','COMPLETE' if horizontal['status']=='PASS' else 'FAILED_RETRYABLE',validation=horizontal['status'])
        if horizontal['status']!='PASS':continue
        vertical=None; plan=[]; execution_error=None
        for attempt,strategy in enumerate(('subject_focus','interaction_aware'),1):
            try:
                plan=build_shot_crop_plan(movie,e,shots,width,height,strategy=strategy); v.unlink(missing_ok=True); ledger.stage(eid,'vertical_reframe','RUNNING',attempt=attempt,strategy=strategy,reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,reframe_fingerprint=reframe_fp,plan=plan); render_vertical(h,v,e,plan); vertical=validate_vertical(v,expected,plan,e)
                ledger.stage(eid,'vertical_reframe','COMPLETE',attempt=attempt,strategy=strategy,reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,reframe_fingerprint=reframe_fp)
            except (OSError,RuntimeError,cv2.error,subprocess.CalledProcessError) as error:
                execution_error=str(error); ledger.stage(eid,'vertical_reframe','FAILED_RETRYABLE',attempt=attempt,error=execution_error,reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,reframe_fingerprint=reframe_fp); break
            if vertical['status']=='PASS':break
        if execution_error:
            failed_retryable+=1; ledger.stage(eid,'vertical_validation','FAILED_RETRYABLE',reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,reframe_fingerprint=reframe_fp,error=execution_error); ledger.stage(eid,'finalization','FAILED_RETRYABLE',asset_hub_ready=False,error=execution_error); continue
        decision='PASS' if vertical and vertical['status']=='PASS' else 'REVIEW_VERTICAL'
        ledger.stage(eid,'vertical_validation','COMPLETE',decision=decision,reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,reframe_fingerprint=reframe_fp,validation=vertical)
        if decision == 'REVIEW_VERTICAL':
            review+=1; v.unlink(missing_ok=True); review_dir.mkdir(parents=True,exist_ok=True); write_json(review_dir/f'{base}.json',{'schema_version':'vertical_review_v1','asset':{'id':aid,'source_asset_id':eid},'asset_hub_ready':False,'analysis':{'semantic_ready':False,'final_asset_semantics_validated':False},'shot_focus_plan':e.get('visual',{}).get('shot_focus_plan',[]),'reframe_algorithm_version':REFRAME_ALGORITHM_VERSION,'vertical_validation_version':VERTICAL_VALIDATION_VERSION,'reframe_fingerprint':reframe_fp,'validation':vertical}); ledger.stage(eid,'finalization','COMPLETE',decision='REVIEW_VERTICAL',asset_hub_ready=False,reframe_fingerprint=reframe_fp); safe_cleanup(work,keep_debug_artifacts); continue
        htime=thumbnail(h,ht); ledger.stage(eid,'horizontal_thumbnail','COMPLETE'); vtime=thumbnail(v,vt); ledger.stage(eid,'vertical_thumbnail','COMPLETE',reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,vertical_validation_version=VERTICAL_VALIDATION_VERSION,reframe_fingerprint=reframe_fp); data={'asset_metadata_v1':'asset_metadata_v1','asset':{'id':aid,'slug':slug,'source_asset_id':eid,'source_movie_id':movie_id},'media':{'filename':h.name,'sha256':sha256_file(h),'duration_seconds':expected,'width':width,'height':height,'fps':fps,'orientation':'landscape','aspect_ratio':'other','horizontal':{'file':h.name,'thumbnail':ht.name,'validated':True},'vertical':{'file':v.name,'thumbnail':vt.name,'sha256':sha256_file(v),'aspect_ratio':'3:4','validated':True}},'analysis':{'semantic_ready':True,'final_asset_semantics_validated':True,'source_video_analyzed':True,'profile':'pilot-finalization-3e.2.2','producer':'movie_broll','producer_version':'0.1.0','generated_at':'durable'},'source_timeline':{'start_seconds':e['start_seconds'],'end_seconds':e['end_seconds'],'visual_event_id':eid,'shot_ids':e.get('source_shot_ids',[])},'visual':{'source_horizontal':e.get('visual',{}),'people':e.get('people',[]),'relationships':e.get('relationships',[]),'final_vertical':{'reframe_algorithm_version':REFRAME_ALGORITHM_VERSION,'vertical_validation_version':VERTICAL_VALIDATION_VERSION,'shot_focus_schema_version':SHOT_FOCUS_SCHEMA_VERSION,'local_detector_version':LOCAL_DETECTOR_VERSION,'reframe_fingerprint':reframe_fp,'reframe':{'shots':plan,'attempts':attempt},'validation':vertical}},'audio':{'speech_present':{'value':False,'source':'export_contract','confidence':1.0}},'narrative':e.get('narrative',{}),'editorial':e.get('editorial',{}),'export':{'orientation':'landscape','aspect_ratio':'other','reframe_applied':True,'reframe_profile':'local shot-aware subject track 3:4 crop'},'thumbnail':{'filename':ht.name,'timestamp_seconds':htime,'vertical_filename':vt.name,'vertical_timestamp_seconds':vtime}}; write_json(md,data)
        for src,dst in zip((h,v,ht,vt,md),final):shutil.move(str(src),dst)
        old.unlink(missing_ok=True); (review_dir/f'{base}.json').unlink(missing_ok=True); ledger.stage(eid,'metadata','COMPLETE',path=str(final[-1])); ledger.stage(eid,'cleanup','COMPLETE',removed_bytes=safe_cleanup(work,keep_debug_artifacts)); ledger.stage(eid,'finalization','COMPLETE',decision='PASS',asset_hub_ready=True,reframe_fingerprint=reframe_fp); completed+=1
    final_bytes=sum(x.stat().st_size for x in assets.glob('*') if x.is_file()); temp_bytes=sum(x.stat().st_size for x in work.rglob('*') if x.is_file()); status='PARTIAL' if failed_retryable or failed_final else 'COMPLETE'; ledger.summary(status=status,final_assets=completed,vertical_review=review,vertical_reused=reused,vertical_review_reused=review_reused,horizontal_reused=horizontal_reused,semantic_reused=semantic_reused,vertical_failed_retryable=failed_retryable,vertical_failed_final=failed_final,disk={'final_assets_bytes':final_bytes,'temporary_bytes':temp_bytes}); return {'status':status,'completed':completed,'review':review,'reused':reused,'review_reused':review_reused,'horizontal_reused':horizontal_reused,'semantic_reused':semantic_reused,'failed_retryable':failed_retryable,'failed_final':failed_final,'assets':assets,'review_dir':review_dir}
