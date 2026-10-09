"""Resumable final packages and local shot-level 3:4 reframing."""
from __future__ import annotations
import copy, json, re, shutil, subprocess, unicodedata, urllib.request, sys
from pathlib import Path
from typing import Any, Callable
import cv2
import numpy as np
from .broll_pilot import boundary_validation, ffmpeg_export_command, probe
from .processing_ledger import ProcessingLedger, fingerprint
from .publication import apply_publication_projection, is_atlas_ready, is_publish_ready
from .utils import sha256_file, write_json
from .active_picture import crop_frame

MOVIE_CODES={"romper-el-circulo":"rc"}; SAFE_MARGIN=.08
# This is deliberately persistent: it is part of every vertical-only reuse key.
REFRAME_ALGORITHM_VERSION="phase-f-canonical-person-bound-reframe-v4"
SHOT_FOCUS_SCHEMA_VERSION="shot_focus_plan_v2"
LOCAL_DETECTOR_VERSION="yolov5n-onnx-person+haar-face-v1"
PERSON_CANDIDATE_CONFIDENCE=.05; PERSON_NMS_IOU=.45
PERSON_MODEL_ID="yolov5n"; PERSON_MODEL_NAME="yolov5n.onnx"; PERSON_MODEL_VERSION="yolov5-v7.0"; PERSON_WEIGHTS_NAME="yolov5n.pt"; PERSON_WEIGHTS_URL="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt"; YOLOV5_EXPORT_REPOSITORY="https://github.com/ultralytics/yolov5.git"
VERTICAL_VALIDATION_VERSION="phase-f-face-head-authoritative-v3"
# Face/head constraints change pixels, so this compatibility value is also a
# vertical-only invalidation boundary.
REFRAME_VALIDATION_COMPATIBILITY_VERSION="face-head-constraint-validation-v3"
POST_RENDER_AUDIT_VERSION="post_render_vertical_audit_v3_editorial_geometry"
POST_RENDER_SAMPLES_PER_SHOT=5
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
    """Return the deterministic package identity for one canonical Visual Event.

    The registry records the identity chosen by the global event manifest; it
    must never allocate an ordinal based on whichever event happens to render
    first.
    """
    ordinal=event.get('timeline_ordinal')
    if isinstance(ordinal,bool) or not isinstance(ordinal,int) or ordinal < 1:
        raise ValueError(f"Visual Event {event.get('visual_event_id', '<unknown>')} has no valid timeline_ordinal")
    path=run/'asset_registry.json'
    data=json.loads(path.read_text()) if path.exists() else {'schema_version':'asset_registry_v1','movie_id':movie_id,'events':{}}
    events=data.setdefault('events',{})
    eid=event['visual_event_id']; code=movie_code(run,movie_id)
    expected_asset_id=f'{code}{ordinal:03d}'
    if event.get('producer_window_id'):
        from .asset_window_handoff import window_identity
        key=window_identity(event['source_visual_event_id'], event['start_frame'], event['end_frame_exclusive'])
        if event['producer_window_id'] != key or eid != event['source_visual_event_id'] + ':window:' + key:
            raise ValueError('producer window identity mismatch')
        expected_asset_id += 'w' + key
    text=event.get('editorial',{}).get('standalone_meaning_es') or event.get('visual',{}).get('summary_es') or 'momento visual'
    entry=events.get(eid)
    if event.get('producer_window_id'):
        if any(k != eid and v.get('asset_id') == expected_asset_id for k,v in events.items()):
            raise ValueError('producer window asset identity collision')
        if entry is not None and (not isinstance(entry,dict) or entry.get('asset_id') != expected_asset_id):
            raise ValueError('immutable producer window registry conflict')
    if not isinstance(entry,dict) or entry.get('asset_id') != expected_asset_id:
        # A completion-order registry entry is incompatible with the canonical
        # timeline mapping.  Replace it rather than silently reusing it.
        entry={'asset_id':expected_asset_id,'slug':slugify(str(text))}
        events[eid]=entry
    entry['timeline_ordinal']=ordinal
    if event.get('producer_window_id'):
        entry.update(source_visual_event_id=event['source_visual_event_id'], producer_window_id=event['producer_window_id'], start_frame=event['start_frame'], end_frame_exclusive=event['end_frame_exclusive'])
    if not isinstance(entry.get('slug'),str) or not entry['slug']:
        entry['slug']=slugify(str(text))
    write_json(path,data)
    return entry['asset_id'],entry['slug']
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
    from .face_safe import preflight as face_preflight
    _PERSON_RUNTIME['face_detector']=face_preflight(provision=provision)
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
def _sample_frames(source_video:Path,start:float,end:float,count:int=5,
                   active_picture:dict[str,Any]|None=None)->list[tuple[float,np.ndarray]]:
    """Sample source-absolute timestamps, then immediately enter active pixels."""
    cap=cv2.VideoCapture(str(source_video)); fps=cap.get(cv2.CAP_PROP_FPS) or 24.; duration=(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)/fps; out=[]
    if start<0 or end<=start or start>=duration+.05: cap.release(); raise RuntimeError(f'source-absolute sampling interval [{start:.3f}, {end:.3f}) is outside source media duration {duration:.3f}')
    for t in np.linspace(start+(end-start)*.12,end-(end-start)*.12,max(1,count)):
        cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,round(t*fps))); ok,frame=cap.read()
        if ok: out.append((float(t),crop_frame(frame,active_picture)))
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

def _binding_candidates(found:list[dict[str,Any]],width:int)->list[dict[str,Any]]:
    """Assign the same shot-local P1/P2/... ordering used by semantic evidence."""
    rows=[]
    for item in found:
        if item.get('detector') != 'yolo_person':
            continue
        box=_bbox(item)
        if box['width']<=1 or box['height']<=1:
            continue
        center=box['x']+box['width']/2
        rows.append({
            'person_id':None,
            'box':box,
            '_center':center,
            '_y':box['y'],
            '_confidence':float(item.get('confidence',0.)),
        })
    rows.sort(key=lambda x:(x['_center'],x['_y'],-x['_confidence']))
    for index,row in enumerate(rows,1):
        row['person_id']=f'P{index}'
        row.pop('_center',None)
        row.pop('_y',None)
        row.pop('_confidence',None)
    return rows


def resolve_shot_person_id(identity: str, shot_id: str, candidates: list[dict[str,Any]], *, allow_legacy_local:bool=False) -> str:
    """Resolve an immutable canonical ID in its owning shot, never by position.

    The planner explicitly opts into bare P IDs for legacy local directives.
    Canonical IDs must contain exactly one owner and one positive P ordinal.
    """
    if not isinstance(identity,str):
        raise ValueError('malformed_person_identity')
    match=re.fullmatch(r'([A-Za-z0-9_][A-Za-z0-9_.-]*):(P[1-9][0-9]*)',identity)
    if match:
        owner,local=match.groups()
        if owner != shot_id:
            raise ValueError('cross_shot_person_identity')
    elif allow_legacy_local and re.fullmatch(r'P[1-9][0-9]*',identity):
        local=identity
    else:
        raise ValueError('malformed_person_identity')
    targets=[row for row in candidates if row.get('person_id')==local]
    if len(targets)!=1:
        raise ValueError('ambiguous_local_target' if targets else 'missing_local_target')
    box=targets[0].get('box',{})
    if not all(np.isfinite(box.get(k,float('nan'))) for k in ('x','y','width','height')) or box['width']<=1 or box['height']<=1:
        raise ValueError('unusable_reference_geometry')
    if any(row is not targets[0] and _bbox_iou(box,row.get('box',{})) >= .95 for row in candidates):
        raise ValueError('ambiguous_reference_geometry')
    return local


def _bbox_iou(a:dict[str,float],b:dict[str,float])->float:
    left=max(a['x'],b['x'])
    top=max(a['y'],b['y'])
    right=min(a['x']+a['width'],b['x']+b['width'])
    bottom=min(a['y']+a['height'],b['y']+b['height'])
    inter=max(0.,right-left)*max(0.,bottom-top)
    if inter<=0:
        return 0.
    area_a=max(1.,a['width']*a['height'])
    area_b=max(1.,b['width']*b['height'])
    return inter/max(1.,area_a+area_b-inter)


def _identity_track_cost(previous:dict[str,float],candidate:dict[str,float],source_width:int)->tuple[float,dict[str,float]]:
    """Geometry-only identity continuity inside one technical shot."""
    import math

    pc=previous['x']+previous['width']/2
    cc=candidate['x']+candidate['width']/2
    center_distance=abs(cc-pc)/max(1.,float(source_width))
    iou=_bbox_iou(previous,candidate)

    previous_area=max(1.,previous['width']*previous['height'])
    candidate_area=max(1.,candidate['width']*candidate['height'])
    area_change=abs(math.log(candidate_area/previous_area))

    previous_ar=max(.001,previous['width']/max(1.,previous['height']))
    candidate_ar=max(.001,candidate['width']/max(1.,candidate['height']))
    aspect_change=abs(math.log(candidate_ar/previous_ar))

    # IoU is useful for short movement; centre/scale continuity allows motion
    # between sparse samples without redefining editorial identity.
    cost=(
        (1.-iou)*.50
        + center_distance*1.75
        + min(area_change,2.)*.18
        + min(aspect_change,2.)*.10
    )

    return cost,{
        'iou':iou,
        'center_distance_ratio':center_distance,
        'area_log_change':area_change,
        'aspect_log_change':aspect_change,
    }


def _track_bound_target(previous:dict[str,float],found:list[dict[str,Any]],width:int)->tuple[dict[str,float]|None,dict[str,Any]]:
    """Continue one already-bound person; never fall back to semantic position."""
    candidates=[x['box'] for x in _binding_candidates(found,width)]

    if not candidates:
        return None,{'resolved':False,'reason':'no_person_candidates'}

    ranked=[]
    for candidate in candidates:
        cost,metrics=_identity_track_cost(previous,candidate,width)
        ranked.append((cost,candidate,metrics))

    ranked.sort(key=lambda x:x[0])
    cost,candidate,metrics=ranked[0]
    if len(ranked)>1 and ranked[1][0]-cost < .05:
        return None,{'resolved':False,'reason':'ambiguous_identity_continuity'}

    # Reject implausible discontinuities rather than silently switching people.
    plausible=(
        metrics['iou'] >= .01
        or metrics['center_distance_ratio'] <= .18
    )
    scale_ok=metrics['area_log_change'] <= 1.40
    aspect_ok=metrics['aspect_log_change'] <= .90

    if not (plausible and scale_ok and aspect_ok):
        return None,{
            'resolved':False,
            'reason':'identity_discontinuity',
            'cost':round(cost,5),
            **{k:round(v,5) for k,v in metrics.items()},
        }

    return candidate,{
        'resolved':True,
        'reason':'geometry_continuity',
        'cost':round(cost,5),
        **{k:round(v,5) for k,v in metrics.items()},
    }


def _bound_single_subject_samples(
    sampled:list[tuple[float,np.ndarray]],
    detections:list[list[dict[str,Any]]],
    target_person_id:str,
    width:int,
)->tuple[list[tuple[float,dict[str,float]]],bool,dict[str,Any]]:
    """Seed at the semantic reference midpoint, then track both directions."""
    if not sampled or len(sampled)!=len(detections):
        return [],False,{'reason':'invalid_samples'}

    midpoint=(sampled[0][0]+sampled[-1][0])/2
    seed_index=min(
        range(len(sampled)),
        key=lambda i:abs(sampled[i][0]-midpoint),
    )

    seed_candidates=_binding_candidates(detections[seed_index],width)
    seed=next(
        (x['box'] for x in seed_candidates if x['person_id']==target_person_id),
        None,
    )

    if seed is None:
        return [],False,{
            'reason':'target_not_found_at_reference_sample',
            'target_person_id':target_person_id,
            'seed_index':seed_index,
            'available_ids':[x['person_id'] for x in seed_candidates],
        }

    resolved={seed_index:seed}
    diagnostics={
        'target_person_id':target_person_id,
        'seed_index':seed_index,
        'seed_time':sampled[seed_index][0],
        'steps':[],
    }
    complete=True

    previous=seed
    for index in range(seed_index+1,len(sampled)):
        target,detail=_track_bound_target(previous,detections[index],width)
        diagnostics['steps'].append({
            'direction':'forward',
            'sample_index':index,
            'time':sampled[index][0],
            **detail,
        })
        if target is None:
            complete=False
            break
        resolved[index]=target
        previous=target

    previous=seed
    for index in range(seed_index-1,-1,-1):
        target,detail=_track_bound_target(previous,detections[index],width)
        diagnostics['steps'].append({
            'direction':'backward',
            'sample_index':index,
            'time':sampled[index][0],
            **detail,
        })
        if target is None:
            complete=False
            break
        resolved[index]=target
        previous=target

    ordered=[
        (sampled[index][0],resolved[index])
        for index in sorted(resolved)
    ]

    diagnostics['resolved_samples']=len(ordered)
    diagnostics['sample_count']=len(sampled)
    diagnostics['complete']=complete and len(ordered)==len(sampled)

    return ordered,diagnostics['complete'],diagnostics


def _bound_multiple_subject_samples(
    sampled:list[tuple[float,np.ndarray]],
    detections:list[list[dict[str,Any]]],
    target_person_ids:list[str],
    width:int,
)->tuple[list[tuple[float,dict[str,float]]],bool,dict[str,Any]]:
    """Track every Gemini-bound person and use their union as the crop target."""
    tracks=[]
    diagnostics={'target_person_ids':list(target_person_ids),'targets':[]}
    for person_id in target_person_ids:
        track,resolved,detail=_bound_single_subject_samples(
            sampled,detections,person_id,width,
        )
        tracks.append(track)
        diagnostics['targets'].append(detail)
        if not resolved:
            diagnostics['complete']=False
            return [],False,diagnostics

    by_time:dict[float,list[dict[str,float]]]={}
    for track in tracks:
        for time,box in track:
            by_time.setdefault(float(time),[]).append(box)
    if any(len(by_time.get(float(time),[])) != len(target_person_ids) for time,_ in sampled):
        diagnostics['complete']=False
        diagnostics['reason']='incomplete_multi_person_tracking'
        return [],False,diagnostics

    if any(_bbox_iou(a,b)>=.95 for boxes in by_time.values() for i,a in enumerate(boxes) for b in boxes[i+1:]):
        diagnostics.update(complete=False,reason='required_participants_collapsed')
        return [],False,diagnostics
    diagnostics['complete']=True
    return [
        (float(time),_union(by_time[float(time)]))
        for time,_ in sampled
    ],True,diagnostics


def _union(boxes:list[dict[str,float]])->dict[str,float]:
    left=min(x['x'] for x in boxes); top=min(x['y'] for x in boxes); right=max(x['x']+x['width'] for x in boxes); bottom=max(x['y']+x['height'] for x in boxes); return {'x':left,'y':top,'width':right-left,'height':bottom-top}
def _anchor(box:dict[str,float],source:int,crop:int)->float: return max(0.,min(float(source-crop),box['x']+box['width']/2-crop/2))
TRACKING_DEAD_ZONE_RATIO=.06; TRACKING_MAX_VELOCITY_RATIO_PER_SECOND=.08
def _smooth(values:list[tuple[float,float]],crop:int,start:float|None=None,end:float|None=None,
            constraints:list[dict[str,float]]|None=None)->list[dict[str,float]]:
    """Smooth anchors, with optional per-time hard crop intervals.

    Movement limits are editorial preferences.  Face/head bounds are safety
    constraints, so a needed bound projection wins over velocity limiting.
    """
    if not values:return []
    if constraints is not None:
        if len(constraints) != len(values):
            raise ValueError('face/head constraints must align with temporal anchors')

        def project(value:float, bound:dict[str,float])->float:
            return max(float(bound['min_x']), min(float(bound['max_x']), float(value)))

        raw=np.array([project(x[1], bound) for x,bound in zip(values,constraints)])
        common_low=max(float(x['min_x']) for x in constraints)
        common_high=min(float(x['max_x']) for x in constraints)
        if float(raw.max()-raw.min())<crop*TRACKING_DEAD_ZONE_RATIO and common_low <= common_high:
            x=max(common_low,min(common_high,float(np.median(raw))))
            return [{'time':float(start if start is not None else values[0][0]),'x':x,
                     'min_x':common_low,'max_x':common_high,'constrained':True}]

        output=[]; old=project(values[0][1],constraints[0]); previous=float(values[0][0])
        first={'time':previous,'x':old,'min_x':float(constraints[0]['min_x']),
               'max_x':float(constraints[0]['max_x']),'constrained':True}
        # The render starts before the first sampled frame.  Keep a separate
        # start anchor, but retain the real sampled-time anchor too; replacing
        # it made interpolation leak outside that sample's legal interval.
        if start is not None and float(start) < previous:
            output.append({**first,'time':float(start)})
        output.append(first)
        for (t,preferred),bound in zip(values[1:],constraints[1:]):
            limit=crop*TRACKING_MAX_VELOCITY_RATIO_PER_SECOND*max(0.,float(t)-previous)
            candidate=project(preferred,bound)
            if abs(candidate-old)<crop*TRACKING_DEAD_ZONE_RATIO:
                candidate=old
            # Project after velocity limiting as well: no smooth camera move
            # may leave this frame's independently computed legal interval.
            old=project(max(old-limit,min(old+limit,candidate)),bound)
            output.append({'time':float(t),'x':old,'min_x':float(bound['min_x']),
                           'max_x':float(bound['max_x']),'constrained':True})
            previous=float(t)
        if end is not None and end>previous:
            bound=constraints[-1]
            output.append({'time':float(end),'x':project(old,bound),'min_x':float(bound['min_x']),
                           'max_x':float(bound['max_x']),'constrained':True})
        return output
    raw=np.array([x[1] for x in values])
    if float(raw.max()-raw.min())<crop*TRACKING_DEAD_ZONE_RATIO:return [{'time':float(start if start is not None else values[0][0]),'x':float(np.median(raw))}]
    output=[]; old=float(raw[0]); previous=float(values[0][0]); output.append({'time':float(start if start is not None else previous),'x':old})
    for t,x in values[1:]:
        delta=float(x)-old; limit=crop*TRACKING_MAX_VELOCITY_RATIO_PER_SECOND*max(0.,float(t)-previous)
        if abs(delta)<crop*TRACKING_DEAD_ZONE_RATIO: x=old
        old=max(old-limit,min(old+limit,float(x))); output.append({'time':float(t),'x':old}); previous=float(t)
    if end is not None and end>previous: output.append({'time':float(end),'x':old})
    return output
def build_shot_crop_plan(source_video:Path,event:dict[str,Any],shots:dict[str,dict[str,Any]],width:int,height:int,detector:Callable[[np.ndarray],list[dict[str,Any]]]=detect_people,strategy:str='subject_focus',sample_count:int=5,active_picture:dict[str,Any]|None=None)->list[dict[str,Any]]:
    """Build active-picture-relative crop geometry with optional semantic binding."""
    active=active_picture or {'x':0,'y':0,'width':width,'height':height,
                              'source_width':width,'source_height':height,
                              'structural_bars':False,'detection_profile':'legacy_full_frame'}
    if (int(active['width']),int(active['height'])) != (int(width),int(height)):
        raise ValueError('crop-plan dimensions must be active-picture dimensions')
    crop=min(width,round(height*3/4))
    plans=[]

    for sid in event.get('source_shot_ids',[]) or ['event']:
        shot=shots.get(sid,{})
        start=float(shot.get('start_seconds',event['start_seconds']))
        end=float(shot.get('end_seconds',event['end_seconds']))
        direct=_directive(event,{**shot,'shot_id':sid})

        sampled=(
            _sample_frames(source_video,start,end,sample_count,active)
            if active_picture is not None
            else _sample_frames(source_video,start,end,sample_count)
        )
        detected=[detector(frame) for _,frame in sampled]

        all_boxes=[
            _bbox(item)
            for frame_found in detected
            for item in frame_found
        ]

        target_ids=direct.get('target_person_ids')
        target_ids=target_ids if isinstance(target_ids,list) else []
        target_confidence=str(direct.get('target_binding_confidence','unclear'))

        required_person=direct['focus_subject'] in {
            'woman','man','multiple_people'
        }

        binding_present='target_person_ids' in direct
        local_ids=[]
        translation_error=None
        if binding_present:
            try:
                if not sampled or not target_ids or len(set(target_ids))!=len(target_ids):
                    raise ValueError('invalid_target_person_ids')
                midpoint=(sampled[0][0]+sampled[-1][0])/2
                seed_index=min(range(len(sampled)),key=lambda i:abs(sampled[i][0]-midpoint))
                candidates=_binding_candidates(detected[seed_index],width)
                local_ids=[resolve_shot_person_id(pid,sid,candidates,allow_legacy_local=True) for pid in target_ids]
                if len(set(local_ids))!=len(local_ids):
                    raise ValueError('ambiguous_local_target')
            except (ValueError,TypeError) as error:
                translation_error=str(error)
        bound_single=(
            required_person
            and direct['focus_subject'] in {'woman','man'}
            and len(target_ids)==1
            and direct['interaction_requirement']!='simultaneous'
            and not direct.get('preserve_secondary_subject')
        )
        bound_multiple=(
            required_person
            and len(target_ids)>=2
        )

        binding_resolved=True
        binding_diagnostics={}
        tracked_boxes=[]
        target_samples=[]

        if binding_present and (translation_error or not (bound_single or bound_multiple)):
            binding_resolved=False
            binding_diagnostics={'reason':translation_error or 'insufficient_required_participants'}
            focus_boxes=[]
            anchor_samples=[]
        elif bound_single:
            tracked_boxes,binding_resolved,binding_diagnostics = \
                _bound_single_subject_samples(
                    sampled,
                    detected,
                    local_ids[0],
                    width,
                )

            focus_boxes=[box for _,box in tracked_boxes]
            target_samples=[
                {'time':float(time),'bbox':box}
                for time,box in tracked_boxes
            ]
            anchor_samples=[
                (time,_anchor(box,width,crop))
                for time,box in tracked_boxes
            ]

        elif bound_multiple:
            tracked_boxes,binding_resolved,binding_diagnostics = \
                _bound_multiple_subject_samples(
                    sampled,
                    detected,
                    local_ids,
                    width,
                )
            focus_boxes=[box for _,box in tracked_boxes]
            target_samples=[
                {'time':float(time),'bbox':box}
                for time,box in tracked_boxes
            ]
            anchor_samples=[
                (time,_anchor(box,width,crop))
                for time,box in tracked_boxes
            ]

        else:
            # Legacy or non-single-subject path. Semantic position remains a
            # fallback only when there is no explicit single-person binding.
            focus_boxes=[]
            anchor_samples=[]
            prior=None

            for (t,_),found_raw in zip(sampled,detected):
                found=[_bbox(x) for x in found_raw]
                target=_choose_target(found,direct,width)

                if prior is not None and found:
                    candidates=[
                        x for x in found
                        if not x.get('foreground',False)
                    ] or found
                    faces=[
                        x for x in candidates
                        if x.get('face_visible',False)
                    ] or candidates
                    target=min(
                        faces,
                        key=lambda x:abs(
                            (x['x']+x['width']/2)
                            -(prior['x']+prior['width']/2)
                        ),
                    )

                if target:
                    target=_bbox(target)
                    focus_boxes.append(target)
                    target_samples.append({'time':float(t),'bbox':target})
                    anchor_samples.append(
                        (t,_anchor(target,width,crop))
                    )
                    prior=target

            # An explicit human binding that cannot be consumed by this
            # strategy must never be treated as silently resolved.
            if (
                binding_present
                and required_person
                and direct['focus_subject'] in {'woman','man'}
            ):
                binding_resolved=False
                binding_diagnostics={
                    'reason':'invalid_single_person_binding',
                    'target_person_ids':target_ids,
                }

        from .face_safe import applies, bind_face, crop_interval, detect_faces, head_proxy
        face_samples=[]
        face_constraints=[]
        face_constraint_review=False
        if applies(direct):
            for sample_index,(time,frame) in enumerate(sampled):
                competitors=[_bbox(p) for p in detected[sample_index] if p.get('detector')=='yolo_person']
                target=next((box for t,box in tracked_boxes if t==time),None)
                # The detector may contribute geometry only after the person
                # tracker has supplied WHO.  It must never select a competitor.
                face=(bind_face(detect_faces(frame),target,competitors)
                      if bound_single and binding_resolved and target else None)
                region=face or (head_proxy(target) if bound_single and binding_resolved else None)
                interval=crop_interval(region,width,crop) if region else None
                row={'time':time,'target':target,'competitors':competitors,'region':region,
                     'constraint':interval,
                     'evidence':'associated_face' if face else 'target_head_proxy' if region else 'unavailable'}
                face_samples.append(row)
                if interval and interval['feasible'] and not interval['source_limited']:
                    face_constraints.append((float(time),interval,region))
                else:
                    # Keep temporal samples aligned even when this one cannot
                    # supply a defensible legal interval.  The broad interval
                    # is explicitly review-only; it cannot erase constraints
                    # from neighbouring samples where associated geometry is
                    # reliable.
                    face_constraints.append((float(time),{
                        'min_x':0., 'max_x':float(width-crop),
                        'margin':None, 'feasible':False,
                        'source_limited':bool(interval and interval.get('source_limited')),
                        'version':interval.get('version') if interval else None,
                        'unconstrained_reason':'source_limit_or_unavailable',
                    },region))
                if (not interval or not interval['feasible'] or interval['source_limited']) and bound_single:
                    # Unknown association/geometry is an editorial review,
                    # never silent permission for a close-person crop.
                    face_constraint_review=True

            # Both strategies use hard associated face/head constraints.  The
            # retry has a real editorial difference: it prefers the bound
            # face/head centre before projection, whereas subject_focus first
            # prefers the normal tracked-person anchor.
            if bound_single and binding_resolved and len(face_constraints)==len(anchor_samples):
                by_time={time:(interval,region) for time,interval,region in face_constraints}
                constrained=[]
                for time,person_preferred in anchor_samples:
                    interval,region=by_time[float(time)]
                    preferred=(
                        _anchor(region,width,crop)
                        if strategy=='face_priority'
                        else person_preferred
                    )
                    constrained.append((float(time),float(preferred),interval))
                anchor_samples=[(time,preferred) for time,preferred,_ in constrained]
                face_constraints=[interval for _,_,interval in constrained]
            else:
                face_constraints=[]

        regions=direct.get('required_action_region',[])
        regions=regions if isinstance(regions,list) else [regions]
        action=[_bbox(x) for x in regions if x]

        focus=_union(focus_boxes) if focus_boxes else None
        preserve=(
            direct['interaction_requirement']=='simultaneous'
            or direct['focus_subject']=='multiple_people'
            or bool(direct.get('preserve_secondary_subject'))
            or bool(direct.get('required_secondary_subjects',[]))
        )

        composition=[focus] if focus else []

        if (preserve or strategy=='interaction_aware') and all_boxes and not bound_multiple:
            composition=[_union(all_boxes)]

        composition+=action
        required=_union(composition) if composition else focus

        impossible=bool(
            required
            and required['width']>crop*(1-2*SAFE_MARGIN)
            and (preserve or action)
        )

        crop_fallback_used=False
        if (
            (preserve or action)
            and required
            and required['width']<=crop*(1-2*SAFE_MARGIN)
        ):
            anchors=_smooth(
                [
                    (t,_anchor(required,width,crop))
                    for t,_ in anchor_samples
                ]
                or [(start,_anchor(required,width,crop))],
                crop,
                start,
                end,
            )
        elif anchor_samples:
            anchors=_smooth(anchor_samples,crop,start,end,face_constraints or None)
        elif required and required['width']<=crop*(1-2*SAFE_MARGIN):
            anchors=[
                {
                    'time':start,
                    'x':float(_anchor(required,width,crop)),
                }
            ]
        else:
            crop_fallback_used=True
            pos=str(
                shot.get(
                    'primary_subject_position',
                    event.get('visual',{}).get(
                        'primary_subject_position',
                        'center',
                    ),
                )
            ).lower()
            anchors=[
                {
                    'time':start,
                    'x':float(crop_x(width,height,pos)),
                }
            ]

        unresolved=required_person and focus is None

        # Explicit semantic binding is authoritative. If it could not be
        # maintained through every sampled time, the shot is not auto-PASS.
        binding_failed=(
            binding_present
            and required_person
            and not binding_resolved
        )

        person_count=sum(
            1 for x in all_boxes
            if x.get('detector')=='yolo_person'
        )
        face_count=sum(
            1 for x in all_boxes
            if x.get('face_visible')
        )

        provenance={
            'person_detector':dict(
                _PERSON_RUNTIME
                or {
                    'model_id':PERSON_MODEL_ID,
                    'loaded':False,
                    'inference_executed':False,
                }
            ),
            'face_detector':dict(_FACE_RUNTIME),
        }

        source_start_frame=int(
            shot.get('start_frame',round(start*24))
        )
        source_end_frame=int(
            shot.get('end_frame_exclusive',round(end*24))
        )
        event_start_frame=int(
            event.get(
                'start_frame',
                round(float(event['start_seconds'])*24),
            )
        )

        plans.append({
            'shot_id':sid,
            'start_seconds':start,
            'end_seconds':end,
            'source_start_frame':source_start_frame,
            'source_end_frame_exclusive':source_end_frame,
            'render_start_frame':max(
                0,
                source_start_frame-event_start_frame,
            ),
            'render_end_frame_exclusive':max(
                0,
                source_end_frame-event_start_frame,
            ),
            'render_start_seconds':max(
                0.,
                start-float(event['start_seconds']),
            ),
            'render_end_seconds':max(
                0.,
                min(
                    float(event['end_seconds'])
                    -float(event['start_seconds']),
                    end-float(event['start_seconds']),
                ),
            ),
            'sampling':{
                'media_role':'source_movie',
                'timeline_basis':'source_absolute',
                'pixel_coordinate_system':'active_picture_relative',
                'active_picture':active,
                'requested_start':start,
                'requested_end':end,
                'sampled_frame_count':len(sampled),
            },
            'tracking':{
                'mode':(
                    'semantic_bound_geometry'
                    if bound_single
                    else 'semantic_bound_multi_geometry'
                    if bound_multiple
                    else 'bounded_linear'
                ),
                'anchor_count':len(anchors),
                'dead_zone_ratio':TRACKING_DEAD_ZONE_RATIO,
                'max_velocity_ratio_per_second':
                    TRACKING_MAX_VELOCITY_RATIO_PER_SECOND,
                'interpolation':'linear',
                'identity_diagnostics':binding_diagnostics,
                'face_head_constraint_version':(
                    face_samples[0].get('constraint',{}).get('version')
                    if face_samples and face_samples[0].get('constraint') else None
                ),
            },
            'focus_subject':direct['focus_subject'],
            'focus_position':direct.get(
                'focus_position',
                'unclear',
            ),
            'focus_role':direct['focus_role'],
            'focus_reason':direct.get(
                'focus_reason',
                'semantic shot focus plus local geometry',
            ),
            'directive_available':direct['directive_available'],
            'required_person_focus':required_person,
            'interaction_requirement':
                direct['interaction_requirement'],
            'preserve_interaction':preserve,
            'required_action_region':regions,
            'focus_bbox':focus,
            # This is sampled source geometry, not a new identity claim.  It
            # lets the post-render observer compare the actual vertical pixels
            # with the WHO target that determined the crop.
            'target_samples':target_samples,
            'subject_bboxes':all_boxes,
            'person_detection_count':person_count,
            'face_detection_count':face_count,
            'geometry_resolved':bool(focus),
            'target_person_ids':target_ids,
            'resolved_local_person_ids':local_ids if not translation_error else [],
            'crop_fallback_used':crop_fallback_used,
            'target_binding_confidence':target_confidence,
            'target_binding_present':binding_present,
            'target_binding_resolved':(
                binding_resolved
                if binding_present and required_person
                else None
            ),
            'anchors':anchors,
            'x':float(np.median([x['x'] for x in anchors])),
            'crop_width':crop,
            'source_width':width,
            'coordinate_system':'active_picture_relative',
            'active_picture':active,
            'strategy':strategy,
            'review_required':(
                impossible
                or unresolved
                or binding_failed
                or face_constraint_review
                or not direct['directive_available']
            ),
            'action_preserved':not action or bool(required),
            'face_safe_required':applies(direct),
            'face_safe_samples':face_samples,
            'face_head_constraint_policy':(
                face_samples[0].get('constraint',{}).get('version')
                if face_samples and face_samples[0].get('constraint') else None
            ),
            'face_safe_source':str(source_video.resolve()),
            'face_safe_event':{k:event[k] for k in ('start_frame','start_seconds') if k in event},
            'detector_version':(
                LOCAL_DETECTOR_VERSION
                if detector is detect_people
                and provenance['person_detector']['loaded']
                else 'injected_test_detector'
            ),
            'detector_provenance':provenance,
        })

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
def render_vertical(source_movie:Path,output:Path,event:dict[str,Any],plan:list[dict[str,Any]],active_picture:dict[str,Any]|None=None)->None:
    """Crop original source pixels and make exactly one final H.264 generation.

    Frames are passed as raw BGR directly to ffmpeg.  In particular, an event
    horizontal MP4 is never a pixel parent and no temporary lossy render exists.
    """
    cap=cv2.VideoCapture(str(source_movie)); fps=cap.get(cv2.CAP_PROP_FPS) or 24.; w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); active=active_picture or {'x':0,'y':0,'width':w,'height':h}; ax,ay,aw,ah=(int(active[k]) for k in ('x','y','width','height')); cw=min(aw,round(ah*3/4)); n=0; start_frame=int(event.get('start_frame',round(float(event['start_seconds'])*fps))); total=int(event.get('end_frame_exclusive',round(float(event['end_seconds'])*fps)))-start_frame; cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
    if not cap.isOpened() or not w or not h or total<=0: cap.release(); raise RuntimeError('vertical reframe cannot open valid source media/event range')
    command=['ffmpeg','-y','-f','rawvideo','-pixel_format','bgr24','-video_size',f'{cw}x{ah}','-framerate',str(fps),'-i','-','-map','0:v:0','-c:v','libx264','-crf','16','-preset','medium','-pix_fmt','yuv420p','-an',str(output)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    def matches(rule:dict[str,Any],index:int,relative:float)->bool:
        if 'render_start_frame' in rule:return rule['render_start_frame']<=index<rule['render_end_frame_exclusive']
        return rule.get('render_start_seconds',rule['start_seconds'])<=relative<rule.get('render_end_seconds',rule['end_seconds'])
    stderr=''; result=-1
    try:
        while n<total:
            ok,frame=cap.read()
            if not ok: break
            relative=n/fps
            matching_rules=[x for x in plan if matches(x,n,relative)]
            if len(matching_rules) != 1:
                raise RuntimeError(
                    'vertical shot plan coverage error: '
                    f'frame={n} matches={len(matching_rules)} '
                    f'shots={[x.get("shot_id") for x in matching_rules]}'
                )
            rule=matching_rules[0]
            # Crop coordinates in a plan are active-picture relative.  Detector
            # geometry remains source-frame based when there is no pillarbox;
            # this explicit origin prevents letterbox Y offsets leaking into
            # output pixels.
            x=max(0,min(aw-cw,_x_at(rule,float(event['start_seconds'])+relative)))
            process.stdin.write(np.ascontiguousarray(frame[ay:ay+ah,ax+x:ax+x+cw]).tobytes())
            n+=1
        process.stdin.close(); stderr=process.stderr.read().decode(errors='replace'); result=process.wait()
    except (BrokenPipeError,OSError) as error:
        stderr=str(error)
        if process.stdin and not process.stdin.closed: process.stdin.close()
        result=process.wait()
    finally:
        cap.release()
        if process.stdin and not process.stdin.closed: process.stdin.close()
    if not n or n!=total or result:
        output.unlink(missing_ok=True); tail='\n'.join(stderr.splitlines()[-8:])
        raise RuntimeError(f'vertical source crop encode failed after {n}/{total} frames: {tail or result}')
def _person_focal_core(focus:dict[str,float], crop:float)->dict[str,float]:
    """Protect the visible head region for oversized close-up person boxes.

    YOLO's person box is intentionally broad.  Once it substantially exceeds a
    3:4 crop, the old torso-centred core made peripheral body loss fatal.  The
    face detector remains optional; this deterministic upper-centre estimate is
    used only for those oversized boxes.
    """
    if focus['width'] >= crop*1.5:
        return {'x':focus['x']+focus['width']*.40,'y':focus['y']+focus['height']*.04,'width':focus['width']*.22,'height':focus['height']*.30}
    return {'x':focus['x']+focus['width']*.30,'y':focus['y']+focus['height']*.05,'width':focus['width']*.40,'height':focus['height']*.50}

def _shot_validation(rule:dict[str,Any])->dict[str,Any]:
    crop=float(rule['crop_width']); x=float(rule['x']); margin=crop*SAFE_MARGIN; focus=rule.get('focus_bbox'); source=float(rule['source_width']); source_clip=False; introduced=False; empty=False; full_clipping=False; critical_clipping=False
    required_person=bool(rule.get('required_person_focus',rule.get('focus_subject') in {'woman','man','multiple_people'}))
    if focus:
        source_clip=focus['x']<=1 or focus['x']+focus['width']>=source-1; full_safe=focus['x']>=x+margin and focus['x']+focus['width']<=x+crop-margin; full_clipping=not full_safe
        core=_person_focal_core(focus,crop)
        # For simultaneous interaction the required union is itself critical;
        # ordinary person shots protect the head/upper-body focal core instead.
        critical=focus if rule.get('preserve_interaction') or rule.get('interaction_requirement')=='simultaneous' else core
        core_safe=critical['x']>=x+margin and critical['x']+critical['width']<=x+crop-margin
        critical_clipping=not core_safe and not source_clip; introduced=critical_clipping if required_person else False; empty=critical_clipping and required_person and abs((critical['x']+critical['width']/2)-(x+crop/2))>crop*.20
    # Actual associated faces replace the invented central body core, including
    # for initial renders. Evaluate each temporal anchor, not a union/median.
    face_samples=rule.get('face_safe_samples',[])
    if face_samples and all(r.get('region',{}).get('kind')=='face' for r in face_samples if r.get('region')) and all(r.get('region') for r in face_samples):
        from .face_safe import assess
        checks=[assess(r['region'],source,float('inf'),_x_at(rule,r['time']),crop,rule.get('target_binding_resolved') is True) for r in face_samples]
        critical_clipping=any(r['decision']=='REPAIR' for r in checks)
        introduced=critical_clipping
        empty=critical_clipping
        source_clip=any(r['source_limited'] for r in checks)
    action_ok=True
    for region in rule.get('required_action_region',[]) or []:
        b=_bbox(region); action_ok=action_ok and b['x']>=x and b['x']+b['width']<=x+crop
    stable=len(rule.get('anchors',[]))<=1 or max(abs(b['x']-a['x']) for a,b in zip(rule['anchors'],rule['anchors'][1:]))<=crop*.18+1; interaction=not rule.get('review_required',False) if rule.get('preserve_interaction') or rule.get('interaction_requirement')=='simultaneous' else True
    requirement='person' if required_person else 'environment' if rule.get('focus_subject')=='environment' else 'action' if rule.get('focus_subject')=='action_region' else 'none'
    present=bool(focus) if required_person else None
    safe=bool(focus) and not introduced if required_person else None
    geometry_ok=(present and safe) if required_person else True
    binding_required=bool(rule.get('target_binding_present')) and required_person
    binding_ok=not binding_required or rule.get('target_binding_resolved') is True
    ok=bool(rule.get('directive_available',True)) and geometry_ok and binding_ok and not introduced and not empty and stable and interaction and rule.get('action_preserved',True) and action_ok
    return {'shot_id':rule['shot_id'],'focus_requirement':requirement,'focus_geometry_resolved':bool(focus),'focus_directive_available':bool(rule.get('directive_available',True)),'focus_subject_present':present,'focus_subject_safe':safe,'target_binding_required':binding_required,'target_binding_resolved':rule.get('target_binding_resolved'),'full_bbox_clipping':full_clipping,'critical_focus_clipping':critical_clipping,'introduced_subject_clipping':introduced,'source_edge_exception':source_clip,'empty_space_while_clipped':empty,'interaction_preserved':interaction,'action_preserved':rule.get('action_preserved',True) and action_ok,'crop_stable':stable,'status':'PASS' if ok else 'FAIL'}

def _post_render_sample_indices(rule:dict[str,Any],frame_count:int,count:int=POST_RENDER_SAMPLES_PER_SHOT)->list[int]:
    """Bound samples to one rendered technical shot; never cross a hard cut."""
    if frame_count<=0:
        return []

    start=max(
        0,
        min(
            frame_count-1,
            int(rule.get('render_start_frame',0)),
        ),
    )
    end=max(
        start+1,
        min(
            frame_count,
            int(rule.get('render_end_frame_exclusive',frame_count)),
        ),
    )

    span=end-start
    wanted=max(1,min(int(count),span))

    if wanted==1:
        return [start+span//2]

    values=np.linspace(
        start+(span-1)*.12,
        start+(span-1)*.88,
        wanted,
    )
    return sorted({
        max(start,min(end-1,int(round(x))))
        for x in values
    })


def _post_render_person_boxes(found:list[dict[str,Any]])->list[dict[str,float]]:
    """Use person geometry only; face candidates never become target authority."""
    boxes=[]

    for item in found:
        # Production detector emits explicit YOLO person provenance.
        # Injected test detectors may omit the detector field.
        detector_name=item.get('detector')
        if detector_name not in {None,'yolo_person'}:
            continue

        try:
            box=_bbox(item)
        except (KeyError,TypeError,ValueError):
            continue

        if box['width']<=1 or box['height']<=1:
            continue

        boxes.append(box)

    return boxes


def _post_render_target(
    rule:dict[str,Any],
    source_time:float,
    people:list[dict[str,float]],
    width:int,
) -> tuple[dict[str,float]|None,bool]:
    """Bind a rendered detection back to the planned source target.

    The vertical detector does not preserve YOLO track IDs.  The persisted
    source-sample geometry is therefore used only to test whether the planned
    target is still the visible person; it never selects a different subject.
    """
    samples=rule.get('target_samples') or []
    if not people:
        return None,False
    if not samples:
        # Legacy plans cannot prove WHO, but still receive the composition gate.
        return min(people,key=lambda box:abs(box['x']+box['width']/2-width/2)),False
    nearest=min(samples,key=lambda row:abs(float(row.get('time',source_time))-source_time))
    source=_bbox(nearest.get('bbox',{}))
    crop=float(rule.get('crop_width',width) or width)
    x=float(_x_at(rule,source_time))
    expected_center=(source['x']+source['width']/2-x)*width/max(1.,crop)
    expected_width=source['width']*width/max(1.,crop)
    selected=min(people,key=lambda box:abs((box['x']+box['width']/2)-expected_center))
    distance=abs((selected['x']+selected['width']/2)-expected_center)
    # This intentionally allows detector box jitter and rule-of-thirds framing,
    # while refusing to relabel the other person across the crop as the target.
    matched=distance<=max(width*.28,expected_width*.9)
    return (selected if matched else None),matched


def _editorial_geometry(
    samples:list[dict[str,Any]],
    rule:dict[str,Any],
    width:int,
    height:int,
) -> dict[str,Any]:
    """Cheap representative-sample editorial usability measurements."""
    decoded=[x for x in samples if x.get('decoded')]
    total=max(1,len(samples))
    targets=[x for x in decoded if x.get('target_bbox')]
    target_visibility=len(targets)/total
    centers=[abs((x['target_bbox']['x']+x['target_bbox']['width']/2)-width/2)/max(1.,width) for x in targets]
    occupancy=[x['target_bbox']['width']*x['target_bbox']['height']/max(1.,width*height) for x in targets]
    edges=[x for x in targets if x['target_bbox']['x']<=width*.04 or x['target_bbox']['x']+x['target_bbox']['width']>=width*.96]
    competing=[]
    for row in targets:
        area=row['target_bbox']['width']*row['target_bbox']['height']
        others=[box for box in row.get('people',[]) if box is not row['target_bbox']]
        competing.append(any(box['width']*box['height']>area*1.15 for box in others))
    empty_center=sum(bool(x.get('empty_center')) for x in decoded)/max(1,len(decoded))
    target_centers=[x['target_bbox']['x']+x['target_bbox']['width']/2 for x in targets]
    jumps=[abs(b-a)/max(1.,width) for a,b in zip(target_centers,target_centers[1:])]
    # Missing target samples are itself binding instability.  Movement within a
    # tracked crop is expected, so only very large sample-to-sample jumps count.
    binding_stability=(len(targets)-sum(j>.38 for j in jumps))/total
    return {
        'target_visibility_ratio':round(target_visibility,4),
        'target_center_distance':round(float(np.median(centers)),4) if centers else None,
        'target_crop_occupancy':round(float(np.median(occupancy)),4) if occupancy else 0.,
        'target_edge_violation_ratio':round(len(edges)/max(1,len(targets)),4),
        'binding_stability':round(max(0.,binding_stability),4),
        'competing_subject_dominance':round(sum(competing)/max(1,len(targets)),4),
        'empty_center_ratio':round(empty_center,4),
    }


def _post_render_person_audit(
    path:Path,
    plan:list[dict[str,Any]],
    detector:Callable[[np.ndarray],list[dict[str,Any]]]=detect_people,
    samples_per_shot:int=POST_RENDER_SAMPLES_PER_SHOT,
)->dict[str,Any]:
    """Observe the actual rendered vertical instead of trusting plan geometry.

    Semantic target binding remains the authority for WHO should be followed.
    This stage answers a different question: did the resulting vertical keep a
    usable human composition through the technical shot?
    """
    cap=cv2.VideoCapture(str(path))
    frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if (
        not path.is_file()
        or not cap.isOpened()
        or frame_count<=0
        or width<=0
        or height<=0
    ):
        cap.release()
        return {
            'version':POST_RENDER_AUDIT_VERSION,
            'decision':'AMBIGUOUS',
            'reason':'vertical_unreadable',
            'shots':[],
        }

    shot_results=[]

    for rule in plan:
        required_person=bool(
            rule.get(
                'required_person_focus',
                rule.get('focus_subject')
                in {'woman','man','multiple_people'},
            )
        )

        if not required_person:
            shot_results.append({
                'shot_id':rule.get('shot_id'),
                'decision':'PASS',
                'reason':'person_audit_not_applicable',
                'sample_count':0,
            })
            continue

        if (
            rule.get('target_binding_present')
            and rule.get('target_binding_resolved') is not True
        ):
            shot_results.append({
                'shot_id':rule.get('shot_id'),
                'decision':'RETRY',
                'reason':'semantic_target_binding_unresolved',
                'sample_count':0,
            })
            continue

        indices=_post_render_sample_indices(
            rule,
            frame_count,
            samples_per_shot,
        )

        if not indices:
            shot_results.append({
                'shot_id':rule.get('shot_id'),
                'decision':'AMBIGUOUS',
                'reason':'no_render_samples',
                'sample_count':0,
            })
            continue

        samples=[]
        detector_error=None

        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame=cap.read()

            if not ok:
                samples.append({
                    'frame':index,
                    'decoded':False,
                    'person_count':0,
                    'centered':False,
                })
                continue

            try:
                people=_post_render_person_boxes(detector(frame))
            except (RuntimeError,cv2.error) as error:
                detector_error=str(error)
                break

            selected=None
            target_matched=False
            centered=False
            center_offset=None

            if people:
                render_start=float(rule.get('render_start_frame',0))
                render_end=float(rule.get('render_end_frame_exclusive',frame_count))
                fraction=(index-render_start)/max(1.,render_end-render_start)
                source_time=float(rule.get('start_seconds',0.))+max(0.,min(1.,fraction))*max(0.,float(rule.get('end_seconds',rule.get('start_seconds',0.)))-float(rule.get('start_seconds',0.)))
                selected,target_matched=_post_render_target(rule,source_time,people,width)

                if selected:
                    center=selected['x']+selected['width']/2
                    center_offset=abs(center-width/2)/max(1.,width)

                    # Diagnostic only: editorial validation deliberately does
                    # not require a centred face/person.
                    centered=center_offset<=.33

            center_has_person=any(
                box['x'] < width*.60 and box['x']+box['width'] > width*.40
                for box in people
            )

            samples.append({
                'frame':index,
                'decoded':True,
                'person_count':len(people),
                'selected_bbox':selected,
                'target_bbox':selected,
                'target_matched':target_matched,
                'people':people,
                'empty_center':len(people)>=2 and not center_has_person,
                'center_offset_ratio':(
                    round(center_offset,5)
                    if center_offset is not None
                    else None
                ),
                'centered':centered,
            })

        if detector_error is not None:
            shot_results.append({
                'shot_id':rule.get('shot_id'),
                'decision':'AMBIGUOUS',
                'reason':'post_render_detector_error',
                'detector_error':detector_error,
                'sample_count':len(indices),
                'samples':samples,
            })
            continue

        total=max(1,len(indices))
        decoded=sum(bool(x.get('decoded')) for x in samples)
        present=sum(x.get('person_count',0)>0 for x in samples)
        centered=sum(bool(x.get('centered')) for x in samples)

        multi=sum(
            x.get('person_count',0)>1
            for x in samples
        )

        enough_people=sum(
            x.get('person_count',0)>=2
            for x in samples
        )

        decoded_ratio=decoded/total
        presence_ratio=present/total
        centered_ratio=centered/total
        multiple_ratio=multi/total
        enough_people_ratio=enough_people/total
        geometry=_editorial_geometry(samples,rule,width,height)

        focus_subject=rule.get('focus_subject')
        interaction=rule.get('interaction_requirement')
        binding_present=bool(rule.get('target_binding_present'))

        focus_bbox=rule.get('focus_bbox')
        close_up_expected=bool(
            focus_bbox
            and float(focus_bbox.get('width',0))
            >= float(rule.get('crop_width',width))*1.40
        )

        decision='PASS'
        reason='rendered_subject_present_and_usable'

        if decoded_ratio < .80:
            decision='AMBIGUOUS'
            reason='insufficient_render_decode'

        elif focus_subject=='multiple_people':
            if (
                interaction=='simultaneous'
                and enough_people_ratio < .60
            ):
                decision='RETRY'
                reason='required_people_not_visible_together'
            elif enough_people_ratio < .60:
                decision='AMBIGUOUS'
                reason='multiple_people_not_reliably_observed'

        elif presence_ratio < .60:
            # Person-only YOLO can be inconclusive on extreme close-ups.
            # Do not manufacture a crop failure when the local observer lacks
            # sufficient evidence; a later face/semantic reviewer can resolve it.
            if close_up_expected:
                decision='AMBIGUOUS'
                reason='close_up_person_detection_inconclusive'
            else:
                decision='RETRY'
                reason='focused_person_lost_in_render'

        elif geometry['target_visibility_ratio'] < .60:
            decision='AMBIGUOUS'
            reason='intended_target_not_reliably_visible'

        elif geometry['binding_stability'] < .60:
            decision='AMBIGUOUS'
            reason='target_binding_unstable'

        elif geometry['target_edge_violation_ratio'] >= .60:
            decision='RETRY'
            reason='focused_person_persistently_at_crop_edge'

        elif (
            interaction != 'simultaneous'
            and focus_subject != 'multiple_people'
            and geometry['empty_center_ratio'] >= .40
        ):
            decision='AMBIGUOUS'
            reason='crop_centered_between_competing_subjects'

        elif (
            interaction != 'simultaneous'
            and focus_subject != 'multiple_people'
            and geometry['competing_subject_dominance'] >= .40
            and geometry['target_edge_violation_ratio'] >= .40
        ):
            decision='AMBIGUOUS'
            reason='competing_subject_dominates_target'

        elif geometry['target_crop_occupancy'] < .025:
            decision='AMBIGUOUS'
            reason='target_too_small_for_editorial_use'

        elif not binding_present and multiple_ratio >= .40:
            # Legacy semantic plans can show a usable person while still giving
            # us no proof it is the intended person. Never call that identity
            # question automatically PASS.
            decision='AMBIGUOUS'
            reason='legacy_multi_person_identity_unbound'

        shot_results.append({
            'shot_id':rule.get('shot_id'),
            'decision':decision,
            'reason':reason,
            'sample_count':len(indices),
            'decoded_ratio':round(decoded_ratio,4),
            'person_presence_ratio':round(presence_ratio,4),
            'centered_ratio':round(centered_ratio,4),
            'multiple_people_ratio':round(multiple_ratio,4),
            'required_people_ratio':round(enough_people_ratio,4),
            'editorial_geometry':geometry,
            'target_binding_present':binding_present,
            'target_binding_resolved':rule.get(
                'target_binding_resolved'
            ),
            'samples':samples,
        })

    cap.release()

    decisions=[x['decision'] for x in shot_results]

    if 'RETRY' in decisions:
        decision='RETRY'
        reason='one_or_more_shots_need_reframe'
    elif 'AMBIGUOUS' in decisions:
        decision='AMBIGUOUS'
        reason='one_or_more_shots_need_semantic_or_identity_review'
    else:
        decision='PASS'
        reason='all_rendered_shots_pass'

    return {
        'version':POST_RENDER_AUDIT_VERSION,
        'decision':decision,
        'reason':reason,
        'width':width,
        'height':height,
        'frame_count':frame_count,
        'shots':shot_results,
    }


def post_render_vertical_audit(path, plan, detector=detect_people, samples_per_shot=POST_RENDER_SAMPLES_PER_SHOT):
    from .face_safe import applies, audit_video
    result=_post_render_person_audit(path,plan,detector,samples_per_shot)
    required=[r for r in plan if applies(r)]
    if required:
        source=required[0].get('face_safe_source')
        event=required[0].get('face_safe_event',{})
        if source and 'start_frame' in event:
            face=audit_video(path,Path(source),event,plan)
        else:
            face={'decision':'AMBIGUOUS','reason_codes':['face_detection_inconclusive'], 'repairable':False}
        result['face_safe_audit']=face
        if face['decision']!='PASS':
            result['decision']='RETRY' if face['decision']=='REPAIR' or result['decision']=='RETRY' else 'AMBIGUOUS'
            result['reason']='face_safe_'+face['decision'].lower()
            result['retry_strategy']='face_priority' if face['decision']=='REPAIR' else None
    return result


def classify_vertical_qa(validation:dict[str,Any])->dict[str,list[str]]:
    """Classify observed vertical QA without treating preferences as failures.

    The render/audit stages intentionally retain their detailed, conservative
    diagnostics.  This final policy projection is the publication gate: a
    concrete usability failure is hard; incomplete aesthetic evidence is a
    warning.  In particular, an observer's inability to prove ideal centering
    must not turn a technically valid semantic KEEP into a review package.
    """
    hard:list[str]=[]; soft:list[str]=[]
    def add(target:list[str], value:str)->None:
        if value not in target: target.append(value)

    if not validation.get('file_exists', True): add(hard,'vertical_render_missing')
    if validation.get('frame_count', 1) <= 0: add(hard,'vertical_render_unreadable')
    if validation.get('width', 0) <= 0 or validation.get('height', 0) <= 0:
        add(hard,'vertical_dimensions_invalid')
    elif validation.get('aspect_ratio') != '3:4': add(hard,'vertical_aspect_ratio_invalid')
    if validation.get('duration_valid') is False: add(hard,'vertical_duration_invalid')
    if validation.get('black_bars'): add(hard,'severe_black_bars_or_corruption')
    if validation.get('crop_plan_valid') is False: add(hard,'crop_plan_invalid')

    for shot in validation.get('shots',[]):
        required=shot.get('focus_requirement') == 'person'
        if required and shot.get('focus_subject_present') is False:
            add(hard,'primary_subject_missing')
        if required and (shot.get('critical_focus_clipping') or shot.get('introduced_subject_clipping')):
            add(hard,'primary_subject_materially_clipped')
        if shot.get('crop_stable') is False: add(hard,'severe_reframe_instability')
        if shot.get('action_preserved') is False: add(hard,'required_action_or_region_lost')
        if shot.get('interaction_preserved') is False:
            add(hard,'required_multi_person_interaction_lost')
        # A broad person box at the crop edge is useful diagnostics but is not
        # proof of head/face loss; the critical-focus and face checks above are.
        if shot.get('full_bbox_clipping') and not shot.get('critical_focus_clipping'):
            add(soft,'noncritical_subject_edge_proximity')
        if shot.get('focus_directive_available') is False:
            add(soft,'focus_directive_unavailable')

    audit=validation.get('post_render_audit',{})
    for shot in audit.get('shots',[]):
        reason=shot.get('reason')
        geometry=shot.get('editorial_geometry',{})
        if reason in {'vertical_unreadable','insufficient_render_decode'}:
            add(hard,'vertical_render_unreadable')
        elif reason == 'required_people_not_visible_together':
            add(hard,'required_multi_person_interaction_lost')
        elif reason == 'focused_person_lost_in_render':
            add(hard,'primary_subject_lost_in_render')
        elif (reason == 'intended_target_not_reliably_visible'
              and geometry.get('target_visibility_ratio',1.) < .20
              and geometry.get('competing_subject_dominance',0.) >= .60):
            add(hard,'wrong_subject_or_region_followed')
        elif reason in {
            'focused_person_persistently_at_crop_edge',
            'close_up_person_detection_inconclusive',
            'multiple_people_not_reliably_observed',
            'intended_target_not_reliably_visible',
            'target_binding_unstable',
            'crop_centered_between_competing_subjects',
            'competing_subject_dominates_target',
            'target_too_small_for_editorial_use',
            'legacy_multi_person_identity_unbound',
            'semantic_target_binding_unresolved',
            'no_render_samples',
            'post_render_detector_error',
        }:
            add(soft,reason)

    face=audit.get('face_safe_audit',{})
    if face.get('decision') == 'REPAIR': add(hard,'important_face_or_head_materially_clipped')
    elif face.get('decision') not in {None,'PASS'}: add(soft,'face_safe_'+str(face.get('decision')).lower())
    return {'hard_failures':hard,'soft_warnings':soft}


def validate_vertical(
    path:Path,
    expected:float,
    plan:list[dict[str,Any]],
    event:dict[str,Any],
    audit_detector:Callable[[np.ndarray],list[dict[str,Any]]]=detect_people,
)->dict[str,Any]:
    cap=cv2.VideoCapture(str(path))
    w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=cap.get(cv2.CAP_PROP_FPS) or 24.
    frames=[]

    for i in {0,max(0,count//2),max(0,count-1)}:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i)
        ok,x=cap.read()
        frames.extend([x] if ok else [])

    cap.release()

    bars=any(
        float(
            np.mean(
                cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)<8
            )
        )>.92
        for x in frames
    )

    per=[_shot_validation(x) for x in plan]

    post_render=post_render_vertical_audit(
        path,
        plan,
        detector=audit_detector,
    )

    post_render_ok=post_render.get('decision')=='PASS'

    ok=(
        path.exists()
        and count>0
        and w*4==h*3
        and abs(count/fps-expected)<=1
        and not bars
        and all(x['status']=='PASS' for x in per)
        and post_render_ok
    )

    sequence=[
        rule
        for rule in plan
        if rule.get('interaction_requirement')=='sequence'
    ]

    sequence_ok=all(
        (
            (
                x['focus_subject_present']
                and x['focus_subject_safe']
            )
            if rule.get(
                'required_person_focus',
                rule.get('focus_subject')
                in {'woman','man','multiple_people'},
            )
            else True
        )
        and x['action_preserved']
        for x,rule in zip(per,plan)
        if rule.get('interaction_requirement')=='sequence'
    )

    # Keep all raw diagnostics, then make the publish/review decision from
    # material failures only.  ``ok`` remains useful evidence for operators;
    # it is intentionally not the publication predicate.
    crop_plan_valid=bool(plan) and all(
        isinstance(rule.get('crop_width'),(int,float))
        and isinstance(rule.get('source_width'),(int,float))
        and isinstance(rule.get('x'),(int,float))
        and float(rule['crop_width']) > 0
        and float(rule['source_width']) >= float(rule['crop_width'])
        and 0 <= float(rule['x']) <= float(rule['source_width'])-float(rule['crop_width'])
        for rule in plan
    )
    provisional={
        'file_exists':path.exists(), 'frame_count':count,
        'duration_valid':abs(count/fps-expected)<=1,
        'width':w, 'height':h, 'aspect_ratio':'3:4' if w*4==h*3 else 'other',
        'black_bars':bars, 'shots':per, 'sequence_interaction_preserved':sequence_ok,
        'post_render_audit':post_render, 'crop_plan_valid':crop_plan_valid,
    }
    qa=classify_vertical_qa(provisional)
    face_decision=(post_render.get('face_safe_audit') or {}).get('decision')
    # A detector that cannot establish face/head safety is not evidence of a
    # safe close-person crop.  It is deliberately a review outcome rather
    # than a fabricated hard geometry failure; concrete repairable clipping is
    # still represented in ``hard_failures`` by classify_vertical_qa.
    face_review_required=face_decision in {'AMBIGUOUS','SOURCE_LIMIT','ERROR'}
    binding_review_required=any(row.get('target_binding_required') and row.get('target_binding_resolved') is not True for row in per)
    final_ok=not qa['hard_failures'] and not face_review_required and not binding_review_required
    review_reason=(
        qa['hard_failures'][0]
        if qa['hard_failures']
        else 'face_safe_'+str(face_decision).lower()
        if face_review_required
        else 'authoritative_subject_binding_unresolved' if binding_review_required
        else None
    )

    return {
        'status':'PASS' if final_ok else 'REVIEW',
        'review_reason':review_reason,
        'hard_failures':qa['hard_failures'],
        'soft_warnings':qa['soft_warnings'],
        'review_required':not final_ok,
        'face_safe_outcome':(
            'HARD_FAIL_REPAIRABLE' if face_decision=='REPAIR'
            else 'REVIEW_AMBIGUOUS' if face_decision in {'AMBIGUOUS','ERROR'}
            else 'SOURCE_LIMIT' if face_decision=='SOURCE_LIMIT'
            else 'PASS'
        ),
        'width':w,
        'height':h,
        'aspect_ratio':'3:4' if w*4==h*3 else 'other',
        'duration_seconds':count/fps,
        'black_bars':bars,
        'crop_plan_valid':crop_plan_valid,
        'shots':per,
        'sequence_interaction_preserved':sequence_ok,
        'sequence_interaction_shots':[
            x['shot_id'] for x in sequence
        ],
        'stable_per_shot':all(
            x['crop_stable'] for x in per
        ),
        'semantic_retained':all(
            x['status']=='PASS' for x in per
        ) and post_render.get('decision')=='PASS',
        'post_render_audit':post_render,
        'post_render_audit_version':POST_RENDER_AUDIT_VERSION,
    }
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
def reframe_fingerprint(event:dict[str,Any],shots:dict[str,dict[str,Any]],width:int,height:int,algorithm_version:str=REFRAME_ALGORITHM_VERSION,active_picture:dict[str,Any]|None=None)->str:
    """Vertical-only cache key.  Semantic/horizontal stages intentionally stay outside it."""
    plan=event.get('visual',{}).get('shot_focus_plan',event.get('shot_focus_plan',[]))
    shot_identity=[{'shot_id':sid,'start':shots.get(sid,{}).get('start_seconds'),'end':shots.get(sid,{}).get('end_seconds')} for sid in event.get('source_shot_ids',[])]
    active=active_picture or {'x':0,'y':0,'width':width,'height':height,'detection_profile':'legacy_full_frame'}
    from .face_safe import FACE_HEAD_CONSTRAINT_VERSION, FACE_HEAD_MARGIN_MIN_PX, FACE_HEAD_MARGIN_RATIO
    return fingerprint({'reframe_algorithm_version':algorithm_version,'vertical_validation_version':REFRAME_VALIDATION_COMPATIBILITY_VERSION,'shot_focus_schema_version':SHOT_FOCUS_SCHEMA_VERSION,'local_detector_version':LOCAL_DETECTOR_VERSION,'active_picture':active,'crop':{'safe_margin':SAFE_MARGIN,'aspect_ratio':'3:4','width':width,'height':height},'face_head_constraint':{'version':FACE_HEAD_CONSTRAINT_VERSION,'margin_ratio':FACE_HEAD_MARGIN_RATIO,'margin_min_px':FACE_HEAD_MARGIN_MIN_PX},'source_shots':shot_identity,'semantic_focus_plan':plan})
def _vertical_reuse_valid(assets:Path,base:str,reframe_fp:str)->bool:
    meta=assets/f'{base}.json'
    try:
        data=json.loads(meta.read_text()); final=data.get('visual',{}).get('final_vertical',{})
        validation_status=final.get('validation_status','PASS')  # Legacy approved packages predate explicit state.
        publication_valid = validation_status=='PASS' or (
            validation_status=='REVIEW' and is_publish_ready(data)
        )
        return _complete_package(assets,base) and publication_valid and final.get('reframe_fingerprint')==reframe_fp and final.get('reframe_algorithm_version')==REFRAME_ALGORITHM_VERSION and final.get('vertical_validation_version')==VERTICAL_VALIDATION_VERSION
    except (OSError,json.JSONDecodeError): return False
def _source_movie_sha256(run:Path,movie:Path)->str:
    """Reuse a persisted source digest; hash only when no trustworthy run record exists."""
    # Production writes this only after hashing canonical movie.mp4.  It takes
    # precedence over older pilot/source manifests, which may describe a prior
    # replacement of the same input filename.
    try:
        digest=json.loads((run/'source_fingerprint.json').read_text()).get('movie_sha256')
        if isinstance(digest,str) and re.fullmatch(r'[0-9a-f]{64}',digest): return digest
    except (OSError,json.JSONDecodeError): pass
    for manifest in (run/'source-v1'/'source_manifest.json',run/'source-inspect-v1'/'source_manifest.json',run/'visual-smoke-v1'/'run_manifest.json'):
        try:
            data=json.loads(manifest.read_text())
            digest=data.get('source',{}).get('movie',{}).get('sha256') or data.get('source_movie_sha256')
            if isinstance(digest,str) and re.fullmatch(r'[0-9a-f]{64}',digest): return digest
        except (OSError,json.JSONDecodeError): pass
    return sha256_file(movie)
def _aspect_ratio(width:int,height:int)->str:
    if width * 4 == height * 3:return '3:4'
    divisor=__import__('math').gcd(width,height)
    return f'{width//divisor}:{height//divisor}'
def _people_semantics(people:list[dict[str,Any]])->dict[str,Any]:
    count=len(people)
    composition='not_applicable' if not count else 'single_subject' if count == 1 else 'two_person' if count == 2 else 'group'
    presentations=[x.get('presentation') for x in people if x.get('presentation')]
    primary=next((x for x in people if x.get('frame_role')=='primary'),people[0] if people else {})
    value={'contains_people':bool(count),'people_count':count,'composition':composition}
    if presentations:value['overall_presentation']=presentations[0] if len(set(presentations)) == 1 else 'mixed'
    if primary:
        value['primary_subject']={k:primary[k] for k in ('presentation','frame_role','position') if k in primary}
    return value
def _vertical_override(event:dict[str,Any],plan:list[dict[str,Any]])->dict[str,Any]:
    """State only crop facts established by the approved focus plan; never call a VLM."""
    people=event.get('people',[])
    focus={x.get('focus_subject') for x in plan if x.get('focus_subject') in {'woman','man'}}
    if len(people) > 1 and len(focus) == 1 and all(not row.get('preserve_interaction') and row.get('target_binding_resolved') is not False for row in plan):
        presentation=next(iter(focus))
        return {'people':{'contains_people':True,'people_count':1,'composition':'single_subject','primary_subject':{'presentation':presentation}}}
    return {}
def _semantic_retention_valid(technical:dict[str,Any])->bool:
    if technical.get('status','PASS') not in {'PASS','REVIEW'} or technical.get('semantic_retained') is False or technical.get('hard_failures'):
        return False
    if technical.get('post_render_audit',{}).get('decision','PASS')!='PASS':
        return False
    return all(row.get('target_binding_resolved') is not False
               and row.get('focus_subject_present') is not False
               and row.get('interaction_preserved') is not False
               and not row.get('critical_focus_clipping')
               for row in technical.get('shots',[]))


def _rendition(file:Path,thumbnail_file:Path,timestamp:float,technical:dict[str,Any],orientation:str)->dict[str,Any]:
    return {'file':file.name,'sha256':sha256_file(file),'size_bytes':file.stat().st_size,'mime_type':'video/mp4','duration_seconds':technical['duration_seconds'],'width':technical['width'],'height':technical['height'],'fps':technical['fps'],'orientation':orientation,'aspect_ratio':_aspect_ratio(technical['width'],technical['height']),'technical_validated':True,'semantic_validated':_semantic_retention_valid(technical),'thumbnail':{'file':thumbnail_file.name,'sha256':sha256_file(thumbnail_file),'size_bytes':thumbnail_file.stat().st_size,'mime_type':'image/jpeg','timestamp_seconds':timestamp}}
def _technical_properties(file:Path,validation:dict[str,Any])->dict[str,Any]:
    cap=cv2.VideoCapture(str(file))
    width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=float(cap.get(cv2.CAP_PROP_FPS) or 24.)
    cap.release()
    return {**validation,'width':width,'height':height,'fps':fps}
def _validated_technical(file:Path,expected:float,frames:int)->dict[str,Any]:
    cap=cv2.VideoCapture(str(file)); width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    return _technical_properties(file,probe(file,width,height,expected,frames))
def _asset_metadata(movie_id:str,source_sha256:str,aid:str,slug:str,event:dict[str,Any],horizontal_file:Path,vertical_file:Path,horizontal_thumbnail:Path,vertical_thumbnail:Path,horizontal_time:float,vertical_time:float,horizontal:dict[str,Any],vertical:dict[str,Any],horizontal_mode:str,horizontal_generation:int,plan:list[dict[str,Any]],attempt:int,reframe_fp:str,active_picture:dict[str,Any]|None=None)->dict[str,Any]:
    source_visual=event.get('visual',{})
    visual={k:source_visual[k] for k in ('summary_es','subjects','objects','actions','visible_emotions','setting') if k in source_visual}
    if 'visible_interactions' in source_visual: visual['interaction_labels']=source_visual['visible_interactions']
    visual['people']=copy.deepcopy(event.get('semantic_people') or source_visual.get('people') or _people_semantics(event.get('people',[])))
    if event.get('relationships'): visual['relationships']=event['relationships']
    overrides=_vertical_override(event,plan)
    if overrides: visual['rendition_overrides']={'vertical':overrides}
    else: visual['rendition_overrides']={}
    hard_failures=list(vertical.get('hard_failures',[]))
    soft_warnings=list(vertical.get('soft_warnings',[]))
    visual['final_vertical']={
        'reframe_algorithm_version':REFRAME_ALGORITHM_VERSION,
        'vertical_validation_version':VERTICAL_VALIDATION_VERSION,
        'reframe_fingerprint':reframe_fp,
        'validation_status':vertical.get('status','PASS'),
        # These fields are deliberately persisted with the package rather than
        # only in the ledger, so a later reconciliation can be media-free.
        'hard_failures':hard_failures,
        'soft_warnings':soft_warnings,
        'publish_ready':vertical.get('status','PASS') == 'PASS' and not hard_failures,
        'review_required':vertical.get('status','PASS') != 'PASS' or bool(hard_failures),
        'active_picture':active_picture,
    }
    if vertical.get('post_render_audit',{}).get('face_safe_audit'):
        from .face_safe import VERSION,MODEL_NAME,MODEL_SHA256
        visual['final_vertical']['face_safe_validation']={'version':VERSION,'decision':vertical['post_render_audit']['face_safe_audit']['decision'],'model':MODEL_NAME,'model_sha256':MODEL_SHA256,'strategy':plan[0].get('strategy') if plan else None}
    data={'schema_version':'asset_metadata_v1','asset':{'id':aid,'slug':slug,'source_movie_id':movie_id},'source':{'movie_sha256':source_sha256,'active_picture':active_picture},'media':{'horizontal':_rendition(horizontal_file,horizontal_thumbnail,horizontal_time,horizontal,'landscape') | {'source_media':'movie.mp4','export_mode':horizontal_mode,'generation_from_source':horizontal_generation},'vertical':_rendition(vertical_file,vertical_thumbnail,vertical_time,vertical,'portrait') | {'source_media':'movie.mp4','export_mode':'source_crop_encode','generation_from_source':1}},'analysis':{'semantic_ready':True,'final_asset_semantics_validated':True,'source_video_analyzed':True,'profile':'production_v1','producer':'movie_broll_extractor','producer_version':'0.1.0','generated_at':'durable'},'source_timeline':{'start_seconds':event['start_seconds'],'end_seconds':event['end_seconds'],'visual_event_id':event.get('source_visual_event_id',event['visual_event_id']),'producer_window_id':event.get('producer_window_id'),'technical_shot_ids':event.get('source_shot_ids',[]),'narrative_segment_ids':event.get('narrative',{}).get('segment_ids',event.get('narrative_segment_ids',[])),'cross_shot_continuity':event.get('continuity',{})},'visual':visual,'audio':{'speech_present':{'value':False,'source':'export_contract','confidence':1.0}},'narrative':event.get('narrative') or {'segment_ids':event.get('narrative_segment_ids',[])},'editorial':event.get('editorial',{}),'export':{'reframe_applied':True,'reframe_profile':'local shot-aware subject track 3:4 crop'}}
    data['analysis']['final_asset_semantics_validated']=data['media']['vertical']['semantic_validated']
    return apply_publication_projection(data)
def _asset_metadata_contract_valid(data:dict[str,Any],assets:Path|None=None)->bool:
    media=data.get('media',{})
    valid=data.get('schema_version') == 'asset_metadata_v1' and 'asset_metadata_v1' not in data and 'source_asset_id' not in data.get('asset',{}) and all(x.get('technical_validated') is True and isinstance(x.get('semantic_validated'),bool) and isinstance(x.get('thumbnail'),dict) for x in (media.get('horizontal',{}),media.get('vertical',{}))) and media.get('horizontal',{}).get('semantic_validated') is True and (media.get('vertical',{}).get('semantic_validated') is True or data.get('visual',{}).get('final_vertical',{}).get('review_required') is True)
    if not valid or assets is None:return valid
    for rendition in (media.get('horizontal',{}),media.get('vertical',{})):
        for record in (rendition,rendition.get('thumbnail',{})):
            file=assets/str(record.get('file',''))
            if not file.is_file() or record.get('sha256') != sha256_file(file) or record.get('size_bytes') != file.stat().st_size:return False
    return True
def _horizontal_reuse_provenance(assets:Path,base:str,active_picture:dict[str,Any]|None=None)->dict[str,Any]|None:
    """A stale vertical may donate horizontal pixels only under this contract."""
    try:
        data=json.loads((assets/f'{base}.json').read_text()); horizontal=data.get('media',{}).get('horizontal',{})
        active_ok=active_picture is None or data.get('source',{}).get('active_picture') == active_picture
        if active_ok and (assets/f'{base}.mp4').is_file() and horizontal.get('source_media')=='movie.mp4' and horizontal.get('export_mode') in {'stream_copy','source_encode','source_crop_encode'} and horizontal.get('generation_from_source') in {0,1}: return horizontal
    except (OSError,json.JSONDecodeError): pass
    return None
def stream_copy_export_command(movie:Path,event:dict[str,Any],output:Path)->list[str]:
    """Attempt a source copy; boundary validation decides whether it is usable."""
    return ['ffmpeg','-y','-i',str(movie),'-ss',str(event['start_seconds']),'-to',str(event['end_seconds']),'-map','0:v:0','-c:v','copy','-an',str(output)]
def export_horizontal_from_source(movie:Path,event:dict[str,Any],output:Path,width:int,height:int,fps:float,active_picture:dict[str,Any]|None=None)->str:
    """Prefer an audited stream copy; otherwise perform one direct source encode."""
    expected=(event['end_frame_exclusive']-event['start_frame'])/fps; frames=event['end_frame_exclusive']-event['start_frame']; copied=output.with_suffix('.copy.tmp.mp4')
    active=active_picture or {'x':0,'y':0,'width':width,'height':height}
    aw,ah=int(active['width']),int(active['height'])
    if active.get('structural_bars'):
        start,end=int(event['start_frame']),int(event['end_frame_exclusive'])
        vf=f"trim=start_frame={start}:end_frame={end},crop={aw}:{ah}:{int(active['x'])}:{int(active['y'])},setpts=PTS-STARTPTS"
        subprocess.run(['ffmpeg','-y','-ss','0','-i',str(movie),'-map','0:v:0','-vf',vf,'-vsync','0','-frames:v',str(frames),'-c:v','libx264','-crf','16','-preset','medium','-pix_fmt','yuv420p','-an',str(output)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        container=probe(output,aw,ah,expected,frames)
        if container['status']!='PASS': output.unlink(missing_ok=True); raise RuntimeError('active-picture horizontal encode failed validation')
        return 'source_crop_encode'
    try:
        subprocess.run(stream_copy_export_command(movie,event,copied),check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        container=probe(copied,width,height,expected,frames); boundaries=boundary_validation(movie,copied,event)
        if container['status']=='PASS' and boundaries['status']=='PASS': copied.replace(output); return 'stream_copy'
    except (OSError,subprocess.CalledProcessError,RuntimeError): pass
    finally: copied.unlink(missing_ok=True)
    subprocess.run(ffmpeg_export_command(movie,event,output,fps),check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    container=probe(output,width,height,expected,frames); boundaries=boundary_validation(movie,output,event)
    if container['status']!='PASS' or boundaries['status']!='PASS':
        output.unlink(missing_ok=True); raise RuntimeError('source horizontal encode failed duration/frame boundary validation')
    return 'source_encode'
def _retire_package(assets:Path,base:str)->None:
    """Remove a stale complete package only after its horizontal source is copied to work."""
    for pre,suffix in (('', '.mp4'),('v','.mp4'),('', '.jpg'),('v','.jpg'),('', '.json')): (assets/f'{pre}{base}{suffix}').unlink(missing_ok=True)
def _remove_incomplete_assets(assets:Path)->None:
    for file in assets.glob('*'):
        if file.is_file():
            base=(file.name[1:] if file.name.startswith('v') else file.name).rsplit('.',1)[0]
            if not _complete_package(assets,base):file.unlink()


def _promote_complete_package(staged:list[Path]|tuple[Path,...],final:list[Path])->None:
    """Publish one flat Atlas-ready producer package as 5/5 or leave assets at 0/5.

    True multi-file filesystem atomicity is impossible.  This helper therefore
    validates the staging set first and rolls back every destination already
    promoted if a normal exception or interrupt occurs mid-promotion.
    A hard SIGKILL is reconciled by _remove_incomplete_assets on next startup.
    """
    staged=list(staged)
    final=list(final)

    if len(staged)!=5 or len(final)!=5:
        raise RuntimeError('asset package promotion requires exactly 5 files')

    if not all(x.is_file() for x in staged):
        missing=[x.name for x in staged if not x.is_file()]
        raise RuntimeError(f'asset package staging incomplete: {missing}')

    if any(x.exists() for x in final):
        dirty=[x.name for x in final if x.exists()]
        raise RuntimeError(f'asset package destination is not clean: {dirty}')

    promoted:list[Path]=[]

    try:
        # Metadata is deliberately last.
        for source,destination in zip(staged,final):
            shutil.move(str(source),str(destination))
            promoted.append(destination)

        if not all(x.is_file() for x in final):
            raise RuntimeError('asset package promotion postcondition failed')

    except BaseException:
        # A published producer package must never retain 1/5 .. 4/5 after a catchable failure.
        for destination in reversed(promoted):
            destination.unlink(missing_ok=True)
        raise
def _persisted_review_qa(run:Path,data:dict[str,Any])->dict[str,list[str]]|None:
    """Return package QA, falling back to the durable finalization ledger.

    Older package metadata only recorded REVIEW/PASS.  We never guess from
    that state: promotion is allowed only when the package itself or its
    ledger contains findings that the current policy can classify.
    """
    final=data.get('visual',{}).get('final_vertical',{})
    if isinstance(final.get('hard_failures'),list) or isinstance(final.get('soft_warnings'),list):
        return {'hard_failures':list(final.get('hard_failures',[])), 'soft_warnings':list(final.get('soft_warnings',[]))}
    timeline=data.get('source_timeline',{})
    event_id=timeline.get('visual_event_id')
    if timeline.get('producer_window_id'):
        event_id=str(event_id) + ':window:' + timeline['producer_window_id']
    try:
        ledger=json.loads((run/'processing_ledger.json').read_text())
        validation=ledger['events'][event_id]['stages']['vertical_validation'].get('validation')
    except (OSError,KeyError,TypeError,json.JSONDecodeError):
        return None
    if not isinstance(validation,dict): return None
    qa=classify_vertical_qa(validation)
    legacy=_legacy_reason_findings(validation)
    for key in ('hard_failures','soft_warnings'):
        for reason in legacy[key]:
            if reason not in qa[key]: qa[key].append(reason)
    return qa


_LEGACY_SOFT_REASONS={
    'POST_RENDER_RETRY':'one_or_more_shots_need_reframe',
    'POST_RENDER_AMBIGUOUS':'post_render_ambiguity',
    'focused_person_persistently_at_crop_edge':'focused_person_persistently_at_crop_edge',
    'face_safe_ambiguous':'face_safe_ambiguous',
    'one_or_more_shots_need_reframe':'one_or_more_shots_need_reframe',
    'crop_centered_between_competing_subjects':'crop_centered_between_competing_subjects',
    'competing_subject_dominates_target':'competing_subject_dominates_target',
    'target_binding_unstable':'target_binding_unstable',
    'post_render_detector_error':'post_render_detector_error',
}
_LEGACY_HARD_REASONS={
    'vertical_unreadable':'vertical_render_unreadable',
    'insufficient_render_decode':'vertical_render_unreadable',
    'focused_person_lost_in_render':'primary_subject_lost_in_render',
    'required_people_not_visible_together':'required_multi_person_interaction_lost',
}


def _legacy_reason_findings(validation:dict[str,Any])->dict[str,list[str]]:
    """Normalize old audit labels while retaining evidence-based hard labels."""
    values=[]
    for value in (validation.get('review_reason'), validation.get('reason')):
        if isinstance(value,str): values.append(value)
    audit=validation.get('post_render_audit',{})
    for value in (audit.get('reason'),):
        if isinstance(value,str): values.append(value)
    for shot in audit.get('shots',[]) if isinstance(audit,dict) else []:
        if isinstance(shot,dict) and isinstance(shot.get('reason'),str): values.append(shot['reason'])
    face=audit.get('face_safe_audit',{}) if isinstance(audit,dict) else {}
    if isinstance(face,dict) and isinstance(face.get('decision'),str): values.append('face_safe_'+face['decision'].lower())
    hard=[]; soft=[]
    for value in values:
        if value in _LEGACY_HARD_REASONS and _LEGACY_HARD_REASONS[value] not in hard: hard.append(_LEGACY_HARD_REASONS[value])
        elif value in _LEGACY_SOFT_REASONS and _LEGACY_SOFT_REASONS[value] not in soft: soft.append(_LEGACY_SOFT_REASONS[value])
    return {'hard_failures':hard,'soft_warnings':soft}


def _persisted_review_plan(run:Path,data:dict[str,Any])->list[dict[str,Any]]:
    timeline=data.get('source_timeline',{})
    event_id=timeline.get('visual_event_id')
    if timeline.get('producer_window_id'):
        event_id=str(event_id) + ':window:' + timeline['producer_window_id']
    try:
        ledger=json.loads((run/'processing_ledger.json').read_text())
        plan=ledger['events'][event_id]['stages']['vertical_reframe'].get('plan',[])
    except (OSError,KeyError,TypeError,json.JSONDecodeError):
        return []
    return plan if isinstance(plan,list) else []


def _local_legacy_vertical_qa(run:Path,video:Path,data:dict[str,Any])->dict[str,Any]:
    """Cheap media-only validation for review packages with lost old QA state.

    It intentionally does not use source frames, semantic providers, or a
    renderer.  A readable 3:4 existing render is publishable by default unless
    observed technical evidence says otherwise.
    """
    timeline=data.get('source_timeline',{})
    try: expected=float(timeline['end_seconds'])-float(timeline['start_seconds'])
    except (KeyError,TypeError,ValueError): expected=None
    # Prefer the original shot/crop plan when it survived in the ledger. This
    # validates rendered pixels only; it never re-encodes them or revisits the
    # semantic provider.
    plan=_persisted_review_plan(run,data)
    if plan and expected is not None:
        try:
            validation=validate_vertical(video,expected,plan,{})
            return {
                **classify_vertical_qa(validation),
                'locally_revalidated':True,
            }
        except (OSError,RuntimeError,cv2.error):
            # Continue with technical validation below. A local detector issue
            # is a warning, not proof that the already rendered asset failed.
            plan_warning='legacy_plan_validation_unavailable'
        else: plan_warning=None
    else: plan_warning=None
    cap=cv2.VideoCapture(str(video))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.)
    frames=[]
    if cap.isOpened() and count>0:
        for index in {0,max(0,count//2),max(0,count-1)}:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index); ok,frame=cap.read()
            if ok: frames.append(frame)
    cap.release()
    bars=any(float(np.mean(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)<8))>.92 for frame in frames)
    validation={
        'file_exists':video.is_file(), 'frame_count':count,
        'width':width, 'height':height,
        'aspect_ratio':'3:4' if width*4==height*3 and width>0 else 'other',
        'duration_valid':None if expected is None or fps<=0 else abs(count/fps-expected)<=1,
        'black_bars':bars, 'crop_plan_valid':True, 'shots':[],
        'post_render_audit':{'shots':[]},
    }
    qa=classify_vertical_qa(validation)
    legacy=_legacy_reason_findings(validation)
    for key in ('hard_failures','soft_warnings'):
        for reason in legacy[key]:
            if reason not in qa[key]: qa[key].append(reason)
    if plan_warning and plan_warning not in qa['soft_warnings']: qa['soft_warnings'].append(plan_warning)
    if expected is None:
        qa['insufficient_reason']='legacy_evidence_insufficient'
    return qa


def reconcile_review_packages(run:Path)->dict[str,Any]:
    """Atomically promote complete, soft-warning-only review packages.

    This copies the validated media and changes only metadata; it never opens
    source media or invokes a renderer.  A second invocation is a no-op.
    """
    review,assets,work=run/'review',run/'assets',run/'.work'
    result={'promoted':0,'hard_review':0,'insufficient_evidence':0,'incomplete':0,'packages':[]}
    if not review.is_dir(): return result
    assets.mkdir(parents=True,exist_ok=True); work.mkdir(parents=True,exist_ok=True)
    members=(('', '.mp4'),('v','.mp4'),('', '.jpg'),('v','.jpg'),('', '.json'))
    for metadata_path in sorted(review.glob('*.json')):
        base=metadata_path.stem
        try: data=json.loads(metadata_path.read_text())
        except (OSError,json.JSONDecodeError):
            result['incomplete']+=1; result['packages'].append({'package':base,'decision':'REMAIN_REVIEW','reason':'metadata_unreadable'}); continue
        if not _complete_package(review,base) or not _asset_metadata_contract_valid(data,review):
            result['incomplete']+=1; result['packages'].append({'package':base,'decision':'REMAIN_REVIEW','reason':'package_incomplete_or_hash_invalid'}); continue
        if data.get('editorial',{}).get('decision') != 'KEEP':
            result['hard_review']+=1; result['packages'].append({'package':base,'decision':'REMAIN_REVIEW','reason':'semantic_not_keep'}); continue
        qa=_persisted_review_qa(run,data)
        qa_source='persisted_qa'
        if qa is None:
            qa=_local_legacy_vertical_qa(run,review/f'v{base}.mp4',data)
            qa_source='local_render_validation'
        projected=copy.deepcopy(data)
        updated=projected.setdefault('visual',{}).setdefault('final_vertical',{})
        insufficient=qa.get('insufficient_reason')
        if insufficient:
            updated.update(validation_status='REVIEW',hard_failures=[],soft_warnings=list(qa['soft_warnings'])+[insufficient],publish_ready=False,review_required=True,review_reason=insufficient)
            write_json(review/f'{base}.json',apply_publication_projection(projected))
            result['insufficient_evidence']+=1; result['packages'].append({'package':base,'decision':'REMAIN_REVIEW','reason':insufficient,'qa_source':qa_source}); continue
        if qa['hard_failures'] or data.get('publication',{}).get('human_review',{}).get('status') == 'REJECTED':
            updated.update(validation_status='REVIEW',hard_failures=qa['hard_failures'],soft_warnings=qa['soft_warnings'],publish_ready=False,review_required=True,review_reason=qa['hard_failures'][0] if qa['hard_failures'] else 'human_rejected')
            write_json(review/f'{base}.json',apply_publication_projection(projected))
            result['hard_review']+=1; result['packages'].append({'package':base,'decision':'REMAIN_REVIEW','reason':updated['review_reason'],'qa_source':qa_source}); continue
        if any((assets/f'{prefix}{base}{suffix}').exists() for prefix,suffix in members):
            result['incomplete']+=1; result['packages'].append({'package':base,'decision':'REMAIN_REVIEW','reason':'asset_destination_not_clean','qa_source':qa_source}); continue
        updated.update(validation_status='PASS',hard_failures=[],soft_warnings=qa['soft_warnings'],publish_ready=True,review_required=False)
        projected=apply_publication_projection(projected)
        stage=work/'review-reconciliation'/base
        if stage.exists(): _retire_package(stage,base)
        stage.mkdir(parents=True,exist_ok=True)
        staged=[stage/f'{prefix}{base}{suffix}' for prefix,suffix in members]
        final_paths=[assets/path.name for path in staged]
        try:
            for source,target in zip((review/f'{base}.mp4',review/f'v{base}.mp4',review/f'{base}.jpg',review/f'v{base}.jpg'),staged[:4]): shutil.copy2(source,target)
            write_json(staged[4],projected)
            if not _complete_package(stage,base) or not _asset_metadata_contract_valid(projected,stage): raise RuntimeError('staged reconciliation package failed validation')
            _promote_complete_package(staged,final_paths)
            _retire_package(review,base)
            result['promoted']+=1; result['packages'].append({'package':base,'decision':'PROMOTED','reason':'no_hard_failures','qa_source':qa_source,'soft_warnings':qa['soft_warnings']})
        except (OSError,RuntimeError):
            _retire_package(stage,base); result['incomplete']+=1; result['packages'].append({'package':base,'decision':'REMAIN_REVIEW','reason':'promotion_staging_failed','qa_source':qa_source})
    write_json(run/'review_reconciliation.json',{'schema_version':'review_reconciliation_v1',**result})
    return result


def _existing_registered_package(run:Path,assets:Path,event:dict[str,Any])->bool:
    """A complete prior package already has validated semantics; metadata changes do not refresh them."""
    try:
        entry=json.loads((run/'asset_registry.json').read_text()).get('events',{}).get(event['visual_event_id'],{})
        return bool(entry) and _complete_package(assets,f"{entry['asset_id']}-{entry['slug']}")
    except (OSError,KeyError,json.JSONDecodeError): return False
def _existing_review_package(run:Path,review:Path,event:dict[str,Any])->bool:
    """Recognize both canonical and pre-5/5 review records for resume."""
    try:
        entry=json.loads((run/'asset_registry.json').read_text()).get('events',{}).get(event['visual_event_id'],{})
        base=f"{entry['asset_id']}-{entry['slug']}"
        return (review/f'{base}.json').is_file() and (review/f'v{base}.mp4').is_file()
    except (OSError,KeyError,json.JSONDecodeError): return False
def next_vertical_strategy(strategy, validation):
    if validation.get('status')=='PASS' or strategy=='face_priority':
        return None
    face=validation.get('post_render_audit',{}).get('face_safe_audit',{})
    if face.get('decision') in {'SOURCE_LIMIT','AMBIGUOUS','ERROR'}:
        return None
    if face.get('repairable'):
        return 'face_priority'
    return 'interaction_aware' if strategy=='subject_focus' else None


def finalize_pilot(input_dir:Path,window_id:str,keep_debug_artifacts:bool=False,
                   candidates:list[dict[str,Any]]|None=None,
                   shots:dict[str,dict[str,Any]]|None=None,
                   detector_preflight:dict[str,Any]|None=None)->dict[str,Any]:
    """Finalize validated events.

    ``candidates`` and ``shots`` are the production entry point.  Keeping the
    original positional pilot interface preserves the regression/debug command
    while ensuring the full-movie runner never needs a pilot window artifact.
    """
    root=input_dir.resolve().parents[1]; movie_id=input_dir.name; run=root/'runs'/movie_id; pilot=run/'broll-pilot-v1'/window_id; candidates_path=pilot/'candidates.json'
    if candidates is None:
        candidates=json.loads(candidates_path.read_text()).get('candidates',[])
    else:
        # Only an owned, bounded scratch path is needed for legacy cleanup of a
        # former pilot export; it is never a source of production pixels.
        pilot=run/'.work'/'production-finalization'; pilot.mkdir(parents=True,exist_ok=True)
        candidates_path=pilot/'candidates.json'
    assets=run/'assets'; assets.mkdir(parents=True,exist_ok=True); _remove_incomplete_assets(assets); work=run/'.work'; work.mkdir(parents=True,exist_ok=True); review_dir=run/'review'
    reconciliation=reconcile_review_packages(run)
    if shots is None:
        shots_path=run/'visual-smoke-v1'/'shots.jsonl'; shots={x['shot_id']:x for x in (json.loads(y) for y in shots_path.read_text().splitlines() if y.strip())} if shots_path.exists() else {}
    # Finalize is self-healing: stale event semantics are refreshed through the
    # existing bounded event request, never by silently using unclear geometry.
    from .broll_pilot import (SEMANTIC_PROMPT_VERSION, SEMANTIC_SCHEMA_VERSION,
                              semantic_validate, shot_focus_compatible, prepublication_event_coherence)
    incoherent=[x for x in candidates if x.get('editorial',{}).get('decision')=='KEEP' and prepublication_event_coherence(x)['status']=='SPLIT_REQUIRED']
    if incoherent:
        ids=', '.join(str(x.get('visual_event_id',x.get('candidate_id','?'))) for x in incoherent)
        raise ValueError(f'visual events require split before rendering: {ids}')
    for candidate in candidates:
        candidate.setdefault('technical_shots',[{'shot_id':sid,'start_seconds':shots.get(sid,{}).get('start_seconds'),'end_seconds':shots.get(sid,{}).get('end_seconds'),'representative_image_index':i} for i,sid in enumerate(candidate.get('source_shot_ids',[]))])
    incompatible=[x for x in candidates if x.get('editorial',{}).get('decision')=='KEEP' and not _existing_registered_package(run,assets,x) and not _existing_review_package(run,review_dir,x) and not shot_focus_compatible({'visual':x.get('visual',{})},x)]
    semantic_reused=0
    if any(x.get('producer_window_id') for x in incompatible):
        raise ValueError('asset-window focus evidence incompatible; explicit semantic review required')
    if incompatible:
        srt=next((input_dir/x for x in ('subtitles.srt',f'{movie_id}.srt') if (input_dir/x).is_file()),None); narrative=run/'narrative-v2'/'narrative_map.json'
        if srt is None or not narrative.is_file(): raise RuntimeError('shot-focus semantic refresh requires canonical SRT and narrative map')
        semantic_reused=semantic_validate(incompatible,input_dir/'movie.mp4',srt,narrative,pilot/'semantic_checkpoints',24.,window_id).get('reused',0)
        write_json(candidates_path,{'schema_version':'broll_pilot_candidates_v4','semantic_schema_version':SEMANTIC_SCHEMA_VERSION,'semantic_prompt_version':SEMANTIC_PROMPT_VERSION,'window_id':window_id,'candidates':candidates})
    movie=input_dir/'movie.mp4'; source=cv2.VideoCapture(str(movie)); source_width,source_height=int(source.get(cv2.CAP_PROP_FRAME_WIDTH)),int(source.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=source.get(cv2.CAP_PROP_FPS) or 24.; source.release()
    from .active_picture import load_or_detect
    from .production_profile import load as load_production_profile
    try: active_picture=load_or_detect(movie,run,load_production_profile()['active_picture'])
    except (RuntimeError, cv2.error, AttributeError):
        # Finalization's historical unit-level API permits an injected capture;
        # production preflight has already persisted the authoritative geometry.
        active_picture={'x':0,'y':0,'width':source_width,'height':source_height,'source_width':source_width,'source_height':source_height,'detection_profile':'fallback_full_frame','structural_bars':False}
    width,height=int(active_picture['width']),int(active_picture['height'])
    source_sha256=_source_movie_sha256(run,movie); ledger=ProcessingLedger(run,movie_id,{'finalization_version':'production_v1','reframe_algorithm_version':REFRAME_ALGORITHM_VERSION,'vertical_validation_version':VERTICAL_VALIDATION_VERSION,'active_picture':active_picture,'movie_code':movie_code(run,movie_id)}); completed=review=reused=review_reused=horizontal_reused=failed_retryable=failed_final=0
    if any(e.get('editorial',{}).get('decision')=='KEEP' and e.get('editorial',{}).get('status')=='VALIDATED' for e in candidates) and detector_preflight is None: person_detector_preflight()
    for e in candidates:
        if e.get('editorial',{}).get('decision')!='KEEP' or e.get('editorial',{}).get('status')!='VALIDATED':continue
        # A Visual Event may cover only a derived subrange of a canonical long
        # technical shot.  Keep its canonical ID while supplying the event-
        # bounded range to crop planning and its vertical reuse fingerprint.
        event_shots=dict(shots)
        for technical in e.get('technical_shots',[]):
            shot_id=technical.get('shot_id')
            if shot_id:
                event_shots[shot_id]={**event_shots.get(shot_id,{}),**technical,'shot_id':shot_id}
        # Do not include reframe config here: register() would incorrectly stale
        # completed horizontal/semantic work.  Vertical gets its own fingerprint.
        fp=fingerprint({'range':[e['start_frame'],e['end_frame_exclusive']],'shots':e.get('source_shot_ids',[]),'semantic':e.get('visual',{}),'active_picture':active_picture,'version':'horizontal-v2'}); ledger.register(e,fp); aid,slug=asset_identity(run,movie_id,e); base=f'{aid}-{slug}'; final=[assets/f'{base}.mp4',assets/f'v{base}.mp4',assets/f'{base}.jpg',assets/f'v{base}.jpg',assets/f'{base}.json']; review_final=[review_dir/f'{base}.mp4',review_dir/f'v{base}.mp4',review_dir/f'{base}.jpg',review_dir/f'v{base}.jpg',review_dir/f'{base}.json']; eid=e['visual_event_id']; expected=(e['end_frame_exclusive']-e['start_frame'])/fps; reframe_fp=reframe_fingerprint(e,event_shots,width,height,active_picture=active_picture)
        if _vertical_reuse_valid(assets,base,reframe_fp):
            old_data=json.loads((assets/f'{base}.json').read_text())
            if not _asset_metadata_contract_valid(old_data,assets):
                h,v,ht,vt=assets/f'{base}.mp4',assets/f'v{base}.mp4',assets/f'{base}.jpg',assets/f'v{base}.jpg'
                horizontal=_technical_properties(h,probe(h,width,height,expected,e['end_frame_exclusive']-e['start_frame']))
                vertical=_validated_technical(v,expected,e['end_frame_exclusive']-e['start_frame'])
                prior_vertical=old_data.get('visual',{}).get('final_vertical',{})
                final_validation={**vertical,**prior_vertical.get('validation',{})}
                htime=old_data.get('thumbnail',{}).get('timestamp_seconds',0.)
                vtime=old_data.get('thumbnail',{}).get('vertical_timestamp_seconds',0.)
                reused_horizontal=_horizontal_reuse_provenance(assets,base,active_picture) or {}
                if horizontal['status']=='PASS' and final_validation['status']=='PASS':
                    write_json(assets/f'{base}.json',_asset_metadata(movie_id,source_sha256,aid,slug,e,h,v,ht,vt,htime,vtime,horizontal,final_validation,str(reused_horizontal.get('export_mode','source_encode')),int(reused_horizontal.get('generation_from_source',1)),prior_vertical.get('reframe',{}).get('shots',[]),int(prior_vertical.get('reframe',{}).get('attempts',0)),reframe_fp))
                    ledger.stage(eid,'metadata','COMPLETE',path=str(assets/f'{base}.json'),reused_media=True)
                    reused+=1; ledger.stage(eid,'finalization','COMPLETE',decision='PASS',publish_ready=True,atlas_ready=True,asset_hub_ready=True,reused=True,reframe_fingerprint=reframe_fp); continue
            else:
                reuse_decision='PASS' if old_data.get('visual',{}).get('final_vertical',{}).get('validation_status')=='PASS' else 'HUMAN_APPROVED'
                ready=is_publish_ready(old_data)
                reused+=1; ledger.stage(eid,'finalization','COMPLETE',decision=reuse_decision,publish_ready=ready,atlas_ready=is_atlas_ready(old_data),asset_hub_ready=ready,reused=True,reframe_fingerprint=reframe_fp); continue
        prior=ledger.data['events'][eid]['stages']
        stage=work/eid; stage.mkdir(parents=True,exist_ok=True); h,v,ht,vt,md=[stage/x.name for x in final]; old=pilot/'exports'/f"{e['candidate_id']}.mp4"; ledger.stage(eid,'horizontal_export','RUNNING')
        # A stale package can still donate its verified horizontal asset, but no
        # stale vertical member may remain in assets during replacement.
        reused_horizontal=_horizontal_reuse_provenance(assets,base,active_picture)
        if reused_horizontal:
            shutil.copy2(assets/f'{base}.mp4',h); _retire_package(assets,base); horizontal_mode=str(reused_horizontal['export_mode']); horizontal_generation=int(reused_horizontal['generation_from_source']); horizontal_reused+=1; ledger.stage(eid,'horizontal_export','COMPLETE',reused=True,export_mode=horizontal_mode)
        else:
            if (assets/f'{base}.mp4').exists(): _retire_package(assets,base)
            horizontal_mode=export_horizontal_from_source(movie,e,h,source_width,source_height,fps,active_picture); horizontal_generation=0 if horizontal_mode=='stream_copy' else 1; ledger.stage(eid,'horizontal_export','COMPLETE',export_mode=horizontal_mode)
        horizontal=_technical_properties(h,probe(h,width,height,expected,e['end_frame_exclusive']-e['start_frame'])); ledger.stage(eid,'horizontal_validation','COMPLETE' if horizontal['status']=='PASS' else 'FAILED_RETRYABLE',validation=horizontal['status'])
        if horizontal['status']!='PASS':continue
        vertical=None; plan=[]; execution_error=None; attempt=0
        # A review MP4 with the same rendering fingerprint is valid pixel work.
        # Revalidate it under the current validator before considering a render.
        prior_review=prior.get('vertical_validation',{})
        review_video=review_dir/f'v{base}.mp4'
        if prior_review.get('status') == 'COMPLETE' and prior_review.get('decision') == 'REVIEW_VERTICAL' and prior_review.get('reframe_fingerprint') == reframe_fp and review_video.is_file():
            plan=list(prior.get('vertical_reframe',{}).get('plan',[])); attempt=int(prior.get('vertical_reframe',{}).get('attempt',0))
            if plan:
                shutil.copy2(review_video,v); vertical=validate_vertical(v,expected,plan,e)
                ledger.stage(eid,'vertical_reframe','COMPLETE',attempt=attempt,reused=True,strategy=prior.get('vertical_reframe',{}).get('strategy'),reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,reframe_fingerprint=reframe_fp)
                review_reused+=1
        if vertical is None:
            strategies=['subject_focus']
            for attempt,strategy in enumerate(strategies,1):
                try:
                    plan=build_shot_crop_plan(movie,e,event_shots,width,height,strategy=strategy,active_picture=active_picture); v.unlink(missing_ok=True); ledger.stage(eid,'vertical_reframe','RUNNING',attempt=attempt,strategy=strategy,reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,reframe_fingerprint=reframe_fp,plan=plan,source_media='movie.mp4',export_mode='source_crop_encode'); render_vertical(movie,v,e,plan,active_picture); vertical=validate_vertical(v,expected,plan,e)
                    ledger.stage(eid,'vertical_reframe','COMPLETE',attempt=attempt,strategy=strategy,reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,reframe_fingerprint=reframe_fp)
                except (OSError,RuntimeError,cv2.error,subprocess.CalledProcessError) as error:
                    execution_error=str(error); ledger.stage(eid,'vertical_reframe','FAILED_RETRYABLE',attempt=attempt,error=execution_error,reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,reframe_fingerprint=reframe_fp); break
                retry=next_vertical_strategy(strategy,vertical)
                if retry is None:break
                strategies.append(retry)
        if execution_error:
            failed_retryable+=1; ledger.stage(eid,'vertical_validation','FAILED_RETRYABLE',reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,reframe_fingerprint=reframe_fp,error=execution_error); ledger.stage(eid,'finalization','FAILED_RETRYABLE',publish_ready=False,atlas_ready=False,asset_hub_ready=False,error=execution_error); continue
        decision='PASS' if vertical and vertical['status']=='PASS' else 'REVIEW_VERTICAL'
        ledger.stage(eid,'vertical_validation','COMPLETE',decision=decision,reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,vertical_validation_version=VERTICAL_VALIDATION_VERSION,reframe_fingerprint=reframe_fp,validation=vertical)
        destination=review_dir if decision == 'REVIEW_VERTICAL' else assets
        destination_final=review_final if decision == 'REVIEW_VERTICAL' else final
        try:
            destination.mkdir(parents=True,exist_ok=True)
            vertical=_technical_properties(v,vertical); htime=thumbnail(h,ht); ledger.stage(eid,'horizontal_thumbnail','COMPLETE'); vtime=thumbnail(v,vt); ledger.stage(eid,'vertical_thumbnail','COMPLETE',reframe_algorithm_version=REFRAME_ALGORITHM_VERSION,vertical_validation_version=VERTICAL_VALIDATION_VERSION,reframe_fingerprint=reframe_fp)
            metadata=_asset_metadata(movie_id,source_sha256,aid,slug,e,h,v,ht,vt,htime,vtime,horizontal,vertical,horizontal_mode,horizontal_generation,plan,attempt,reframe_fp,active_picture)
            write_json(md,metadata)
            staged_metadata=json.loads(md.read_text())
            if not _asset_metadata_contract_valid(staged_metadata,stage):
                raise RuntimeError('staged asset package failed metadata/hash validation')

            # Retire only this event's prior package (including a legacy 2/5
            # review) after a complete, validated replacement is staged.
            _retire_package(destination,base)
            _promote_complete_package((h,v,ht,vt,md),destination_final)

            published_metadata=json.loads(destination_final[-1].read_text())
            if not _complete_package(destination,base):
                raise RuntimeError('published asset package is not 5/5')
            if not _asset_metadata_contract_valid(published_metadata,destination):
                raise RuntimeError('published asset package failed metadata/hash validation')

        except (OSError,RuntimeError,json.JSONDecodeError) as error:
            _retire_package(destination,base)
            failed_retryable+=1
            ledger.stage(
                eid,
                'metadata',
                'FAILED_RETRYABLE',
                error=str(error),
            )
            ledger.stage(
                eid,
                'finalization',
                'FAILED_RETRYABLE',
                publish_ready=False,
                atlas_ready=False,
                asset_hub_ready=False,
                error=str(error),
                reframe_fingerprint=reframe_fp,
            )
            continue

        if decision == 'REVIEW_VERTICAL':
            review+=1; ledger.stage(eid,'metadata','COMPLETE',path=str(review_final[-1])); ledger.stage(eid,'cleanup','COMPLETE',removed_bytes=safe_cleanup(work,keep_debug_artifacts)); ledger.stage(eid,'finalization','COMPLETE',decision='REVIEW_VERTICAL',publish_ready=False,atlas_ready=False,asset_hub_ready=False,reframe_fingerprint=reframe_fp); continue
        old.unlink(missing_ok=True); _retire_package(review_dir,base); ledger.stage(eid,'metadata','COMPLETE',path=str(final[-1])); ledger.stage(eid,'cleanup','COMPLETE',removed_bytes=safe_cleanup(work,keep_debug_artifacts)); ledger.stage(eid,'finalization','COMPLETE',decision='PASS',publish_ready=True,atlas_ready=True,asset_hub_ready=True,reframe_fingerprint=reframe_fp); completed+=1
    final_bytes=sum(x.stat().st_size for x in assets.glob('*') if x.is_file()); temp_bytes=sum(x.stat().st_size for x in work.rglob('*') if x.is_file()); status='PARTIAL' if failed_retryable or failed_final else 'COMPLETE'; ledger.summary(status=status,final_assets=completed,reconciled_promotions=reconciliation['promoted'],vertical_review=review,vertical_reused=reused,vertical_review_reused=review_reused,horizontal_reused=horizontal_reused,semantic_reused=semantic_reused,vertical_failed_retryable=failed_retryable,vertical_failed_final=failed_final,disk={'final_assets_bytes':final_bytes,'temporary_bytes':temp_bytes}); return {'status':status,'completed':completed,'review':review,'reconciled_promotions':reconciliation['promoted'],'reconciliation':reconciliation,'reused':reused,'review_reused':review_reused,'horizontal_reused':horizontal_reused,'semantic_reused':semantic_reused,'failed_retryable':failed_retryable,'failed_final':failed_final,'assets':assets.relative_to(run).as_posix(),'review_dir':review_dir.relative_to(run).as_posix()}
