"""Single-window B-roll pilot: technical candidates, semantics, exact exports."""
from __future__ import annotations
import json, math, os, subprocess
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from .srt import parse_srt_file
from .utils import write_json
from .broll_semantics import GeminiBrollSemanticProvider, PROMPT as SEMANTIC_PROMPT, SemanticProvider, validate_response

PILOT_WINDOW="SW_02"; SAMPLE_FPS=3.0; KEEP=70; REVIEW=50
SEMANTIC_SCHEMA_VERSION="broll_semantics_v2"; SEMANTIC_PROMPT_VERSION="broll_semantic_prompt_v2"

def _root(input_dir:Path)->Path: return input_dir.resolve().parents[1]
def _overlap(a:float,b:float,c:float,d:float)->float: return max(0.,min(b,d)-max(a,c))
def _num(v:float)->float: return round(float(v),4)

def discover(input_dir:Path, window_id:str=PILOT_WINDOW)->dict[str,Path|dict[str,Any]|str]:
    root=_root(input_dir); smoke=root/'runs'/input_dir.name/'visual-smoke-v1'; narrative=root/'runs'/input_dir.name/'narrative-v2'/'narrative_map.json'
    required={'movie':input_dir/'movie.mp4','windows':smoke/'windows.json','shots':smoke/'shots.jsonl','profile':smoke/'selected_profile.json','narrative':narrative}
    for name,path in required.items():
        if not path.is_file(): raise FileNotFoundError(f"required {name} artifact does not exist: {path}")
    srt=next((input_dir/x for x in ('subtitles.srt',f'{input_dir.name}.srt') if (input_dir/x).is_file()),None)
    if srt is None: raise FileNotFoundError('canonical SRT not found (expected subtitles.srt or movie-id.srt)')
    windows=json.loads(required['windows'].read_text())['windows']; available=[x.get('window_id') for x in windows if x.get('window_id')]
    window=next((x for x in windows if x.get('window_id')==window_id),None)
    if not window: raise ValueError(f"requested visual smoke window {window_id!r} is absent; available window IDs: {', '.join(available) or '(none)'}")
    profile=json.loads(required['profile'].read_text())
    if float(profile.get('selected_threshold',-1)) != 24.: raise ValueError('pilot requires selected threshold 24')
    return {**required,'srt':srt,'window':window,'window_id':window_id,'root':root}

def load_shots(paths:dict[str,Any])->list[dict[str,Any]]:
    w=paths['window']; shots=[json.loads(x) for x in Path(paths['shots']).read_text().splitlines() if x.strip()]
    window_id=str(paths['window_id'])
    result=[x for x in shots if x.get('window_id')==window_id and float(x.get('detector',{}).get('threshold',-1))==24.]
    result.sort(key=lambda x:(x['start_seconds'],x['shot_id']))
    if not result: raise ValueError(f'no selected threshold-24 {window_id} shots')
    prior=float(w['start_seconds'])
    for shot in result:
        start,end=float(shot['start_seconds']),float(shot['end_seconds'])
        if end<=start or abs(start-prior)>0.05 or start<float(w['start_seconds'])-.05 or end>float(w['end_seconds'])+.05: raise ValueError(f'technical shots are not ordered, continuous, positive, and inside {window_id}')
        prior=end
    if abs(prior-float(w['end_seconds']))>.05: raise ValueError(f'technical shots do not cover {window_id}')
    return result

def visual_signals(movie:Path, shots:list[dict[str,Any]], sample_fps:float=SAMPLE_FPS)->list[dict[str,Any]]:
    cap=cv2.VideoCapture(str(movie)); out=[]
    for shot in shots:
        frames=[]; t=float(shot['start_seconds']); end=float(shot['end_seconds'])
        while t<end:
            cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,frame=cap.read()
            if ok: frames.append(frame)
            t+=1/sample_fps
        if not frames: raise RuntimeError(f"cannot decode {shot['shot_id']}")
        gray=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in frames]
        values=np.concatenate([g.ravel() for g in gray]); sharp=float(np.mean([cv2.Laplacian(g,cv2.CV_64F).var() for g in gray]))
        motion=float(np.mean([np.mean(cv2.absdiff(a,b)) for a,b in zip(gray,gray[1:])])) if len(gray)>1 else 0.
        hist=cv2.calcHist([cv2.cvtColor(frames[len(frames)//2],cv2.COLOR_BGR2HSV)],[0,1],None,[16,16],[0,180,0,256]); cv2.normalize(hist,hist)
        out.append({'brightness_mean':float(values.mean()),'brightness_std':float(values.std()),'sharpness_score':sharp,'motion_score':motion,'near_black_fraction':float((values<20).mean()),'_hist':hist})
    cap.release(); return out

def add_context(shots:list[dict[str,Any]], srt:Path, narrative:Path)->None:
    cues=parse_srt_file(srt).cues; segments=json.loads(narrative.read_text()).get('segments',[])
    for s in shots:
        a,b=float(s['start_seconds']),float(s['end_seconds']); duration=b-a
        s['subtitle_overlap_seconds']=sum(_overlap(a,b,c.start_seconds,c.end_seconds) for c in cues)
        s['subtitle_occupancy_ratio']=min(1.,s['subtitle_overlap_seconds']/duration)
        s['narrative_segment_ids']=[x['segment_id'] for x in segments if _overlap(a,b,float(x['start_seconds']),float(x['end_seconds']))>0]

def _similarity(a:dict,b:dict)->float: return max(0.,min(1.,float(cv2.compareHist(a['_hist'],b['_hist'],cv2.HISTCMP_CORREL)+1)/2))
def generate_groups(shots:list[dict[str,Any]])->list[list[int]]:
    groups=[]; i=0
    while i<len(shots):
        duration=float(shots[i]['duration_seconds']); group=[i]
        # Keep useful standalones; greedily join only short shots with direct successor.
        while duration<4 and i+1<len(shots) and duration+float(shots[i+1]['duration_seconds'])<=12:
            if _similarity(shots[i],shots[i+1])<.12: break
            i+=1; group.append(i); duration+=float(shots[i]['duration_seconds'])
        if 3<=duration<=15: groups.append(group)
        i+=1
    return groups

def score_candidate(candidate:dict[str,Any])->dict[str,float]:
    d=candidate['duration_seconds']; duration=25*max(0.,1-abs(d-7)/8)
    sig=candidate['signals']; quality=25*(.45*min(1,sig['sharpness']/150)+.35*(1-abs(sig['brightness']-110)/145)+.20*(1-sig['near_black_fraction']))
    continuity=20*(.6*sig['visual_continuity']+.4*(1-min(1,sig['subtitle_occupancy'])))
    motion=15*(1-min(1,abs(sig['motion']-12)/20))
    simple=15*(1/(1+.18*(len(candidate['source_shot_ids'])-1)))
    values={'duration_fit':duration,'visual_quality':quality,'continuity':continuity,'motion_usefulness':motion,'structural_simplicity':simple}
    values={k:round(max(0,min(100,v)),2) for k,v in values.items()}; values['total']=round(sum(values.values()),2); return values

def candidates(shots:list[dict[str,Any]])->list[dict[str,Any]]:
    result=[]
    for group in generate_groups(shots):
        part=[shots[i] for i in group]; a,b=float(part[0]['start_seconds']),float(part[-1]['end_seconds']); signals={'brightness':float(np.mean([x['brightness_mean'] for x in part])),'sharpness':float(np.mean([x['sharpness_score'] for x in part])),'motion':float(np.mean([x['motion_score'] for x in part])),'near_black_fraction':float(np.mean([x['near_black_fraction'] for x in part])),'subtitle_occupancy':float(np.mean([x['subtitle_occupancy_ratio'] for x in part])),'visual_continuity':float(np.mean([_similarity(x,y) for x,y in zip(part,part[1:])])) if len(part)>1 else 1.}
        c={'start_frame':int(part[0].get('start_frame', round(a*24))), 'end_frame_exclusive':int(part[-1].get('end_frame_exclusive',round(b*24))), 'start_seconds':_num(a),'end_seconds':_num(b),'duration_seconds':_num(b-a),'source_shot_ids':[x['shot_id'] for x in part],'narrative_segment_ids':list(dict.fromkeys(y for x in part for y in x['narrative_segment_ids'])),'signals':{k:_num(v) for k,v in signals.items()}}
        c['score']=score_candidate(c); structural='KEEP' if c['score']['total']>=KEEP else 'REVIEW' if c['score']['total']>=REVIEW else 'REJECT'; c['structural_decision']=structural; c['editorial']={'decision':structural, 'status':'PROVISIONAL'}; result.append(c)
    return dedupe(result)

def dedupe(items:list[dict[str,Any]])->list[dict[str,Any]]:
    """Retain every technical candidate; semantic diversity is decided later.

    Deleting overlap variants before their meaning is known made the former cap both
    arbitrary and unauditable.
    """
    result=sorted(items,key=lambda x:(x['start_seconds'],x['end_seconds']))
    for i,x in enumerate(result,1):
        x['candidate_id']=f'BRC_{i:04d}'
        x.setdefault('semantic_redundancy',{'status':'NOT_EVALUATED'})
    return result

def ffmpeg_export_command(movie:Path,c:dict[str,Any],output:Path, fps:float=24.0)->list[str]:
    """Coarse accurate decode, then trim by frame number relative to that decode.

    A second output timestamp seek was susceptible to timestamp rounding and could
    emit the preceding frame.  The sole input seek starts a small decoded segment at
    a known coarse frame; ``trim`` then owns exact [start, end) frame selection.
    """
    start=int(c['start_frame']) if 'start_frame' in c else round(float(c['start_seconds'])*fps)
    end=int(c.get('end_frame_exclusive',start+round(float(c.get('duration_seconds', 0))*fps)))
    if end <= start: raise ValueError('end_frame_exclusive must be greater than start_frame')
    coarse_frame=max(0,start-max(1,round(fps)))
    relative_start=start-coarse_frame; count=end-start
    vf=f"trim=start_frame={relative_start}:end_frame={relative_start + count},setpts=PTS-STARTPTS"
    return ['ffmpeg','-y','-ss',f'{coarse_frame/fps:.9f}','-i',str(movie),'-map','0:v:0','-vf',vf,'-vsync','0','-frames:v',str(count),'-c:v','libx264','-crf','19','-preset','medium','-an',str(output)]
def probe(path:Path, width:int,height:int, expected:float, expected_frame_count:int|None=None)->dict[str,Any]:
    raw=subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries','stream=codec_type,codec_name,width,height,nb_read_frames:format=duration','-of','json',str(path)],text=True); data=json.loads(raw); streams=data.get('streams',[]); video=[x for x in streams if x['codec_type']=='video']; audio=[x for x in streams if x['codec_type']=='audio']; duration=float(data.get('format',{}).get('duration',0)); actual=int(video[0]['nb_read_frames']) if video and str(video[0].get('nb_read_frames','')).isdigit() else None
    count_ok=expected_frame_count is None or actual==expected_frame_count
    ok=path.is_file() and path.stat().st_size>0 and len(video)==1 and video[0].get('codec_name')=='h264' and video[0].get('width')==width and video[0].get('height')==height and not audio and abs(duration-expected)<=1.0 and count_ok
    return {'path':str(path),'status':'PASS' if ok else 'FAIL','duration_seconds':duration,'video_streams':len(video),'audio_streams':len(audio),'codec':video[0].get('codec_name') if video else None,'expected_frame_count':expected_frame_count,'actual_frame_count':actual}
def review_reel_command(exports:list[Path], output:Path)->list[str]:
    # black gaps are omitted deliberately: all exports have the same encoded format.
    return ['ffmpeg','-y',*sum((['-i',str(x)] for x in exports),[]),'-filter_complex',f'concat=n={len(exports)}:v=1:a=0','-c:v','libx264','-crf','19','-an',str(output)]
def contact_sheet(movie:Path, items:list[dict[str,Any]], output:Path)->None:
    cap=cv2.VideoCapture(str(movie)); tiles=[]
    for c in items:
        cap.set(cv2.CAP_PROP_POS_MSEC,((c['start_seconds']+c['end_seconds'])/2)*1000); ok,f=cap.read()
        if ok:
            f=cv2.resize(f,(320,134)); cv2.putText(f,f"{c['candidate_id']} {c['duration_seconds']:.1f}s {c['score']['total']:.0f} {c['editorial']['decision']}",(5,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA); tiles.append(f)
    cap.release()
    if not tiles: raise RuntimeError('could not create contact sheet')
    cols=3; blank=np.zeros_like(tiles[0]); tiles += [blank]*((-len(tiles))%cols); cv2.imwrite(str(output),cv2.vconcat([cv2.hconcat(tiles[i:i+cols]) for i in range(0,len(tiles),cols)]),[cv2.IMWRITE_JPEG_QUALITY,85])

def _cue_context(cues: list[Any], start: float, end: float, padding: float=5.) -> dict[str, Any]:
    def row(x: Any) -> dict[str, Any]: return {'cue_id':x.cue_id,'start_seconds':x.start_seconds,'end_seconds':x.end_seconds,'text':x.text}
    return {'asset_overlap':[row(x) for x in cues if _overlap(start,end,x.start_seconds,x.end_seconds)>0], 'context_window':[row(x) for x in cues if _overlap(start-padding,end+padding,x.start_seconds,x.end_seconds)>0], 'literal_transcription':False}

def _field_values(value:Any)->list[Any]:
    if value is None: return []
    if isinstance(value,dict): return _field_values(value.get('value'))
    if isinstance(value,list): return [y for x in value for y in _field_values(x)]
    return [value] if value != '' else []

def _narrative_context(segments:list[dict[str,Any]], start:float,end:float)->dict[str,Any]:
    selected=[x for x in segments if _overlap(start,end,float(x['start_seconds']),float(x['end_seconds']))>0]
    def collect(*names:str)->list[Any]: return [v for x in selected for name in names for v in _field_values(x.get(name))]
    return {'segment_ids':[x['segment_id'] for x in selected], 'summary_es':collect('narrative_summary_es','narrative_summary','summary_es'), 'tone':collect('narrative_tone','tone'), 'themes':collect('themes'), 'interaction_context':collect('interaction_context','narrative_function'), 'literal_transcription':False}

def candidate_contact_sheet(movie:Path,c:dict[str,Any],fps:float)->bytes:
    """Build an in-memory five-frame sheet; never samples outside [start,end)."""
    start,end=int(c['start_frame']),int(c['end_frame_exclusive']); positions=[start, start+(end-start)//4, start+(end-start)//2, start+3*(end-start)//4, end-1]
    cap=cv2.VideoCapture(str(movie)); frames=[]
    for frame_no in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no); ok,frame=cap.read()
        if not ok: cap.release(); raise RuntimeError(f"cannot decode candidate frame {frame_no}")
        frames.append(cv2.resize(frame,(320,180)))
    cap.release(); ok, encoded=cv2.imencode('.jpg',cv2.hconcat(frames),[cv2.IMWRITE_JPEG_QUALITY,85])
    if not ok: raise RuntimeError('cannot encode candidate contact sheet')
    return encoded.tobytes()

def boundary_validation(movie:Path, exported:Path, c:dict[str,Any])->dict[str,Any]:
    """Authoritatively reject an export whose boundary provenance is wrong."""
    source=cv2.VideoCapture(str(movie)); result={'candidate_id':c['candidate_id'],'source_frame_immediately_before':int(c['start_frame'])-1,'source_first_frame':int(c['start_frame']),'source_last_candidate_frame':int(c['end_frame_exclusive'])-1,'source_frame_immediately_after':int(c['end_frame_exclusive'])}
    def read(cap:Any,n:int)->Any:
        if n<0:return None
        cap.set(cv2.CAP_PROP_POS_FRAMES,n); ok,x=cap.read(); return x if ok else None
    before,first,last,after=(read(source,n) for n in (result['source_frame_immediately_before'],result['source_first_frame'],result['source_last_candidate_frame'],result['source_frame_immediately_after'])); source.release()
    expected_count=int(c['end_frame_exclusive'])-int(c['start_frame'])
    out=cv2.VideoCapture(str(exported)); actual_count=int(out.get(cv2.CAP_PROP_FRAME_COUNT)); exp_first,exp_last=read(out,0),read(out,max(0,expected_count-1)); out.release()
    def distance(a:Any,b:Any)->float|None:
        if a is None or b is None:return None
        return round(float(np.mean(cv2.absdiff(cv2.resize(a,(160,90)),cv2.resize(b,(160,90))))),3)
    result['export_first_frame']=0; result['export_last_frame']=expected_count-1
    differences={'export_first_to_source_first':distance(exp_first,first),'export_first_to_source_previous':distance(exp_first,before),'export_last_to_source_last':distance(exp_last,last),'export_last_to_source_next':distance(exp_last,after)}
    # A material margin makes static adjacent frames inconclusive rather than false
    # failures, while the observed "previous nearly exact / target distant" class fails.
    def target_not_beaten(target:float|None, outside:float|None)->bool:
        if target is None: return False
        if outside is None: return True
        return not (outside + max(2.0,target*.20) < target)
    first_ok=target_not_beaten(differences['export_first_to_source_first'],differences['export_first_to_source_previous'])
    last_ok=target_not_beaten(differences['export_last_to_source_last'],differences['export_last_to_source_next'])
    result.update({'diagnostic_mean_abs_difference':differences,'expected_frame_count':expected_count,'actual_frame_count':actual_count,'first_frame_matches_target':first_ok,'last_frame_matches_target':last_ok,'boundary_validation':'PASS' if first_ok and last_ok and actual_count==expected_count else 'FAIL','status':'PASS' if first_ok and last_ok and actual_count==expected_count else 'FAIL','frame_index_authority':'source start inclusive; end exclusive; decoded count and provenance comparisons are authoritative'})
    return result

def _semantic_checkpoint(path:Path, candidate:dict[str,Any], model:str, window_id:str)->dict[str,Any]|None:
    try:
        item=json.loads(path.read_text()); identity={'window_id':window_id,'candidate_id':candidate['candidate_id'],'start_frame':candidate['start_frame'],'end_frame_exclusive':candidate['end_frame_exclusive']}; expected={**identity,'candidate_identity':identity,'model':model,'semantic_schema_version':SEMANTIC_SCHEMA_VERSION,'semantic_prompt_version':SEMANTIC_PROMPT_VERSION}
        return item['response'] if all(item.get(k)==v for k,v in expected.items()) and not validate_response(item['response']) else None
    except (OSError,KeyError,TypeError,json.JSONDecodeError): return None

def semantic_validate(items:list[dict[str,Any]], movie:Path, srt:Path, narrative:Path, checkpoint_dir:Path, fps:float, window_id:str, provider:SemanticProvider|None=None, model:str='gemini-3.6-flash')->dict[str,Any]:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2]/'.env'); key=os.getenv('GEMINI_API_KEY')
    active=provider or (GeminiBrollSemanticProvider(key,model) if key else None); cues=parse_srt_file(srt).cues; segments=json.loads(narrative.read_text()).get('segments',[]); checkpoint_dir.mkdir(parents=True,exist_ok=True); usage={'prompt_tokens':0,'response_tokens':0,'thinking_tokens':0,'cached_tokens':0,'total_tokens':0}; reused=requests=0
    for c in items: # every compact technical candidate is eligible, irrespective of structural rank.
        c['narrative']=_narrative_context(segments,c['start_seconds'],c['end_seconds']); c['srt_context']=_cue_context(cues,c['start_seconds'],c['end_seconds']); cp=checkpoint_dir/f"{c['candidate_id']}.json"; response=_semantic_checkpoint(cp,c,model,window_id)
        if response is not None: reused+=1
        elif active is None: c['visual']={}; c['editorial']={'decision':'REVIEW','status':'SEMANTIC_INCOMPLETE','reason':'GEMINI_API_KEY is not configured'}; continue
        else:
            try:
                sheet=candidate_contact_sheet(movie,c,fps); response_obj=active.generate(SEMANTIC_PROMPT,{'window_id':window_id,'candidate_id':c['candidate_id'],'candidate_identity':{'window_id':window_id,'candidate_id':c['candidate_id'],'start_frame':c['start_frame'],'end_frame_exclusive':c['end_frame_exclusive']},'narrative':c['narrative'],'srt_context':c['srt_context'],'instruction':'Images are visual authority; context is separate narrative evidence.'},sheet); errors=validate_response(response_obj.data)
                if errors: raise ValueError('; '.join(errors))
                response=response_obj.data; identity={'window_id':window_id,'candidate_id':c['candidate_id'],'start_frame':c['start_frame'],'end_frame_exclusive':c['end_frame_exclusive']}; write_json(cp,{**identity,'candidate_identity':identity,'model':model,'semantic_schema_version':SEMANTIC_SCHEMA_VERSION,'semantic_prompt_version':SEMANTIC_PROMPT_VERSION,'response':response,'usage':response_obj.usage}); requests+=1
                for name,value in response_obj.usage.items():
                    if value is not None: usage[name]+=value
            except Exception as error:
                c['visual']={}; c['editorial']={'decision':'REVIEW','status':'SEMANTIC_INCOMPLETE','reason':str(error)}; continue
        c['visual']=response['visual']; c['people']=response['visual']['people']; c['relationships']=response['relationships']; c['editorial']={**response['editorial'],'status':'VALIDATED'}
    return {'provider':active.identifier if active else 'unavailable','model':model,'requests':requests,'reused':reused,'usage':usage}

def _words(value:Any)->set[str]:
    import re
    return {x for x in re.findall(r"[\wáéíóúñ]+",str(value).lower()) if len(x)>2}

def _semantic_signature(c:dict[str,Any])->set[str]:
    visual=c.get('visual',{}); editorial=c.get('editorial',{})
    values=[visual.get('setting',''),visual.get('actions',[]),visual.get('visible_interactions',[]),editorial.get('standalone_meaning_es',''),editorial.get('use_cases_es',[]),[x.get('presentation') for x in c.get('people',[])],[x.get('type') for x in c.get('relationships',[])]]
    return set().union(*(_words(x) for x in values))

def apply_semantic_scarcity(items:list[dict[str,Any]])->None:
    """Suppress only adjacent, semantically near-identical validated KEEP variants."""
    keepers=[]
    for c in sorted(items,key=lambda x:(-x.get('score',{}).get('total',0),x['start_seconds'],x['candidate_id'])):
        if c.get('editorial',{}).get('decision') != 'KEEP': continue
        signature=_semantic_signature(c); rival=None
        for old in keepers:
            adjacent=float(c['start_seconds']) <= float(old['end_seconds'])+12 and float(old['start_seconds']) <= float(c['end_seconds'])+12
            a,b=signature,_semantic_signature(old); similarity=len(a&b)/len(a|b) if a or b else 0.
            same_setting=c.get('visual',{}).get('setting') == old.get('visual',{}).get('setting')
            if adjacent and same_setting and similarity >= .55: rival=old; break
        if rival is None:
            keepers.append(c); c['semantic_redundancy']={'status':'DISTINCT'}
        else:
            c['editorial']={**c['editorial'],'decision':'REJECT','status':'VALIDATED','rejection_reason':'semantic_redundancy','redundant_with':rival['candidate_id']}
            c['semantic_redundancy']={'status':'SUPPRESSED','redundant_with':rival['candidate_id'],'reason':'semantic_redundancy'}

def run_broll_pilot(input_dir:Path, provider:SemanticProvider|None=None, model:str='gemini-3.6-flash', window_id:str=PILOT_WINDOW)->dict[str,Any]:
    paths=discover(input_dir,window_id); shots=load_shots(paths); signals=visual_signals(Path(paths['movie']),shots)
    for shot,signal in zip(shots,signals): shot.update(signal)
    add_context(shots,Path(paths['srt']),Path(paths['narrative'])); items=candidates(shots)
    for item in items: item['window_id']=window_id
    output=Path(paths['root'])/'runs'/input_dir.name/'broll-pilot-v1'/window_id; exports=output/'exports'; exports.mkdir(parents=True,exist_ok=True)
    # This run owns only these pilot outputs; remove stale files before regeneration.
    for path in [*exports.glob('BRC_*.mp4'),output/'review_reel.mp4',output/'review_contact_sheet.jpg',output/'candidates.json',output/'export_validation.json']:
        if path.is_file(): path.unlink()
    source= cv2.VideoCapture(str(paths['movie'])); width,height=int(source.get(cv2.CAP_PROP_FRAME_WIDTH)),int(source.get(cv2.CAP_PROP_FRAME_HEIGHT)); source.release()
    # The source fps makes the frame boundaries canonical; semantic analysis is
    # intentionally before selecting final exports and covers all candidates.
    fps=float(cv2.VideoCapture(str(paths['movie'])).get(cv2.CAP_PROP_FPS)) or 24.0
    semantic=semantic_validate(items,Path(paths['movie']),Path(paths['srt']),Path(paths['narrative']),output/'semantic_checkpoints',fps,window_id,provider,model)
    apply_semantic_scarcity(items)
    exported=[]; validations=[]
    for c in items:
        if c['editorial']['decision']=='KEEP':
            p=exports/f"{c['candidate_id']}.mp4"; expected_count=c['end_frame_exclusive']-c['start_frame']
            subprocess.run(ffmpeg_export_command(Path(paths['movie']),c,p,fps),check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            container=probe(p,width,height,expected_count/fps,expected_count)
            validation={**container,**boundary_validation(Path(paths['movie']),p,c),'container_validation':container['status']}
            validation['production_validation']='PASS' if validation['container_validation']=='PASS' and validation['boundary_validation']=='PASS' else 'FAIL'
            validations.append(validation)
            c['boundary_validation']={k:validation[k] for k in ('boundary_validation','expected_frame_count','actual_frame_count','first_frame_matches_target','last_frame_matches_target')}
            if validation['production_validation']=='PASS': exported.append(p)
            else:
                c['editorial']={**c['editorial'],'decision':'REVIEW','status':'BOUNDARY_FAILED','rejection_reason':'boundary_validation_failed'}
                p.unlink(missing_ok=True)
    for c in items:
        c['final_decision']=c['editorial']['decision']
    reel=output/'review_reel.mp4'
    if exported: subprocess.run(review_reel_command(exported,reel),check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); validations.append(probe(reel,width,height,sum(c['duration_seconds'] for c in items if c['editorial']['decision']=='KEEP')))
    if exported: contact_sheet(Path(paths['movie']),[c for c in items if c['editorial']['decision']=='KEEP'],output/'review_contact_sheet.jpg')
    write_json(output/'candidates.json',{'schema_version':'broll_pilot_candidates_v3','semantic_schema_version':SEMANTIC_SCHEMA_VERSION,'semantic_prompt_version':SEMANTIC_PROMPT_VERSION,'window_id':window_id,'frame_semantics':'start_frame inclusive; end_frame_exclusive exclusive','semantic_run':semantic,'candidates':items})
    write_json(output/'export_validation.json',{'schema_version':'broll_pilot_export_validation_v3','frame_semantics':'start_frame inclusive; end_frame_exclusive exclusive','exports':validations})
    keep_items=[x for x in items if x['editorial']['decision']=='KEEP']
    return {'window':window_id,'shots':len(shots),'candidates':len(items),'KEEP':len(keep_items),'REVIEW':sum(x['editorial']['decision']=='REVIEW' for x in items),'REJECT':sum(x['editorial']['decision']=='REJECT' for x in items),'exported':len(exported),'average_keep_duration':round(sum(x['duration_seconds'] for x in keep_items)/len(keep_items),2) if keep_items else 0.0,'output':output}
