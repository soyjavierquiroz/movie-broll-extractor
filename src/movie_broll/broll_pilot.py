"""Deterministic, local-only single-window B-roll pilot (not final assets)."""
from __future__ import annotations
import json, math, subprocess
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from .srt import parse_srt_file
from .utils import write_json

PILOT_WINDOW="SW_02"; SAMPLE_FPS=3.0; KEEP=70; REVIEW=50; KEEP_CAP=8

def _root(input_dir:Path)->Path: return input_dir.resolve().parents[1]
def _overlap(a:float,b:float,c:float,d:float)->float: return max(0.,min(b,d)-max(a,c))
def _num(v:float)->float: return round(float(v),4)

def discover(input_dir:Path)->dict[str,Path|dict[str,Any]]:
    root=_root(input_dir); smoke=root/'runs'/input_dir.name/'visual-smoke-v1'; narrative=root/'runs'/input_dir.name/'narrative-v2'/'narrative_map.json'
    required={'movie':input_dir/'movie.mp4','windows':smoke/'windows.json','shots':smoke/'shots.jsonl','profile':smoke/'selected_profile.json','narrative':narrative}
    for name,path in required.items():
        if not path.is_file(): raise FileNotFoundError(f"required {name} artifact does not exist: {path}")
    srt=next((input_dir/x for x in ('subtitles.srt',f'{input_dir.name}.srt') if (input_dir/x).is_file()),None)
    if srt is None: raise FileNotFoundError('canonical SRT not found (expected subtitles.srt or movie-id.srt)')
    windows=json.loads(required['windows'].read_text())['windows']; window=next((x for x in windows if x['window_id']==PILOT_WINDOW),None)
    if not window: raise ValueError('SW_02 is absent from persisted visual smoke windows')
    profile=json.loads(required['profile'].read_text())
    if float(profile.get('selected_threshold',-1)) != 24.: raise ValueError('pilot requires selected threshold 24')
    return {**required,'srt':srt,'window':window,'root':root}

def load_shots(paths:dict[str,Any])->list[dict[str,Any]]:
    w=paths['window']; shots=[json.loads(x) for x in Path(paths['shots']).read_text().splitlines() if x.strip()]
    result=[x for x in shots if x.get('window_id')==PILOT_WINDOW and float(x.get('detector',{}).get('threshold',-1))==24.]
    result.sort(key=lambda x:(x['start_seconds'],x['shot_id']))
    if not result: raise ValueError('no selected threshold-24 SW_02 shots')
    prior=float(w['start_seconds'])
    for shot in result:
        start,end=float(shot['start_seconds']),float(shot['end_seconds'])
        if end<=start or abs(start-prior)>0.05 or start<float(w['start_seconds'])-.05 or end>float(w['end_seconds'])+.05: raise ValueError('technical shots are not ordered, continuous, positive, and inside SW_02')
        prior=end
    if abs(prior-float(w['end_seconds']))>.05: raise ValueError('technical shots do not cover SW_02')
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
        c={'start_seconds':_num(a),'end_seconds':_num(b),'duration_seconds':_num(b-a),'source_shot_ids':[x['shot_id'] for x in part],'narrative_segment_ids':list(dict.fromkeys(y for x in part for y in x['narrative_segment_ids'])),'signals':{k:_num(v) for k,v in signals.items()}}
        c['score']=score_candidate(c); c['editorial']={'decision':'KEEP' if c['score']['total']>=KEEP else 'REVIEW' if c['score']['total']>=REVIEW else 'REJECT'}; result.append(c)
    return dedupe(result)

def dedupe(items:list[dict[str,Any]])->list[dict[str,Any]]:
    chosen=[]
    for item in sorted(enumerate(items),key=lambda x:(-x[1]['score']['total'],x[0])):
        if any(len(set(item[1]['source_shot_ids'])&set(old['source_shot_ids']))/min(len(item[1]['source_shot_ids']),len(old['source_shot_ids']))>=.75 for old in chosen): continue
        chosen.append(item[1])
    chosen.sort(key=lambda x:(x['start_seconds'],x['end_seconds']));
    keeps=0
    for x in chosen:
        if x['editorial']['decision']=='KEEP':
            keeps+=1
            if keeps>KEEP_CAP: x['editorial']['decision']='REVIEW'
    for i,x in enumerate(chosen,1): x['candidate_id']=f'BRC_{i:04d}'
    return chosen

def ffmpeg_export_command(movie:Path,c:dict[str,Any],output:Path)->list[str]: return ['ffmpeg','-y','-ss',str(c['start_seconds']),'-i',str(movie),'-t',str(c['duration_seconds']),'-map','0:v:0','-c:v','libx264','-crf','19','-preset','medium','-an',str(output)]
def probe(path:Path, width:int,height:int, expected:float)->dict[str,Any]:
    raw=subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_type,codec_name,width,height:format=duration','-of','json',str(path)],text=True); data=json.loads(raw); streams=data.get('streams',[]); video=[x for x in streams if x['codec_type']=='video']; audio=[x for x in streams if x['codec_type']=='audio']; duration=float(data.get('format',{}).get('duration',0)); ok=path.is_file() and path.stat().st_size>0 and len(video)==1 and video[0].get('codec_name')=='h264' and video[0].get('width')==width and video[0].get('height')==height and not audio and abs(duration-expected)<=1.0
    return {'path':str(path),'status':'PASS' if ok else 'FAIL','duration_seconds':duration,'video_streams':len(video),'audio_streams':len(audio),'codec':video[0].get('codec_name') if video else None}
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

def run_broll_pilot(input_dir:Path)->dict[str,Any]:
    paths=discover(input_dir); shots=load_shots(paths); signals=visual_signals(Path(paths['movie']),shots)
    for shot,signal in zip(shots,signals): shot.update(signal)
    add_context(shots,Path(paths['srt']),Path(paths['narrative'])); items=candidates(shots); output=Path(paths['root'])/'runs'/input_dir.name/'broll-pilot-v1'; exports=output/'exports'; exports.mkdir(parents=True,exist_ok=True)
    # This run owns only these pilot outputs; remove stale files before regeneration.
    for path in [*exports.glob('BRC_*.mp4'),output/'review_reel.mp4',output/'review_contact_sheet.jpg',output/'candidates.json',output/'export_validation.json']:
        if path.is_file(): path.unlink()
    write_json(output/'candidates.json',{'schema_version':'broll_pilot_candidates_v1','window_id':PILOT_WINDOW,'candidates':items})
    source= cv2.VideoCapture(str(paths['movie'])); width,height=int(source.get(cv2.CAP_PROP_FRAME_WIDTH)),int(source.get(cv2.CAP_PROP_FRAME_HEIGHT)); source.release()
    exported=[]; validations=[]
    for c in items:
        if c['editorial']['decision']=='KEEP':
            p=exports/f"{c['candidate_id']}.mp4"; subprocess.run(ffmpeg_export_command(Path(paths['movie']),c,p),check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); exported.append(p); validations.append(probe(p,width,height,c['duration_seconds']))
    reel=output/'review_reel.mp4'
    if exported: subprocess.run(review_reel_command(exported,reel),check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); validations.append(probe(reel,width,height,sum(c['duration_seconds'] for c in items if c['editorial']['decision']=='KEEP')))
    if exported: contact_sheet(Path(paths['movie']),[c for c in items if c['editorial']['decision']=='KEEP'],output/'review_contact_sheet.jpg')
    write_json(output/'export_validation.json',{'schema_version':'broll_pilot_export_validation_v1','exports':validations})
    return {'window':PILOT_WINDOW,'shots':len(shots),'candidates':len(items),'KEEP':sum(x['editorial']['decision']=='KEEP' for x in items),'REVIEW':sum(x['editorial']['decision']=='REVIEW' for x in items),'REJECT':sum(x['editorial']['decision']=='REJECT' for x in items),'exported':len(exported),'output':output}
