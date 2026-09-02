"""CPU-first technical shot detection smoke pipeline (no asset semantics)."""
from __future__ import annotations
import json, math, statistics, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .inspect_source import inspect_movie
from .utils import sha256_file, write_json, write_jsonl

THRESHOLDS=(20.0,24.0,27.0)
EDGE_TOLERANCE_FRAMES=1

def _value(segment:dict[str,Any], key:str)->Any:
    value=segment.get(key)
    return value.get("value") if isinstance(value,dict) else value
def _utc()->str: return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def _frame(seconds:float,fps:float)->int: return round(seconds*fps)
def _seconds(frame:int,fps:float)->float: return frame/fps

@dataclass(frozen=True)
class Window:
    window_id:str; start_seconds:float; end_seconds:float; selection_reason:str; source_narrative_segment_ids:list[str]
    def as_dict(self): return {"window_id":self.window_id,"start_seconds":self.start_seconds,"end_seconds":self.end_seconds,"selection_reason":self.selection_reason,"source_narrative_segment_ids":self.source_narrative_segment_ids}

def _centered(segment:dict[str,Any], duration:float, movie_duration:float)->tuple[float,float]:
    center=(float(segment["start_seconds"])+float(segment["end_seconds"]))/2
    start=max(0.0,min(center-duration/2,movie_duration-duration)); return start,start+duration
def _nonoverlap(window:tuple[float,float], selected:list[Window], separation:float=0)->bool:
    return all(window[1] <= old.start_seconds or window[0] >= old.end_seconds for old in selected) and all(abs((window[0]+window[1])/2-(old.start_seconds+old.end_seconds)/2)>=separation for old in selected)

def select_smoke_windows(narrative:dict[str,Any], movie_duration:float, window_seconds:float=60.0)->list[Window]:
    """Select A dialogue, B sparse/transition, C second-half diverse deterministically."""
    segments=narrative.get("segments",[])
    if movie_duration <= 0 or not segments: raise ValueError("narrative map requires segments and positive movie duration")
    duration=min(window_seconds,movie_duration)
    selected=[]
    # Long, low-context conversations nearest the interior (ranking includes deterministic id tie-break).
    def a_rank(s): return (_value(s,"segment_type") == "conversation", _value(s,"dialogue_density") == "high", _value(s,"context_dependency") != "high", float(s["end_seconds"])-float(s["start_seconds"]), -abs((float(s["start_seconds"])+float(s["end_seconds"]))/2-movie_duration/2))
    a=max(segments,key=lambda s:(a_rank(s),str(s.get("segment_id",""))))
    start,end=_centered(a,duration,movie_duration); selected.append(Window("SW_01",start,end,"preferred interior conversation: long segment, high dialogue when annotated, and context dependency below high when available",[a["segment_id"]]))
    sparse=[s for s in segments if _value(s,"segment_type") in {"sparse_dialogue","transition"} or _value(s,"narrative_function")=="transition"]
    pool=sparse or segments
    b=next((s for s in sorted(pool,key=lambda x:(-(float(x["end_seconds"])-float(x["start_seconds"])),str(x["segment_id"]))) if _nonoverlap(_centered(s,duration,movie_duration),selected)),None)
    if b is None: raise ValueError("unable to select non-overlapping sparse/transition window")
    start,end=_centered(b,duration,movie_duration); selected.append(Window("SW_02",start,end,"sparse-dialogue or transition segment (falling back to the longest available non-overlapping narrative segment only if absent)",[b["segment_id"]]))
    functions={_value(a,"narrative_function"),_value(b,"narrative_function")}
    cpool=[s for s in segments if (float(s["start_seconds"])+float(s["end_seconds"]))/2 >= movie_duration/2 and _value(s,"narrative_function") not in functions]
    c=next((s for s in sorted(cpool,key=lambda x:(-(float(x["end_seconds"])-float(x["start_seconds"])),str(x["segment_id"]))) if _nonoverlap(_centered(s,duration,movie_duration),selected,900)),None)
    if c is None: c=next((s for s in sorted(segments,key=lambda x:(-float(x["start_seconds"]),str(x["segment_id"]))) if _nonoverlap(_centered(s,duration,movie_duration),selected,900)),None)
    if c is None: raise ValueError("unable to select temporally diverse third window")
    start,end=_centered(c,duration,movie_duration); selected.append(Window("SW_03",start,end,"second-half segment with a narrative function distinct from the first two and at least 15 minutes from both when structure permits",[c["segment_id"]]))
    return selected

def detect_cuts(movie:Path, start_frame:int, end_frame:int, threshold:float)->list[int]:
    from scenedetect import ContentDetector, open_video, SceneManager
    video=open_video(str(movie)); video.seek(start_frame)
    manager=SceneManager(); manager.add_detector(ContentDetector(threshold=threshold)); manager.detect_scenes(video, end_time=end_frame)
    return [cut.get_frames() for _,cut in manager.get_scene_list() if start_frame < cut.get_frames() < end_frame]

def build_shots(window:Window, fps:float, cuts:list[int], threshold:float)->list[dict[str,Any]]:
    start,end=_frame(window.start_seconds,fps),_frame(window.end_seconds,fps); boundaries=[start]+sorted(set(c for c in cuts if start<c<end))+[end]
    return [{"schema_version":"shot_v1","window_id":window.window_id,"shot_id":f"{window.window_id}_SHOT_{i:04d}","start_frame":left,"end_frame_exclusive":right,"start_seconds":_seconds(left,fps),"end_seconds":_seconds(right,fps),"duration_seconds":_seconds(right-left,fps),"detector":{"name":"content_detector","threshold":threshold},"representative_frame_seconds":_seconds(left+(right-left)//2,fps)} for i,(left,right) in enumerate(zip(boundaries,boundaries[1:]),1)]

def metrics(shots:list[dict[str,Any]])->dict[str,Any]:
    durations=[s["duration_seconds"] for s in shots]
    return {"shot_count":len(shots),"shots_per_minute":len(shots)/(sum(durations)/60),"minimum_duration":min(durations),"median_duration":statistics.median(durations),"mean_duration":statistics.mean(durations),"p90_duration":sorted(durations)[max(0,math.ceil(len(durations)*.9)-1)],"maximum_duration":max(durations),"count_lt_0_5s":sum(d<.5 for d in durations),"count_lt_1_0s":sum(d<1 for d in durations),"count_gt_15s":sum(d>15 for d in durations),"count_gt_30s":sum(d>30 for d in durations)}
def validate_shots(shots:list[dict[str,Any]], windows:list[Window],fps:float)->dict[str,Any]:
    errors=[]
    for window in windows:
        part=[s for s in shots if s["window_id"]==window.window_id]; expected_start=_frame(window.start_seconds,fps); expected_end=_frame(window.end_seconds,fps)
        if not part: errors.append(f"{window.window_id}: no shots"); continue
        if part[0]["start_frame"] != expected_start: errors.append(f"{window.window_id}: first shot does not start at window")
        if part[-1]["end_frame_exclusive"] != expected_end: errors.append(f"{window.window_id}: last shot does not end at window")
        for previous,current in zip(part,part[1:]):
            if previous["end_frame_exclusive"] != current["start_frame"]: errors.append(f"{window.window_id}: gap or overlap")
        for s in part:
            if s["end_frame_exclusive"]<=s["start_frame"]: errors.append(f"{s['shot_id']}: non-positive duration")
            if not s["start_seconds"] <= s["representative_frame_seconds"] < s["end_seconds"]: errors.append(f"{s['shot_id']}: representative outside shot")
            if abs(s["duration_seconds"]-(s["end_frame_exclusive"]-s["start_frame"])/fps)>1e-8: errors.append(f"{s['shot_id']}: frame/time mismatch")
    return {"schema_version":"shot_validation_v1","frame_semantics":"start_frame is inclusive; end_frame_exclusive is exclusive","ordering":"PASS" if not any("gap or overlap" in e for e in errors) else "FAIL","gaps":"PASS" if not any("gap or overlap" in e for e in errors) else "FAIL","overlaps":"PASS" if not any("gap or overlap" in e for e in errors) else "FAIL","frame_time_consistency":"PASS" if not any("mismatch" in e for e in errors) else "FAIL","status":"PASS" if not errors else "FAIL","errors":errors}

def choose_threshold(results:dict[float,list[dict[str,Any]]])->tuple[float,str]:
    # A technical shot detector preserves credible boundaries; short shots are evidence, never invalidation.
    ordered=sorted(results)
    if len(ordered) < 3: return ordered[len(ordered)//2],"deterministic central threshold from available sweep"
    return ordered[len(ordered)//2],"deterministic pilot compromise: central sweep threshold balances sensitivity and false-positive risk across heterogeneous smoke windows; short shots are retained as technical boundaries"

def score_trace(movie:Path,start_frame:int,end_frame:int)->dict[int,float]:
    from scenedetect import ContentDetector, SceneManager, StatsManager, open_video
    stats=StatsManager(); video=open_video(str(movie)); video.seek(start_frame)
    manager=SceneManager(stats_manager=stats); manager.add_detector(ContentDetector(threshold=999.0)); manager.detect_scenes(video,end_time=end_frame)
    return {frame:float(value[0]) for frame in range(start_frame,end_frame) if (value:=stats.get_metrics(frame,["content_val"])) and value[0] is not None}

def _local_peaks(trace:dict[int,float],limit:int=15)->list[tuple[int,float]]:
    points=sorted(trace.items()); peaks=[(frame,score) for i,(frame,score) in enumerate(points) if score>0 and (i==0 or score>=points[i-1][1]) and (i==len(points)-1 or score>=points[i+1][1])]
    return sorted(peaks,key=lambda item:(-item[1],item[0]))[:limit]

def boundary_strip(movie:Path,frame:int,score:float,fps:float,output:Path)->None:
    import cv2
    cap=cv2.VideoCapture(str(movie)); frames=[]
    for offset in (-.25,-.05,.05,.25):
        cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,round(frame+offset*fps))); ok,image=cap.read()
        if ok: frames.append(cv2.resize(image,(280,117)))
    cap.release()
    if len(frames)==4:
        canvas=cv2.hconcat(frames); cv2.putText(canvas,f"{frame/fps:.3f}s score {score:.2f}",(8,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA); output.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(output),canvas,[cv2.IMWRITE_JPEG_QUALITY,82])

def run_threshold_audit(input_dir:Path)->dict[str,Any]:
    movie=input_dir/"movie.mp4"; output=input_dir.resolve().parents[1]/"runs"/input_dir.name/"visual-smoke-v1"
    if not movie.is_file() or not (output/"windows.json").is_file(): raise FileNotFoundError("existing movie and visual-smoke-v1/windows.json are required")
    source=inspect_movie(movie); fps=float(source["video"]["fps"]); windows=[Window(**x) for x in json.loads((output/"windows.json").read_text())["windows"]]
    cuts={}; matrix={}
    for window in windows:
        start,end=_frame(window.start_seconds,fps),_frame(window.end_seconds,fps); matrix[window.window_id]={}; cuts[window.window_id]={}
        for threshold in (20.,24.,27.):
            value=detect_cuts(movie,start,end,threshold); cuts[window.window_id][threshold]=value; matrix[window.window_id][str(int(threshold))]=metrics(build_shots(window,fps,value,threshold))
    target=windows[2]; start,end=_frame(target.start_seconds,fps),_frame(target.end_seconds,fps); trace=score_trace(movie,start,end)
    extended={}; target_cuts={}
    for threshold in (18.,20.,22.,24.,27.):
        value=detect_cuts(movie,start,end,threshold); target_cuts[threshold]=value; extended[str(int(threshold))]={"boundaries":[{"frame":x,"seconds":_seconds(x,fps),"score":trace.get(x)} for x in value],"metrics":metrics(build_shots(target,fps,value,threshold))}
    peaks=[{"frame":f,"seconds":_seconds(f,fps),"score":s,"would_trigger":{"18":s>=18,"20":s>=20,"22":s>=22,"24":s>=24,"27":s>=27}} for f,s in _local_peaks(trace)]
    disagreement=sorted(set(target_cuts[20.])-set(target_cuts[27.]),key=lambda x:(-trace.get(x,0),x))[:12]
    strips=[]
    for index,frame in enumerate(disagreement,1):
        path=output/"threshold_audit"/f"SW_03_boundary_{index:02d}.jpg"; boundary_strip(movie,frame,trace.get(frame,0),fps,path); strips.append(str(path))
    adaptive={"attempted":False,"shot_count":None,"boundary_seconds":[],"notes":"not run"}
    try:
        from scenedetect import AdaptiveDetector, SceneManager, open_video
        video=open_video(str(movie)); video.seek(start); manager=SceneManager(); manager.add_detector(AdaptiveDetector()); manager.detect_scenes(video,end_time=end); values=[b.get_frames() for _,b in manager.get_scene_list() if start<b.get_frames()<end]; adaptive={"attempted":True,"shot_count":len(values)+1,"boundary_seconds":[_seconds(x,fps) for x in values],"notes":"default AdaptiveDetector, corroboration only"}
    except Exception as error: adaptive["notes"]=str(error)
    cls="LIKELY_TRUE_LONG_TAKE" if not target_cuts[20.] and sum(p["would_trigger"]["20"] for p in peaks)==0 and adaptive.get("shot_count",1)<=1 else "THRESHOLD_27_TOO_CONSERVATIVE" if len(target_cuts[20.])-len(target_cuts[27.])>=2 else "INCONCLUSIVE"
    audit={"schema_version":"threshold_audit_v1","per_window_threshold_matrix":matrix,"sw_03_extended_sweep":extended,"sw_03_score_peaks":peaks,"sw_03_score_counts":{str(t):sum(s>=t for s in trace.values()) for t in (18,20,22,24,27)},"detector_comparison":{"adaptive_detector":adaptive},"sw_03_classification":cls,"old_threshold":27,"new_threshold":24,"decision_reason":"threshold 24 is the central, sensitivity-preserving pilot profile; no minimum-duration filtering is applied","sw_01_sw_02_disagreements":{w.window_id:{"at20_not27":[_seconds(x,fps) for x in sorted(set(cuts[w.window_id][20.])-set(cuts[w.window_id][27.]))],"at24_not27":[_seconds(x,fps) for x in sorted(set(cuts[w.window_id][24.])-set(cuts[w.window_id][27.]))]} for w in windows[:2]},"diagnostic_strips":strips}
    write_json(output/"threshold_audit.json",audit); return audit

def contact_sheet(movie:Path, shots:list[dict[str,Any]], output:Path)->dict[str,Any]:
    import cv2
    cap=cv2.VideoCapture(str(movie)); selected=shots if len(shots)<=64 else [shots[round(i*(len(shots)-1)/63)] for i in range(64)]
    thumbs=[]
    for ordinal,s in enumerate(selected,1):
        cap.set(cv2.CAP_PROP_POS_FRAMES,round(s["representative_frame_seconds"]*cap.get(cv2.CAP_PROP_FPS))); ok,frame=cap.read()
        if not ok: continue
        frame=cv2.resize(frame,(240,101)); cv2.putText(frame,f"{ordinal} {s['start_seconds']:.1f}s {s['duration_seconds']:.1f}s",(4,16),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA); thumbs.append(frame)
    cap.release(); cols=4; rows=math.ceil(len(thumbs)/cols); canvas=cv2.copyMakeBorder(cv2.vconcat([cv2.hconcat(thumbs[r*cols:(r+1)*cols]+[thumbs[-1]*0 for _ in range(max(0,cols-len(thumbs[r*cols:(r+1)*cols])))]) for r in range(rows)]),0,0,0,0,cv2.BORDER_CONSTANT) if thumbs else None
    if canvas is None: raise RuntimeError("could not decode contact-sheet frames")
    output.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(output),canvas,[cv2.IMWRITE_JPEG_QUALITY,80]); return {"path":str(output),"source_shots":len(shots),"displayed_frames":len(thumbs),"capped":len(shots)>64}

def run_visual_smoke(input_dir:Path, threshold_override:float|None=None, window_seconds:float=60.0)->dict[str,Any]:
    movie=input_dir/"movie.mp4"; narrative_path=input_dir.parent.parent/"runs"/input_dir.name/"narrative-v2"/"narrative_map.json"
    # Input dir is conventionally input/<movie-id>; resolve repository root robustly.
    narrative_path=input_dir.resolve().parents[1]/"runs"/input_dir.name/"narrative-v2"/"narrative_map.json"
    if not movie.is_file(): raise FileNotFoundError(f"movie does not exist: {movie}")
    if not narrative_path.is_file(): raise FileNotFoundError(f"narrative map does not exist: {narrative_path}")
    started=_utc(); clock=time.monotonic(); source=inspect_movie(movie); fps=float(source["video"]["fps"]); duration=float(source["duration_seconds"]); narrative=json.loads(narrative_path.read_text()); windows=select_smoke_windows(narrative,duration,window_seconds)
    output=input_dir.resolve().parents[1]/"runs"/input_dir.name/"visual-smoke-v1"; output.mkdir(parents=True,exist_ok=True); write_json(output/"windows.json",{"schema_version":"visual_smoke_windows_v1","windows":[w.as_dict() for w in windows]})
    sweep={}; all_by_threshold={}
    for threshold in THRESHOLDS:
        by_window={}; flat=[]
        for window in windows:
            shots=build_shots(window,fps,detect_cuts(movie,_frame(window.start_seconds,fps),_frame(window.end_seconds,fps),threshold),threshold); by_window[window.window_id]=metrics(shots); flat.extend(shots)
        sweep[str(int(threshold))]={"windows":by_window,"aggregate":metrics(flat)}; all_by_threshold[threshold]=flat
    write_json(output/"threshold_sweep.json",{"schema_version":"threshold_sweep_v1","detector":"content_detector","thresholds":sweep})
    selected,reason=(threshold_override,"debug override") if threshold_override is not None else choose_threshold(all_by_threshold)
    final=all_by_threshold.get(selected)
    if final is None:
        final=[]
        for window in windows: final.extend(build_shots(window,fps,detect_cuts(movie,_frame(window.start_seconds,fps),_frame(window.end_seconds,fps),selected),selected))
    profile={"schema_version":"shot_detection_profile_v1","selected_threshold":selected,"selection_reason":reason,"detector":"content_detector","detector_version":__import__('scenedetect').__version__}; write_json(output/"selected_profile.json",profile); write_jsonl(output/"shots.jsonl",final)
    validation=validate_shots(final,windows,fps); write_json(output/"shot_validation.json",validation)
    sheets={w.window_id:contact_sheet(movie,[s for s in final if s["window_id"]==w.window_id],output/"contact_sheets"/f"{w.window_id}.jpg") for w in windows}
    manifest={"schema_version":"visual_smoke_run_v1","movie_id":input_dir.name,"source_movie_sha256":sha256_file(movie),"narrative_map_sha256":sha256_file(narrative_path),"profile":profile,"windows":[w.as_dict() for w in windows],"started_at":started,"completed_at":_utc(),"status":"COMPLETE" if validation["status"]=="PASS" else "FAILED","runtime_seconds":round(time.monotonic()-clock,3),"shot_count":len(final),"cpu_oriented_observation":"source MP4 decoded directly; no persistent proxy or per-frame extraction","errors":validation["errors"],"contact_sheets":sheets}; write_json(output/"run_manifest.json",manifest); return manifest
