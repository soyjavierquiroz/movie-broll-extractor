"""Single-window B-roll pilot: technical candidates, semantics, exact exports."""
from __future__ import annotations
import inspect, json, math, os, subprocess
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from .srt import parse_srt_file
from .utils import write_json, sha256_file
from .processing_ledger import ProcessingLedger, fingerprint
from .active_picture import crop_frame, full_frame, load_or_detect
from .production_profile import load as load_production_profile
from .broll_semantics import (PROMPT as SEMANTIC_PROMPT, SemanticProvider, FOCUS_SUBJECTS, INTERACTION_REQUIREMENTS, TARGET_BINDING_CONFIDENCE, directive_validation_errors, validate_response, build_semantic_provider_from_env, classify_provider_error, estimate_openai_cost, redact_provider_error, DEFAULT_OPENAI_MODEL, OpenAIProviderError, OpenAIStructuredOutputError)

PILOT_WINDOW="SW_02"; SAMPLE_FPS=3.0; KEEP=70; REVIEW=50
# This is deliberately separate from editorial duration policy.  It exists
# only for a decoder's empty, final EOF sliver (two frames in the E02 case).
TERMINAL_SLIVER_MAX_SECONDS=.25
SEMANTIC_SCHEMA_VERSION="broll_semantics_v9"; SEMANTIC_PROMPT_VERSION="broll_semantic_prompt_v9"
TARGET_BINDING_VERSION="semantic_target_binding_v1"
STRUCTURAL_SCORING_VERSION="visual_event_duration_v1"
COHERENCE_VERSION="visual_event_coherence_v4_scene_interaction"
MAX_PROVIDER_RESPONSE_ATTEMPTS=2

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
    # A selector window has no technical artifacts yet.  On the explicit B-roll
    # command only, add bounded threshold-24 shot detection for that one window;
    # this never scans the feature and keeps selection/extraction separable.
    if window_id not in available:
        registry=root/'runs'/input_dir.name/'pilot_windows.json'
        selected=None
        if registry.is_file():
            selected=next((x for x in json.loads(registry.read_text()).get('windows',[]) if x.get('window_id')==window_id),None)
        if selected is not None:
            from .inspect_source import inspect_movie
            from .visual import Window, build_shots, detect_cuts
            fps=float(inspect_movie(required['movie'])['video']['fps'])
            window=Window(window_id,float(selected['start_seconds']),float(selected['end_seconds']),'pilot_selector',list(selected.get('narrative_segment_ids',[])))
            cuts=detect_cuts(required['movie'],round(window.start_seconds*fps),round(window.end_seconds*fps),24.)
            additions=build_shots(window,fps,cuts,24.)
            windows.append(window.as_dict())
            write_json(required['windows'],{'schema_version':'visual_smoke_windows_v1','windows':windows})
            with required['shots'].open('a',encoding='utf-8') as handle:
                for shot in additions: handle.write(json.dumps(shot,separators=(',',':'))+'\n')
            available.append(window_id)
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

def visual_signals(movie:Path, shots:list[dict[str,Any]], sample_fps:float=SAMPLE_FPS,
                   on_complete:Any=None, active_picture:dict[str,Any]|None=None)->list[dict[str,Any]]:
    """Compute signals in one decode pass per bounded analysis unit.

    Only small HSV histograms are retained until the successful middle sample is
    known.  Pixel statistics and the preceding gray frame are streaming state.
    ``on_complete`` is intentionally per-unit so callers can durably checkpoint.
    """
    cap=cv2.VideoCapture(str(movie)); out=[]
    try:
        try: source_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        except AttributeError: source_frames=0
        for shot in shots:
            start,end=float(shot['start_seconds']),float(shot['end_seconds'])
            sample_count=0; t=start
            while t<end:
                sample_count+=1; t+=1/sample_fps
            pixels=dark=frames=0; mean=m2=sharp_total=motion_total=0.; prev_gray=None; histograms=[]
            t=start
            for _ in range(sample_count):
                cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,frame=cap.read()
                t+=1/sample_fps
                if not ok: continue
                frame=crop_frame(frame,active_picture)
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); count=gray.size
                # Parallel Welford merge preserves population std semantics while
                # avoiding concatenating all sampled pixels.
                batch_mean=float(gray.mean()); delta=batch_mean-mean
                batch_m2=float(np.sum((gray.astype(np.float64)-batch_mean)**2))
                new_pixels=pixels+count
                mean+=delta*count/new_pixels
                m2+=batch_m2+delta*delta*pixels*count/new_pixels
                pixels=new_pixels; dark+=int(np.count_nonzero(gray<20)); frames+=1
                sharp_total+=float(cv2.Laplacian(gray,cv2.CV_64F).var())
                if prev_gray is not None: motion_total+=float(np.mean(cv2.absdiff(prev_gray,gray)))
                prev_gray=gray
                hist=cv2.calcHist([cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)],[0,1],None,[16,16],[0,180,0,256]); cv2.normalize(hist,hist)
                histograms.append(hist)
            if not frames:
                final=bool(shot.get('is_final_technical_shot'))
                duration=float(shot.get('duration_seconds',end-start))
                end_frame=int(shot.get('end_frame_exclusive',-1))
                # Frame count is preferred.  A one-frame tolerance is only for
                # container timestamp/frame-count rounding at source EOF.
                at_eof=source_frames>0 and end_frame>=source_frames-1
                if final and at_eof and duration<TERMINAL_SLIVER_MAX_SECONDS:
                    value={'status':'TERMINAL_SLIVER_SKIPPED','reason':'eof_microshot_no_decodable_sample'}
                    out.append(value)
                    if on_complete: on_complete(shot,value)
                    continue
                raise RuntimeError(f"cannot decode {shot['shot_id']}")
            value={'status':'COMPLETE','brightness_mean':mean,'brightness_std':float(math.sqrt(m2/pixels)),'sharpness_score':sharp_total/frames,'motion_score':motion_total/(frames-1) if frames>1 else 0.,'near_black_fraction':dark/pixels,'_hist':histograms[frames//2]}
            out.append(value)
            if on_complete: on_complete(shot,value)
        return out
    finally:
        cap.release()

def add_context(shots:list[dict[str,Any]], srt:Path, narrative:Path)->None:
    cues=parse_srt_file(srt).cues; segments=json.loads(narrative.read_text()).get('segments',[])
    for s in shots:
        a,b=float(s['start_seconds']),float(s['end_seconds']); duration=b-a
        s['subtitle_overlap_seconds']=sum(_overlap(a,b,c.start_seconds,c.end_seconds) for c in cues)
        s['subtitle_occupancy_ratio']=min(1.,s['subtitle_overlap_seconds']/duration)
        s['narrative_segment_ids']=[x['segment_id'] for x in segments if _overlap(a,b,float(x['start_seconds']),float(x['end_seconds']))>0]

def _similarity(a:dict,b:dict)->float: return max(0.,min(1.,float(cv2.compareHist(a['_hist'],b['_hist'],cv2.HISTCMP_CORREL)+1)/2))

def _ids(shot:dict[str,Any], *names:str)->set[str]:
    """Read supplied scene-analysis IDs without inferring them from pixels.

    ``scene_continuity`` is the production-facing source-metadata contract.
    The top-level aliases retain compatibility with a small number of existing
    callers while producers migrate their shot analysis to the nested form.
    """
    values=[]
    continuity=shot.get('scene_continuity',{})
    if not isinstance(continuity,dict): continuity={}
    for name in names:
        for value in (continuity.get(name),shot.get(name)):
            values.extend(value if isinstance(value,list) else [value])
    return {str(value) for value in values if value not in (None,'','unknown')}

def _flag(shot:dict[str,Any], *names:str)->bool:
    continuity=shot.get('scene_continuity',{})
    continuity=continuity if isinstance(continuity,dict) else {}
    return any(bool(continuity.get(name,shot.get(name))) for name in names)

def _shared(current:dict[str,Any], nxt:dict[str,Any], *names:str)->bool:
    return bool(_ids(current,*names) & _ids(nxt,*names))

def adjacent_continuity(current:dict[str,Any], nxt:dict[str,Any])->dict[str,Any]:
    """Return deterministic continuity diagnostics for one adjacent cut.

    Scene metadata is asymmetric: reliable discontinuity vetoes a join, but
    missing positive metadata never turns a camera cut into an asset boundary.
    The established visual/subtitle grouping rule remains responsible for the
    normal join decision.
    """
    boundary_scores=[
        current.get('boundary_scene_change_score'),
        nxt.get('boundary_scene_change_score'),
        current.get('scene_change_score'),
        nxt.get('scene_change_score'),
    ]
    strong_cut=any(float(x)>=.65 for x in boundary_scores if x is not None)
    current_subjects=_ids(current,'tracked_subject_ids','subject_track_ids','primary_subject_id')
    next_subjects=_ids(nxt,'tracked_subject_ids','subject_track_ids','primary_subject_id')
    shared_subject=bool(current_subjects & next_subjects)
    same_interaction=_shared(current,nxt,'interaction_id','conversation_id','conversation_pair_id','shot_reverse_shot_id')
    same_action=_shared(current,nxt,'action_id','continuous_action_id','action_track_id','object_action_id')
    same_scene=_shared(current,nxt,'scene_id')
    same_establishing_sequence=_shared(current,nxt,'establishing_sequence_id')
    same_setting=_shared(current,nxt,'setting_id','background_id','location_id')
    same_participant_set=_shared(current,nxt,'participant_set_id')
    validated_reverse_shot=(
        _flag(current,'validated_shot_reverse_shot','valid_shot_reverse_shot')
        and _flag(nxt,'validated_shot_reverse_shot','valid_shot_reverse_shot')
    )
    composition_changed=(
        current.get('composition_signature') is not None
        and nxt.get('composition_signature') is not None
        and current.get('composition_signature') != nxt.get('composition_signature')
    )
    subject_changed=bool(current_subjects and next_subjects and not shared_subject)
    reasons=[]
    if same_interaction or validated_reverse_shot:
        reasons.append('same_interaction')
        if validated_reverse_shot: reasons.append('validated_shot_reverse_shot')
    if same_action:
        reasons.append('continuous_action')
    if same_scene:
        reasons.append('same_scene')
    if same_establishing_sequence:
        reasons.append('establishing_or_insert_continuity')
    # A face/subject is useful corroboration but deliberately cannot merge two
    # scenes by itself: the same person can appear in an unrelated later scene.
    if same_setting and same_participant_set:
        reasons.append('stable_setting_and_participants')
    elif same_setting and shared_subject:
        reasons.append('stable_setting_and_shared_subject')

    explicit_break=(
        _flag(current,'scene_break_after','location_jump_after','temporal_jump_after','action_discontinuity_after')
        or _flag(nxt,'scene_break_before','location_jump_before','temporal_jump_before','action_discontinuity_before')
    )
    # Incompatible settings/participant sets are meaningful only when both
    # sides actually provide that evidence.  Missing data is not a difference.
    setting_changed=bool(_ids(current,'setting_id','background_id','location_id') and _ids(nxt,'setting_id','background_id','location_id') and not same_setting)
    participant_set_changed=bool(_ids(current,'participant_set_id') and _ids(nxt,'participant_set_id') and not same_participant_set)
    invalid=[]
    if explicit_break: invalid.append('explicit_scene_or_time_break')
    if setting_changed: invalid.append('setting_or_location_discontinuity')
    if participant_set_changed: invalid.append('participant_set_discontinuity')
    dialogue=(float(current.get('subtitle_occupancy_ratio',0))+float(nxt.get('subtitle_occupancy_ratio',0))) >= .45
    similarity=_num(_similarity(current,nxt))
    if strong_cut and not reasons: invalid.append('strong_scene_change_without_continuity')
    if composition_changed and not reasons and not dialogue: invalid.append('major_composition_change_without_interaction')
    if subject_changed and not reasons and not dialogue: invalid.append('major_subject_identity_change_without_interaction')
    evidence={
        'same_interaction':same_interaction,
        'validated_shot_reverse_shot':validated_reverse_shot,
        'same_action':same_action,
        'same_scene':same_scene,
        'same_establishing_sequence':same_establishing_sequence,
        'same_setting':same_setting,
        'same_participant_set':same_participant_set,
        'shared_tracked_subject':shared_subject,
    }
    # Explicitly supplied scene/time breaks and incompatible scene evidence win
    # even over an accidentally stale interaction label.  Absence of IDs is not
    # a break: existing technical shots deliberately have no such IDs.
    if invalid:
        return {'merge':False,'merge_reasons':reasons,'split_reasons':invalid,'evidence':evidence,'scene_similarity':similarity}
    return {'merge':True,'merge_reasons':reasons,'split_reasons':[],'evidence':evidence,'scene_similarity':similarity}


def _baseline_join(current:dict[str,Any], nxt:dict[str,Any])->tuple[bool,dict[str,Any]]:
    """The v1 event predicate, with hard scene evidence used only as a veto."""
    edge=adjacent_continuity(current,nxt)
    if not edge['merge']:
        return False,edge
    same_narrative=bool(set(current.get('narrative_segment_ids',[])) & set(nxt.get('narrative_segment_ids',[])))
    dialogue=(float(current.get('subtitle_occupancy_ratio',0))+float(nxt.get('subtitle_occupancy_ratio',0))) >= .45
    similar=edge['scene_similarity'] >= .12
    positive=bool(edge['merge_reasons'])
    # Narrative context supports a continuation but never supplies a hard
    # boundary.  A genuinely dissimilar, non-dialogue pair remains a split,
    # preventing inserts/photo material from absorbing an unrelated scene.
    if not similar and not dialogue and not positive:
        edge={**edge,'merge':False,'split_reasons':['strong_visual_discontinuity_without_dialogue_or_interaction']}
        return False,edge
    edge={**edge,'same_narrative_context':same_narrative,'dialogue_continuity':dialogue,'visual_similarity':similar}
    return True,edge

def generate_groups(shots:list[dict[str,Any]])->list[list[int]]:
    """Construct bounded v1 Visual Events before semantic enrichment."""
    groups=[]; i=0
    while i<len(shots):
        duration=float(shots[i]['duration_seconds']); group=[i]
        while i+1<len(shots) and duration+float(shots[i+1]['duration_seconds'])<=18:
            current,nxt=shots[i],shots[i+1]
            joined,edge=_baseline_join(current,nxt)
            if not joined: break
            conversational=edge['dialogue_continuity']
            if (
                (current.get('autonomous_broll_value') is True and len(group)==1)
                or (duration>=5 and not conversational and len(group)==1 and not edge['merge_reasons'])
            ):
                break
            i+=1; group.append(i); duration+=float(shots[i]['duration_seconds'])
            if duration>=15 and not conversational and not edge['merge_reasons']: break
        if 3<=duration<=18: groups.append(group)
        i+=1
    return _rescue_orphans(groups,shots)


def _rescue_orphans(groups:list[list[int]], shots:list[dict[str,Any]])->list[list[int]]:
    """Locally absorb a short event only into an immediate safe neighbour."""
    result=[list(group) for group in groups]
    index=0
    while index<len(result):
        group=result[index]
        duration=sum(float(shots[item]['duration_seconds']) for item in group)
        if not 3<=duration<=4.5 or any(shots[item].get('autonomous_broll_value') is True for item in group):
            index+=1; continue
        neighbours=[]
        if index:
            neighbours.append(('previous',index-1))
        if index+1<len(result):
            neighbours.append(('next',index+1))
        merged=False
        for direction,other_index in neighbours:
            other=result[other_index]
            if any(shots[item].get('autonomous_broll_value') is True for item in other):
                continue
            combined=other+group if direction=='previous' else group+other
            if sum(float(shots[item]['duration_seconds']) for item in combined)>18:
                continue
            left,right=(other[-1],group[0]) if direction=='previous' else (group[-1],other[0])
            joined,_=_baseline_join(shots[left],shots[right])
            if not joined:
                continue
            if direction=='previous':
                result[other_index]=combined
                result.pop(index)
                index=max(0,index-1)
            else:
                result[index]=combined
                result.pop(other_index)
            merged=True
            break
        if not merged:
            index+=1
    return result

def _duration_fit(duration_seconds:float, event_type_hint:str='mixed')->float:
    """Smooth, event-aware provisional duration evidence (maximum 25 points)."""
    d=max(0.,float(duration_seconds)); hint=event_type_hint.lower()
    ranges={
        'reaction':(3.,7.,10.), 'action':(4.,9.,12.),
        'movement':(5.,10.,14.), 'activity':(5.,10.,14.),
        'conversation':(7.,15.,18.), 'interaction':(7.,15.,18.),
    }
    low,high,allowed=ranges.get(hint,(5.,10.,15.))
    if low<=d<=high: return 25.
    if d<low:
        # Useful short evidence fades gently rather than being binary.
        return 25*(.45+.55*d/low)
    if d<=allowed:
        # Long conversations/interactions remain legitimate, just less efficient.
        return 25*(1-.35*(d-high)/(allowed-high))
    # No cliff at 15/18 seconds: progressively discount overly broad events.
    return max(0.,25*.65*(1-(d-allowed)/max(8.,allowed)))

def score_candidate(candidate:dict[str,Any])->dict[str,float]:
    d=candidate['duration_seconds']; duration=_duration_fit(d,str(candidate.get('event_type_hint','mixed')))
    sig=candidate['signals']; quality=25*(.45*min(1,sig['sharpness']/150)+.35*(1-abs(sig['brightness']-110)/145)+.20*(1-sig['near_black_fraction']))
    continuity=20*(.6*sig['visual_continuity']+.4*(1-min(1,sig['subtitle_occupancy'])))
    motion=15*(1-min(1,abs(sig['motion']-12)/20))
    simple=15*(1/(1+.18*(len(candidate['source_shot_ids'])-1)))
    values={'duration_fit':duration,'visual_quality':quality,'continuity':continuity,'motion_usefulness':motion,'structural_simplicity':simple}
    values={k:round(max(0,min(100,v)),2) for k,v in values.items()}; values['total']=round(sum(values.values()),2); return values

def _candidate_from_part(part:list[dict[str,Any]], *, continuity:dict[str,Any]|None=None)->dict[str,Any]:
    """Build one candidate without deciding whether its shots are a beat.

    This is deliberately shared by the legacy conservative group builder and
    the post-semantic Beat Builder.  The latter does not invent semantic fields
    here; it sends the selected multi-shot range through the existing semantic
    validator before it can be published.
    """
    def signal(item:dict[str,Any], raw:str, aggregate:str)->float:
        return float(item.get(raw, item.get('signals', {}).get(aggregate, 0.)))
    def technical(item:dict[str,Any])->list[dict[str,Any]]:
        if item.get('technical_shots'):
            return list(item['technical_shots'])
        # Analysis-unit IDs are internal preparation provenance, never camera
        # identities.  A derived range still carries its canonical parent.
        shot_id=item.get('parent_shot_id') or item.get('shot_id') or next(iter(item.get('source_shot_ids',[])), None)
        if shot_id is None:
            raise ValueError('candidate has no technical shot identity')
        return [{'shot_id':shot_id,'start_seconds':item['start_seconds'],'end_seconds':item['end_seconds']}]
    a,b=float(part[0]['start_seconds']),float(part[-1]['end_seconds'])
    signals={'brightness':float(np.mean([signal(x,'brightness_mean','brightness') for x in part])),'sharpness':float(np.mean([signal(x,'sharpness_score','sharpness') for x in part])),'motion':float(np.mean([signal(x,'motion_score','motion') for x in part])),'near_black_fraction':float(np.mean([signal(x,'near_black_fraction','near_black_fraction') for x in part])),'subtitle_occupancy':float(np.mean([signal(x,'subtitle_occupancy_ratio','subtitle_occupancy') for x in part])),'visual_continuity':float(np.mean([_similarity(x,y) for x,y in zip(part,part[1:])])) if len(part)>1 and all('_hist' in x for x in part) else float(np.mean([signal(x,'visual_continuity','visual_continuity') for x in part])) if len(part)>1 else 1.}
    reasons=['technical_shot_continuity']
    pair_evidence=[]
    if continuity is not None:
        reasons=list(continuity.get('merge_reasons') or reasons)
        pair_evidence=list(continuity.get('adjacent_pairs') or [])
    elif len(part)>1:
        pair_evidence=[{'from_shot_id':left['shot_id'],'to_shot_id':right['shot_id'],**adjacent_continuity(left,right)} for left,right in zip(part,part[1:])]
    if len(part)>1:
        reasons.append('camera_cut_not_treated_as_asset_boundary')
    narrative_ids=list(dict.fromkeys(y for x in part for y in x.get('narrative_segment_ids',[])))
    if len(narrative_ids)==1:
        reasons.append('same_narrative_segment')
    if float(np.mean([x.get('subtitle_occupancy_ratio',x.get('signals',{}).get('subtitle_occupancy',0)) for x in part])) >= .22:
        reasons.append('subtitle_continuity')
    event_hint='conversation' if 'subtitle_continuity' in reasons and len(part)>1 else 'interaction' if len(part)>1 else 'action'
    technical_entries=[]
    for entry in (entry for x in part for entry in technical(x)):
        current=next((x for x in technical_entries if x['shot_id']==entry['shot_id']),None)
        if current is None:
            technical_entries.append(dict(entry))
        else:
            # Multiple adjacent analysis units from one long take remain one
            # technical shot directive, bounded to this event's actual range.
            current['start_seconds']=min(float(current['start_seconds']),float(entry['start_seconds']))
            current['end_seconds']=max(float(current['end_seconds']),float(entry['end_seconds']))
    source_ids=list(dict.fromkeys(entry['shot_id'] for entry in technical_entries))
    c={'start_frame':int(part[0].get('start_frame', round(a*24))), 'end_frame_exclusive':int(part[-1].get('end_frame_exclusive',round(b*24))), 'start_seconds':_num(a),'end_seconds':_num(b),'duration_seconds':_num(b-a),'source_shot_ids':source_ids,'technical_shots':technical_entries, 'narrative_segment_ids':narrative_ids,'event_type_hint':event_hint,'grouping_reason':reasons,'continuity':{'version':COHERENCE_VERSION,'merge_reasons':reasons,'adjacent_pairs':pair_evidence,'narrative_context_only':True,'requires_positive_evidence':False},'autonomous_broll_value':any(x.get('autonomous_broll_value') is True for x in part),'structural_scoring_version':STRUCTURAL_SCORING_VERSION,'signals':{k:_num(v) for k,v in signals.items()}}
    c['score']=score_candidate(c); structural='KEEP' if c['score']['total']>=KEEP else 'REVIEW' if c['score']['total']>=REVIEW else 'REJECT'; c['structural_decision']=structural; c['editorial']={'decision':structural, 'status':'PROVISIONAL'}
    return c

def candidates(shots:list[dict[str,Any]])->list[dict[str,Any]]:
    result=[]
    for group in generate_groups(shots):
        result.append(_candidate_from_part([shots[i] for i in group]))
    return dedupe(result)

def technical_candidates(shots:list[dict[str,Any]])->list[dict[str,Any]]:
    """One semantic work item per technical shot for the Beat Builder prepass."""
    return dedupe([_candidate_from_part([shot]) for shot in shots])


def prepublication_event_coherence(event:dict[str,Any])->dict[str,Any]:
    """Reject only persisted cross-shot events containing a hard-break edge."""
    shot_ids=list(event.get('source_shot_ids',[]))
    if len(shot_ids)<=1:
        return {'status':'PASS','reason':'single_technical_shot','adjacent_pairs':[]}
    continuity=event.get('continuity',{})
    pairs=continuity.get('adjacent_pairs',[])
    expected=list(zip(shot_ids,shot_ids[1:]))
    if not pairs:
        return {'status':'LEGACY_UNVERIFIED','reason':'legacy_event_without_persisted_edges','adjacent_pairs':pairs}
    valid=(
        len(pairs)==len(expected)
        and all(
            pair.get('merge') is True
            and pair.get('from_shot_id')==left
            and pair.get('to_shot_id')==right
            for pair,(left,right) in zip(pairs,expected)
        )
    )
    return {
        'status':'PASS' if valid else 'SPLIT_REQUIRED',
        'reason':'no_hard_cross_shot_break_evidence' if valid else 'persisted_hard_break_or_missing_edge',
        'adjacent_pairs':pairs,
    }

def dedupe(items:list[dict[str,Any]])->list[dict[str,Any]]:
    """Retain every technical candidate; semantic diversity is decided later.

    Deleting overlap variants before their meaning is known made the former cap both
    arbitrary and unauditable.
    """
    result=sorted(items,key=lambda x:(x['start_seconds'],x['end_seconds']))
    for i,x in enumerate(result,1):
        x['candidate_id']=f'BRC_{i:04d}'
        x['visual_event_id']=f'VE_{i:06d}'
        x.setdefault('semantic_redundancy',{'status':'NOT_EVALUATED'})
    return result

def pilot_event_id(window_id:str, ordinal:int)->str:
    """Deterministic ledger identity for a visual event in one pilot window."""
    return f'{window_id}_VE_{ordinal:06d}'

def ffmpeg_export_command(movie:Path,c:dict[str,Any],output:Path, fps:float=24.0)->list[str]:
    """Coarse accurate decode, then trim by frame number relative to that decode.

    A second output timestamp seek was susceptible to timestamp rounding and could
    emit the preceding frame.  The sole input seek starts a small decoded segment at
    a known coarse frame; ``trim`` then owns exact [start, end) frame selection.
    """
    start=int(c['start_frame']) if 'start_frame' in c else round(float(c['start_seconds'])*fps)
    end=int(c.get('end_frame_exclusive',start+round(float(c.get('duration_seconds', 0))*fps)))
    if end <= start: raise ValueError('end_frame_exclusive must be greater than start_frame')
    count=end-start
    # Decode from the source origin.  Input seeking is deliberately not used as
    # frame authority: variable PTS/keyframe seeking can otherwise yield N-1.
    vf=f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS"
    return ['ffmpeg','-y','-ss','0','-i',str(movie),'-map','0:v:0','-vf',vf,'-vsync','0','-frames:v',str(count),'-c:v','libx264','-crf','16','-preset','medium','-pix_fmt','yuv420p','-an',str(output)]
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

def semantic_request_context(candidate:dict[str,Any], cues:list[Any], segments:list[dict[str,Any]], evidence:dict[str,Any], window_id:str)->dict[str,Any]:
    """Build the one canonical semantic request context used in production."""
    return {
        'window_id':window_id,
        'candidate_id':candidate['candidate_id'],
        'visual_event_id':candidate['visual_event_id'],
        'candidate_identity':{'window_id':window_id,'candidate_id':candidate['candidate_id'],'start_frame':candidate['start_frame'],'end_frame_exclusive':candidate['end_frame_exclusive']},
        'source_shot_ids':candidate['source_shot_ids'],
        'technical_shots':candidate.get('technical_shots',[]),
        'narrative':_narrative_context(segments,candidate['start_seconds'],candidate['end_seconds']),
        'srt_context':_cue_context(cues,candidate['start_seconds'],candidate['end_seconds']),
        'target_binding_version':TARGET_BINDING_VERSION,
        'person_candidates':evidence.get('technical_shots',[]),
        'instruction':(
            'Images are visual authority. technical_shots and labelled image order map deterministically. '
            'P1/P2/... labels are shot-local detected people and restart for every technical shot. '
            'For every shot_focus_plan directive return target_person_ids and target_binding_confidence. '
            'The editorially important subject must be bound only to IDs visibly labelled in that same shot. '
            'Do not treat the largest box, nearest face, or highest detector confidence as automatically primary. '
            'For woman/man use at most one target ID. For multiple_people use the required visible people. '
            'For action_region/environment/unclear return an empty target_person_ids list. '
            'If the intended human subject cannot be confidently matched to a labelled person, return [] with target_binding_confidence=unclear. '
            'Return exactly one shot_focus_plan directive for each listed technical shot; this is one event request, never one request per shot.'
        ),
    }

def _person_candidates(found:list[dict[str,Any]], width:int, height:int)->list[dict[str,Any]]:
    """Expose finalization's canonical shot-local P1/P2/... ordering."""
    from .finalization import _binding_candidates

    result=[]
    for item in _binding_candidates(found,width):
        box=item['box']
        center=box['x']+box['width']/2
        original=next(
            (
                row for row in found
                if row.get('detector')=='yolo_person'
                and _approximately_same_bbox(row.get('bbox',row),box)
            ),
            {},
        )
        result.append({
            'person_id':item['person_id'],
            'bbox':{
                'x':round(box['x'],3),
                'y':round(box['y'],3),
                'width':round(box['width'],3),
                'height':round(box['height'],3),
            },
            'bbox_normalized':{
                'x':round(box['x']/max(1,width),5),
                'y':round(box['y']/max(1,height),5),
                'width':round(box['width']/max(1,width),5),
                'height':round(box['height']/max(1,height),5),
            },
            'confidence':round(float(original.get('confidence',0.)),5),
            'position':(
                'left' if center < width*.40
                else 'right' if center > width*.60
                else 'center'
            ),
        })
    return result


def _approximately_same_bbox(a:dict[str,Any],b:dict[str,Any])->bool:
    try:
        return all(
            abs(float(a[key])-float(b[key])) < .01
            for key in ('x','y','width','height')
        )
    except (KeyError,TypeError,ValueError):
        return False
def candidate_contact_sheet(movie:Path,c:dict[str,Any],fps:float,evidence:dict[str,Any]|None=None,
                            active_picture:dict[str,Any]|None=None, *, evidence_profile:str='midpoint_v1')->bytes:
    """Representative technical-shot frames with deterministic local person IDs."""
    start,end=int(c['start_frame']),int(c['end_frame_exclusive'])
    technical=c.get('technical_shots',[])

    # Every technical shot represented in the semantic request must be visible.
    # P1/P2/... identities are local to one technical shot and restart after cuts.
    if technical and evidence_profile != 'temporal_evidence_v2':
        selected=technical
        labelled=[
            (
                str(x['shot_id']),
                round(((float(x['start_seconds'])+float(x['end_seconds']))/2)*fps),
            )
            for x in selected
        ]
    else:
        positions=[
            start,
            start+(end-start)//4,
            start+(end-start)//2,
            start+3*(end-start)//4,
            end-1,
        ]
        labelled=[(f'SHOT {i+1}',p) for i,p in enumerate(positions)]

    temporal_samples=[]
    if evidence_profile == 'temporal_evidence_v2':
        from .temporal_evidence import sample_plan
        temporal_samples=sample_plan(c,fps)
        labelled=[(row['shot_id'],row['frame']) for row in temporal_samples]
    elif evidence_profile != 'midpoint_v1':
        raise ValueError(f'unknown semantic evidence profile: {evidence_profile}')

    # Person geometry is local evidence only.  It does NOT decide editorial focus.
    # Gemini binds editorial intent to one of these visible IDs.
    from .finalization import person_detector_preflight, detect_people
    person_detector_preflight()

    cap=cv2.VideoCapture(str(movie))
    frames=[]
    evidence_rows=[]

    for sample_index,(label,frame_no) in enumerate(labelled):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no)
        ok,frame=cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"cannot decode candidate frame {frame_no}")

        frame=crop_frame(frame,active_picture)
        source_h,source_w=frame.shape[:2]
        reference=not temporal_samples or 'middle' in temporal_samples[sample_index]['roles']
        candidates=_person_candidates(detect_people(frame),source_w,source_h) if reference else []

        if reference: evidence_rows.append({
            'shot_id':label,
            'reference_frame':int(frame_no),
            'reference_time_seconds':round(float(frame_no)/fps,4),
            'candidates':candidates,
        })

        # Preserve source aspect ratio.  1920x800 becomes 480x200, not 320x180.
        tile_w=480
        tile_h=max(1,round(source_h*tile_w/source_w))
        if temporal_samples:
            scale=min(480/source_w,240/source_h)
            tile_w=max(1,round(source_w*scale))
            tile_h=max(1,round(source_h*scale))
        tile=cv2.resize(frame,(tile_w,tile_h))
        sx=tile_w/source_w
        sy=tile_h/source_h

        for person in candidates:
            box=person['bbox']
            x1=max(0,min(tile_w-1,round(box['x']*sx)))
            y1=max(0,min(tile_h-1,round(box['y']*sy)))
            x2=max(0,min(tile_w-1,round((box['x']+box['width'])*sx)))
            y2=max(0,min(tile_h-1,round((box['y']+box['height'])*sy)))
            cv2.rectangle(tile,(x1,y1),(x2,y2),(255,255,255),1)
            cv2.putText(
                tile,
                person['person_id'],
                (x1,max(14,y1+14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                .48,
                (255,255,255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            tile,
            label,
            (8,20),
            cv2.FONT_HERSHEY_SIMPLEX,
            .5,
            (255,255,255),
            1,
            cv2.LINE_AA,
        )
        if temporal_samples:
            sample=temporal_samples[sample_index]
            tile=cv2.copyMakeBorder(tile,0,240-tile_h,0,480-tile_w,cv2.BORDER_CONSTANT,value=(0,0,0))
            caption=np.zeros((38,480,3),dtype=np.uint8)
            cv2.putText(caption,f"{sample['sample_id']} {label} {sample['role']}",(5,14),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1)
            cv2.putText(caption,f"frame={frame_no} t={sample['timestamp_seconds']:.4f}s",(5,30),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1)
            tile=cv2.vconcat([caption,tile])
        frames.append(tile)

    cap.release()

    if not frames:
        raise RuntimeError('cannot create candidate contact sheet')

    # A bounded grid remains readable when an event contains more than 8 shots.
    cols=min(4,len(frames))
    blank=np.zeros_like(frames[0])
    frames += [blank]*((-len(frames))%cols)
    sheet=cv2.vconcat([
        cv2.hconcat(frames[i:i+cols])
        for i in range(0,len(frames),cols)
    ])

    if evidence is not None:
        evidence.clear()
        evidence.update({
            'version':TARGET_BINDING_VERSION,
            'technical_shots':evidence_rows,
        })
        if temporal_samples:
            evidence.update(evidence_profile=evidence_profile,samples=temporal_samples,fps=fps)

    ok,encoded=cv2.imencode(
        '.jpg',
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY,88],
    )
    if not ok:
        raise RuntimeError('cannot encode candidate contact sheet')
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

def _semantic_checkpoint(path:Path, candidate:dict[str,Any], model:str, window_id:str, candidate_fingerprint:str|None=None, provider_identifier:str|None=None)->dict[str,Any]|None:
    try:
        item=json.loads(path.read_text()); identity={'window_id':window_id,'candidate_id':candidate['candidate_id'],'start_frame':candidate['start_frame'],'end_frame_exclusive':candidate['end_frame_exclusive']}; expected={**identity,'candidate_identity':identity,'semantic_schema_version':SEMANTIC_SCHEMA_VERSION,'semantic_prompt_version':SEMANTIC_PROMPT_VERSION}
        # Provider/model provenance must never invalidate otherwise canonical
        # work.  Older fingerprints included those operational settings, so a
        # provider/model change can legitimately have a different fingerprint.
        fingerprint_matches=(candidate_fingerprint is None or item.get('candidate_fingerprint') == candidate_fingerprint
                             or bool(item.get('provider')) and (
                                 item.get('model') != model or item.get('provider') != provider_identifier
                             ))
        return item['response'] if fingerprint_matches and all(item.get(k)==v for k,v in expected.items()) and shot_focus_compatible(item['response'],candidate) and not validate_response(item['response']) else None
    except (OSError,KeyError,TypeError,json.JSONDecodeError): return None

def _unique_reasons(values:list[str])->list[str]:
    """Keep validation evidence deterministic without repeating a reason."""
    return list(dict.fromkeys(values))

def shot_focus_diagnostics(response:dict[str,Any],candidate:dict[str,Any])->dict[str,Any]:
    """Explain the strict per-canonical-shot focus-plan compatibility check.

    This is diagnostic-only evidence.  ``shot_focus_compatible`` remains the
    boolean gate used by checkpoint reuse and semantic acceptance.
    """
    expected=list(candidate.get('source_shot_ids',[]))
    visual=response.get('visual') if isinstance(response,dict) else None
    plan_present=isinstance(visual,dict) and 'shot_focus_plan' in visual
    plan=visual.get('shot_focus_plan') if isinstance(visual,dict) else None
    result={
        'expected_canonical_shot_ids':expected,
        'expected_focus_plan_count':len(expected),
        'actual_focus_plan_count':len(plan) if isinstance(plan,list) else None,
        'actual_focus_plan_shot_ids':[],
        'missing_shot_ids':[],
        'duplicate_shot_ids':[],
        'unexpected_shot_ids':[],
        'shot_focus_plan_missing':not plan_present,
        'shot_focus_plan_not_list':plan_present and not isinstance(plan,list),
        'invalid_directives':[],
        'validation_reasons':[],
    }
    if result['shot_focus_plan_missing']:
        result['validation_reasons']=['shot_focus_plan_missing']
        return result
    if result['shot_focus_plan_not_list']:
        result['validation_reasons']=['shot_focus_plan_not_list']
        return result

    # From here ``plan`` is a list.  Preserve every supplied shot ID where it
    # can be represented in JSON; malformed directives are separately noted.
    actual=[]
    invalid_directive=False
    for directive in plan:
        if not isinstance(directive,dict):
            actual.append(None)
            result['invalid_directives'].append({
                'shot_id':None,
                'invalid_fields':directive_validation_errors(directive),
            })
            invalid_directive=True
            continue
        shot_id=directive.get('shot_id')
        actual.append(shot_id if isinstance(shot_id,str) else None)
        invalid_fields=directive_validation_errors(directive)
        if invalid_fields:
            result['invalid_directives'].append({
                'shot_id':shot_id if isinstance(shot_id,str) else None,
                'invalid_fields':invalid_fields,
            })
            invalid_directive=True
    result['actual_focus_plan_shot_ids']=actual

    reasons=[]
    if len(plan)!=len(expected): reasons.append('cardinality_mismatch')
    if invalid_directive: reasons.append('invalid_directive')

    actual_ids=[x for x in actual if isinstance(x,str)]
    expected_set=set(expected)
    actual_set=set(actual_ids)
    result['missing_shot_ids']=[x for x in expected if x not in actual_set]
    if result['missing_shot_ids']: reasons.append('missing_canonical_shot')
    result['duplicate_shot_ids']=list(dict.fromkeys(x for x in actual_ids if actual_ids.count(x)>1))
    if result['duplicate_shot_ids']: reasons.append('duplicate_shot_id')
    result['unexpected_shot_ids']=list(dict.fromkeys(x for x in actual_ids if x not in expected_set))
    if result['unexpected_shot_ids']:
        reasons.append('unexpected_shot_id')
        # Visual analysis unit IDs are provenance, not canonical technical IDs.
        if any('__VAU_' in x and x.split('__VAU_',1)[0] in expected_set for x in result['unexpected_shot_ids']):
            reasons.append('derived/noncanonical_id')
    result['validation_reasons']=_unique_reasons(reasons)
    return result

def shot_focus_compatible(response:dict[str,Any],candidate:dict[str,Any])->bool:
    """Content compatibility, not merely a checkpoint version label."""
    return not shot_focus_diagnostics(response,candidate)['validation_reasons']

def _semantic_failure_path(checkpoint_dir:Path,visual_event_id:str)->Path:
    """Failure evidence is deliberately adjacent to, never inside, checkpoints."""
    return checkpoint_dir.parent/'semantic_failures'/f'{visual_event_id}.json'

def _write_semantic_failure_diagnostic(checkpoint_dir:Path,candidate:dict[str,Any],
                                       response:dict[str,Any],focus:dict[str,Any],
                                       validation_reasons:list[str],target_binding_errors:list[dict[str,Any]],provider:str,model:str,
                                       provider_attempts:int,provider_trace:list[dict[str,Any]],
                                       usage:dict[str,Any],candidate_fingerprint:str)->Path:
    """Atomically retain a parsed-but-rejected provider semantic response."""
    path=_semantic_failure_path(checkpoint_dir,candidate['visual_event_id'])
    write_json(path,{
        'schema_version':'semantic_failure_diagnostic_v1',
        'visual_event_id':candidate['visual_event_id'],
        'candidate_id':candidate['candidate_id'],
        'semantic_schema_version':SEMANTIC_SCHEMA_VERSION,
        'semantic_prompt_version':SEMANTIC_PROMPT_VERSION,
        'provider':provider,
        'model':model,
        'provider_attempts':provider_attempts,
        'provider_trace':provider_trace,
        'failure_stage':'semantic_validation',
        'validation_reasons':_unique_reasons(validation_reasons),
        'expected_canonical_shot_ids':focus['expected_canonical_shot_ids'],
        'expected_focus_plan_count':focus['expected_focus_plan_count'],
        'actual_focus_plan_count':focus['actual_focus_plan_count'],
        'actual_focus_plan_shot_ids':focus['actual_focus_plan_shot_ids'],
        'missing_shot_ids':focus['missing_shot_ids'],
        'duplicate_shot_ids':focus['duplicate_shot_ids'],
        'unexpected_shot_ids':focus['unexpected_shot_ids'],
        'shot_focus_plan_missing':focus['shot_focus_plan_missing'],
        'shot_focus_plan_not_list':focus['shot_focus_plan_not_list'],
        'invalid_directives':focus['invalid_directives'],
        'invalid_target_bindings':target_binding_errors,
        'candidate_fingerprint':candidate_fingerprint,
        'provider_semantic_response':response,
        'usage':usage,
    })
    return path

def _write_provider_output_diagnostic(checkpoint_dir:Path,candidate:dict[str,Any],error:Exception,
                                      provider:str,model:str,candidate_fingerprint:str)->Path:
    """Persist only safe malformed-provider evidence, never the image request."""
    source=error.error if isinstance(error,OpenAIProviderError) else error
    diagnostic=dict(source.diagnostic) if isinstance(source,OpenAIStructuredOutputError) else {}
    path=checkpoint_dir.parent/'provider_diagnostics'/f"{candidate['visual_event_id']}.json"
    write_json(path,{
        'schema_version':'provider_output_diagnostic_v1',
        'failure_stage':'structured_output_parse',
        'visual_event_id':candidate['visual_event_id'],
        'candidate_id':candidate['candidate_id'],
        'provider':provider,'model':model,'candidate_fingerprint':candidate_fingerprint,
        'parser_error':redact_provider_error(source),
        **diagnostic,
    })
    return path

def _has_terminal_semantic_failure(checkpoint_dir:Path,visual_event_id:str)->bool:
    path=_semantic_failure_path(checkpoint_dir,visual_event_id)
    try:
        value=json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    return value.get('visual_event_id')==visual_event_id and value.get('failure_stage')=='semantic_validation'

def target_binding_diagnostics(response:dict[str,Any],candidate:dict[str,Any],evidence:dict[str,Any])->list[dict[str,Any]]:
    """Return field-level grounding errors against the exact Gemini label namespace."""
    evidence_shots=evidence.get('technical_shots',[]) if isinstance(evidence,dict) else []
    if not evidence_shots:
        # Backward compatibility for old checkpoints/tests with no binding evidence.
        return []

    available={
        str(row.get('shot_id')):{
            str(x.get('person_id'))
            for x in row.get('candidates',[])
            if isinstance(x,dict) and x.get('person_id')
        }
        for row in evidence_shots
        if isinstance(row,dict)
    }

    plan=response.get('visual',{}).get('shot_focus_plan',[])
    if not isinstance(plan,list):
        return [{'shot_id':None,'invalid_fields':[{
            'field':'shot_focus_plan','value':plan,'reason':'invalid_type',
            'expected':'array[directive]',
        }]}]

    errors=[]
    for directive in plan:
        if not isinstance(directive,dict):
            errors.append({'shot_id':None,'invalid_fields':[{
                'field':'$directive','value':directive,'reason':'invalid_type',
                'expected':'object',
            }]})
            continue

        shot_id=directive.get('shot_id')
        if not isinstance(shot_id,str) or shot_id not in available:
            errors.append({'shot_id':shot_id if isinstance(shot_id,str) else None,'invalid_fields':[{
                'field':'shot_id','value':shot_id,'reason':'unknown_canonical_shot_id',
                'allowed':sorted(available),
            }]})
            continue

        ids=directive.get('target_person_ids')
        confidence=directive.get('target_binding_confidence')
        invalid_fields=[]

        if not isinstance(ids,list) or any(not isinstance(x,str) for x in ids):
            invalid_fields.append({'field':'target_person_ids','value':ids,'reason':'invalid_type','expected':'unique array[string]'})
        elif len(ids)!=len(set(ids)):
            invalid_fields.append({'field':'target_person_ids','value':ids,'reason':'duplicate_person_id','expected':'unique array[string]'})
        if confidence not in TARGET_BINDING_CONFIDENCE:
            invalid_fields.append({'field':'target_binding_confidence','value':confidence,'reason':'unsupported_enum','allowed':TARGET_BINDING_CONFIDENCE})

        if invalid_fields:
            errors.append({'shot_id':shot_id,'invalid_fields':invalid_fields})
            continue

        valid=available[shot_id]
        unknown=sorted(x for x in ids if x not in valid)
        if unknown:
            errors.append({'shot_id':shot_id,'invalid_fields':[{
                'field':'target_person_ids','value':unknown,'reason':'unknown_person_id',
                'allowed':sorted(valid),
            }]})
            continue

        focus=directive.get('focus_subject')
        human=focus in {'woman','man','multiple_people'}

        if not human:
            if ids:
                errors.append({'shot_id':shot_id,'invalid_fields':[{
                    'field':'target_person_ids','value':ids,'reason':'non_person_focus_has_target_ids',
                    'expected':'[] for non-person focus_subject',
                }]})
            continue

        if not ids:
            # If people were detected but Gemini cannot bind the requested
            # semantic target, uncertainty must be explicit rather than guessed.
            if valid and confidence != 'unclear':
                errors.append({'shot_id':shot_id,'invalid_fields':[{
                    'field':'target_binding_confidence','value':confidence,
                    'reason':'unbound_detected_person_requires_unclear_confidence',
                    'expected':'unclear when target_person_ids is [] and people are labelled',
                }]})
            continue

        if focus in {'woman','man'} and len(ids)!=1:
            errors.append({'shot_id':shot_id,'invalid_fields':[{
                'field':'target_person_ids','value':ids,'reason':'single_person_focus_requires_one_target',
                'expected':'exactly one shot-local person ID',
            }]})

        if focus=='multiple_people' and len(ids)<2:
            errors.append({'shot_id':shot_id,'invalid_fields':[{
                'field':'target_person_ids','value':ids,'reason':'multiple_people_requires_two_or_more_targets',
                'expected':'at least two shot-local person IDs',
            }]})

    return errors


def target_binding_compatible(response:dict[str,Any],candidate:dict[str,Any],evidence:dict[str,Any])->bool:
    """Validate shot-local semantic target IDs against the exact labelled detections Gemini saw."""
    return not target_binding_diagnostics(response,candidate,evidence)

def _quota_error(error: Exception) -> bool:
    return bool(classify_provider_error(error).get('quota_exhausted'))

def _retryable_error(error: Exception) -> bool:
    return bool(classify_provider_error(error).get('retryable'))

class ProviderResponseValidationError(ValueError):
    """The provider replied, but its structured content failed our contract."""

def _legacy_provider_response_failure(stage:dict[str,Any])->bool:
    """Narrow non-destructive compatibility repair for pre-attempt-count output failures."""
    return (stage.get('status') == 'FAILED_FINAL' and not stage.get('failure_kind')
            and stage.get('error') == 'incomplete or mismatched shot focus plan')

def semantic_validate(items:list[dict[str,Any]], movie:Path, srt:Path, narrative:Path, checkpoint_dir:Path, fps:float, window_id:str, provider:SemanticProvider|None=None, model:str=DEFAULT_OPENAI_MODEL, preserve_event_ids:bool=False, active_picture:dict[str,Any]|None=None)->dict[str,Any]:
    active=provider or build_semantic_provider_from_env(model, env_file=Path(__file__).resolve().parents[2]/'.env'); cues=parse_srt_file(srt).cues; segments=json.loads(narrative.read_text()).get('segments',[]); checkpoint_dir.mkdir(parents=True,exist_ok=True); usage={'prompt_tokens':0,'response_tokens':0,'thinking_tokens':0,'cached_tokens':0,'total_tokens':0}; reused=requests=0; quota=False; unavailable=False; blocked=False; failed=0; failure=None
    movie_id=movie.parent.name
    # Pilot checkpoints live under runs/<movie>/broll-pilot-v1/<window>; unit
    # callers may pass an isolated directory, for which its parent is the run.
    lineage=(checkpoint_dir, *checkpoint_dir.parents)
    movie_run=next((x.parent for x in lineage if x.name == 'broll-pilot-v1'), checkpoint_dir.parent)
    # Provider/model are provenance on completed results, never semantic input
    # identity.  Their absence here preserves valid mixed-provider checkpoints.
    ledger=ProcessingLedger(movie_run,movie_id,{'movie_sha256':sha256_file(movie),'srt_sha256':sha256_file(srt),'narrative_sha256':sha256_file(narrative),'semantic_schema_version':SEMANTIC_SCHEMA_VERSION,'semantic_prompt_version':SEMANTIC_PROMPT_VERSION,'active_picture':active_picture})
    for ordinal,c in enumerate(items,1): # every visual event is eligible, irrespective of structural rank.
        # Never let an ordinal-only legacy ID address movie-wide ledger state.
        if not preserve_event_ids:
            c['visual_event_id']=pilot_event_id(window_id,ordinal)
        c['narrative']=_narrative_context(segments,c['start_seconds'],c['end_seconds']); c['srt_context']=_cue_context(cues,c['start_seconds'],c['end_seconds']); cp=checkpoint_dir/f"{c['candidate_id']}.json"
        # A checkpoint's identity plus the event/config fingerprint is the reuse key.
        event_fp=fingerprint({'range':[c['start_frame'],c['end_frame_exclusive']],'shots':c['source_shot_ids'],'event_type_hint':c.get('event_type_hint'),'inputs':ledger.data['inputs'],'window_id':window_id})
        record=ledger.register(c,event_fp); semantic_stage=record['stages']['semantic']; response=_semantic_checkpoint(cp,c,model,window_id,event_fp,getattr(active,'identifier',None))
        if record['stages']['semantic'].get('status') == 'COMPLETE' and response is not None: reused+=1
        elif response is not None:
            ledger.stage(c['visual_event_id'],'semantic','COMPLETE',checkpoint=str(cp),model=model,candidate_fingerprint=event_fp); reused+=1
        elif _legacy_provider_response_failure(semantic_stage):
            # Earlier versions marked this provider-output validation error final
            # without retaining an attempt count. Preserve the record, but give it
            # one explicit, resume-safe retry budget rather than poisoning the ID.
            ledger.stage(c['visual_event_id'],'semantic','FAILED_RETRYABLE',error=semantic_stage['error'],failure_kind='provider_response_validation',provider_response_attempts=1,candidate_fingerprint=event_fp)
            c['visual']={}; c['editorial']={'decision':'REVIEW','status':'SEMANTIC_INCOMPLETE','reason':semantic_stage['error']}; continue
        elif semantic_stage.get('status') == 'FAILED_FINAL' and not _has_terminal_semantic_failure(checkpoint_dir,c['visual_event_id']):
            # A completed batch is never allowed to rely on an un-artifacted
            # provider/parse failure.  This repairs historical interrupted
            # state as well as any future partial write without touching valid
            # checkpoints or terminal semantic-validation artifacts.
            ledger.stage(c['visual_event_id'],'semantic','FAILED_RETRYABLE',
                         error=semantic_stage.get('error'),failure_kind='resume_integrity_repair',
                         provider=semantic_stage.get('provider'),model=semantic_stage.get('model',model),
                         candidate_fingerprint=event_fp)
            semantic_stage=record['stages']['semantic']
        if response is not None:
            c['visual']=response['visual']; c['people']=response['visual']['people']; c['relationships']=response['relationships']; c['editorial']={**response['editorial'],'status':'VALIDATED'}; continue
        if (
            semantic_stage.get('status') == 'FAILED_FINAL'
            and semantic_stage.get('failure_kind') == 'auth_error'
        ):
            ledger.stage(
                c['visual_event_id'],
                'semantic',
                'FAILED_RETRYABLE',
                error=semantic_stage.get('error'),
                failure_kind='auth_error',
                provider=semantic_stage.get('provider'),
                model=semantic_stage.get('model', model),
                http_status=semantic_stage.get('http_status'),
                retryable=True,
                candidate_fingerprint=event_fp,
            )
            semantic_stage = record['stages']['semantic']

        if semantic_stage.get('status') == 'FAILED_FINAL':
            c['visual']={}; c['editorial']={'decision':'REVIEW','status':'SEMANTIC_INCOMPLETE','reason':str(semantic_stage.get('error','semantic validation failed'))}; continue
        elif quota or unavailable:
            reason=(failure or {}).get('reason') or ('quota_exceeded' if quota else 'provider_unavailable')
            c['visual']={}; c['editorial']={'decision':'REVIEW','status':'PROVIDER_DEFERRED','reason':f'{reason}; safe to resume'}; continue
        elif active is None:
            message='semantic provider configuration is not configured'
            ledger.stage(c['visual_event_id'],'semantic','FAILED_RETRYABLE',error=message,failure_kind='provider_configuration')
            c['visual']={}; c['editorial']={'decision':'REVIEW','status':'PROVIDER_DEFERRED','reason':message}
            unavailable=True
            failure={'failure_stage':'semantic','failed_segment':window_id,'failed_event_id':c['visual_event_id'],'candidate_id':c['candidate_id'],'provider':'unavailable','model':model,'http_status':None,'reason':'provider_configuration','retryable':True,'retry_after_seconds':None,'checkpoint_saved':False,'ledger_saved':True,'resume_safe':True,'message':message}
            continue
        else:
            try:
                ledger.stage(c['visual_event_id'],'semantic','RUNNING',model=model,candidate_fingerprint=event_fp)
                evidence={}
                sheet=(candidate_contact_sheet(movie,c,fps,evidence,active_picture)
                       if active_picture is not None
                       else candidate_contact_sheet(movie,c,fps,evidence))
                context=semantic_request_context(c,cues,segments,evidence,window_id)
                response_obj=active.generate(SEMANTIC_PROMPT,context,sheet)
                resolved_provider=response_obj.provider or active.identifier
                resolved_model=response_obj.model or model
                provider_attempts=max(1,int(response_obj.attempts or 1))
                provider_trace=list(response_obj.provider_trace)
                focus=shot_focus_diagnostics(response_obj.data,c)
                errors=validate_response(response_obj.data)
                # Retain the established ledger error text for bounded retry
                # compatibility; the failure artifact records the precise cause.
                if focus['validation_reasons']: errors.append('incomplete or mismatched shot focus plan')
                target_binding_errors=target_binding_diagnostics(response_obj.data,c,evidence)
                target_binding_valid=not target_binding_errors
                if not target_binding_valid: errors.append('invalid or mismatched semantic target binding')
                if errors:
                    diagnostic_reasons=_unique_reasons([
                        *validate_response(response_obj.data),
                        *focus['validation_reasons'],
                        *([] if target_binding_valid else ['invalid_semantic_target_binding']),
                    ])
                    _write_semantic_failure_diagnostic(
                        checkpoint_dir,c,response_obj.data,focus,diagnostic_reasons,target_binding_errors,
                        resolved_provider,resolved_model,provider_attempts,provider_trace,
                        response_obj.usage,event_fp,
                    )
                    raise ProviderResponseValidationError('; '.join(_unique_reasons(errors)))
                response=response_obj.data
                identity={'window_id':window_id,'candidate_id':c['candidate_id'],'start_frame':c['start_frame'],'end_frame_exclusive':c['end_frame_exclusive']}
                write_json(cp,{**identity,'candidate_identity':identity,'visual_event_id':c['visual_event_id'],'candidate_fingerprint':event_fp,'provider':resolved_provider,'model':resolved_model,'provider_attempts':provider_attempts,'provider_trace':provider_trace,'semantic_schema_version':SEMANTIC_SCHEMA_VERSION,'semantic_prompt_version':SEMANTIC_PROMPT_VERSION,'response':response,'usage':response_obj.usage})
                # A successful checkpoint supersedes only this event's stale
                # failure evidence; failure artifacts are never reuse inputs.
                try: _semantic_failure_path(checkpoint_dir,c['visual_event_id']).unlink(missing_ok=True)
                except OSError: pass  # Cleanup must never invalidate a checkpoint.
                ledger.stage(c['visual_event_id'],'semantic','COMPLETE',checkpoint=str(cp),provider=resolved_provider,model=resolved_model,provider_attempts=provider_attempts,candidate_fingerprint=event_fp)
                requests+=provider_attempts
                for name,value in response_obj.usage.items():
                    if value is not None: usage[name]+=value
            except Exception as error:
                if isinstance(error,ProviderResponseValidationError):
                    attempts=int(semantic_stage.get('provider_response_attempts',0))+1
                    status='FAILED_FINAL' if attempts >= MAX_PROVIDER_RESPONSE_ATTEMPTS else 'FAILED_RETRYABLE'
                    ledger.stage(c['visual_event_id'],'semantic',status,error=str(error),failure_kind='provider_response_validation',provider_response_attempts=attempts,candidate_fingerprint=event_fp)
                else:
                    detail=classify_provider_error(error)
                    status='FAILED_RETRYABLE' if detail.get('retryable') else 'FAILED_FINAL'
                    resolved_provider=detail.get('provider') or getattr(active,'identifier','unavailable')
                    resolved_model=detail.get('model') or getattr(active,'model',model)
                    safe_error=redact_provider_error(error)
                    fields={'error':safe_error,'failure_kind':detail.get('reason'),'provider':resolved_provider,'model':resolved_model,'http_status':detail.get('http_status'),'retryable':bool(detail.get('retryable')),'retry_after_seconds':detail.get('retry_after_seconds'),'candidate_fingerprint':event_fp}
                    providers_attempted=detail.get('providers_attempted')
                    if providers_attempted: fields['providers_attempted']=providers_attempted
                    ledger.stage(c['visual_event_id'],'semantic',status,**fields)

                    if detail.get('reason')=='structured_output_invalid':
                        _write_provider_output_diagnostic(checkpoint_dir,c,error,resolved_provider,resolved_model,event_fp)

                    attempts_used=detail.get('attempts')
                    requests+=(int(attempts_used) if attempts_used is not None else 1)

                    quota=bool(detail.get('quota_exhausted') or detail.get('reason')=='quota_exceeded')
                    blocked=bool(detail.get('reason') == 'auth_error' and not detail.get('retryable'))
                    unavailable=bool(not blocked and not quota and (detail.get('retryable') or detail.get('reason')=='auth_error'))

                    if quota or unavailable or blocked:
                        failure={'failure_stage':'semantic','failed_segment':window_id,'failed_event_id':c['visual_event_id'],'candidate_id':c['candidate_id'],'provider':resolved_provider,'model':resolved_model,'http_status':detail.get('http_status'),'reason':detail.get('reason'),'retryable':bool(detail.get('retryable')),'retry_after_seconds':detail.get('retry_after_seconds'),'providers_attempted':providers_attempted if providers_attempted is not None else [resolved_provider],'checkpoint_saved':False,'ledger_saved':True,'resume_safe':True,'message':safe_error}
                failed+=1
                c['visual']={}
                # Only a persisted terminal semantic-validation artifact is an
                # editorial semantic failure. Transport, timeout and strict
                # parser problems remain provider-deferred and resumable.
                c['editorial']={'decision':'REVIEW','status':('SEMANTIC_INCOMPLETE' if isinstance(error,ProviderResponseValidationError) else 'PROVIDER_DEFERRED'),'reason':redact_provider_error(error)}
                continue
        c['visual']=response['visual']; c['people']=response['visual']['people']; c['relationships']=response['relationships']; c['editorial']={**response['editorial'],'status':'VALIDATED'}
    # Semantic status is scoped to the current pilot window, never the movie-wide ledger.
    current_event_ids=[c['visual_event_id'] for c in items]
    stages=[ledger.data['events'][eid]['stages']['semantic'].get('status') for eid in current_event_ids]
    complete=stages.count('COMPLETE'); retryable=stages.count('FAILED_RETRYABLE'); final=stages.count('FAILED_FINAL')
    pending=stages.count('PENDING')+stages.count('RUNNING'); remaining=len(stages)-complete
    # A returned but invalid response is editorial/schema failure, not provider
    # availability.  FAILED_FINAL response validation is terminal and permits
    # the rest of the segment/movie to continue.
    status='BLOCKED_PROVIDER' if blocked else ('PARTIAL_QUOTA' if quota else ('PROVIDER_DEFERRED' if unavailable else 'COMPLETE'))
    ledger.summary(status=status, window_id=window_id, visual_events_total=len(items), semantic_complete=complete,
                   semantic_pending=pending, semantic_failed=retryable+final,
                   semantic_failed_retryable=retryable, semantic_failed_final=final,
                   semantic_reused=reused, last_completed_event=next((c['visual_event_id'] for c in reversed(items) if c.get('editorial',{}).get('status')=='VALIDATED'),None),
                   remaining_count=remaining, remaining_work_definition='current window: PENDING + RUNNING + FAILED_RETRYABLE + FAILED_FINAL', failure=failure)
    return {'provider':active.identifier if active else 'unavailable','model':getattr(active,'model',model) if active else model,'requests':requests,'reused':reused,'usage':usage,'estimated_cost_usd':estimate_openai_cost(usage) if getattr(active,'identifier',None)=='openai' else 0.0,'status':status,'complete':complete,'pending':pending,'semantic_pending':pending,'semantic_failed_retryable':retryable,'semantic_failed_final':final,'remaining_count':remaining,'quota_exhausted':quota,'provider_unavailable':unavailable,'blocked_provider':blocked,'failure':failure}

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
    paths=discover(input_dir,window_id); movie=Path(paths['movie']); run=Path(paths['root'])/'runs'/input_dir.name
    try:
        active_picture=load_or_detect(movie,run,load_production_profile()['active_picture'])
    except AttributeError:  # Lightweight legacy callers may only provide stream metadata.
        source=cv2.VideoCapture(str(movie))
        active_picture=full_frame(int(source.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  int(source.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        source.release()
    shots=load_shots(paths)
    signals=(visual_signals(movie,shots,active_picture=active_picture)
             if 'active_picture' in inspect.signature(visual_signals).parameters
             else visual_signals(movie,shots))
    for shot,signal in zip(shots,signals): shot.update(signal)
    add_context(shots,Path(paths['srt']),Path(paths['narrative'])); items=candidates(shots)
    for ordinal,item in enumerate(items,1):
        item['window_id']=window_id
        item['visual_event_id']=pilot_event_id(window_id,ordinal)
    output=Path(paths['root'])/'runs'/input_dir.name/'broll-pilot-v1'/window_id; exports=output/'exports'; exports.mkdir(parents=True,exist_ok=True)
    # Never delete durable outputs on resume. Stable event IDs/checkpoints make a
    # rerun idempotent; regenerated manifests describe the current state.
    source= cv2.VideoCapture(str(paths['movie'])); width,height=int(source.get(cv2.CAP_PROP_FRAME_WIDTH)),int(source.get(cv2.CAP_PROP_FRAME_HEIGHT)); source.release()
    # The source fps makes the frame boundaries canonical; semantic analysis is
    # intentionally before selecting final exports and covers all candidates.
    fps=float(cv2.VideoCapture(str(paths['movie'])).get(cv2.CAP_PROP_FPS)) or 24.0
    # Event identities exist before any provider work, so an interruption never
    # requires rediscovering a semantic worklist.
    write_json(output/'visual_events_manifest.json',{'schema_version':'visual_events_manifest_v1','window_id':window_id,'frame_semantics':'start_frame inclusive; end_frame_exclusive exclusive','events':items})
    semantic=(semantic_validate(items,movie,Path(paths['srt']),Path(paths['narrative']),output/'semantic_checkpoints',fps,window_id,provider,model,active_picture=active_picture)
              if 'active_picture' in inspect.signature(semantic_validate).parameters
              else semantic_validate(items,movie,Path(paths['srt']),Path(paths['narrative']),output/'semantic_checkpoints',fps,window_id,provider,model))
    apply_semantic_scarcity(items)
    exported=[]; validations=[]
    for c in items:
        if c['editorial']['decision']=='KEEP':
            p=exports/f"{c['candidate_id']}.mp4"; expected_count=c['end_frame_exclusive']-c['start_frame']
            if not p.exists():
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
    # The semantic ledger is authoritative for stage resumption; export/validation
    # results are mirrored there without storing large payloads.
    ledger=ProcessingLedger(Path(paths['root'])/'runs'/input_dir.name,input_dir.name,{})
    for c in items:
        event=ledger.data['events'].get(c['visual_event_id'])
        if event and c['editorial']['decision']=='KEEP':
            ledger.stage(c['visual_event_id'],'export','COMPLETE' if (exports/f"{c['candidate_id']}.mp4").exists() else 'PENDING',path=str(exports/f"{c['candidate_id']}.mp4"))
            result=next((v for v in validations if v.get('candidate_id')==c['candidate_id']),None)
            ledger.stage(c['visual_event_id'],'validation','COMPLETE' if result and result.get('production_validation')=='PASS' else 'FAILED_RETRYABLE',validation=result.get('production_validation') if result else 'not_run')
    write_json(output/'candidates.json',{'schema_version':'broll_pilot_candidates_v4','semantic_schema_version':SEMANTIC_SCHEMA_VERSION,'semantic_prompt_version':SEMANTIC_PROMPT_VERSION,'window_id':window_id,'frame_semantics':'start_frame inclusive; end_frame_exclusive exclusive','semantic_run':semantic,'candidates':items})
    write_json(output/'export_validation.json',{'schema_version':'broll_pilot_export_validation_v3','frame_semantics':'start_frame inclusive; end_frame_exclusive exclusive','exports':validations})
    keep_items=[x for x in items if x['editorial']['decision']=='KEEP']
    report={'window':window_id,'shots':len(shots),'candidates':len(items),'visual_events':len(items),'KEEP':len(keep_items),'REVIEW':sum(x['editorial']['decision']=='REVIEW' for x in items),'REJECT':sum(x['editorial']['decision']=='REJECT' for x in items),'exported':len(exported),'average_keep_duration':round(sum(x['duration_seconds'] for x in keep_items)/len(keep_items),2) if keep_items else 0.0,'status':semantic.get('status','COMPLETE'),'semantic_complete':semantic.get('complete',len(items)),'semantic_pending':semantic.get('semantic_pending',0),'semantic_failed_retryable':semantic.get('semantic_failed_retryable',0),'semantic_failed_final':semantic.get('semantic_failed_final',0),'semantic_remaining_count':semantic.get('remaining_count',0),'semantic_reused':semantic.get('reused',0),'output':output}
    from .pilot_selector import mark_attempted
    mark_attempted(input_dir,window_id,report['status'])
    return report
