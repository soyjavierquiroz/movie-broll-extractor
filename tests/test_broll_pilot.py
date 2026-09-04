import json
from pathlib import Path
import pytest
from movie_broll.broll_pilot import (PILOT_WINDOW, _narrative_context, apply_semantic_scarcity,
    boundary_validation, candidates, dedupe, discover, ffmpeg_export_command, generate_groups,
    pilot_event_id, probe, review_reel_command, score_candidate, _duration_fit)
from movie_broll.cli import main
from movie_broll.broll_semantics import validate_response

def shot(n, start, end, similarity=1):
    return {'shot_id':f'S_{n}', 'start_seconds':start,'end_seconds':end,'duration_seconds':end-start,
      'brightness_mean':110,'sharpness_score':120,'motion_score':12,'near_black_fraction':.05,
      'subtitle_occupancy_ratio':.1,'narrative_segment_ids':['N1'], '_hist':None}

def fake_similarity(monkeypatch):
    import movie_broll.broll_pilot as m
    monkeypatch.setattr(m,'_similarity',lambda a,b:.8)

def test_cli_pilot_broll_exists(monkeypatch,tmp_path,capsys):
    import movie_broll.broll_pilot as b
    calls=[]
    monkeypatch.setattr(b,'run_broll_pilot',lambda x,window_id: calls.append(window_id) or {'window':window_id,'shots':16,'candidates':8,'KEEP':5,'REVIEW':2,'REJECT':1,'exported':5,'average_keep_duration':7.2,'output':tmp_path})
    assert main(['pilot','broll',str(tmp_path)]) == 0
    assert calls == ['SW_02']
    assert '[broll-pilot] window: SW_02' in capsys.readouterr().out

@pytest.mark.parametrize('window_id', ['SW_01', 'SW_03'])
def test_cli_pilot_broll_passes_selected_window(monkeypatch,tmp_path,window_id):
    import movie_broll.broll_pilot as b
    calls=[]
    monkeypatch.setattr(b,'run_broll_pilot',lambda x,window_id: calls.append(window_id) or {'window':window_id,'shots':0,'candidates':0,'KEEP':0,'REVIEW':0,'REJECT':0,'exported':0,'average_keep_duration':0.,'output':tmp_path})
    assert main(['pilot','broll',str(tmp_path),'--window',window_id]) == 0
    assert calls == [window_id]

def test_cli_pilot_broll_help_lists_window(capsys):
    with pytest.raises(SystemExit) as error: main(['pilot','broll','--help'])
    assert error.value.code == 0
    assert '--window WINDOW' in capsys.readouterr().out

def test_discovery_uses_persisted_sw02(tmp_path):
    inp=tmp_path/'input'/'film'; inp.mkdir(parents=True); (inp/'movie.mp4').write_bytes(b'x'); (inp/'subtitles.srt').write_text('')
    smoke=tmp_path/'runs'/'film'/'visual-smoke-v1'; smoke.mkdir(parents=True)
    (smoke/'windows.json').write_text(json.dumps({'windows':[{'window_id':'SW_02','start_seconds':1,'end_seconds':61}]})); (smoke/'shots.jsonl').write_text(''); (smoke/'selected_profile.json').write_text(json.dumps({'selected_threshold':24}))
    nar=tmp_path/'runs'/'film'/'narrative-v2'; nar.mkdir(); (nar/'narrative_map.json').write_text('{}')
    assert discover(inp)['window']['window_id']==PILOT_WINDOW

def test_discovery_accepts_existing_window_and_rejects_unknown_without_analysis(tmp_path):
    inp=tmp_path/'input'/'film'; inp.mkdir(parents=True); (inp/'movie.mp4').write_bytes(b'x'); (inp/'subtitles.srt').write_text('')
    smoke=tmp_path/'runs'/'film'/'visual-smoke-v1'; smoke.mkdir(parents=True)
    (smoke/'windows.json').write_text(json.dumps({'windows':[{'window_id':'SW_01','start_seconds':1,'end_seconds':61},{'window_id':'SW_03','start_seconds':121,'end_seconds':181}]})); (smoke/'shots.jsonl').write_text(''); (smoke/'selected_profile.json').write_text(json.dumps({'selected_threshold':24}))
    nar=tmp_path/'runs'/'film'/'narrative-v2'; nar.mkdir(); (nar/'narrative_map.json').write_text('{}')
    assert discover(inp,'SW_03')['window']['window_id'] == 'SW_03'
    with pytest.raises(ValueError, match=r"available window IDs: SW_01, SW_03"): discover(inp,'SW_02')

def test_cli_invalid_window_returns_nonzero_before_analysis(monkeypatch,tmp_path,capsys):
    import movie_broll.broll_pilot as b
    inp=tmp_path/'input'/'film'; inp.mkdir(parents=True); (inp/'movie.mp4').write_bytes(b'x'); (inp/'subtitles.srt').write_text('')
    smoke=tmp_path/'runs'/'film'/'visual-smoke-v1'; smoke.mkdir(parents=True)
    (smoke/'windows.json').write_text(json.dumps({'windows':[{'window_id':'SW_01','start_seconds':1,'end_seconds':61}]})); (smoke/'shots.jsonl').write_text(''); (smoke/'selected_profile.json').write_text(json.dumps({'selected_threshold':24}))
    nar=tmp_path/'runs'/'film'/'narrative-v2'; nar.mkdir(); (nar/'narrative_map.json').write_text('{}')
    monkeypatch.setattr(b,'visual_signals',lambda *args: pytest.fail('analysis must not run'))
    assert main(['pilot','broll',str(inp),'--window','SW_99']) == 2
    assert 'available window IDs: SW_01' in capsys.readouterr().err

def test_grouping_and_consecutive_max(monkeypatch):
    fake_similarity(monkeypatch)
    xs=[shot(1,0,2),shot(2,2,5),shot(3,5,11),shot(4,11,27)]
    assert generate_groups(xs)==[[0,1,2],[3]] # cuts do not force an event boundary

def test_candidate_order_scores_and_decisions(monkeypatch):
    fake_similarity(monkeypatch); xs=[shot(1,0,6),shot(2,6,12)]
    out=candidates(xs)
    assert [x['candidate_id'] for x in out]==['BRC_0001','BRC_0002']
    assert all(0<=x['score']['total']<=100 and set(x['score'])=={'duration_fit','visual_quality','continuity','motion_usefulness','structural_simplicity','total'} for x in out)
    assert score_candidate({'duration_seconds':7,'source_shot_ids':['x'],'signals':{'sharpness':120,'brightness':110,'near_black_fraction':0,'subtitle_occupancy':0,'visual_continuity':1,'motion':12}})['total']>=70
    assert candidates(xs)==candidates(xs) # stable IDs and no random scoring

def test_adaptive_visual_event_duration_scoring_is_smooth_and_type_aware():
    conversation=lambda seconds: _duration_fit(seconds,'conversation')
    assert conversation(13)==25 and conversation(14.8)==25 and conversation(15)==25
    assert 0 < conversation(17.3) < conversation(15)
    assert conversation(19) < conversation(17.3) and conversation(27) < conversation(19)
    assert _duration_fit(5,'reaction')==25
    assert _duration_fit(5,'action')==25 and _duration_fit(13,'action') < _duration_fit(13,'conversation')

def test_conversation_duration_is_not_structural_garbage():
    signals={'sharpness':120,'brightness':110,'near_black_fraction':0,'subtitle_occupancy':.2,'visual_continuity':.9,'motion':12}
    score=score_candidate({'duration_seconds':14.8,'event_type_hint':'conversation','source_shot_ids':['a','b'],'signals':signals})
    assert score['duration_fit']==25 and score['total']>=70

def test_technical_dedupe_retains_candidates_for_semantic_audit():
    def item(ids,score,start): return {'source_shot_ids':ids,'score':{'total':score},'editorial':{'decision':'KEEP'},'start_seconds':start,'end_seconds':start+1}
    assert len(dedupe([item(['a','b'],90,0),item(['a','b'],80,1)]))==2
    many=dedupe([item([str(i)],99,i) for i in range(10)])
    assert sum(x['editorial']['decision']=='KEEP' for x in many)==10

def test_ffmpeg_export_is_h264_no_audio_and_review_path(tmp_path):
    c={'start_seconds':1,'duration_seconds':5}; cmd=ffmpeg_export_command(Path('movie.mp4'),c,tmp_path/'x.mp4')
    assert 'libx264' in cmd and '-an' in cmd and '-c:v' in cmd
    assert 'concat=n=2:v=1:a=0' in review_reel_command([tmp_path/'a.mp4',tmp_path/'b.mp4'],tmp_path/'reel.mp4')

def test_probe_validation_mock(monkeypatch,tmp_path):
    p=tmp_path/'x.mp4'; p.write_bytes(b'x')
    import movie_broll.broll_pilot as b
    monkeypatch.setattr(b.subprocess,'check_output',lambda *a,**k: json.dumps({'streams':[{'codec_type':'video','codec_name':'h264','width':1720,'height':720}],'format':{'duration':'5.1'}}))
    assert probe(p,1720,720,5)['status']=='PASS'

def _semantic(decision='KEEP'):
    return {'visual':{'summary_es':'Una persona mira una puerta','subjects':['persona'],'objects':['puerta'],'actions':['mirar una puerta'],'people_count_estimate':'1','setting':'interior','visible_interactions':[],'visible_emotions':['neutral'],'people':[{'presentation':'unclear','apparent_age_group':'unclear','frame_role':'primary','position':'center'}],'primary_subject_position':'center','primary_subject_description':'persona','visual_focus':'persona y puerta','shot_focus_plan':[{'shot_id':'S1','focus_subject':'environment','focus_reason':'puerta','preserve_secondary_subject':False,'interaction_requirement':'none','focus_position':'unclear'}]},'relationships':[],'editorial':{'standalone_meaning_es':'Una persona espera ante una puerta','reusable_broll':True,'action_or_moment_complete':'true','use_cases_es':['esperar ante una puerta'],'negative_use_cases_es':['no afirmar una llamada'],'search_terms_es':['esperar'],'editorial_confidence':'high','reason':'acción visible','decision':decision}}

def test_frame_bounds_are_authoritative_and_no_blind_offset(tmp_path):
    cmd=ffmpeg_export_command(Path('movie.mp4'),{'start_frame':240,'end_frame_exclusive':360},tmp_path/'x.mp4',24)
    assert '-frames:v' in cmd and cmd[cmd.index('-frames:v')+1]=='120'
    assert '0.083' not in ' '.join(cmd) and cmd.count('-ss') == 1
    assert 'trim=start_frame=240:end_frame=360' in cmd[cmd.index('-vf')+1]

def test_semantic_contract_keeps_visual_and_narrative_separate():
    assert validate_response(_semantic()) == []
    data=_semantic(); data['visual']['subjects']=['wife']; assert 'hallucination' in validate_response(data)[0]

def test_visible_emotion_is_conservative_enum():
    data=_semantic(); data['visual']['visible_emotions']=['heartbroken']; assert 'invalid visible emotion' in validate_response(data)

def test_final_keep_requires_semantic_gates_and_provider_failure_is_not_keep():
    data=_semantic(); data['editorial']['action_or_moment_complete']='unclear'; assert 'KEEP lacks semantic usefulness' in validate_response(data)
    data=_semantic(); data['editorial']['reusable_broll']=False; assert 'KEEP lacks semantic usefulness' in validate_response(data)

def test_semantic_checkpoint_reuse_and_reframe_metadata(tmp_path):
    import movie_broll.broll_pilot as b
    c={'candidate_id':'BRC_0001','start_frame':10,'end_frame_exclusive':20}
    identity={'window_id':'SW_01',**c}
    (tmp_path/'BRC_0001.json').write_text(json.dumps({**identity,'candidate_identity':identity,'model':'gemini-3.6-flash','semantic_schema_version':'broll_semantics_v2','semantic_prompt_version':'broll_semantic_prompt_v2','response':_semantic()}))
    response=b._semantic_checkpoint(tmp_path/'BRC_0001.json',c,'gemini-3.6-flash','SW_01')
    assert response is None  # old schema lacks the required per-shot focus plan
    assert b._semantic_checkpoint(tmp_path/'BRC_0001.json',c,'gemini-3.6-flash','SW_02') is None

def test_shot_focus_compatibility_requires_exact_complete_unique_coverage():
    import movie_broll.broll_pilot as b
    candidate={'source_shot_ids':['S1','S2']}
    response=_semantic(); response['visual']['shot_focus_plan']=[
        {'shot_id':'S1','focus_subject':'man','focus_reason':'speaker','preserve_secondary_subject':False,'interaction_requirement':'sequence','focus_position':'left'},
        {'shot_id':'S2','focus_subject':'woman','focus_reason':'reaction','preserve_secondary_subject':False,'interaction_requirement':'sequence','focus_position':'right'}]
    assert b.shot_focus_compatible(response,candidate)
    response['visual']['shot_focus_plan'][1]['shot_id']='S1'; assert not b.shot_focus_compatible(response,candidate)

def test_selected_window_flows_to_isolated_output_and_semantic_checkpoint(monkeypatch,tmp_path):
    import movie_broll.broll_pilot as b
    inp=tmp_path/'input'/'film'; inp.mkdir(parents=True); (inp/'movie.mp4').write_bytes(b'x'); (inp/'subtitles.srt').write_text('')
    smoke=tmp_path/'runs'/'film'/'visual-smoke-v1'; smoke.mkdir(parents=True)
    windows=[{'window_id':name,'start_seconds':i*100.,'end_seconds':i*100.+60} for i,name in enumerate(('SW_01','SW_02','SW_03'))]
    (smoke/'windows.json').write_text(json.dumps({'windows':windows}))
    (smoke/'shots.jsonl').write_text('\n'.join(json.dumps({'window_id':x['window_id'],'shot_id':f"{x['window_id']}_S",'start_seconds':x['start_seconds'],'end_seconds':x['end_seconds'],'detector':{'threshold':24}}) for x in windows))
    (smoke/'selected_profile.json').write_text(json.dumps({'selected_threshold':24}))
    nar=tmp_path/'runs'/'film'/'narrative-v2'; nar.mkdir(); (nar/'narrative_map.json').write_text('{}')
    monkeypatch.setattr(b,'visual_signals',lambda movie,shots:[{} for _ in shots]); monkeypatch.setattr(b,'add_context',lambda *args:None)
    monkeypatch.setattr(b,'candidates',lambda shots:[{'candidate_id':'BRC_0001','start_frame':0,'end_frame_exclusive':24,'start_seconds':0.,'end_seconds':1.,'duration_seconds':1.,'editorial':{'decision':'REVIEW'}}])
    class Capture:
        def get(self, prop): return 24 if prop == b.cv2.CAP_PROP_FPS else 1920
        def release(self): pass
    monkeypatch.setattr(b.cv2,'VideoCapture',lambda _:Capture())
    calls=[]
    def validate(items,movie,srt,narrative,checkpoint_dir,fps,window_id,provider,model):
        calls.append((window_id,checkpoint_dir))
        return {'provider':'fake','model':'fake','requests':0,'reused':0,'usage':{}}
    monkeypatch.setattr(b,'semantic_validate',validate)
    report=b.run_broll_pilot(inp,window_id='SW_03')
    other=b.run_broll_pilot(inp,window_id='SW_01')
    assert report['window'] == 'SW_03'
    assert report['output'] == tmp_path/'runs'/'film'/'broll-pilot-v1'/'SW_03'
    assert other['output'] == tmp_path/'runs'/'film'/'broll-pilot-v1'/'SW_01'
    assert calls == [('SW_03', report['output']/'semantic_checkpoints'),('SW_01', other['output']/'semantic_checkpoints')]
    saved=json.loads((report['output']/'candidates.json').read_text())
    assert saved['window_id'] == 'SW_03' and saved['candidates'][0]['window_id'] == 'SW_03'
    assert saved['candidates'][0]['visual_event_id'] == 'SW_03_VE_000001'
    other_saved=json.loads((other['output']/'candidates.json').read_text())
    assert other_saved['candidates'][0]['visual_event_id'] == 'SW_01_VE_000001'

def test_pilot_event_ids_are_window_scoped_and_deterministic():
    assert pilot_event_id('SW_05',1) == 'SW_05_VE_000001'
    assert pilot_event_id('SW_05',1) == pilot_event_id('SW_05',1)
    assert pilot_event_id('SW_05',1) != pilot_event_id('SW_06',1)

def _semantic_item(candidate_id='BRC_0001', source_shot_id='S1'):
    return {'candidate_id':candidate_id,'visual_event_id':'VE_000001','start_frame':10,'end_frame_exclusive':20,
            'start_seconds':0.,'end_seconds':1.,'duration_seconds':1.,'source_shot_ids':[source_shot_id]}

def _semantic_files(tmp_path):
    movie=tmp_path/'film.mp4'; srt=tmp_path/'subtitles.srt'; narrative=tmp_path/'map.json'
    movie.write_bytes(b'movie'); srt.write_text(''); narrative.write_text('{"segments":[]}')
    return movie,srt,narrative

def test_semantic_status_is_scoped_to_current_window_and_legacy_ids_do_not_collide(monkeypatch,tmp_path):
    import movie_broll.broll_pilot as b
    from movie_broll.broll_semantics import SemanticResponse
    from movie_broll.processing_ledger import ProcessingLedger
    movie,srt,narrative=_semantic_files(tmp_path); movie_id=movie.parent.name; run=tmp_path/'film'; pilot=run/'broll-pilot-v1'; ledger=ProcessingLedger(run,movie_id,{})
    legacy=_semantic_item(); ledger.register(legacy,'old'); ledger.stage('VE_000001','semantic','FAILED_FINAL',error='historical')
    other=_semantic_item('BRC_0002'); other['visual_event_id']='VE_000002'; ledger.register(other,'other'); ledger.stage('VE_000002','semantic','PENDING')
    monkeypatch.setattr(b,'candidate_contact_sheet',lambda *args:b'jpg')
    class Provider:
        identifier='fake'; model='fake'
        def generate(self,*args):
            data=_semantic(); data['visual']['shot_focus_plan'][0]['shot_id']=args[1]['source_shot_ids'][0]
            return SemanticResponse(data,{'prompt_tokens':0,'response_tokens':0,'thinking_tokens':0,'cached_tokens':0,'total_tokens':0})
    item=_semantic_item()
    report=b.semantic_validate([item],movie,srt,narrative,pilot/'SW_06'/'semantic_checkpoints',24,'SW_06',Provider())
    assert item['visual_event_id'] == 'SW_06_VE_000001'
    assert report['status'] == 'COMPLETE' and report['complete'] == 1
    assert report['pending'] == report['semantic_pending'] == report['semantic_failed_retryable'] == report['semantic_failed_final'] == 0
    assert report['remaining_count'] == 0
    summary=json.loads((run/'progress_summary.json').read_text())
    assert summary['window_id'] == 'SW_06' and summary['status'] == 'COMPLETE'
    assert 'SW_06_VE_000001' in ProcessingLedger(run,movie_id,{}).data['events']

def test_invalid_provider_focus_plan_has_two_bounded_resume_safe_attempts(monkeypatch,tmp_path):
    import movie_broll.broll_pilot as b
    from movie_broll.broll_semantics import SemanticResponse
    movie,srt,narrative=_semantic_files(tmp_path); monkeypatch.setattr(b,'candidate_contact_sheet',lambda *args:b'jpg')
    class Provider:
        identifier='fake'; model='fake'
        def __init__(self): self.calls=0
        def generate(self,*args):
            self.calls+=1
            return SemanticResponse(_semantic(),{'prompt_tokens':0,'response_tokens':0,'thinking_tokens':0,'cached_tokens':0,'total_tokens':0})
    provider=Provider(); checkpoint=tmp_path/'broll-pilot-v1'/'SW_06'/'semantic_checkpoints'
    first=b.semantic_validate([_semantic_item(source_shot_id='SW_06_SHOT_0001')],movie,srt,narrative,checkpoint,24,'SW_06',provider)
    assert first['status'] == 'COMPLETE' and first['semantic_pending'] == 0 and first['semantic_failed_retryable'] == 1 and first['semantic_failed_final'] == 0 and first['remaining_count'] == 1
    second=b.semantic_validate([_semantic_item(source_shot_id='SW_06_SHOT_0001')],movie,srt,narrative,checkpoint,24,'SW_06',provider)
    assert second['status'] == 'COMPLETE' and provider.calls == 2 and second['semantic_failed_retryable'] == 0 and second['semantic_failed_final'] == 1
    third=b.semantic_validate([_semantic_item(source_shot_id='SW_06_SHOT_0001')],movie,srt,narrative,checkpoint,24,'SW_06',provider)
    assert provider.calls == 2 and third['semantic_failed_final'] == 1

def test_namespaced_event_reuses_matching_checkpoint_fingerprint(monkeypatch,tmp_path):
    import movie_broll.broll_pilot as b
    from movie_broll.broll_semantics import SemanticResponse
    movie,srt,narrative=_semantic_files(tmp_path); monkeypatch.setattr(b,'candidate_contact_sheet',lambda *args:b'jpg')
    class Provider:
        identifier='fake'; model='fake'
        def __init__(self): self.calls=0
        def generate(self,*args):
            self.calls+=1; data=_semantic(); data['visual']['shot_focus_plan'][0]['shot_id']=args[1]['source_shot_ids'][0]
            return SemanticResponse(data,{'prompt_tokens':0,'response_tokens':0,'thinking_tokens':0,'cached_tokens':0,'total_tokens':0})
    provider=Provider(); checkpoint=tmp_path/'broll-pilot-v1'/'SW_06'/'semantic_checkpoints'
    b.semantic_validate([_semantic_item()],movie,srt,narrative,checkpoint,24,'SW_06',provider)
    rerun=_semantic_item(); rerun['visual_event_id']='VE_000001'
    report=b.semantic_validate([rerun],movie,srt,narrative,checkpoint,24,'SW_06',provider)
    assert rerun['visual_event_id'] == 'SW_06_VE_000001' and report['reused'] == 1 and provider.calls == 1

def test_structural_review_can_be_promoted_and_structural_keep_can_be_demoted():
    reviewed={'structural_decision':'REVIEW','editorial':_semantic()['editorial']}
    kept={'structural_decision':'KEEP','editorial':{**_semantic()['editorial'],'decision':'REVIEW'}}
    assert reviewed['editorial']['decision']=='KEEP'
    assert kept['editorial']['decision']=='REVIEW'

def test_narrative_bridge_reads_canonical_value_wrappers_and_flattens_themes():
    context=_narrative_context([{'segment_id':'NARR_1','start_seconds':0,'end_seconds':5,
        'narrative_summary':{'value':'Resumen canónico'},'narrative_tone':{'value':'sad'},
        'themes':[{'value':'duelo'},'familia'],'narrative_function':{'value':'conversation'}}],1,2)
    assert context == {'segment_ids':['NARR_1'],'summary_es':['Resumen canónico'],'tone':['sad'],
        'themes':['duelo','familia'],'interaction_context':['conversation'],'literal_transcription':False}
    assert _narrative_context([],1,2)['themes'] == []

def test_people_and_relationship_provenance_contract():
    data=_semantic(); data['visual']['people']=[{'presentation':'woman','apparent_age_group':'young_adult','frame_role':'primary','position':'left'},{'presentation':'man','apparent_age_group':'adult','frame_role':'secondary','position':'right'}]
    data['relationships']=[{'type':'romantic_partner','source':'narrative','confidence':.78}]
    assert validate_response(data) == []
    data['relationships'][0]['source']='visual'
    assert 'overreach' in validate_response(data)[0]

def test_semantic_scarcity_suppresses_only_nearby_same_meaning():
    def candidate(identifier,start,meaning,action='hablar',presentation='woman'):
        return {'candidate_id':identifier,'start_seconds':start,'end_seconds':start+5,'score':{'total':90-start},
          'visual':{'setting':'azotea','actions':[action],'visible_interactions':[]},'people':[{'presentation':presentation}], 'relationships':[],
          'editorial':{'decision':'KEEP','standalone_meaning_es':meaning,'use_cases_es':[meaning]}}
    same=candidate('BRC_0002',6,'mujer hablando en azotea')
    winner=candidate('BRC_0001',0,'mujer hablando en azotea')
    distinct=candidate('BRC_0003',12,'mujer se disculpa en azotea','disculparse')
    apply_semantic_scarcity([winner,same,distinct])
    assert same['editorial']['decision']=='REJECT' and same['semantic_redundancy']['redundant_with']=='BRC_0001'
    assert distinct['editorial']['decision']=='KEEP'

def test_boundary_validation_fails_previous_frame_provenance(monkeypatch):
    import movie_broll.broll_pilot as b
    frame=lambda value: __import__('numpy').full((10,10,3),value,dtype='uint8')
    class Capture:
        def __init__(self, frames): self.frames=frames; self.pos=0
        def set(self, _, value): self.pos=int(value)
        def read(self): return (self.pos in self.frames, self.frames.get(self.pos))
        def get(self, _): return len(self.frames)
        def release(self): pass
    source={9:frame(1),10:frame(50),11:frame(60),12:frame(70)}; exported={0:frame(1),1:frame(60)}
    calls=iter([Capture(source),Capture(exported)])
    monkeypatch.setattr(b.cv2,'VideoCapture',lambda _:next(calls))
    result=boundary_validation(Path('source.mp4'),Path('export.mp4'),{'candidate_id':'BRC_1','start_frame':10,'end_frame_exclusive':12})
    assert result['boundary_validation']=='FAIL' and not result['first_frame_matches_target'] and result['actual_frame_count']==2

def test_frame_exact_export_on_synthetic_movie(tmp_path):
    """A real ffmpeg regression using only generated, uniquely coloured frames."""
    import cv2
    import movie_broll.broll_pilot as b
    source=tmp_path/'source.avi'; out=tmp_path/'clip.mp4'; fps=24
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),fps,(64,48))
    for number in range(48): writer.write(__import__('numpy').full((48,64,3),(number*5 % 255, number*11 % 255, number*17 % 255),dtype='uint8'))
    writer.release()
    # Starts beyond the one-second preroll so the coarse seek + relative trim path
    # (not merely a decode-from-zero path) is exercised.
    candidate={'candidate_id':'BRC_SYN','start_frame':36,'end_frame_exclusive':46}
    b.subprocess.run(ffmpeg_export_command(source,candidate,out,fps),check=True,stdout=b.subprocess.DEVNULL,stderr=b.subprocess.DEVNULL)
    validation=boundary_validation(source,out,candidate)
    assert validation['boundary_validation']=='PASS'
    assert validation['expected_frame_count']==validation['actual_frame_count']==10

def test_quota_stops_later_events_and_resume_reuses_checkpoints(monkeypatch,tmp_path):
    import movie_broll.broll_pilot as b
    from movie_broll.broll_semantics import SemanticResponse
    movie=tmp_path/'film.mp4'; srt=tmp_path/'subtitles.srt'; narrative=tmp_path/'map.json'
    movie.write_bytes(b'movie'); srt.write_text(''); narrative.write_text('{"segments":[]}')
    items=[{'candidate_id':f'BRC_{i:04d}','visual_event_id':f'VE_{i:06d}','start_frame':i*10,'end_frame_exclusive':i*10+10,'start_seconds':float(i),'end_seconds':float(i+1),'duration_seconds':1.,'source_shot_ids':[f'S{i}']} for i in range(1,6)]
    monkeypatch.setattr(b,'candidate_contact_sheet',lambda *args:b'jpg')
    class Provider:
        identifier='fake'
        def __init__(self, quota=False): self.calls=[]; self.quota=quota
        def generate(self,*args):
            self.calls.append(args[1]['candidate_id'])
            if self.quota and len(self.calls)==3: raise RuntimeError('429 free tier request quota exceeded')
            response=_semantic(); response['visual']['shot_focus_plan'][0]['shot_id']=args[1]['source_shot_ids'][0]
            return SemanticResponse(response,{'prompt_tokens':1,'response_tokens':1,'thinking_tokens':0,'cached_tokens':0,'total_tokens':2})
    first=Provider(True)
    report=b.semantic_validate(items,movie,srt,narrative,tmp_path/'broll-pilot-v1'/'SW_01'/'semantic_checkpoints',24,'SW_01',first)
    assert first.calls==['BRC_0001','BRC_0002','BRC_0003'] and report['status']=='PARTIAL_QUOTA'
    assert report['complete']==2 and report['pending']==2 and report['remaining_count']==3
    second=Provider()
    report=b.semantic_validate(items,movie,srt,narrative,tmp_path/'broll-pilot-v1'/'SW_01'/'semantic_checkpoints',24,'SW_01',second)
    assert second.calls==['BRC_0003','BRC_0004','BRC_0005']
    assert report['reused']==2 and report['status']=='COMPLETE'



def test_pool_provider_checkpoint_provenance_and_reuse(monkeypatch,tmp_path):
    import json
    import movie_broll.broll_pilot as b
    from movie_broll.broll_semantics import SemanticResponse

    movie,srt,narrative=_semantic_files(tmp_path)
    monkeypatch.setattr(b,'candidate_contact_sheet',lambda *args:b'jpg')

    class Provider:
        identifier='gemini'
        model='gemini-3.6-flash'

        def __init__(self):
            self.calls=0

        def generate(self,*args):
            self.calls+=1
            data=_semantic()
            data['visual']['shot_focus_plan'][0]['shot_id']=args[1]['source_shot_ids'][0]
            return SemanticResponse(
                data,
                {'prompt_tokens':1,'response_tokens':1,'thinking_tokens':0,'cached_tokens':0,'total_tokens':2},
                provider='gemini-primary-2',
                model=self.model,
                attempts=2,
                provider_trace=(
                    {'provider':'gemini-primary-1','status':429},
                    {'provider':'gemini-primary-2','status':'COMPLETE'},
                ),
            )

    provider=Provider()
    checkpoint=tmp_path/'broll-pilot-v1'/'SW_06'/'semantic_checkpoints'
    item=_semantic_item()

    first=b.semantic_validate([item],movie,srt,narrative,checkpoint,24,'SW_06',provider)

    assert first['status']=='COMPLETE'
    assert provider.calls==1
    assert first['requests']==2

    data=json.loads((checkpoint/f"{item['candidate_id']}.json").read_text())

    assert data['provider']=='gemini-primary-2'
    assert data['model']=='gemini-3.6-flash'
    assert data['provider_attempts']==2
    assert len(data['provider_trace'])==2

    rerun=_semantic_item()
    second=b.semantic_validate([rerun],movie,srt,narrative,checkpoint,24,'SW_06',provider)

    assert second['reused']==1
    assert provider.calls==1


def test_pool_exhaustion_returns_partial_quota_with_failure(monkeypatch,tmp_path):
    import movie_broll.broll_pilot as b
    from movie_broll.broll_semantics import GeminiProviderPool

    movie,srt,narrative=_semantic_files(tmp_path)
    monkeypatch.setattr(b,'candidate_contact_sheet',lambda *args:b'jpg')

    class QuotaProvider:
        model='gemini-3.6-flash'

        def __init__(self,identifier):
            self.identifier=identifier

        def generate(self,*args):
            raise RuntimeError(
                'Error code: 429 '
                'generate_content_free_tier_requests '
                'quota exceeded. Please retry in: 17.25s'
            )

    pool=GeminiProviderPool(
        [
            QuotaProvider('gemini-primary-1'),
            QuotaProvider('gemini-primary-2'),
            QuotaProvider('gemini-primary-3'),
        ],
        QuotaProvider('gemini-backup'),
    )

    item=_semantic_item()

    report=b.semantic_validate(
        [item],
        movie,
        srt,
        narrative,
        tmp_path/'broll-pilot-v1'/'SW_06'/'semantic_checkpoints',
        24,
        'SW_06',
        pool,
    )

    assert report['status']=='PARTIAL_QUOTA'
    assert report['quota_exhausted'] is True
    assert report['provider_unavailable'] is False
    assert report['requests']==4

    failure=report['failure']

    assert failure['http_status']==429
    assert failure['reason']=='quota_exceeded'
    assert failure['retryable'] is True
    assert failure['retry_after_seconds']==17.25
    assert failure['resume_safe'] is True
    assert failure['ledger_saved'] is True
    assert failure['failed_event_id']==item['visual_event_id']
    assert failure['providers_attempted']==[
        'gemini-primary-1',
        'gemini-primary-2',
        'gemini-primary-3',
        'gemini-backup',
    ]
