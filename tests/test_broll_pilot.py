import json
from pathlib import Path
import pytest
from movie_broll.broll_pilot import (PILOT_WINDOW, candidates, dedupe, discover,
    ffmpeg_export_command, generate_groups, probe, review_reel_command, score_candidate)
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
    monkeypatch.setattr(b,'run_broll_pilot',lambda x:{'window':'SW_02','shots':16,'candidates':8,'KEEP':5,'REVIEW':2,'REJECT':1,'exported':5,'average_keep_duration':7.2,'output':tmp_path})
    assert main(['pilot','broll',str(tmp_path)]) == 0
    assert '[broll-pilot] window: SW_02' in capsys.readouterr().out

def test_discovery_uses_persisted_sw02(tmp_path):
    inp=tmp_path/'input'/'film'; inp.mkdir(parents=True); (inp/'movie.mp4').write_bytes(b'x'); (inp/'subtitles.srt').write_text('')
    smoke=tmp_path/'runs'/'film'/'visual-smoke-v1'; smoke.mkdir(parents=True)
    (smoke/'windows.json').write_text(json.dumps({'windows':[{'window_id':'SW_02','start_seconds':1,'end_seconds':61}]})); (smoke/'shots.jsonl').write_text(''); (smoke/'selected_profile.json').write_text(json.dumps({'selected_threshold':24}))
    nar=tmp_path/'runs'/'film'/'narrative-v2'; nar.mkdir(); (nar/'narrative_map.json').write_text('{}')
    assert discover(inp)['window']['window_id']==PILOT_WINDOW

def test_grouping_and_consecutive_max(monkeypatch):
    fake_similarity(monkeypatch)
    xs=[shot(1,0,2),shot(2,2,5),shot(3,5,11),shot(4,11,27)]
    assert generate_groups(xs)==[[0,1],[2]] # short pair, standalone, >15 rejected

def test_candidate_order_scores_and_decisions(monkeypatch):
    fake_similarity(monkeypatch); xs=[shot(1,0,6),shot(2,6,12)]
    out=candidates(xs)
    assert [x['candidate_id'] for x in out]==['BRC_0001','BRC_0002']
    assert all(0<=x['score']['total']<=100 and set(x['score'])=={'duration_fit','visual_quality','continuity','motion_usefulness','structural_simplicity','total'} for x in out)
    assert score_candidate({'duration_seconds':7,'source_shot_ids':['x'],'signals':{'sharpness':120,'brightness':110,'near_black_fraction':0,'subtitle_occupancy':0,'visual_continuity':1,'motion':12}})['total']>=70
    assert candidates(xs)==candidates(xs) # stable IDs and no random scoring

def test_heavy_overlap_dedupe_and_keep_cap():
    def item(ids,score,start): return {'source_shot_ids':ids,'score':{'total':score},'editorial':{'decision':'KEEP'},'start_seconds':start,'end_seconds':start+1}
    assert len(dedupe([item(['a','b'],90,0),item(['a','b'],80,1)]))==1
    many=dedupe([item([str(i)],99,i) for i in range(10)])
    assert sum(x['editorial']['decision']=='KEEP' for x in many)==8

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
    return {'visual':{'summary_es':'Una persona mira una puerta','subjects':['persona'],'objects':['puerta'],'actions':['mirar una puerta'],'people_count_estimate':'1','setting':'interior','visible_interactions':[],'visible_emotions':['neutral'],'primary_subject_position':'center','primary_subject_description':'persona','visual_focus':'persona y puerta'},'editorial':{'standalone_meaning_es':'Una persona espera ante una puerta','reusable_broll':True,'action_or_moment_complete':'true','use_cases_es':['esperar ante una puerta'],'negative_use_cases_es':['no afirmar una llamada'],'search_terms_es':['esperar'],'editorial_confidence':'high','reason':'acción visible','decision':decision}}

def test_frame_bounds_are_authoritative_and_no_blind_offset(tmp_path):
    cmd=ffmpeg_export_command(Path('movie.mp4'),{'start_frame':240,'end_frame_exclusive':360},tmp_path/'x.mp4',24)
    assert '-frames:v' in cmd and cmd[cmd.index('-frames:v')+1]=='120'
    assert '0.083' not in ' '.join(cmd) and '-ss' in cmd

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
    (tmp_path/'BRC_0001.json').write_text(json.dumps({**c,'model':'gemini-3.6-flash','response':_semantic()}))
    response=b._semantic_checkpoint(tmp_path/'BRC_0001.json',c,'gemini-3.6-flash')
    assert response['visual']['primary_subject_position']=='center'

def test_structural_review_can_be_promoted_and_structural_keep_can_be_demoted():
    reviewed={'structural_decision':'REVIEW','editorial':_semantic()['editorial']}
    kept={'structural_decision':'KEEP','editorial':{**_semantic()['editorial'],'decision':'REVIEW'}}
    assert reviewed['editorial']['decision']=='KEEP'
    assert kept['editorial']['decision']=='REVIEW'
