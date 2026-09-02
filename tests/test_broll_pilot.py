import json
from pathlib import Path
import pytest
from movie_broll.broll_pilot import (PILOT_WINDOW, candidates, dedupe, discover,
    ffmpeg_export_command, generate_groups, probe, review_reel_command, score_candidate)
from movie_broll.cli import main

def shot(n, start, end, similarity=1):
    return {'shot_id':f'S_{n}', 'start_seconds':start,'end_seconds':end,'duration_seconds':end-start,
      'brightness_mean':110,'sharpness_score':120,'motion_score':12,'near_black_fraction':.05,
      'subtitle_occupancy_ratio':.1,'narrative_segment_ids':['N1'], '_hist':None}

def fake_similarity(monkeypatch):
    import movie_broll.broll_pilot as m
    monkeypatch.setattr(m,'_similarity',lambda a,b:.8)

def test_cli_pilot_broll_exists(monkeypatch,tmp_path,capsys):
    import movie_broll.broll_pilot as b
    monkeypatch.setattr(b,'run_broll_pilot',lambda x:{'window':'SW_02','shots':16,'candidates':8,'KEEP':5,'REVIEW':2,'REJECT':1,'exported':5,'output':tmp_path})
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
