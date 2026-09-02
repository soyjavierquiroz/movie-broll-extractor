import json
from pathlib import Path
import pytest
from movie_broll.visual import Window, build_shots, choose_threshold, select_smoke_windows, validate_shots
from movie_broll.cli import main

def seg(identifier,start,end,kind="conversation",function="setup",context="low",density=None):
    return {"segment_id":identifier,"start_seconds":start,"end_seconds":end,"segment_type":{"value":kind},"narrative_function":{"value":function},"context_dependency":{"value":context},"dialogue_density":{"value":density} if density else None}
def test_window_selection_is_deterministic_and_diverse():
    narrative={"segments":[seg("a",100,260,"conversation","setup"),seg("b",1200,1280,"sparse_dialogue","transition"),seg("c",2500,2700,"conversation","conflict"),seg("d",4000,4200,"conversation","decision")]}
    one=select_smoke_windows(narrative,5000); two=select_smoke_windows(narrative,5000)
    assert one==two and one[0].source_narrative_segment_ids==["c"] and one[1].source_narrative_segment_ids==["b"] and one[2].source_narrative_segment_ids==["d"]
    assert all(0<=w.start_seconds<w.end_seconds<=5000 for w in one)
    assert all(a.end_seconds<=b.start_seconds or b.end_seconds<=a.start_seconds for a in one for b in one if a != b)
def test_shot_frame_semantics_and_validation():
    window=Window("SW_01",10,20,"test",["x"]); shots=build_shots(window,24,[300,360],24)
    assert shots[0]["start_frame"]==240 and shots[0]["end_frame_exclusive"]==300 and shots[0]["duration_seconds"]==2.5
    assert validate_shots(shots,[window],24)["status"]=="PASS"
    broken=[dict(x) for x in shots]; broken[1]["start_frame"]+=1
    assert validate_shots(broken,[window],24)["status"]=="FAIL"
    broken=[dict(x) for x in shots]; broken[0]["representative_frame_seconds"]=100
    assert validate_shots(broken,[window],24)["status"]=="FAIL"
    broken=[dict(x) for x in shots]; broken[0]["end_frame_exclusive"]=broken[0]["start_frame"]
    assert validate_shots(broken,[window],24)["status"]=="FAIL"
def test_threshold_choice_prefers_less_aggressive_tie():
    shots={t:build_shots(Window("W",0,10,"",[]),24,[120],t) for t in (20.0,24.0,27.0)}
    assert choose_threshold(shots)[0]==27.0
def test_cli_missing_movie(tmp_path,capsys):
    assert main(["visual","smoke",str(tmp_path)])==2
    assert "movie does not exist" in capsys.readouterr().err
def test_cli_missing_narrative(tmp_path,capsys):
    (tmp_path/"movie.mp4").write_bytes(b"placeholder")
    assert main(["visual","smoke",str(tmp_path)])==2
    assert "narrative map does not exist" in capsys.readouterr().err
def test_cli_successful_mocked_smoke(monkeypatch,tmp_path,capsys):
    import movie_broll.visual
    monkeypatch.setattr(movie_broll.visual,"run_visual_smoke",lambda *args,**kwargs:{"status":"COMPLETE","shot_count":7})
    assert main(["visual","smoke",str(tmp_path)])==0
    assert "shots: 7" in capsys.readouterr().out
