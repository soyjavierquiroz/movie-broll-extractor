import json
from copy import deepcopy
from pathlib import Path

import pytest

from movie_broll import production
from movie_broll.broll_semantics import SemanticResponse
from movie_broll.semantic_observations import observation_path
from movie_broll.finalization import _source_movie_sha256


def fixture(tmp_path: Path):
    source = tmp_path / "input" / "film"; source.mkdir(parents=True)
    (source / "movie.mp4").write_bytes(b"movie")
    (source / "subtitles.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHola\n")
    run = tmp_path / "runs" / "film" / "narrative-v2"; run.mkdir(parents=True)
    (run / "narrative_map.json").write_text(json.dumps({"segments": [{"segment_id": "N1", "start_seconds": 0, "end_seconds": 1}]}))
    return source


def test_preflight_requires_canonical_inputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="movie.mp4"):
        production.preflight(tmp_path / "input" / "missing")


def test_preflight_requires_manual_external_narrative_when_map_is_missing(monkeypatch, tmp_path):
    source=tmp_path/'input'/'film'; source.mkdir(parents=True)
    (source/'movie.mp4').write_bytes(b'movie')
    (source/'subtitles.srt').write_text('1\n00:00:00,000 --> 00:00:01,000\nHola\n')
    monkeypatch.setattr(production,'inspect_movie',lambda _: {'video':{'width':160,'height':90}})
    monkeypatch.setattr(production,'load_or_detect',lambda *_: {'x':0,'y':0,'width':160,'height':90,'structural_bars':False})
    with pytest.raises(production.ProductionFailure, match='external Narrative Mapper handoff'):
        production.preflight(source)


def test_preflight_does_not_replace_an_incompatible_manual_narrative(monkeypatch, tmp_path):
    source=tmp_path/'input'/'film'; source.mkdir(parents=True)
    (source/'movie.mp4').write_bytes(b'movie')
    (source/'subtitles.srt').write_text('1\n00:00:00,000 --> 00:00:01,000\nHola\n')
    run=tmp_path/'runs'/'film'/'narrative-v2'; run.mkdir(parents=True)
    (run/'narrative_map.json').write_text(json.dumps({'analysis':{'chunk_profile':'1200s_90s_overlap'},'segments':[{'segment_id':'old'}]}))
    (run/'narrative_run.json').write_text(json.dumps({'window_seconds':1200,'overlap_seconds':90}))
    monkeypatch.setattr(production,'inspect_movie',lambda _: {'video':{'width':160,'height':90}})
    monkeypatch.setattr(production,'load_or_detect',lambda *_: {'x':0,'y':0,'width':160,'height':90,'structural_bars':False})
    before=(run/'narrative_map.json').read_bytes()
    with pytest.raises(production.ProductionFailure, match='external Narrative Mapper handoff'):
        production.preflight(source)
    assert (run/'narrative_map.json').read_bytes() == before


def test_process_orchestrates_full_movie_and_reuses_same_source(monkeypatch, tmp_path):
    source = fixture(tmp_path)
    info = {"movie": source / "movie.mp4", "srt": source / "subtitles.srt", "run": tmp_path / "runs" / "film",
            "narrative": tmp_path / "runs" / "film" / "narrative-v2" / "narrative_map.json", "movie_sha256": "a" * 64,
            "srt_sha256": "b" * 64, "metadata": {"video": {"fps": 24, "width": 160, "height": 120}, "duration_seconds": 1}}
    monkeypatch.setattr(production, "preflight", lambda _: info)
    monkeypatch.setattr(production, "_semantic_terminal_contract", lambda *_: (True, []))
    shots = [{"shot_id": "FULL_S_1", "start_seconds": 0, "end_seconds": 1, "start_frame": 0, "end_frame_exclusive": 24}]
    event = {"candidate_id": "BRC_0001", "visual_event_id": "VE_stable", "start_seconds": 0, "end_seconds": 1,
             "start_frame": 0, "end_frame_exclusive": 24, "source_shot_ids": ["FULL_S_1"], "editorial": {"decision": "KEEP", "status": "VALIDATED"}}
    monkeypatch.setattr(production, "_technical_shots", lambda _: shots)
    monkeypatch.setattr(production, "_visual_events", lambda *_: [event])
    calls = []
    def semantics(items, *args, **kwargs):
        calls.append(kwargs["preserve_event_ids"])
        for item in items:
            item["editorial"] = {"decision": "KEEP", "status": "VALIDATED"}
        return {"quota_exhausted": False, "status": "COMPLETE"}
    monkeypatch.setattr(production, "semantic_validate", semantics)
    monkeypatch.setattr(production, "apply_semantic_scarcity", lambda _: None)
    detector_calls=[]
    monkeypatch.setattr(production, "person_detector_preflight", lambda: detector_calls.append(True) or {"loaded": True})
    finalized=[]
    monkeypatch.setattr(production, "finalize_pilot", lambda *a, **k: finalized.append(k) or {"status": "COMPLETE", "completed": 1, "review": 0, "failed_final": 0})
    bounded_labels = []
    def bounded(label, action, report, ledger):
        bounded_labels.append(label)
        return action()
    monkeypatch.setattr(production, "_bounded_operation", bounded)
    lines=[]; result = production.process(source, provider=object(), reporter=lines.append)
    assert result["status"] == "COMPLETE" and calls == [True]
    assert "technical-shots" in bounded_labels
    assert "person-detector-preflight" in bounded_labels and detector_calls == [True]
    assert finalized[0]["detector_preflight"] == {"loaded": True}
    assert (info["run"] / "progress_summary.json").is_file()
    assert any("building Visual Events" in line for line in lines)
    assert "PRODUCTION_STARTED" in (info["run"] / "progress.jsonl").read_text()
    assert production.process(source, provider=object())["status"] == "COMPLETE"
    assert calls == [True]  # completed Visual Events skip semantic work on resume


def test_normal_process_observes_only_missing_then_applies_local_policy(monkeypatch, tmp_path):
    source = fixture(tmp_path)
    info = {"movie": source / "movie.mp4", "srt": source / "subtitles.srt", "run": tmp_path / "runs" / "film",
            "narrative": tmp_path / "runs" / "film" / "narrative-v2" / "narrative_map.json",
            "movie_sha256": "a" * 64, "srt_sha256": "b" * 64,
            "metadata": {"video": {"fps": 24, "width": 160, "height": 120}, "duration_seconds": 2}}
    monkeypatch.setattr(production, "preflight", lambda _: info)
    shots = [{"shot_id": "S1", "start_seconds": 0, "end_seconds": 1, "start_frame": 0, "end_frame_exclusive": 24},
             {"shot_id": "S2", "start_seconds": 1, "end_seconds": 2, "start_frame": 24, "end_frame_exclusive": 48}]
    events = [{"candidate_id": f"BRC_{n}", "visual_event_id": f"VE_{n}", "start_seconds": n - 1,
               "end_seconds": n, "start_frame": (n - 1) * 24, "end_frame_exclusive": n * 24,
               "source_shot_ids": [f"S{n}"], "editorial": {"decision": "REVIEW", "status": "PROVISIONAL"}}
              for n in (1, 2)]
    monkeypatch.setattr(production, "_technical_shots", lambda _: shots)
    monkeypatch.setattr(production, "_visual_events", lambda *_: events)
    from movie_broll.temporal_evidence import PROFILE, sample_plan
    from movie_broll.temporal_semantics import latest_record, record_path
    def temporal_sheet(_movie, item, fps, evidence, _active, **kwargs):
        assert kwargs['evidence_profile'] == PROFILE
        evidence.update(evidence_profile=PROFILE, samples=sample_plan(item, fps), technical_shots=[], fps=fps)
        return b"sheet"
    monkeypatch.setattr(production, "candidate_contact_sheet", temporal_sheet)
    monkeypatch.setattr(production, "_bounded_operation", lambda _label, action, _report, _ledger: action())
    monkeypatch.setattr(production, "apply_semantic_scarcity", lambda _: None)
    monkeypatch.setattr(production, "finalize_pilot", lambda *a, **k: {"status": "COMPLETE", "completed": 0, "review": 0, "failed_final": 0})
    class Observer:
        identifier = "fake"; model = "observation-model"
        def __init__(self): self.calls = 0
        def generate(self, prompt, context, jpeg):
            self.calls += 1
            assert "Do not return KEEP" in prompt
            event_id = context["visual_event_id"]
            return SemanticResponse({"event_id": event_id, "represented_shot_ids": context["source_shot_ids"],
                "people_count": "0", "visible_person_ids": [], "action_evidence_ids": [],
                "object_evidence_ids": [], "visible_states": [], "movement": "present", "physical_interactions": [],
                "visible_reactions": [], "conversation_present": "false", "conversation_visual_signal": "none",
                "visual_utility_kind": "concrete_action", "action_or_moment_complete": "true", "context_dependency": "low",
                "technical_observations": {"title_card": False, "credits": False, "logo": False, "dominant_text": False,
                                           "black_or_empty": False, "corrupt_or_unusable": False},
                "shot_focus_plan": [{"shot_id": sid, "focus_subject": "environment", "focus_reason": "visible setting",
                                     "focus_position": "center", "preserve_secondary_subject": False,
                                     "interaction_requirement": "none", "target_person_ids": [], "target_binding_confidence": "unclear"}
                                    for sid in context["source_shot_ids"]],
                "moment_status": "complete_action", "visual_actions": ["visible action"],
                "temporal_support_sample_ids": [row['sample_id'] for row in context['temporal_evidence']['samples']],
                "observed_actions": [{"canonical_label": "visible action", "evidence_type": "distinct_visible_action_or_reaction",
                                      "sample_ids": [row['sample_id'] for row in context['temporal_evidence']['samples']]}]},
                {"prompt_tokens": 1, "cached_tokens": 0, "response_tokens": 1, "thinking_tokens": 0, "total_tokens": 2}, "fake", self.model)
    observer = Observer()
    assert production.process(source, provider=observer, policy_version="broll_policy_v1")["status"] == "COMPLETE"
    assert observer.calls == 2
    assert all(latest_record(info["run"], item) is not None for item in events)
    # A complete normal rerun has no provider work; remove no canonical state.
    assert production.process(source, provider=observer, policy_version="broll_policy_v1")["status"] == "COMPLETE"
    assert observer.calls == 2
    # Simulate one interrupted/unpersisted observation in the isolated fixture.
    second_record=latest_record(info["run"],events[1])
    record_path(info["run"], "VE_2", second_record['observation_fingerprint']).unlink()
    store_path = info["run"] / "visual_event_segments_v1.json"
    store = json.loads(store_path.read_text())
    store["production_status"] = "PARTIAL"; store["status"] = "RUNNING"
    store["batches"]["PBATCH_0002"]["status"] = "PENDING"
    store["batches"]["PBATCH_0002"]["semantic_status"] = "PENDING"
    store_path.write_text(json.dumps(store))
    assert production.process(source, provider=observer, policy_version="broll_policy_v1")["status"] == "COMPLETE"
    assert observer.calls == 2 and latest_record(info["run"],events[1]) is not None


def test_completed_process_reconciles_without_constructing_semantic_provider(monkeypatch, tmp_path):
    source=fixture(tmp_path)
    run=tmp_path/'runs'/'film'
    info={'movie':source/'movie.mp4','srt':source/'subtitles.srt','run':run,
          'narrative':run/'narrative-v2'/'narrative_map.json','movie_sha256':'a'*64,
          'srt_sha256':'b'*64,'metadata':{'video':{'fps':24,'width':160,'height':120}}}
    (run/'source_fingerprint.json').write_text(json.dumps({'movie_sha256':'a'*64}))
    event={'visual_event_id':'VE_1','editorial':{'decision':'KEEP','status':'VALIDATED'}}
    (run/'visual_event_segments_v1.json').write_text(json.dumps({
        'production_status':'COMPLETE','production_batch_size':1,'events':[event],
        'batches':{'PBATCH_0001':{'status':'COMPLETE'}},
    }))
    monkeypatch.setattr(production,'preflight',lambda _:info)
    monkeypatch.setattr(production,'build_semantic_provider_from_env',lambda *_args,**_kwargs:pytest.fail('semantic provider must not be constructed'))
    monkeypatch.setattr(production,'_technical_shots',lambda *_:pytest.fail('technical stage must not run'))
    calls=[]
    monkeypatch.setattr(production,'reconcile_review_packages',lambda path:calls.append(path) or {'promoted':1,'hard_review':0,'insufficient_evidence':0,'incomplete':0})
    result=production.process(source)
    assert result['status']=='COMPLETE' and calls == [run]
    assert result['finalization'][0]['reconciliation']['promoted'] == 1


def test_heartbeat_is_visible_and_logged(monkeypatch, tmp_path):
    run = tmp_path / "runs" / "film"; run.mkdir(parents=True)
    from movie_broll.processing_ledger import ProcessingLedger
    ledger = ProcessingLedger(run, "film", {})
    lines = []
    monkeypatch.setattr(production, "HEARTBEAT_SECONDS", .001)
    def slow():
        import time; time.sleep(.01); return "done"
    assert production._bounded_operation("test", slow, lines.append, ledger) == "done"
    assert any("working" in line for line in lines)
    assert "PRODUCTION_HEARTBEAT" in (run / "progress.jsonl").read_text()


def test_process_reuses_detector_preflight_across_segments(monkeypatch, tmp_path):
    source=fixture(tmp_path)
    narrative=tmp_path/'runs'/'film'/'narrative-v2'/'narrative_map.json'
    narrative.write_text(json.dumps({'segments':[{'segment_id':'N1','start_seconds':0,'end_seconds':1},{'segment_id':'N2','start_seconds':1,'end_seconds':2}]}))
    info={'movie':source/'movie.mp4','srt':source/'subtitles.srt','run':tmp_path/'runs'/'film','narrative':narrative,'movie_sha256':'a'*64,'srt_sha256':'b'*64,'metadata':{'video':{'fps':24,'width':160,'height':120},'duration_seconds':2}}
    monkeypatch.setattr(production,'preflight',lambda _:info)
    monkeypatch.setattr(production, "_semantic_terminal_contract", lambda *_: (True, []))
    shots=[{'shot_id':'S1','start_seconds':0,'end_seconds':2,'start_frame':0,'end_frame_exclusive':48}]
    monkeypatch.setattr(production,'_technical_shots',lambda _:shots)
    def visual_events(_info, _shots):
        return [{'candidate_id':'BRC_0001','visual_event_id':'VE_FULL','start_seconds':0,'end_seconds':2,'start_frame':0,'end_frame_exclusive':48,'source_shot_ids':['S1'],'editorial':{'decision':'KEEP','status':'VALIDATED'}}]
    monkeypatch.setattr(production,'_visual_events',visual_events)
    def semantics(items, *args, **kwargs):
        for item in items:
            item['editorial']={'decision':'KEEP','status':'VALIDATED'}
        return {'quota_exhausted':False,'status':'COMPLETE'}
    monkeypatch.setattr(production,'semantic_validate',semantics)
    monkeypatch.setattr(production,'apply_semantic_scarcity',lambda _:None)
    calls=[]; monkeypatch.setattr(production,'person_detector_preflight',lambda:calls.append(True) or {'loaded':True})
    monkeypatch.setattr(production,'finalize_pilot',lambda *a,**k:{'status':'COMPLETE','completed':0,'review':0,'failed_final':0})
    monkeypatch.setattr(production,'_bounded_operation',lambda _label,action,_report,_ledger:action())
    assert production.process(source,provider=object())['status']=='COMPLETE'
    assert calls == [True]


def test_normal_production_runs_one_final_semantic_pass_per_visual_event(monkeypatch, tmp_path):
    source=fixture(tmp_path)
    info={'movie':source/'movie.mp4','srt':source/'subtitles.srt','run':tmp_path/'runs'/'film','narrative':tmp_path/'runs'/'film'/'narrative-v2'/'narrative_map.json','movie_sha256':'a'*64,'srt_sha256':'b'*64,'metadata':{'video':{'fps':24,'width':160,'height':120},'duration_seconds':16}}
    monkeypatch.setattr(production,'preflight',lambda _:info)
    monkeypatch.setattr(production, "_semantic_terminal_contract", lambda *_: (True, []))
    shots=[{'shot_id':f'S{i}','start_seconds':i*4,'end_seconds':i*4+4,'start_frame':i*96,'end_frame_exclusive':i*96+96} for i in range(4)]
    events=[
        {'candidate_id':'BRC_0001','visual_event_id':'VE_1','start_seconds':0,'end_seconds':8,'start_frame':0,'end_frame_exclusive':192,'source_shot_ids':['S0','S1'],'editorial':{'decision':'REVIEW','status':'PROVISIONAL'}},
        {'candidate_id':'BRC_0002','visual_event_id':'VE_2','start_seconds':8,'end_seconds':16,'start_frame':192,'end_frame_exclusive':384,'source_shot_ids':['S2','S3'],'editorial':{'decision':'REVIEW','status':'PROVISIONAL'}},
    ]
    monkeypatch.setattr(production,'_technical_shots',lambda _:shots)
    monkeypatch.setattr(production,'_visual_events',lambda *_:events)
    semantic_calls=[]
    def semantics(items,*_args,**kwargs):
        semantic_calls.append((list(items),kwargs['window_id'] if 'window_id' in kwargs else _args[5]))
        for event in items: event['editorial']={'decision':'REJECT','status':'VALIDATED'}
        return {'status':'COMPLETE','quota_exhausted':False,'provider_unavailable':False}
    monkeypatch.setattr(production,'semantic_validate',semantics)
    monkeypatch.setattr(production,'apply_semantic_scarcity',lambda _:None)
    monkeypatch.setattr(production,'finalize_pilot',lambda *args,**kwargs:{'status':'COMPLETE','completed':0,'review':0,'failed_final':0})
    monkeypatch.setattr(production,'_bounded_operation',lambda _label,action,_report,_ledger:action())

    assert production.process(source,provider=object())['status']=='COMPLETE'
    assert [[event['source_shot_ids'] for event in items] for items, _window in semantic_calls] == [
        [['S0', 'S1']], [['S2', 'S3']],
    ]
    assert [window for _items, window in semantic_calls] == ['FULL', 'FULL']


def test_source_change_retires_only_source_artifacts(tmp_path):
    run = tmp_path / "runs" / "film"; (run / "assets").mkdir(parents=True); (run / "review").mkdir(); (run / ".work").mkdir()
    for path in (run / "technical_shots.json", run / "visual_events.json", run / "assets" / "old.mp4", run / "review" / "old.mp4"):
        path.write_bytes(b"x")
    narrative = run / "narrative-v2"; narrative.mkdir(); (narrative / "narrative_map.json").write_text("{}")
    production._invalidate_source_media(run, "old", "new")
    assert not (run / "technical_shots.json").exists() and not list((run / "assets").iterdir())
    assert (narrative / "narrative_map.json").exists()


def test_v1_event_store_does_not_reuse_or_overwrite_v5_drafts(tmp_path):
    run=tmp_path/'runs'/'film'; run.mkdir(parents=True)
    legacy=run/'production_segments.json'
    legacy.write_text(json.dumps({'fingerprint':'v5','segments':{'N1':{'draft_events':[{'candidate_id':'DRAFT_BRC_0001'}]}}}))
    path,store=production._event_store({'run':run,'movie_sha256':'a'*64})
    assert path.name == 'visual_event_segments_v1.json'
    assert store['events'] == []
    assert json.loads(legacy.read_text())['segments']['N1']['draft_events'][0]['candidate_id'] == 'DRAFT_BRC_0001'


def test_production_fingerprint_beats_legacy_source_manifest(tmp_path):
    movie = tmp_path / "movie.mp4"; movie.write_bytes(b"current")
    run = tmp_path / "runs" / "film"; (run / "source-v1").mkdir(parents=True)
    (run / "source-v1" / "source_manifest.json").write_text(json.dumps({"source": {"movie": {"sha256": "a" * 64}}}))
    (run / "source_fingerprint.json").write_text(json.dumps({"movie_sha256": "b" * 64}))
    assert _source_movie_sha256(run, movie) == "b" * 64



def test_process_builds_provider_once_for_all_segments(monkeypatch, tmp_path):
    source = fixture(tmp_path)

    narrative = (
        tmp_path
        / "runs"
        / "film"
        / "narrative-v2"
        / "narrative_map.json"
    )
    narrative.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "segment_id": "N1",
                        "start_seconds": 0,
                        "end_seconds": 1,
                    },
                    {
                        "segment_id": "N2",
                        "start_seconds": 1,
                        "end_seconds": 2,
                    },
                ]
            }
        )
    )

    info = {
        "movie": source / "movie.mp4",
        "srt": source / "subtitles.srt",
        "run": tmp_path / "runs" / "film",
        "narrative": narrative,
        "movie_sha256": "a" * 64,
        "srt_sha256": "b" * 64,
        "metadata": {
            "video": {
                "fps": 24,
                "width": 160,
                "height": 120,
            },
            "duration_seconds": 2,
        },
    }

    monkeypatch.setattr(
        production,
        "preflight",
        lambda _: info,
    )
    monkeypatch.setattr(production, "_semantic_terminal_contract", lambda *_: (True, []))

    shots = [
        {
            "shot_id": "FULL_S_1",
            "start_seconds": 0,
            "end_seconds": 2,
            "start_frame": 0,
            "end_frame_exclusive": 48,
        }
    ]

    monkeypatch.setattr(
        production,
        "_technical_shots",
        lambda _: shots,
    )

    def visual_events(_info, _shots):
        return [
            {
                "candidate_id": "BRC_FULL",
                "visual_event_id": "VE_FULL",
                "start_seconds": 0,
                "end_seconds": 2,
                "start_frame": 0,
                "end_frame_exclusive": 24,
                "source_shot_ids": ["FULL_S_1"],
                "editorial": {
                    "decision": "KEEP",
                    "status": "VALIDATED",
                },
            }
        ]

    monkeypatch.setattr(
        production,
        "_visual_events",
        visual_events,
    )

    pool = object()
    build_calls = []

    def build(*args, **kwargs):
        build_calls.append((args, kwargs))
        return pool

    monkeypatch.setattr(
        production,
        "build_semantic_provider_from_env",
        build,
    )

    semantic_providers = []

    def semantics(items, *args, **kwargs):
        semantic_providers.append(kwargs["provider"])
        return {
            "status": "COMPLETE",
            "quota_exhausted": False,
            "provider_unavailable": False,
        }

    monkeypatch.setattr(
        production,
        "semantic_validate",
        semantics,
    )
    monkeypatch.setattr(
        production,
        "apply_semantic_scarcity",
        lambda _: None,
    )
    monkeypatch.setattr(
        production,
        "person_detector_preflight",
        lambda: {},
    )
    monkeypatch.setattr(
        production,
        "finalize_pilot",
        lambda *args, **kwargs: {
            "status": "COMPLETE",
            "completed": 0,
            "review": 0,
            "failed_final": 0,
        },
    )

    result = production.process(
        source,
        reporter=lambda _: None,
        policy_version="broll_policy_v1",
    )

    assert result["status"] == "COMPLETE"
    assert len(build_calls) == 1
    # One constructed Visual Event gets one final semantic request even though
    # it crosses N1/N2; there is no technical-shot prepass.
    assert semantic_providers == [pool]


def test_partial_quota_survives_production_with_resume_failure(
    monkeypatch,
    tmp_path,
):
    source = fixture(tmp_path)

    info = {
        "movie": source / "movie.mp4",
        "srt": source / "subtitles.srt",
        "run": tmp_path / "runs" / "film",
        "narrative": (
            tmp_path
            / "runs"
            / "film"
            / "narrative-v2"
            / "narrative_map.json"
        ),
        "movie_sha256": "a" * 64,
        "srt_sha256": "b" * 64,
        "metadata": {
            "video": {
                "fps": 24,
                "width": 160,
                "height": 120,
            },
            "duration_seconds": 1,
        },
    }

    monkeypatch.setattr(
        production,
        "preflight",
        lambda _: info,
    )

    shots = [
        {
            "shot_id": "FULL_S_1",
            "start_seconds": 0,
            "end_seconds": 1,
            "start_frame": 0,
            "end_frame_exclusive": 24,
        }
    ]

    event = {
        "candidate_id": "BRC_0009",
        "visual_event_id": "VE_QUOTA",
        "start_seconds": 0,
        "end_seconds": 1,
        "start_frame": 0,
        "end_frame_exclusive": 24,
        "source_shot_ids": ["FULL_S_1"],
        "editorial": {
            "decision": "REVIEW",
            "status": "SEMANTIC_INCOMPLETE",
        },
    }

    monkeypatch.setattr(
        production,
        "_technical_shots",
        lambda _: shots,
    )
    monkeypatch.setattr(
        production,
        "_visual_events",
        lambda *_: [event],
    )

    def semantics(*args, **kwargs):
        return {
            "status": "PARTIAL_QUOTA",
            "quota_exhausted": True,
            "provider_unavailable": False,
            "failure": {
                "failure_stage": "semantic",
                "failed_segment": "N1",
                "failed_event_id": "VE_QUOTA",
                "candidate_id": "BRC_0009",
                "provider": "gemini-primary-3",
                "model": "gemini-3.6-flash",
                "http_status": 429,
                "reason": "quota_exceeded",
                "retryable": True,
                "retry_after_seconds": 17.25,
                "checkpoint_saved": False,
                "ledger_saved": True,
                "resume_safe": True,
                "message": "quota exceeded",
            },
        }

    monkeypatch.setattr(
        production,
        "semantic_validate",
        semantics,
    )

    # Must never reach finalization after operational semantic stop.
    monkeypatch.setattr(
        production,
        "finalize_pilot",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("finalization must not run")
        ),
    )

    lines = []

    result = production.process(
        source,
        provider=object(),
        reporter=lines.append,
    )

    assert result["status"] == "PARTIAL_QUOTA"

    failure = result["failure"]

    assert failure["failed_segment"] == "PBATCH_0001"
    assert failure["segment_index"] == 1
    assert failure["segments_total"] == 1
    assert failure["failed_event_id"] == "VE_QUOTA"
    assert failure["provider"] == "gemini-primary-3"
    assert failure["http_status"] == 429
    assert failure["reason"] == "quota_exceeded"
    assert failure["retryable"] is True
    assert failure["retry_after_seconds"] == 17.25
    assert failure["resume_safe"] is True

    assert result["summary"]["status"] == "PARTIAL_QUOTA"
    assert result["summary"]["failure"]["reason"] == "quota_exceeded"

    assert any(
        line.startswith("[provider-error]")
        and "http_status=429" in line
        and "reason=quota_exceeded" in line
        and "event=VE_QUOTA" in line
        for line in lines
    )

    store = json.loads(
        (
            info["run"]
            / "visual_event_segments_v1.json"
        ).read_text()
    )

    assert store["event_discovery_status"] == "COMPLETE"
    assert store["production_status"] == "PARTIAL_QUOTA"
    assert store["status"] == "RUNNING"
    assert store["batches"]["PBATCH_0001"]["status"] == "PARTIAL"

    summary = json.loads(
        (
            info["run"]
            / "progress_summary.json"
        ).read_text()
    )

    assert summary["status"] == "PARTIAL_QUOTA"
    assert summary["failure"]["reason"] == "quota_exceeded"

    log = (
        info["run"]
        / "progress.jsonl"
    ).read_text()

    assert "PRODUCTION_PARTIAL_QUOTA" in log



def test_failed_retryable_counts_as_remaining_work(tmp_path):
    from movie_broll.processing_ledger import ProcessingLedger

    run = tmp_path / "runs" / "film"
    run.mkdir(parents=True)

    ledger = ProcessingLedger(
        run,
        "film",
        {},
    )

    ledger.data["events"]["VE_RETRY"] = {
        "visual_event_id": "VE_RETRY",
        "stages": {
            "semantic": {
                "status": "FAILED_RETRYABLE",
            }
        },
    }

    info = {
        "run": run,
        "movie_sha256": "a" * 64,
        "srt_sha256": "b" * 64,
    }

    summary = production._summary(
        info,
        ledger,
        [],
        "PARTIAL_QUOTA",
    )

    assert summary["semantic"]["retryable"] == 1
    assert summary["semantic"]["remaining"] == 1


def test_event_granular_production_finalizes_before_next_provider_call_and_resumes(monkeypatch, tmp_path):
    """Each global Visual Event is finalized before the next semantic request."""
    source = fixture(tmp_path)
    narrative = tmp_path / "runs" / "film" / "narrative-v2" / "narrative_map.json"
    narrative.write_text(json.dumps({"segments": [{"segment_id": "N1", "start_seconds": 0, "end_seconds": 80}]}))
    info = {"movie": source / "movie.mp4", "srt": source / "subtitles.srt", "run": tmp_path / "runs" / "film",
            "narrative": narrative, "movie_sha256": "a" * 64, "srt_sha256": "b" * 64,
            "metadata": {"video": {"fps": 24, "width": 160, "height": 120}, "duration_seconds": 80}}
    monkeypatch.setattr(production, "preflight", lambda _: info)
    monkeypatch.setattr(production, "_semantic_terminal_contract", lambda *_: (True, []))
    assert production.SEMANTIC_PRODUCTION_BATCH_SIZE == 1
    shots = [{"shot_id": f"S{i}", "start_seconds": i * 20, "end_seconds": i * 20 + 20,
              "start_frame": i * 480, "end_frame_exclusive": i * 480 + 480} for i in range(4)]
    monkeypatch.setattr(production, "_technical_shots", lambda _: shots)
    events = [{"candidate_id": f"BRC_{i + 1:04d}", "visual_event_id": f"VE_{i + 1:04d}",
               "start_seconds": i * 20, "end_seconds": i * 20 + 8,
               "start_frame": i * 480, "end_frame_exclusive": i * 480 + 192,
               "source_shot_ids": [f"S{i}"], "narrative_segment_ids": ["N1"],
               "score": {"total": 1000 - i}, "editorial": {"decision": "REVIEW", "status": "PROVISIONAL"}}
              for i in range(4)]
    monkeypatch.setattr(production, "_visual_events", lambda *_: [dict(event) for event in events])
    monkeypatch.setattr(production, "_bounded_operation", lambda _label, action, _report, _ledger: action())
    monkeypatch.setattr(production, "person_detector_preflight", lambda: {"loaded": True})

    phase = {"partial": True}
    semantic_calls, provider_calls, rendered, scarcity_inputs = [], [], [], []
    def semantics(items, *_args, **_kwargs):
        semantic_calls.append([item["candidate_id"] for item in items])
        assert len(items) == 1
        item = items[0]
        candidate_id = item["candidate_id"]
        if candidate_id == "BRC_0002":
            assert (info["run"] / "assets" / "VE_0001.mp4").is_file()
        if candidate_id == "BRC_0004" and phase["partial"]:
            assert (info["run"] / "assets" / "VE_0003.mp4").is_file()
            return {"status": "PARTIAL_PROVIDER", "provider_unavailable": True,
                    "failure": {"failed_event_id": "VE_0004", "candidate_id": "BRC_0004", "retryable": True,
                                "provider": "fake", "model": "fake", "reason": "provider_unavailable"}}
        provider_calls.append(candidate_id)
        keep = candidate_id in {"BRC_0001", "BRC_0003"}
        item["editorial"] = {"decision": "KEEP" if keep else "REJECT", "status": "VALIDATED"}
        item["visual"] = {"setting": candidate_id, "actions": ["accion"]}
        checkpoint_dir = _args[3]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / f"{candidate_id}.json").write_text(candidate_id)
        return {"status": "COMPLETE", "quota_exhausted": False, "provider_unavailable": False}
    monkeypatch.setattr(production, "semantic_validate", semantics)
    real_scarcity = production.apply_semantic_scarcity
    def scarcity(items):
        scarcity_inputs.append([item["candidate_id"] for item in items])
        real_scarcity(items)
    monkeypatch.setattr(production, "apply_semantic_scarcity", scarcity)

    def finalize(_input, _batch, *, candidates, **_kwargs):
        (info["run"] / "assets").mkdir(parents=True, exist_ok=True)
        completed = 0
        for event in candidates:
            if event.get("editorial", {}).get("decision") != "KEEP":
                continue
            package = info["run"] / "assets" / event["visual_event_id"]
            if package.with_suffix(".mp4").exists():
                continue
            for suffix in (".mp4", ".vertical.mp4", ".jpg", ".vertical.jpg", ".json"):
                package.with_suffix(suffix).write_text("package")
            rendered.append(event["visual_event_id"]); completed += 1
        return {"status": "COMPLETE", "completed": completed, "review": 0, "failed_final": 0}
    monkeypatch.setattr(production, "finalize_pilot", finalize)

    first = production.process(source, provider=object())
    assert first["status"] == "PARTIAL_PROVIDER"
    assert semantic_calls == [[f"BRC_{i:04d}"] for i in range(1, 5)]
    assert rendered == ["VE_0001", "VE_0003"]
    assert scarcity_inputs == [["BRC_0001"], ["BRC_0001", "BRC_0003"]]
    store = json.loads((info["run"] / "visual_event_segments_v1.json").read_text())
    assert store["event_discovery_status"] == "COMPLETE"
    assert store["batches"]["PBATCH_0001"]["status"] == "COMPLETE"
    assert store["batches"]["PBATCH_0004"]["status"] == "PARTIAL"
    assert store["production_queue"] == [f"VE_{i:04d}" for i in range(1, 5)]
    assert store["events"][3]["narrative_segment_ids"] == ["N1"]  # discovery remains global
    accepted = [event for event in store["events"] if event["candidate_id"] in {"BRC_0001", "BRC_0003"}]
    globally_scarce = deepcopy(accepted)
    real_scarcity(globally_scarce)
    assert [(event["candidate_id"], event["editorial"]["decision"], event["semantic_redundancy"]["status"])
            for event in accepted] == [
        (event["candidate_id"], event["editorial"]["decision"], event["semantic_redundancy"]["status"])
        for event in globally_scarce
    ]
    assert len(list((info["run"] / "assets").glob("*"))) == 10  # two canonical 5-file packages
    checkpoints = info["run"] / "semantic_checkpoints"
    assert {path.name for path in checkpoints.glob("*.json")} == {"BRC_0001.json", "BRC_0002.json", "BRC_0003.json"}
    preserved_checkpoints = {path.name: path.read_text() for path in checkpoints.glob("*.json")}

    phase["partial"] = False
    second = production.process(source, provider=object())
    assert second["status"] == "COMPLETE"
    assert rendered == ["VE_0001", "VE_0003"]  # completed packages were never rendered again
    assert provider_calls == ["BRC_0001", "BRC_0002", "BRC_0003", "BRC_0004"]
    assert {path.name: path.read_text() for path in checkpoints.glob("*.json") if path.name != "BRC_0004.json"} == preserved_checkpoints
    assert [event["visual_event_id"] for event in json.loads((info["run"] / "visual_events.json").read_text())["events"]] == [f"VE_{i:04d}" for i in range(1, 5)]
    assert {len(list((info["run"] / "assets").glob(f"{event_id}*"))) for event_id in rendered} == {5}


def test_batch_size_migration_rebuilds_only_metadata_and_keeps_global_events(tmp_path):
    run = tmp_path / "runs" / "film"
    checkpoints = run / "semantic_checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "BRC_0001.json"
    checkpoint.write_text('{"preserved": true}')
    events = [
        {"candidate_id": f"BRC_{index:04d}", "visual_event_id": f"VE_{index:04d}",
         "start_seconds": index, "end_seconds": index + 1, "score": {"total": 10 - index},
         "editorial": {"decision": "KEEP", "status": "VALIDATED"}}
        for index in range(1, 5)
    ]
    queue = production._production_queue(events)
    store = {
        "events": events,
        "production_queue": queue,
        "production_batch_size": 8,
        "batches": {"PBATCH_0001": {"event_ids": queue, "status": "PARTIAL"}},
    }
    original_events = store["events"]

    production._ensure_production_batches(store, events)

    assert store["events"] is original_events
    assert store["production_queue"] == queue
    assert store["production_batch_size"] == 1
    assert [batch["event_ids"] for batch in store["batches"].values()] == [[event_id] for event_id in queue]
    assert all(batch["status"] == "PENDING" for batch in store["batches"].values())
    assert checkpoint.read_text() == '{"preserved": true}'


def test_timeline_ordinals_are_chronological_while_queue_remains_quality_first():
    events=[
        {'visual_event_id':f'VE_{index}','candidate_id':f'BRC_{index}',
         'start_seconds':float(index),'end_seconds':float(index+1),
         'score':{'total':score}}
        for index,score in ((1,10),(2,20),(3,40),(4,30),(5,50))
    ]
    assert production._assign_timeline_ordinals(events)
    assert [event['timeline_ordinal'] for event in events] == [1,2,3,4,5]
    assert production._production_queue(events) == ['VE_5','VE_3','VE_4','VE_2','VE_1']
    assert not production._assign_timeline_ordinals(events)
