import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from movie_broll.broll_semantics import PROMPT, SemanticResponse
from movie_broll.active_picture import VERSION, load_cached_active_picture, load_or_detect
from movie_broll.processing_ledger import fingerprint
from movie_broll.utils import sha256_file
from movie_broll.semantic_v9_benchmark import (
    BenchmarkSystemicFailure,
    DIALOGUE_NEGATIVE_IDS,
    FALSE_NEGATIVE_IDS,
    HARD_NEGATIVE_IDS,
    _protected_signature,
    run_semantic_v9_benchmark,
    select_benchmark_events,
)


def _event(event_id, ordinal, decision="REJECT", actions=None):
    return {
        "visual_event_id": event_id, "candidate_id": f"BRC_{ordinal:04d}",
        "timeline_ordinal": ordinal, "start_frame": ordinal * 10,
        "end_frame_exclusive": ordinal * 10 + 9, "start_seconds": float(ordinal),
        "end_seconds": float(ordinal + 1), "source_shot_ids": [], "technical_shots": [],
        "visual": {"actions": actions or ["acción visible"], "visible_interactions": [],
                   "visible_emotions": []},
        "editorial": {"decision": decision, "standalone_meaning_es": "estado visible"},
    }


def _benchmark_events():
    controls = [
        _event(event_id, ordinal) for ordinal, event_id in enumerate(
            (*FALSE_NEGATIVE_IDS, *HARD_NEGATIVE_IDS, *DIALOGUE_NEGATIVE_IDS), 1
        )
    ]
    keeps = [
        _event("VE_KEEP_MOVE", 101, "KEEP", ["walking"]),
        _event("VE_KEEP_OBJECT", 102, "KEEP", ["applying makeup"]),
        _event("VE_KEEP_INTERACTION", 103, "KEEP", ["touching shoulders"]),
        _event("VE_KEEP_REACTION", 104, "KEEP", ["observa con tristeza"]),
        _event("VE_KEEP_STATE", 105, "KEEP", ["resting head on table"]),
        _event("VE_KEEP_ACTION", 106, "KEEP", ["carrying a tray with food"]),
    ]
    return controls + keeps


def _valid_response():
    return {
        "visual": {"summary_es": "Persona caminando", "subjects": ["persona"], "objects": [],
                   "actions": ["caminar"], "people_count_estimate": "1", "setting": "exterior",
                   "visible_interactions": [], "visible_emotions": ["neutral"], "people": [],
                   "primary_subject_position": "center", "primary_subject_description": "persona",
                   "visual_focus": "persona caminando", "shot_focus_plan": []},
        "relationships": [],
        "editorial": {"standalone_meaning_es": "Una persona camina.", "reusable_broll": True,
                      "action_or_moment_complete": "true", "use_cases_es": ["persona caminando"],
                      "negative_use_cases_es": [], "search_terms_es": ["caminar"],
                      "editorial_confidence": "high",
                      "keep_qualification": {"action_evidence_type": "generic_presence_or_movement",
                                               "action_evidence_es": "caminar",
                                               "reusable_intent_type": "generic_ambience_or_context",
                                               "reusable_use_case_es": "persona caminando"},
                      "reason": "Actividad visible, clara y buscable.", "decision": "KEEP"},
    }


def test_v9_policy_explicitly_allows_ordinary_searchable_actions_and_keeps_hard_rejects():
    policy = PROMPT.casefold()
    assert "does not mean distinctive" in policy
    assert "generic enum automatically means reject" in policy
    assert "walking or walking away" in policy and "writing/reading/working" in policy
    assert "credits, logos, title cards" in policy
    assert "generic alternating dialogue" in policy


def test_benchmark_selection_is_24_events_with_dynamic_diverse_keep_controls():
    selected = select_benchmark_events(_benchmark_events())
    assert len(selected) == 24
    assert len({event["visual_event_id"] for event in selected}) == 24
    assert all(event["editorial"]["decision"] == "KEEP" for event in selected[-6:])
    assert {event["_benchmark_category"] for event in selected[-6:]} == {
        "positive_control:movement", "positive_control:object_activity", "positive_control:interaction",
        "positive_control:reaction", "positive_control:emotional_state", "positive_control:action",
    }


def test_benchmark_only_writes_its_own_output_and_never_creates_checkpoint_or_asset(tmp_path, monkeypatch):
    input_dir = tmp_path / "input" / "e02"
    run = tmp_path / "runs" / "e02"
    input_dir.mkdir(parents=True); run.mkdir(parents=True)
    (input_dir / "movie.mp4").write_bytes(b"not decoded in this test")
    (input_dir / "subtitles.srt").write_text("", encoding="utf-8")
    (run / "narrative-v2").mkdir()
    (run / "narrative-v2" / "narrative_map.json").write_text(json.dumps({"segments": []}))
    (run / "visual_event_segments_v1.json").write_text(json.dumps({"events": _benchmark_events()}))
    (run / "active_picture.json").write_text(json.dumps({
        "schema_version": "active_picture_v1", "fingerprint": "test",
        "active_picture": {"x": 0, "y": 0, "width": 1920, "height": 1080,
                           "source_width": 1920, "source_height": 1080},
    }))
    (run / "semantic_checkpoints").mkdir()
    (run / "semantic_checkpoints" / "BRC_0001.json").write_text("{}")
    (run / "assets").mkdir(); (run / "assets" / "existing.mp4").write_bytes(b"asset")
    before = _protected_signature(run)

    import movie_broll.semantic_v9_benchmark as benchmark
    monkeypatch.setattr(benchmark, "parse_srt_file", lambda path: SimpleNamespace(cues=[]))
    monkeypatch.setattr("movie_broll.inspect_source.inspect_movie", lambda path: {"video": {"fps": 24}})

    class Provider:
        identifier = "openai"; model = "gpt-6-luna"; reasoning_effort = "none"; image_detail = "low"
        def generate(self, prompt, context, jpeg):
            return SemanticResponse(_valid_response(), {"prompt_tokens": 10, "cached_tokens": 0,
                                                        "response_tokens": 5, "thinking_tokens": 0,
                                                        "total_tokens": 15}, provider="openai", model=self.model)

    report = run_semantic_v9_benchmark(
        input_dir, provider=Provider(), run_id="test-run",
        contact_sheet=lambda movie, event, fps, evidence, active_picture: b"jpeg",
    )
    assert report["read_only_verified"] is True
    assert report["provider_requests"] == report["semantic_results"] == 24
    assert report["locally_valid_results"] == 24
    assert _protected_signature(run) == before
    assert (run / "benchmarks" / "semantic-v9" / "test-run" / "summary.json").is_file()
    assert len(list((run / "benchmarks" / "semantic-v9" / "test-run" / "events").glob("*.json"))) == 24
    assert len(list((run / "semantic_checkpoints").glob("*.json"))) == 1
    assert (run / "assets" / "existing.mp4").read_bytes() == b"asset"


def test_benchmark_uses_normalized_active_picture_and_canonical_request_context(tmp_path, monkeypatch):
    input_dir = tmp_path / "input" / "e02"; run = tmp_path / "runs" / "e02"
    input_dir.mkdir(parents=True); run.mkdir(parents=True)
    (input_dir / "movie.mp4").write_bytes(b"movie"); (input_dir / "subtitles.srt").write_text("")
    (run / "narrative-v2").mkdir(); (run / "narrative-v2" / "narrative_map.json").write_text(json.dumps({"segments": []}))
    (run / "visual_event_segments_v1.json").write_text(json.dumps({"events": _benchmark_events()}))
    geometry = {"x": 7, "y": 9, "width": 1900, "height": 1000, "source_width": 1920, "source_height": 1080}
    (run / "active_picture.json").write_text(json.dumps({"schema_version": "active_picture_v1", "fingerprint": "cache", "active_picture": geometry}))
    import movie_broll.semantic_v9_benchmark as benchmark
    monkeypatch.setattr(benchmark, "parse_srt_file", lambda path: SimpleNamespace(cues=[]))
    monkeypatch.setattr("movie_broll.inspect_source.inspect_movie", lambda path: {"video": {"fps": 24}})
    seen = []
    def sheet(movie, event, fps, evidence, active_picture):
        seen.append(active_picture)
        evidence.update({"technical_shots": []})
        return b"jpeg"
    report = run_semantic_v9_benchmark(input_dir, run_id="dry", contact_sheet=sheet, dry_run=True)
    assert report["selected_events"] == 24 and report["provider_requests"] == 0
    assert report["failed_events"] == 0 and len(seen) == 24
    assert report["stage_counts"]["input_preparation"] == report["stage_counts"]["contact_sheet"] == 24
    assert all(active == geometry for active in seen)


def test_benchmark_active_picture_loader_matches_production_cache_normalization(tmp_path):
    movie = tmp_path / "movie.mp4"; movie.write_bytes(b"canonical source")
    run = tmp_path / "run"; run.mkdir()
    profile = {"enabled": True}
    geometry = {"x": 0, "y": 60, "width": 1920, "height": 960,
                "source_width": 1920, "source_height": 1080}
    source_sha = sha256_file(movie)
    cache_key = fingerprint({"version": VERSION, "source_movie_sha256": source_sha, "settings": {}})
    cache = run / "active_picture.json"
    cache.write_text(json.dumps({"schema_version": VERSION, "fingerprint": cache_key,
                                 "source_movie_sha256": source_sha, "active_picture": geometry}))
    assert load_cached_active_picture(cache) == geometry
    assert load_or_detect(movie, run, profile) == geometry


def test_preprovider_systemic_failure_is_persisted_and_cannot_look_successful(tmp_path, monkeypatch):
    input_dir = tmp_path / "input" / "e02"; run = tmp_path / "runs" / "e02"
    input_dir.mkdir(parents=True); run.mkdir(parents=True)
    (input_dir / "movie.mp4").write_bytes(b"movie"); (input_dir / "subtitles.srt").write_text("")
    (run / "narrative-v2").mkdir(); (run / "narrative-v2" / "narrative_map.json").write_text(json.dumps({"segments": []}))
    (run / "visual_event_segments_v1.json").write_text(json.dumps({"events": _benchmark_events()}))
    (run / "active_picture.json").write_text(json.dumps({"active_picture": {"x": 0, "y": 0, "width": 1, "height": 1}}))
    import movie_broll.semantic_v9_benchmark as benchmark
    monkeypatch.setattr(benchmark, "parse_srt_file", lambda path: SimpleNamespace(cues=[]))
    monkeypatch.setattr("movie_broll.inspect_source.inspect_movie", lambda path: {"video": {"fps": 24}})
    class Provider:
        identifier = "openai"; model = "gpt-6-luna"; reasoning_effort = "none"
        def generate(self, *args): raise AssertionError("must not reach provider")
    with pytest.raises(BenchmarkSystemicFailure) as raised:
        run_semantic_v9_benchmark(input_dir, provider=Provider(), run_id="failed",
                                  contact_sheet=lambda *args: (_ for _ in ()).throw(KeyError("x")))
    report = raised.value.report
    assert report["provider_requests"] == report["semantic_results"] == 0
    assert report["counts"] == {} and report["dominant_failure"] == "contact_sheet / KeyError: 'x'"
    event = json.loads(next((report["output"] / "events").glob("*.json")).read_text())
    assert event["failure_stage"] == "contact_sheet"
    assert event["error_type"] == "KeyError" and event["error_message"] == "'x'"


def test_cli_reports_systemic_preprovider_failure_without_calibration_counts(tmp_path, monkeypatch, capsys):
    import movie_broll.semantic_v9_benchmark as benchmark
    from movie_broll.cli import main
    report = {"provider_requests": 0, "semantic_results": 0,
              "dominant_failure": "contact_sheet / KeyError: 'x'", "output": tmp_path}
    monkeypatch.setattr(benchmark, "run_semantic_v9_benchmark",
                        lambda *args, **kwargs: (_ for _ in ()).throw(BenchmarkSystemicFailure(report)))
    assert main(["benchmark", "semantic-v9", "input/e02"]) == 2
    captured = capsys.readouterr()
    assert "BENCHMARK FAILED" in captured.err and "provider requests: 0" in captured.err
    assert "KEEP:" not in captured.out and "REJECT:" not in captured.out
