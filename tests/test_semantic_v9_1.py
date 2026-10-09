import copy
import json

import pytest

from movie_broll.broll_semantics import SemanticResponse
from movie_broll.semantic_reclassification import (
    V9_1_SPEC, promote_reclassification_v9_1, run_reclassification_v9_1, workspace_path,
)
from movie_broll.semantic_v9_1 import (
    PROMPT_V9_1, SEMANTIC_CONTRACT_V9_1, SEMANTIC_PROMPT_V9_1,
    effective_decision, validate_response_v9_1,
)
from movie_broll.semantic_v9_1_benchmark import (
    CONTROLS, run_semantic_v9_1_benchmark, select_v9_1_benchmark_events,
)
from movie_broll.utils import write_json


def response(kind="concrete_action", decision="KEEP", *, action="caminar", reusable=True):
    return {
        "visual": {"summary_es": "Acción visible", "subjects": ["persona"], "objects": [],
                   "actions": [action], "people_count_estimate": "1", "setting": "interior",
                   "visible_interactions": [], "visible_emotions": ["neutral"], "people": [],
                   "primary_subject_position": "center", "primary_subject_description": "persona",
                   "visual_focus": "acción", "shot_focus_plan": []},
        "relationships": [],
        "editorial": {"standalone_meaning_es": "Acción reutilizable.", "reusable_broll": reusable,
                      "action_or_moment_complete": "true", "use_cases_es": [action],
                      "negative_use_cases_es": [], "search_terms_es": [action], "editorial_confidence": "high",
                      "keep_qualification": {"action_evidence_type": "distinct_visible_action_or_reaction",
                                               "action_evidence_es": action,
                                               "reusable_intent_type": "specific_visual_need",
                                               "reusable_use_case_es": action},
                      "visual_utility_kind": kind, "conversation_visual_signal": "none",
                      "reason": "El objetivo visual está presente.", "decision": decision},
    }


@pytest.mark.parametrize("kind,action", [
    ("concrete_action", "caminar"), ("concrete_action", "escribir"),
    ("object_activity", "usar teléfono"), ("useful_state", "descansar en cama"),
    ("clear_reaction", "escuchar con reacción"), ("physical_interaction", "tomar la mano"),
    ("strong_nonverbal_interaction", "consolar"),
])
def test_v9_1_grounded_reusable_utility_is_keep_eligible(kind, action):
    assert validate_response_v9_1(response(kind, action=action)) == []


@pytest.mark.parametrize("kind", ["generic_dialogue_only", "generic_presence_only", "unclear"])
def test_v9_1_generic_dialogue_or_presence_cannot_be_effective_keep(kind):
    data = response(kind, action="hablar")
    errors = validate_response_v9_1(data)
    assert "KEEP has ineligible visual utility kind" in errors
    assert effective_decision(data, {"valid": not errors, "errors": errors}) == "REVIEW"


def test_v9_1_identity_and_policy_are_distinct_from_v9():
    assert SEMANTIC_CONTRACT_V9_1 == "semantic_contract_v9_1"
    assert SEMANTIC_PROMPT_V9_1 == "semantic_prompt_v9_1"
    assert "generic dialogue alone" in PROMPT_V9_1.casefold()
    assert "title cards, credits, logos" in PROMPT_V9_1.casefold()


def test_v9_1_hard_negative_model_rejection_is_preserved():
    data = response("unclear", decision="REJECT", action="pantalla negra", reusable=False)
    assert validate_response_v9_1(data) == []
    assert effective_decision(data, {"valid": True, "errors": []}) == "REJECT"


def _event(ordinal):
    return {"visual_event_id": f"VE_{ordinal}", "candidate_id": f"BRC_{ordinal:04d}",
            "timeline_ordinal": ordinal, "start_frame": ordinal * 10,
            "end_frame_exclusive": ordinal * 10 + 9, "start_seconds": float(ordinal),
            "end_seconds": float(ordinal + 1), "source_shot_ids": [], "technical_shots": [],
            "visual": {}, "editorial": {"decision": "REJECT", "status": "VALIDATED"}}


def _fixture(tmp_path, ordinals=(1, 2)):
    source = tmp_path / "input" / "e02"; run = tmp_path / "runs" / "e02"
    source.mkdir(parents=True); run.mkdir(parents=True)
    (source / "movie.mp4").write_bytes(b"movie"); (source / "subtitles.srt").write_text("")
    write_json(run / "visual_event_segments_v1.json", {"events": [_event(n) for n in ordinals], "batches": {}})
    write_json(run / "narrative-v2" / "narrative_map.json", {"segments": []})
    write_json(run / "active_picture.json", {"active_picture": {"x": 0, "y": 0, "width": 100, "height": 100, "source_width": 100, "source_height": 100}})
    return source, run


def test_v9_1_selection_is_thirty_controls_in_required_categories():
    events = [_event(ordinal) for _category, ordinal, _label in CONTROLS]
    selected = select_v9_1_benchmark_events(events)
    assert len(selected) == 30
    assert {x["_benchmark_category"] for x in selected} == {
        "ordinary_action", "generic_dialogue_or_presence", "borderline_reaction_interaction", "hard_negative"}
    assert {x["timeline_ordinal"] for x in selected if x["_benchmark_category"] == "generic_dialogue_or_presence"} == {13, 54, 58, 81, 87, 91, 101, 106, 127, 141}


def test_v9_1_workspace_is_separate_resumable_and_promotion_refuses_partial(monkeypatch, tmp_path):
    source, run = _fixture(tmp_path)
    before = (run / "visual_event_segments_v1.json").read_bytes()
    historical_v9 = workspace_path(source)
    historical_v9.mkdir(parents=True)
    (historical_v9 / "historical-sentinel.json").write_text('{"state":"NOT_PROMOTED"}')
    import movie_broll.semantic_reclassification as module
    monkeypatch.setattr(module, "inspect_movie", lambda _: {"video": {"fps": 24}})
    monkeypatch.setattr(module, "parse_srt_file", lambda _: type("Srt", (), {"cues": []})())
    monkeypatch.setattr(module, "load_cached_active_picture", lambda _: {"x": 0, "y": 0, "width": 100, "height": 100, "source_width": 100, "source_height": 100})
    class Provider:
        identifier = "openai"; model = "gpt-6-luna"; reasoning_effort = "none"; calls = 0
        def generate(self, *_):
            self.calls += 1
            return SemanticResponse(response(), {"prompt_tokens": 1, "cached_tokens": 0, "response_tokens": 1, "thinking_tokens": 0, "total_tokens": 2}, "openai", self.model)
    provider = Provider()
    result = run_reclassification_v9_1(source, provider=provider, contact_sheet=lambda *_: b"jpg", max_events=1)
    assert result["complete"] is False and provider.calls == 1
    assert workspace_path(source, V9_1_SPEC).name == "semantic-v9.1"
    assert (historical_v9 / "historical-sentinel.json").read_text() == '{"state":"NOT_PROMOTED"}'
    assert (run / "visual_event_segments_v1.json").read_bytes() == before
    with pytest.raises(RuntimeError, match="incomplete"):
        promote_reclassification_v9_1(source, finalize=False)
    run_reclassification_v9_1(source, provider=provider, contact_sheet=lambda *_: b"jpg")
    assert provider.calls == 2
    assert run_reclassification_v9_1(source, provider=provider, contact_sheet=lambda *_: b"jpg")["reused"] == 2


def test_v9_1_invalid_model_keep_is_auditable_but_effectively_review(monkeypatch, tmp_path):
    source, run = _fixture(tmp_path, (1,))
    import movie_broll.semantic_reclassification as module
    monkeypatch.setattr(module, "inspect_movie", lambda _: {"video": {"fps": 24}})
    monkeypatch.setattr(module, "parse_srt_file", lambda _: type("Srt", (), {"cues": []})())
    monkeypatch.setattr(module, "load_cached_active_picture", lambda _: {"x": 0, "y": 0, "width": 100, "height": 100, "source_width": 100, "source_height": 100})
    class Provider:
        identifier = "openai"; model = "gpt-6-luna"; reasoning_effort = "none"
        def generate(self, *_):
            return SemanticResponse(response("generic_dialogue_only", action="hablar"), {"prompt_tokens": 1, "cached_tokens": 0, "response_tokens": 1, "thinking_tokens": 0, "total_tokens": 2}, "openai", self.model)
    run_reclassification_v9_1(source, provider=Provider(), contact_sheet=lambda *_: b"jpg")
    record = json.loads(next((workspace_path(source, V9_1_SPEC) / "events").glob("*.json")).read_text())
    assert record["model_decision"] == "KEEP" and record["effective_decision"] == "REVIEW"
    assert record["local_validation"]["valid"] is False


def test_v9_1_benchmark_dry_run_is_read_only_and_prepares_all_controls(monkeypatch, tmp_path):
    ordinals = tuple(ordinal for _category, ordinal, _label in CONTROLS)
    source, run = _fixture(tmp_path, ordinals)
    before = (run / "visual_event_segments_v1.json").read_bytes()
    import movie_broll.semantic_v9_benchmark as shared
    monkeypatch.setattr(shared, "parse_srt_file", lambda _: type("Srt", (), {"cues": []})())
    monkeypatch.setattr("movie_broll.inspect_source.inspect_movie", lambda _: {"video": {"fps": 24}})
    report = run_semantic_v9_1_benchmark(source, run_id="dry", dry_run=True, contact_sheet=lambda *_: b"jpg")
    assert report["selected_events"] == 30 and report["provider_requests"] == 0
    assert report["stage_counts"]["input_preparation"] == report["stage_counts"]["contact_sheet"] == 30
    assert report["read_only_verified"] is True
    assert (run / "visual_event_segments_v1.json").read_bytes() == before


def test_v9_1_benchmark_counts_effective_not_invalid_model_keep(monkeypatch, tmp_path):
    ordinals = tuple(ordinal for _category, ordinal, _label in CONTROLS)
    source, run = _fixture(tmp_path, ordinals)
    import movie_broll.semantic_v9_benchmark as shared
    monkeypatch.setattr(shared, "parse_srt_file", lambda _: type("Srt", (), {"cues": []})())
    monkeypatch.setattr("movie_broll.inspect_source.inspect_movie", lambda _: {"video": {"fps": 24}})
    class Provider:
        identifier = "openai"; model = "gpt-6-luna"; reasoning_effort = "none"
        def generate(self, *_):
            return SemanticResponse(response("generic_dialogue_only", action="hablar"),
                                    {"prompt_tokens": 1, "cached_tokens": 0, "response_tokens": 1, "thinking_tokens": 0, "total_tokens": 2},
                                    "openai", self.model)
    report = run_semantic_v9_1_benchmark(source, provider=Provider(), run_id="effective", contact_sheet=lambda *_: b"jpg")
    assert report["counts"] == {"KEEP": 0, "REJECT": 0, "REVIEW": 30}
    first = json.loads(next((run / "benchmarks" / "semantic-v9.1" / "effective" / "events").glob("*.json")).read_text())
    assert first["model_decision"] == "KEEP" and first["effective_decision"] == "REVIEW"
