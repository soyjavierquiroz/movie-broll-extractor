import copy
import json

from movie_broll.broll_semantics import SemanticResponse
from movie_broll.semantic_reclassification import (
    promote_reclassification,
    reclassification_report,
    run_reclassification,
    workspace_path,
)
from movie_broll.utils import write_json


def _response(decision):
    return {
        "visual": {"summary_es": "Persona camina", "subjects": ["persona"], "objects": [],
                   "actions": ["caminar"], "people_count_estimate": "1", "setting": "calle",
                   "visible_interactions": [], "visible_emotions": ["neutral"], "people": [],
                   "primary_subject_position": "center", "primary_subject_description": "persona",
                   "visual_focus": "persona", "shot_focus_plan": []},
        "relationships": [],
        "editorial": {"standalone_meaning_es": "Una persona camina.", "reusable_broll": decision == "KEEP",
                      "action_or_moment_complete": "true", "use_cases_es": ["caminar"],
                      "negative_use_cases_es": [], "search_terms_es": ["caminar"],
                      "editorial_confidence": "high",
                      "keep_qualification": {"action_evidence_type": "generic_presence_or_movement",
                                               "action_evidence_es": "caminar",
                                               "reusable_intent_type": "generic_ambience_or_context",
                                               "reusable_use_case_es": "caminar"},
                      "reason": "usable", "decision": decision},
    }


def _event(number, decision):
    return {"visual_event_id": f"VE_{number}", "candidate_id": f"BRC_{number:04d}",
            "timeline_ordinal": number, "start_frame": number * 10,
            "end_frame_exclusive": number * 10 + 9, "start_seconds": float(number),
            "end_seconds": float(number + 1), "source_shot_ids": [], "technical_shots": [],
            "visual": {}, "editorial": {"decision": decision, "status": "VALIDATED"}}


def _fixture(tmp_path, events):
    source = tmp_path / "input" / "film"; run = tmp_path / "runs" / "film"
    source.mkdir(parents=True); run.mkdir(parents=True)
    (source / "movie.mp4").write_bytes(b"movie")
    (source / "subtitles.srt").write_text("")
    write_json(run / "visual_event_segments_v1.json", {"events": events, "batches": {}})
    write_json(run / "narrative-v2" / "narrative_map.json", {"segments": []})
    write_json(run / "active_picture.json", {"active_picture": {"x": 0, "y": 0, "width": 100, "height": 100,
                                                                  "source_width": 100, "source_height": 100}})
    return source, run


def _patch_runtime(monkeypatch):
    import movie_broll.semantic_reclassification as module
    monkeypatch.setattr(module, "inspect_movie", lambda _: {"video": {"fps": 24}})
    monkeypatch.setattr(module, "parse_srt_file", lambda _: type("Srt", (), {"cues": []})())
    monkeypatch.setattr(module, "load_cached_active_picture", lambda _: {"x": 0, "y": 0, "width": 100, "height": 100,
                                                                            "source_width": 100, "source_height": 100})


class Provider:
    identifier = "openai"
    model = "gpt-6-luna"
    reasoning_effort = "none"

    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls = 0

    def generate(self, *_args):
        self.calls += 1
        return SemanticResponse(_response(next(self.decisions)), {"prompt_tokens": 2, "cached_tokens": 0,
            "response_tokens": 1, "thinking_tokens": 0, "total_tokens": 3}, provider="openai", model=self.model)


def test_v9_sidecar_resume_is_idempotent_and_never_mutates_canonical(monkeypatch, tmp_path):
    source, run = _fixture(tmp_path, [_event(1, "REJECT"), _event(2, "KEEP")])
    before = (run / "visual_event_segments_v1.json").read_bytes()
    _patch_runtime(monkeypatch)
    provider = Provider(["KEEP", "REJECT"])
    first = run_reclassification(source, provider=provider, contact_sheet=lambda *_: b"jpeg", max_events=1)
    assert first["complete"] is False and provider.calls == 1
    assert (run / "visual_event_segments_v1.json").read_bytes() == before
    second = run_reclassification(source, provider=provider, contact_sheet=lambda *_: b"jpeg")
    assert second["complete"] is True and provider.calls == 2
    again = run_reclassification(source, provider=provider, contact_sheet=lambda *_: b"jpeg")
    assert again["reused"] == 2 and provider.calls == 2
    assert reclassification_report(source)["totals"]["requests"] == 2
    assert len(list((workspace_path(source) / "events").glob("*.json"))) == 2


def test_promotion_archives_v8_keep_reject_and_preserves_v8_history(monkeypatch, tmp_path):
    source, run = _fixture(tmp_path, [_event(1, "KEEP"), _event(2, "REJECT")])
    _patch_runtime(monkeypatch)
    provider = Provider(["REJECT", "KEEP"])
    run_reclassification(source, provider=provider, contact_sheet=lambda *_: b"jpeg")
    registry = {"events": {"VE_1": {"asset_id": "mf001", "slug": "old"}}}
    write_json(run / "asset_registry.json", registry)
    assets = run / "assets"; assets.mkdir()
    for name in ("mf001-old.mp4", "vmf001-old.mp4", "mf001-old.jpg", "vmf001-old.jpg", "mf001-old.json"):
        (assets / name).write_bytes(b"old")
    checkpoints = run / "semantic_checkpoints"; checkpoints.mkdir()
    (checkpoints / "legacy.json").write_text("{}")
    result = promote_reclassification(source, finalize=False)
    store = json.loads((run / "visual_event_segments_v1.json").read_text())
    assert result["status"] == "PROMOTED"
    assert [x["editorial"]["decision"] for x in store["events"]] == ["REJECT", "KEEP"]
    assert store["events"][1]["timeline_ordinal"] == 2
    assert not list(assets.glob("mf001-old.*"))
    assert (run / "superseded" / "semantic-v8-to-v9" / "VE_1" / "mf001-old.mp4").is_file()
    assert list((run / "semantic_history" / "v8").rglob("legacy.json"))
    contract = json.loads((run / "canonical_semantic_contract.json").read_text())
    assert contract["active_contract"] == "V9"


def test_status_reads_only_active_v9_decisions(monkeypatch, tmp_path):
    from movie_broll.production_run import print_status, read_status

    source, _run = _fixture(tmp_path, [_event(1, "REJECT")])
    _patch_runtime(monkeypatch)
    run_reclassification(source, provider=Provider(["KEEP"]), contact_sheet=lambda *_: b"jpeg")
    promote_reclassification(source, finalize=False)
    status = read_status(source)
    lines = []
    print_status(status, lines.append)
    assert status["active_semantic_contract"] == "V9"
    assert status["keep"] == 1 and status["reject"] == 0
    assert "ACTIVE SEMANTIC CONTRACT: V9" in lines


def test_promotion_preserves_compatible_keep_and_finalizes_only_new_keep(monkeypatch, tmp_path):
    import movie_broll.semantic_reclassification as module

    source, run = _fixture(tmp_path, [_event(1, "KEEP"), _event(2, "REJECT")])
    _patch_runtime(monkeypatch)
    run_reclassification(source, provider=Provider(["KEEP", "KEEP"]), contact_sheet=lambda *_: b"jpeg")
    write_json(run / "asset_registry.json", {"events": {"VE_1": {"asset_id": "mf001", "slug": "kept"}}})
    assets = run / "assets"; assets.mkdir()
    for name in ("mf001-kept.mp4", "vmf001-kept.mp4", "mf001-kept.jpg", "vmf001-kept.jpg", "mf001-kept.json"):
        (assets / name).write_bytes(b"preserve")
    seen = []
    monkeypatch.setattr(module, "finalize_pilot", lambda _source, _window, **kwargs: seen.append(kwargs["candidates"]) or {"status": "COMPLETE"})
    promote_reclassification(source)
    assert (assets / "mf001-kept.mp4").read_bytes() == b"preserve"
    assert [event["visual_event_id"] for event in seen[0]] == ["VE_2"]
