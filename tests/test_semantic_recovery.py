import hashlib
import json
from pathlib import Path

from movie_broll.semantic_recovery import recover_e02_report, recover_observation_from_historical_semantics
from movie_broll.semantic_observations import policy_evaluate
from movie_broll.utils import write_json


def _event(event_id="VE_000001", ordinal=1, old="REJECT"):
    return {"visual_event_id": event_id, "candidate_id": "BRC_0001", "timeline_ordinal": ordinal,
            "source_shot_ids": ["S1"], "editorial": {"decision": old},
            "technical_shots": [{"shot_id": "S1"}]}


def _v9(event_id, actions, *, interactions=None, reusable=True, action_type="distinct_visible_action_or_reaction"):
    return {"event_id": event_id, "state": "VALID", "response": {
        "visual": {"actions": actions, "objects": ["papel"], "visible_interactions": interactions or [],
                   "visible_emotions": ["neutral"], "people": [{"presentation": "woman"}], "shot_focus_plan": []},
        "relationships": [], "editorial": {"decision": "KEEP", "reusable_broll": reusable,
            "action_or_moment_complete": "true", "keep_qualification": {
                "action_evidence_type": action_type, "reusable_intent_type": "specific_visual_need"}}}}


def test_recovery_uses_v9_structured_action_and_local_policy():
    item = _event()
    record = recover_observation_from_historical_semantics(item, v9=_v9(item["visual_event_id"], ["escribiendo en un papel"]), existing=None, v91=None)
    body = record["observation"]
    assert body["visual_utility_kind"] == "concrete_action"
    assert body["action_or_moment_complete"] == "true"
    assert body["context_dependency"] == "low"
    assert body["action_evidence_ids"]
    assert policy_evaluate(record)["policy_decision"] == "KEEP"


def test_recovery_does_not_blindly_promote_v9_dialogue():
    item = _event()
    record = recover_observation_from_historical_semantics(
        item, v9=_v9(item["visual_event_id"], ["hablando"], interactions=["talking_face_to_face"]), existing=None, v91=None)
    assert record["observation"]["visual_utility_kind"] == "generic_dialogue_only"
    assert policy_evaluate(record)["policy_decision"] == "REJECT"


def test_recovery_report_is_deterministic_free_and_preserves_historical_sidecars(tmp_path: Path):
    input_dir = tmp_path / "input" / "movie"; input_dir.mkdir(parents=True)
    run = tmp_path / "runs" / "movie"; (run / "semantic_reclassifications" / "semantic-v9" / "events").mkdir(parents=True)
    action, dialogue = _event("VE_000001", 1), _event("VE_000002", 2)
    dialogue["candidate_id"] = "BRC_0002"
    write_json(run / "visual_event_segments_v1.json", {"events": [action, dialogue]})
    first = run / "semantic_reclassifications" / "semantic-v9" / "events" / "VE_000001.json"
    second = run / "semantic_reclassifications" / "semantic-v9" / "events" / "VE_000002.json"
    write_json(first, _v9("VE_000001", ["escribiendo en un papel"]))
    write_json(second, _v9("VE_000002", ["hablando"], interactions=["talking_face_to_face"]))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (first, second, run / "visual_event_segments_v1.json")}
    one = recover_e02_report(input_dir)
    encoded = (run / "e02_recovery_report.json").read_bytes()
    two = recover_e02_report(input_dir)
    assert one == two
    assert encoded == (run / "e02_recovery_report.json").read_bytes()
    assert one["zero_api_usage"] == {"provider_requests": 0, "api_cost_usd": 0.0}
    assert one["final_local_policy"] == {"KEEP": 1, "REJECT": 1, "REVIEW": 0}
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before} == before


def test_recovery_detects_existing_exact_five_file_package(tmp_path: Path):
    input_dir = tmp_path / "input" / "movie"; input_dir.mkdir(parents=True)
    run = tmp_path / "runs" / "movie"; (run / "assets").mkdir(parents=True); (run / "semantic_reclassifications" / "semantic-v9" / "events").mkdir(parents=True)
    item = _event(); write_json(run / "visual_event_segments_v1.json", {"events": [item]})
    write_json(run / "semantic_reclassifications" / "semantic-v9" / "events" / "VE_000001.json", _v9("VE_000001", ["escribiendo en un papel"]))
    base = "m001-writing"
    for name in (f"{base}.mp4", f"v{base}.mp4", f"{base}.jpg", f"v{base}.jpg"):
        (run / "assets" / name).write_bytes(b"x")
    write_json(run / "assets" / f"{base}.json", {"asset": {"id": "m001", "slug": "writing"}, "source_timeline": {"visual_event_id": "VE_000001"}})
    report = recover_e02_report(input_dir)
    assert report["existing_rendered_v8_packages"] == 1
    assert report["final_keep"][0]["existing_package"] == "yes"
