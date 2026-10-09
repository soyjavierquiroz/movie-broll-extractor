"""Temporal contracts tested with synthetic pixels and in-memory responses only."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from movie_broll import broll_pilot as pilot
from movie_broll import semantic_observations as legacy
from movie_broll import temporal_semantics as temporal
from movie_broll.broll_semantics import SemanticResponse
from movie_broll.temporal_evidence import PROFILE, MAX_FRAMES, sample_plan
from movie_broll.utils import write_json


def event(identifier="VE_TEST", shots=1):
    return {"visual_event_id": identifier, "candidate_id": "BRC_TEST", "start_frame": 0,
            "end_frame_exclusive": shots * 96, "source_shot_ids": [f"SHOT_{i}" for i in range(shots)],
            "technical_shots": [{"shot_id": f"SHOT_{i}", "start_frame": i * 96,
                                 "end_frame_exclusive": (i + 1) * 96} for i in range(shots)]}


def evidence(item):
    return {"evidence_profile": PROFILE, "samples": sample_plan(item, 24), "technical_shots": [], "fps": 24}


def payload(item, status="complete_action"):
    return {"event_id": item["visual_event_id"], "represented_shot_ids": item["source_shot_ids"],
            "people_count": "0", "visible_person_ids": [], "action_evidence_ids": [], "object_evidence_ids": [],
            "visible_states": ["A visible sustained posture"], "movement": "present", "physical_interactions": [],
            "visible_reactions": [], "conversation_present": "false", "conversation_visual_signal": "none",
            "visual_utility_kind": "useful_state" if status == "reusable_state" else "concrete_action",
            "action_or_moment_complete": {"complete_action": "true", "incomplete_action": "false",
                                          "reusable_state": "false", "unclear": "unclear"}[status],
            "context_dependency": "low", "technical_observations": dict.fromkeys(legacy.HARD_REJECT_FLAGS, False),
            "shot_focus_plan": [{"shot_id": sid, "focus_subject": "environment", "focus_reason": "visible setting",
                                 "preserve_secondary_subject": False, "interaction_requirement": "none",
                                 "focus_position": "center", "target_person_ids": [], "target_binding_confidence": "unclear"}
                                for sid in item["source_shot_ids"]],
            "moment_status": status, "temporal_support_sample_ids": [row["sample_id"] for row in evidence(item)["samples"]],
            "visual_actions": ["Visible structured action"],
            "observed_actions": [{"canonical_label": "Visible structured action",
                                  "evidence_type": "distinct_visible_action_or_reaction",
                                  "sample_ids": [row["sample_id"] for row in evidence(item)["samples"]]}]}


def record(item, body):
    return {"observation": body, "evidence_catalog": temporal.grounded_catalog(item, body, evidence(item))}


def setup(tmp_path, items):
    source = tmp_path / "input" / "film"; source.mkdir(parents=True)
    run = tmp_path / "runs" / "film"
    write_json(run / "visual_event_segments_v1.json", {"events": items})
    return source, run


def callbacks():
    def sheet(item, output):
        output.update(evidence(item))
        return b"synthetic-jpeg"
    return {"make_contact_sheet": sheet, "make_context": lambda *_: {}, "movie_sha256": "synthetic-source"}


class ScriptedObserver:
    """No SDK, client, credentials or network; only local response fixtures."""
    def __init__(self, items, statuses=None):
        self.items = {item["visual_event_id"]: item for item in items}
        self.calls = []
        self.statuses = statuses or {}

    def generate(self, prompt, context, jpeg):
        identifier = context["expected_event_id"]
        self.calls.append(copy.deepcopy(context))
        return SemanticResponse(payload(self.items[identifier], self.statuses.get(identifier, "complete_action")),
                                {"total_tokens": 1}, "synthetic", "fixture")


class Offline:
    def generate(self, *args):
        raise AssertionError("completed/historical evidence must not request a response")


def test_single_shot_bounded_begin_middle_end_and_deterministic():
    item = event()
    rows = sample_plan(item, 24)
    assert rows == sample_plan(copy.deepcopy(item), 24)
    assert len(rows) == 3
    assert [(row["frame"], row["role"]) for row in rows] == [(0, "begin"), (47, "middle"), (95, "end")]
    assert all(row["timestamp_seconds"] == row["frame"] / 24 for row in rows)
    item["event_type_hint"] = "action"
    assert len(sample_plan(item, 24)) == 5


def test_multi_shot_identity_and_global_cap():
    item = event(shots=6)
    item["signals"] = {"motion": 20}
    rows = sample_plan(item, 24)
    assert len(rows) == MAX_FRAMES
    for i in range(6):
        own = [row for row in rows if row["shot_id"] == f"SHOT_{i}"]
        assert any("middle" in row["roles"] for row in own)
        assert all(i * 96 <= row["frame"] < (i + 1) * 96 for row in own)
    with pytest.raises(ValueError, match="cap"):
        sample_plan(event(shots=17), 24)


def test_short_event_deduplicates_roles_without_claiming_complete():
    item = event(); item["end_frame_exclusive"] = 1
    item["technical_shots"][0]["end_frame_exclusive"] = 1
    assert len(sample_plan(item, 24)) == 1
    assert "insufficient_temporal_support" in temporal.validate_response(payload(item), item, evidence(item))


def test_sheet_preserves_pixels_aspect_and_records_sample_roles(monkeypatch):
    class Capture:
        def __init__(self, *args): self.frames = []
        def set(self, prop, frame): self.frames.append(frame)
        def read(self): return True, np.zeros((120, 160, 3), np.uint8)
        def release(self): pass
    monkeypatch.setattr(pilot.cv2, "VideoCapture", Capture)
    import movie_broll.finalization as finalization
    monkeypatch.setattr(finalization, "person_detector_preflight", lambda: None)
    monkeypatch.setattr(finalization, "detect_people", lambda frame: [])
    output = {}
    jpeg = pilot.candidate_contact_sheet(Path("synthetic"), event(shots=2), 24, output, evidence_profile=PROFILE)
    decoded = pilot.cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1)
    assert decoded.shape[:2] == (556, 1920)
    assert len(output["samples"]) == 6
    assert len(output["technical_shots"]) == 2  # only middle-reference identity catalog
    assert {row["shot_id"] for row in output["samples"]} == {"SHOT_0", "SHOT_1"}


@pytest.mark.parametrize("status,decision", [("complete_action", "KEEP"), ("incomplete_action", "REJECT"),
                                           ("reusable_state", "KEEP"), ("unclear", "REVIEW")])
def test_moment_contract_replays_unchanged_policy(status, decision):
    item = event(); body = payload(item, status)
    assert not temporal.validate_response(body, item, evidence(item))
    projected = temporal.policy_projection(record(item, body))
    assert legacy.policy_evaluate(projected)["policy_decision"] == decision
    if status == "reusable_state":
        assert body["action_or_moment_complete"] == "false"
        assert projected["observation"]["moment_status"] == "reusable_state"


def test_event_end_or_single_midpoint_is_not_completion():
    item = event(); body = payload(item)
    body["temporal_support_sample_ids"] = ["SAMPLE_02", "SAMPLE_03"]
    assert "complete_action_requires_begin_development_end" in temporal.validate_response(body, item, evidence(item))
    body = payload(item, "reusable_state"); body["action_or_moment_complete"] = "true"
    assert "reusable_state_raw_action_completeness_must_be_false_or_unclear" in temporal.validate_response(body, item, evidence(item))


def test_actions_require_exact_visual_label_and_temporal_citations():
    item = event(); body = payload(item)
    catalog = temporal.grounded_catalog(item, body, evidence(item))
    assert len(catalog["actions"]) == 1
    assert catalog["actions"][0]["source"] == "temporal_visual_observation"
    body["observed_actions"][0]["canonical_label"] = "Invented from subtitle"
    with pytest.raises(ValueError, match="structured_action"):
        temporal.grounded_catalog(item, body, evidence(item))
    body = payload(item); body["observed_actions"][0]["sample_ids"] = ["SAMPLE_02"]
    with pytest.raises(ValueError, match="temporal_visual_support"):
        temporal.grounded_catalog(item, body, evidence(item))
    body = payload(item); body["observed_actions"] = []
    assert temporal.grounded_catalog(item, body, evidence(item))["actions"] == []
    assert legacy.policy_evaluate(temporal.policy_projection(record(item, body)))["policy_decision"] == "REVIEW"
    body["observed_actions"] = payload(item)["observed_actions"]
    body["observed_actions"][0]["evidence_type"] = "generic_presence_or_movement"
    assert temporal.grounded_catalog(item, body, evidence(item))["actions"] == []


def legacy_body(item):
    body = payload(item, "unclear")
    for key in ("moment_status", "temporal_support_sample_ids", "visual_actions", "observed_actions"):
        body.pop(key)
    return body


def save_legacy(run, item):
    return legacy.persist_observation(run, item, legacy_body(item), observation_fingerprint_value="midpoint-only",
                                     evidence_catalog={}, provider_provenance={"request_count": 1})


def test_old_evidence_not_v2_and_immutable_during_enrichment(tmp_path):
    item = event(); source, run = setup(tmp_path, [item]); save_legacy(run, item)
    old = legacy.observation_path(run, item["visual_event_id"]); before = old.read_bytes()
    assert temporal.latest_record(run, item) is None
    assert temporal.observe_events(source, [item], provider=Offline(), **callbacks())["reused"] == 1
    fake = ScriptedObserver([item])
    temporal.observe_events(source, [item], provider=fake, preserve_legacy=False, **callbacks())
    assert old.read_bytes() == before
    assert temporal.latest_record(run, item)["evidence_profile"] == PROFILE
    assert legacy.policy_evaluate(legacy.effective_observation(run, item))["policy_decision"] == "KEEP"
    a = dict(movie_sha256="source", event=item, contact_sheet_sha256="same", active_picture=None,
             narrative_context_fingerprint=None)
    assert legacy.observation_fingerprint(**a) != legacy.observation_fingerprint(**a, evidence_profile=PROFILE, input_evidence=evidence(item))


def test_targeted_plan_uses_authoritative_final_review_and_local_exclusions(tmp_path):
    items = [event(f"VE_{i}") for i in range(5)]
    source, run = setup(tmp_path, items)
    for item in items:
        save_legacy(run, item)
        item["editorial"] = {"decision": "REVIEW", "status": "VALIDATED", "policy_version": legacy.POLICY_VERSION,
                             "policy_reasons": ["insufficient_observation"]}
    items[1]["editorial"]["decision"] = "KEEP"
    items[2]["editorial"]["decision"] = "REJECT"
    items[3]["editorial"]["policy_reasons"] = ["other_reason"]
    path = legacy.observation_path(run, items[4]["visual_event_id"])
    old = json.loads(path.read_text()); old["observation"]["context_dependency"] = "high"; write_json(path, old)
    write_json(run / "visual_event_segments_v1.json", {"events": items})
    before = {p: p.read_bytes() for p in run.rglob("*.json")}
    plan = temporal.enrichment_plan(source)
    assert plan["eligible_event_ids"] == ["VE_0"]
    assert len(plan["locally_resolved"]) == 1 and plan["maximum_generate_attempts"] == 2
    assert before == {p: p.read_bytes() for p in run.rglob("*.json")}
    temporal.enrich_reviews(source, provider=ScriptedObserver(items), **callbacks())
    assert temporal.enrichment_plan(source)["eligible"] == 0


def test_resume_and_saved_response_recovery_are_free(tmp_path, monkeypatch):
    items = [event("VE_FIRST"), event("VE_SECOND")]
    source, run = setup(tmp_path, items)
    fake = ScriptedObserver(items)
    original = temporal.persist
    def interrupted(run, item, *args, **kwargs):
        if item["visual_event_id"] == "VE_SECOND":
            raise RuntimeError("interruption after response persistence")
        return original(run, item, *args, **kwargs)
    monkeypatch.setattr(temporal, "persist", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        temporal.observe_events(source, items, provider=fake, **callbacks())
    first = temporal.latest_record(run, items[0])
    monkeypatch.setattr(temporal, "persist", original)
    resumed = temporal.observe_events(source, items, provider=Offline(), **callbacks())
    assert resumed["reused"] == 2 and resumed["requests"] == 0
    assert temporal.latest_record(run, items[0]) == first
    assert len(fake.calls) == 2


def test_identity_validation_retry_is_bounded_across_resumes(tmp_path):
    item = event(); source, run = setup(tmp_path, [item])
    class Wrong(ScriptedObserver):
        def generate(self, *args):
            response = super().generate(*args)
            response.data["event_id"] = "invented"
            return response
    fake = Wrong([item])
    temporal.observe_events(source, [item], provider=fake, **callbacks())
    assert len(fake.calls) == 2
    assert fake.calls[-1]["validation_feedback"]["returned_event_id"] == "invented"
    assert len(list((run / "semantic_observations/v2/responses").rglob("[0-9][0-9][0-9][0-9].json"))) == 2
    temporal.observe_events(source, [item], provider=Offline(), **callbacks())
    assert temporal.latest_record(run, item) is None


def test_changed_input_retains_historical_v2_and_requires_new_evidence(tmp_path):
    item = event(); source, run = setup(tmp_path, [item]); fake = ScriptedObserver([item])
    temporal.observe_events(source, [item], provider=fake, **callbacks())
    before = {p: p.read_bytes() for p in (run / "semantic_observations/v2/events").rglob("*.json")}
    changed = copy.deepcopy(item); changed["end_frame_exclusive"] -= 1
    changed["technical_shots"][0]["end_frame_exclusive"] -= 1
    assert temporal.latest_record(run, changed) is None
    temporal.observe_events(source, [changed], provider=ScriptedObserver([changed]), **callbacks())
    assert len(list((run / "semantic_observations/v2/events").rglob("*.json"))) == 2
    assert all(p.read_bytes() == data for p, data in before.items())


def test_dry_enrichment_does_not_construct_provider_or_modify_files(tmp_path, monkeypatch):
    item = event(); source, run = setup(tmp_path, [item]); save_legacy(run, item)
    item["editorial"] = {"decision": "REVIEW", "status": "VALIDATED", "policy_version": legacy.POLICY_VERSION,
                         "policy_reasons": ["insufficient_observation"]}
    write_json(run / "visual_event_segments_v1.json", {"events": [item]})
    import movie_broll.broll_semantics as providers
    monkeypatch.setattr(providers, "build_semantic_provider_from_env", lambda **kwargs: pytest.fail("dry plan must be local"))
    before = {p: p.read_bytes() for p in run.rglob("*.json")}
    assert temporal.run_enrichment(source)["eligible"] == 1
    assert before == {p: p.read_bytes() for p in run.rglob("*.json")}


def test_retry_succeeds_and_preserves_paid_attempt_provenance(tmp_path):
    item = event(); source, run = setup(tmp_path, [item])
    class Corrected(ScriptedObserver):
        def generate(self, *args):
            response = super().generate(*args)
            if len(self.calls) == 1:
                response.data.pop("event_id")
            return response
    fake = Corrected([item])
    result = temporal.observe_events(source, [item], provider=fake, **callbacks())
    assert result["requests"] == 2
    assert "event_id" in fake.calls[-1]["validation_feedback"]["errors"][0]
    assert temporal.latest_record(run, item)["provider_provenance"]["request_count"] == 2


def test_resume_recovers_raw_response_without_completed_diagnostic(tmp_path, monkeypatch):
    item = event(); source, run = setup(tmp_path, [item]); fake = ScriptedObserver([item])
    original = temporal.write_json
    def interrupt(path, value):
        if "diagnostic" in value:
            raise RuntimeError("after raw response")
        return original(path, value)
    monkeypatch.setattr(temporal, "write_json", interrupt)
    with pytest.raises(RuntimeError):
        temporal.observe_events(source, [item], provider=fake, **callbacks())
    monkeypatch.setattr(temporal, "write_json", original)
    assert temporal.observe_events(source, [item], provider=Offline(), **callbacks())["reused"] == 1


def test_targeted_enrichment_resumes_first_unresolved_event(tmp_path):
    items = [event("VE_ONE"), event("VE_TWO")]; source, run = setup(tmp_path, items)
    for item in items:
        save_legacy(run, item)
        item["editorial"] = {"decision": "REVIEW", "status": "VALIDATED", "policy_version": legacy.POLICY_VERSION,
                             "policy_reasons": ["insufficient_observation"]}
    write_json(run / "visual_event_segments_v1.json", {"events": items})
    fake = ScriptedObserver(items)
    temporal.enrich_reviews(source, provider=fake, max_events=1, **callbacks())
    assert temporal.enrichment_plan(source)["eligible_event_ids"] == ["VE_TWO"]
    temporal.enrich_reviews(source, provider=fake, **callbacks())
    assert [c["expected_event_id"] for c in fake.calls] == ["VE_ONE", "VE_TWO"]


@pytest.mark.parametrize("raw", ["false", "unclear"])
def test_sustained_state_preserves_raw_and_materializes_effective_completeness(raw):
    item = event("VE_GENERIC_STATE"); body = payload(item, "reusable_state")
    body["action_or_moment_complete"] = raw
    original = record(item, body); before = copy.deepcopy(original)
    projected = temporal.policy_projection(original)
    result = legacy.policy_evaluate(projected)
    assert result["policy_decision"] == "KEEP"
    legacy.apply_policy_result(item, {**projected, "event_id": item["visual_event_id"]}, result)
    assert original == before
    assert item["editorial"]["temporal_completeness"] == {
        "moment_status": "reusable_state", "raw_action_or_moment_complete": raw,
        "effective_action_or_moment_complete": "true"}


@pytest.mark.parametrize("utility", ["strong_nonverbal_interaction", "generic_dialogue_only"])
def test_reusable_state_utility_failure_is_not_raw_completeness_failure(utility):
    item = event(); body = payload(item, "reusable_state")
    body["visual_utility_kind"] = utility
    assert temporal.validate_response(body, item, evidence(item)) == ["reusable_state_requires_useful_state_utility"]


@pytest.mark.parametrize("states", [[], ["  "]])
def test_reusable_state_requires_substantive_state(states):
    item = event(); body = payload(item, "reusable_state"); body["visible_states"] = states
    assert "reusable_state_requires_visible_state" in temporal.validate_response(body, item, evidence(item))


def test_reusable_state_requires_multiple_temporal_samples():
    item = event(); body = payload(item, "reusable_state")
    body["temporal_support_sample_ids"] = ["SAMPLE_02"]
    assert "insufficient_temporal_support" in temporal.validate_response(body, item, evidence(item))


def test_saved_historical_rejection_revalidated_offline_preserves_all_paid_provenance(tmp_path):
    item = event(); source, run = setup(tmp_path, [item])
    fake = ScriptedObserver([item], {item["visual_event_id"]: "reusable_state"})
    temporal.observe_events(source, [item], provider=fake, **callbacks())
    completed = next((run / "semantic_observations/v2/events").rglob("*.json"))
    completed.unlink()  # Synthetic interrupted persistence, never production artifacts.
    saved = next((run / "semantic_observations/v2/responses").rglob("*.json"))
    attempt = json.loads(saved.read_text())
    attempt["diagnostic"]["errors"] = ["reusable_state_not_complete_action"]
    write_json(saved, attempt)
    later = copy.deepcopy(attempt)
    later["response"]["event_id"] = "invalid"
    later["cumulative_provider_provenance"]["request_count"] = 2
    write_json(saved.with_name("0002.json"), later)
    before = {p: p.read_bytes() for p in saved.parent.glob("*.json")}
    result = temporal.observe_events(source, [item], provider=Offline(), **callbacks())
    assert result["requests"] == 0 and result["reused"] == 1
    assert temporal.latest_record(run, item)["provider_provenance"]["request_count"] == 2
    assert all(p.read_bytes() == data for p, data in before.items())


def test_batch_failure_retains_prior_success_and_does_not_pay_on_restart(tmp_path):
    items = [event("VE_FIRST"), event("VE_FAIL"), event("VE_LATER")]
    source, run = setup(tmp_path, items)
    class BadState(ScriptedObserver):
        def generate(self, *args):
            response = super().generate(*args)
            if response.data["event_id"] == "VE_FAIL":
                response.data["visual_utility_kind"] = "generic_dialogue_only"
            return response
    fake = BadState(items, {"VE_FAIL": "reusable_state"})
    temporal.observe_events(source, items, provider=fake, **callbacks())
    assert [c["expected_event_id"] for c in fake.calls] == ["VE_FIRST", "VE_FAIL", "VE_FAIL", "VE_LATER"]
    assert temporal.latest_record(run, items[0]) is not None
    assert temporal.latest_record(run, items[2]) is not None
    temporal.observe_events(source, items, provider=Offline(), **callbacks())
    attempts = list((run / "semantic_observations/v2/responses/VE_FAIL").rglob("*.json"))
    assert len(attempts) == 2
    assert all("input_evidence" in json.loads(p.read_text()) for p in attempts)


class InvalidState(ScriptedObserver):
    identifier = "synthetic"
    model = "fixture"

    def generate(self, *args):
        response = super().generate(*args)
        response.data["moment_status"] = "reusable_state"
        response.data["action_or_moment_complete"] = "false"
        response.data["visual_utility_kind"] = "generic_dialogue_only"
        return response


def test_revision_grants_one_budget_and_returning_to_exhausted_revision_blocks(tmp_path, monkeypatch):
    item = event(); source, run = setup(tmp_path, [item])
    fake = InvalidState([item])
    authorized_revision = temporal.TEMPORAL_VALIDATION_RECOVERY_REVISION
    assert authorized_revision == "temporal_validation_recovery_v2"
    original_revision = temporal._LEGACY_RECOVERY_REVISION
    monkeypatch.setattr(temporal, 'TEMPORAL_VALIDATION_RECOVERY_REVISION', original_revision)
    temporal.observe_events(source, [item], provider=fake, **callbacks())
    assert len(fake.calls) == 2
    before = {p: p.read_bytes() for p in run.rglob('*.json')}
    temporal.observe_events(source, [item], provider=fake, **callbacks())
    assert len(fake.calls) == 2
    feedback = fake.calls[1]['validation_feedback']['validation_errors'][0]
    assert feedback['code'] == 'reusable_state_requires_useful_state_utility'
    assert feedback['path'] == 'visual_utility_kind'
    assert 'moment_status=reusable_state requires visual_utility_kind=useful_state' in feedback['message']
    assert 'Do not change moment_status merely to satisfy validation' in feedback['message']
    assert 'otherwise choose the truthful moment/utility classification' in feedback['message']
    monkeypatch.setattr(temporal, 'TEMPORAL_VALIDATION_RECOVERY_REVISION', 'temporal_validation_recovery_v2')
    temporal.observe_events(source, [item], provider=fake, **callbacks())
    assert len(fake.calls) == 4
    temporal.observe_events(source, [item], provider=fake, **callbacks())
    assert len(fake.calls) == 4
    monkeypatch.setattr(temporal, 'TEMPORAL_VALIDATION_RECOVERY_REVISION', original_revision)
    temporal.observe_events(source, [item], provider=fake, **callbacks())
    assert len(fake.calls) == 4
    assert all(p.read_bytes() == data for p, data in before.items())
    attempts = list((run / 'semantic_observations/v2/responses').rglob('*.json'))
    assert len(attempts) == 4
    for path in attempts:
        row = json.loads(path.read_text())
        assert row['evidence_fingerprint'] == temporal.fingerprint(row['input_evidence'])
        assert row['attempt_number'] in (1, 2)
        assert row['expected_event_id'] == row['returned_event_id'] == item['visual_event_id']
        assert row['validation_recovery_revision']
        assert row['diagnostic']['validation_errors']


@pytest.mark.parametrize('changed', ['model', 'identifier'])
def test_provider_model_switch_is_bounded_and_return_is_blocked(tmp_path, changed):
    item = event(); source, run = setup(tmp_path, [item]); fake = InvalidState([item])
    temporal.observe_events(source, [item], provider=fake, **callbacks())
    original = getattr(fake, changed)
    setattr(fake, changed, 'different')
    temporal.observe_events(source, [item], provider=fake, **callbacks())
    temporal.observe_events(source, [item], provider=fake, **callbacks())
    assert len(fake.calls) == 4
    setattr(fake, changed, original)
    temporal.observe_events(source, [item], provider=fake, **callbacks())
    assert len(fake.calls) == 4


def test_revision_reuses_five_valid_records_then_repairs_and_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(temporal, 'TEMPORAL_VALIDATION_RECOVERY_REVISION', temporal._LEGACY_RECOVERY_REVISION)
    items = [event(f'VE_SYNTHETIC_{i}') for i in range(8)]
    source, run = setup(tmp_path, items)
    class Batch(InvalidState):
        def generate(self, *args):
            if args[1]['expected_event_id'] == items[5]['visual_event_id']:
                return super().generate(*args)
            return ScriptedObserver.generate(self, *args)
    fake = Batch(items)
    temporal.observe_events(source, items, provider=fake, **callbacks())
    assert len(fake.calls) == 9
    before = {p: p.read_bytes() for p in run.rglob('*.json')}
    monkeypatch.setattr(temporal, 'TEMPORAL_VALIDATION_RECOVERY_REVISION', 'temporal_validation_recovery_v2')
    fixed = ScriptedObserver(items)
    result = temporal.observe_events(source, items, provider=fixed, **callbacks())
    assert result['reused'] == 7 and len(fixed.calls) == 1
    assert [c['expected_event_id'] for c in fixed.calls] == [items[5]['visual_event_id']]
    assert temporal.observe_events(source, items, provider=Offline(), **callbacks())['reused'] == 8
    assert all(p.read_bytes() == data for p, data in before.items())


def test_transport_failure_does_not_consume_semantic_response_budget(tmp_path):
    item = event(); source, run = setup(tmp_path, [item])
    class Transport(ScriptedObserver):
        def generate(self, *args):
            raise TimeoutError('synthetic transport failure')
    for _ in range(3):
        result = temporal.observe_events(source, [item], provider=Transport([item]), **callbacks())
        assert result['status'] == 'PARTIAL_PROVIDER' and result['remaining'] == 1
        assert result['validation_blocked'] == result['provider_blocked'] == 0
    assert not [p for p in (run / 'semantic_observations/v2/responses').rglob('0001.json') if p.parent.name != 'transport_failures']
    assert len(list((run / 'semantic_observations/v2/responses').rglob('transport_failures/*.json'))) == 3
    fake = InvalidState([item])
    temporal.observe_events(source, [item], provider=fake, **callbacks())
    assert len(fake.calls) == 2


def test_pre_revision_archives_are_adopted_without_repay_or_rewrite(tmp_path, monkeypatch):
    monkeypatch.setattr(temporal, 'TEMPORAL_VALIDATION_RECOVERY_REVISION', temporal._LEGACY_RECOVERY_REVISION)
    item = event(); source, run = setup(tmp_path, [item]); fake = InvalidState([item])
    temporal.observe_events(source, [item], provider=fake, **callbacks())
    paths = list((run / 'semantic_observations/v2/responses').rglob('*.json'))
    for path in paths:
        row = json.loads(path.read_text())
        # Match the minimal historical archive format, including stale diagnostics.
        row = {k: row[k] for k in ('response', 'cumulative_provider_provenance', 'provider_trace', 'created_at')}
        row['diagnostic'] = {'errors': ['reusable_state_not_complete_action']}
        write_json(path, row)
    before = {p: p.read_bytes() for p in paths}
    temporal.observe_events(source, [item], provider=fake, **callbacks())
    assert len(fake.calls) == 2
    assert all(p.read_bytes() == data for p, data in before.items())


def test_returned_model_alias_does_not_reopen_configured_execution(tmp_path):
    item = event(); source, run = setup(tmp_path, [item])
    class ResolvedModel(InvalidState):
        def generate(self, *args):
            response = super().generate(*args)
            return SemanticResponse(response.data, response.usage, response.provider,
                                    'fixture-resolved-version', response.attempts, response.provider_trace)
    fake = ResolvedModel([item])
    temporal.observe_events(source, [item], provider=fake, **callbacks())
    temporal.observe_events(source, [item], provider=fake, **callbacks())
    assert len(fake.calls) == 2


@pytest.mark.parametrize('raw', ['false', 'unclear'])
def test_unclear_moment_allows_known_incomplete_or_unknown_action(raw):
    item = event(); body = payload(item, 'unclear')
    body['action_or_moment_complete'] = raw
    assert temporal.validate_response(body, item, evidence(item)) == []
    if raw == 'unclear':
        result = legacy.policy_evaluate(temporal.policy_projection(record(item, body)))
        assert result['policy_decision'] == 'REVIEW'
        assert result['policy_reasons'] == ['insufficient_observation']
    body['action_or_moment_complete'] = 'true'
    assert 'unclear_moment_cannot_claim_complete_action' in temporal.validate_response(body, item, evidence(item))


@pytest.mark.parametrize('field', ['people_count', 'movement', 'conversation_present',
                                  'conversation_visual_signal', 'visual_utility_kind', 'context_dependency'])
def test_other_schema_valid_unclear_fields_are_evidence(field):
    item = event(); body = payload(item)
    body[field] = 'unclear'
    assert temporal.validate_response(body, item, evidence(item)) == []
    if field in {'visual_utility_kind', 'context_dependency'}:
        assert legacy.policy_evaluate(temporal.policy_projection(record(item, body)))['policy_decision'] == 'REVIEW'


def test_unclear_action_evidence_is_valid_but_not_canonical():
    item = event(); body = payload(item)
    body['observed_actions'][0].update(evidence_type='unclear', sample_ids=[])
    assert temporal.validate_response(body, item, evidence(item)) == []
    assert temporal.grounded_catalog(item, body, evidence(item))['actions'] == []
    body['observed_actions'][0]['sample_ids'] = ['invented']
    assert 'unknown_action_sample_id' in temporal.validate_response(body, item, evidence(item))


def test_review_continues_batch_without_retry_and_is_reused(tmp_path):
    items = [event('VE_UNCERTAIN'), event('VE_AFTER')]
    source, run = setup(tmp_path, items)
    fake = ScriptedObserver(items, {'VE_UNCERTAIN': 'unclear'})
    result = temporal.observe_events(source, items, provider=fake, **callbacks())
    assert result['requests'] == 2 and len(fake.calls) == 2
    first = temporal.latest_record(run, items[0])
    assert legacy.policy_evaluate(temporal.policy_projection(first))['policy_decision'] == 'REVIEW'
    assert temporal.latest_record(run, items[1]) is not None
    assert temporal.observe_events(source, items, provider=Offline(), **callbacks())['reused'] == 2


@pytest.mark.parametrize('raw,decision', [('unclear', 'REVIEW'), ('false', 'REJECT')])
def test_corrected_unclear_archive_recovery_without_rendering_or_provider(tmp_path, monkeypatch, raw, decision):
    item = event('VE_GENERIC_UNRESOLVED'); source, run = setup(tmp_path, [item])
    write_json(run / 'source_fingerprint.json', {'movie_sha256': 'synthetic-source'})
    class SavedUncertain(ScriptedObserver):
        def generate(self, *args):
            response = super().generate(*args)
            response.data['action_or_moment_complete'] = raw
            return response
    fake = SavedUncertain([item], {item['visual_event_id']: 'unclear'})
    original = temporal.validate_response
    def old_validator(body, event, evidence):
        return original(body, event, evidence) + ['unclear_completeness']
    monkeypatch.setattr(temporal, 'validate_response', old_validator)
    temporal.observe_events(source, [item], provider=fake, **callbacks())
    paths = sorted((run / 'semantic_observations/v2/responses').rglob('*.json'))
    before = {p: p.read_bytes() for p in paths}
    monkeypatch.setattr(temporal, 'validate_response', original)
    recovered = temporal.recover_saved_response_offline(run, item, paths[-1])
    assert recovered['observation'] == json.loads(paths[-1].read_text())['response']
    assert recovered['provider_provenance']['request_count'] == 2
    assert legacy.policy_evaluate(temporal.policy_projection(recovered))['policy_decision'] == decision
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert temporal.observe_events(source, [item], provider=Offline(), **callbacks())['reused'] == 1
    changed = copy.deepcopy(item); changed['end_frame_exclusive'] -= 1
    with pytest.raises(ValueError, match='mismatch'):
        temporal.recover_saved_response_offline(run, changed, paths[-1])


@pytest.mark.parametrize('field', ['focus_subject', 'interaction_requirement', 'focus_position', 'target_binding_confidence'])
def test_contract_valid_unclear_focus_fields(field):
    item = event(); body = payload(item)
    body['shot_focus_plan'][0][field] = 'unclear'
    assert temporal.validate_response(body, item, evidence(item)) == []


def test_malformed_schema_and_impossible_enum_remain_invalid():
    item = event(); body = payload(item)
    body['context_dependency'] = 'invented'
    assert temporal.validate_response(body, item, evidence(item))
    body = payload(item); body.pop('moment_status')
    assert temporal.validate_response(body, item, evidence(item))


class InputRejected(RuntimeError):
    status_code = 400
    code = 'invalid_prompt'
    body = {'error': {'code': 'invalid_prompt', 'message': 'private input sk-secret'}}


class RejectInputs(ScriptedObserver):
    identifier = 'synthetic'
    model = 'fixture'

    def generate(self, *args):
        if args[1]['expected_event_id'].startswith('VE_BLOCK'):
            self.calls.append(copy.deepcopy(args[1]))
            from movie_broll.broll_semantics import OpenAIProviderError
            raise OpenAIProviderError(InputRejected('private input sk-secret'), 1)
        return super().generate(*args)


def test_provider_rejection_durable_fail_soft_and_free_resume(tmp_path, monkeypatch):
    items = [event('VE_BLOCK_A'), event('VE_GOOD'), event('VE_BLOCK_B'), event('VE_REVIEW')]
    source, run = setup(tmp_path, items)
    for item in items:
        save_legacy(run, item)
    originals = {p: p.read_bytes() for p in (run / 'semantic_observations/v1').rglob('*.json')}
    fake = RejectInputs(items, {'VE_REVIEW': 'unclear'})
    result = temporal.observe_events(source, items, provider=fake, preserve_legacy=False, **callbacks())
    assert len(fake.calls) == 4
    assert result['completed'] == 2 and result['review'] == 1
    assert result['provider_blocked'] == 2 and result['validation_blocked'] == 0
    assert result['remaining'] == 0 and result['status'] == 'COMPLETE_WITH_REVIEW'
    blocks = list((run / 'semantic_observations/v2/terminal').rglob('*.json'))
    assert len(blocks) == 2
    for path in blocks:
        row = json.loads(path.read_text())
        assert row['status'] == 'PROVIDER_BLOCKED'
        assert row['semantic_attempts'] == 0
        assert row['diagnostic']['http_status'] == 400
        assert row['diagnostic']['error_code'] == 'invalid_prompt'
        assert row['editorial_decision'] == 'REVIEW' and row['renderable'] is False
        assert 'private' not in path.read_text() and 'sk-secret' not in path.read_text()
        assert row['execution_identity']['provider'] == 'synthetic'
        assert row['execution_identity']['model'] == 'fixture'
        assert row['observation_fingerprint'] and row['evidence_fingerprint']
    before = {p: p.read_bytes() for p in run.rglob('*.json')}
    monkeypatch.setattr(temporal, 'TEMPORAL_VALIDATION_RECOVERY_REVISION', 'fixture-new-validation')
    result = temporal.observe_events(source, items, provider=fake, preserve_legacy=False, **callbacks())
    assert len(fake.calls) == 4 and result['reused'] == 2 and result['provider_blocked'] == 2
    assert all(p.read_bytes() == data for p, data in before.items())
    assert all(p.read_bytes() == data for p, data in originals.items())
    for item in (items[0], items[2]):
        assert temporal.latest_record(run, item) is None
        assert legacy.policy_evaluate(legacy.effective_observation(run, item))['policy_decision'] == 'REVIEW'
        assert not list((run / 'semantic_observations/v2/responses' / item['visual_event_id']).rglob('*.json'))
    fake.model = 'legitimate-new-model'
    temporal.observe_events(source, items, provider=fake, preserve_legacy=False, **callbacks())
    assert len(fake.calls) == 6


def test_validation_block_is_distinct_and_batch_continues(tmp_path):
    items = [event('VE_INVALID_A'), event('VE_INVALID_B')]
    source, run = setup(tmp_path, items)
    fake = InvalidState(items)
    result = temporal.observe_events(source, items, provider=fake, **callbacks())
    assert len(fake.calls) == 4
    assert result['validation_blocked'] == 2 and result['provider_blocked'] == 0
    assert result['completed'] == 0 and result['remaining'] == 0
    assert result['status'] == 'COMPLETE_WITH_REVIEW'
    resumed = temporal.observe_events(source, items, provider=fake, **callbacks())
    assert len(fake.calls) == 4 and resumed['validation_blocked'] == 2


def test_adapter_invalid_prompt_has_no_transport_retry():
    from types import SimpleNamespace
    from movie_broll.broll_semantics import OpenAISemanticProvider, OpenAIProviderError, classify_provider_error
    calls, sleeps = [], []
    def parse(**kwargs):
        calls.append(1)
        raise InputRejected('invalid_prompt')
    adapter = OpenAISemanticProvider('synthetic-key',
        client=SimpleNamespace(responses=SimpleNamespace(parse=parse)), sleep=sleeps.append, max_retries=3)
    with pytest.raises(OpenAIProviderError) as raised:
        adapter.generate('synthetic prompt', {'visual_event_id': 'VE_SYNTHETIC'}, b'synthetic')
    assert len(calls) == 1 and sleeps == []
    detail = classify_provider_error(raised.value)
    assert detail['reason'] == 'provider_input_rejection' and detail['retryable'] is False
    assert detail['attempts'] == 1


def test_rejection_after_invalid_response_consumes_no_additional_semantic_attempt(tmp_path):
    item = event('VE_SYNTHETIC'); source, run = setup(tmp_path, [item])
    class InvalidThenBlocked(InvalidState):
        def generate(self, *args):
            if self.calls:
                raise InputRejected('invalid_prompt')
            return super().generate(*args)
    fake = InvalidThenBlocked([item])
    result = temporal.observe_events(source, [item], provider=fake, **callbacks())
    assert result['provider_blocked'] == 1 and result['validation_blocked'] == 0
    block = json.loads(next((run / 'semantic_observations/v2/terminal').rglob('*.json')).read_text())
    assert block['semantic_attempts'] == 1
    assert len(list((run / 'semantic_observations/v2/responses').rglob('*.json'))) == 1
    result = temporal.observe_events(source, [item], provider=fake, **callbacks())
    assert result['requests'] == 0 and len(fake.calls) == 1


def test_offline_historical_quarantine_skips_without_request_or_pixel_generation(tmp_path):
    item = event('VE_SYNTHETIC_QUARANTINE'); source, run = setup(tmp_path, [item])
    fake = InvalidState([item])
    path = temporal.quarantine_reported_rejection_offline(run, item, movie_sha256='synthetic-source',
                                                        provider_id=fake.identifier, model=fake.model)
    row = json.loads(path.read_text())
    assert row['exact_request_fingerprint'] is None and row['input_snapshot_fingerprint']
    result = temporal.observe_events(source, [item], provider=fake,
        **{**callbacks(), 'make_contact_sheet': lambda *_: pytest.fail('quarantined input must be free')})
    assert result['provider_blocked'] == 1 and result['requests'] == 0 and not fake.calls
    fake.model = 'legitimate-model-revision'
    result = temporal.observe_events(source, [item], provider=fake, **callbacks())
    assert result['validation_blocked'] == 1 and len(fake.calls) == 2
