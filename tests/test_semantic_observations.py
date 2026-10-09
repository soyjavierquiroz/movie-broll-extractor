import json
from pathlib import Path

from movie_broll.semantic_observations import (
    OBSERVATION_SCHEMA_VERSION, canonical_evidence_catalog, evaluate_cached_observations,
    migrate_existing_observations, observation_fingerprint, observation_path, persist_observation,
    policy_evaluate, observe_missing_events,
)
from movie_broll.broll_semantics import SemanticResponse
from movie_broll.utils import write_json


def event(event_id="VE_000001"):
    return {"visual_event_id": event_id, "candidate_id": "BRC_0001", "start_frame": 1,
            "end_frame_exclusive": 24, "source_shot_ids": ["FULL_SHOT_0001"],
            "technical_shots": [{"shot_id": "FULL_SHOT_0001", "start_frame": 1, "end_frame_exclusive": 24}]}


def body(event_id="VE_000001", *, utility="concrete_action", signal="none", flags=None):
    return {"event_id": event_id, "represented_shot_ids": ["FULL_SHOT_0001"], "people_count": "1",
            "visible_person_ids": ["FULL_SHOT_0001:P1"], "action_evidence_ids": [f"{event_id}:ACTION_01"],
            "object_evidence_ids": [], "visible_states": [], "movement": "present", "physical_interactions": [],
            "visible_reactions": [], "conversation_present": "false", "conversation_visual_signal": signal,
            "visual_utility_kind": utility, "action_or_moment_complete": "true", "context_dependency": "low",
            "technical_observations": flags or {"title_card": False, "credits": False, "logo": False,
                                                    "dominant_text": False, "black_or_empty": False,
                                                    "corrupt_or_unusable": False},
            "shot_focus_plan": []}


def test_fingerprint_has_no_policy_input():
    value = event()
    first = observation_fingerprint(movie_sha256="movie", event=value, contact_sheet_sha256="sheet",
                                    active_picture={"x": 0}, narrative_context_fingerprint="narrative")
    second = observation_fingerprint(movie_sha256="movie", event=value, contact_sheet_sha256="sheet",
                                     active_picture={"x": 0}, narrative_context_fingerprint="narrative")
    assert first == second


def test_policy_is_local_and_uses_ids_not_action_wording():
    record = {"schema_version": OBSERVATION_SCHEMA_VERSION, "observation": body()}
    assert policy_evaluate(record)["policy_decision"] == "KEEP"
    # No prose label is present in the policy input; changing display metadata
    # cannot change the outcome.
    record["evidence_catalog"] = {"actions": [{"id": "VE_000001:ACTION_01", "canonical_label": "anything"}]}
    assert policy_evaluate(record)["policy_decision"] == "KEEP"
    dialogue = {"schema_version": OBSERVATION_SCHEMA_VERSION,
                "observation": body(utility="generic_dialogue_only", signal="generic_dialogue_only")}
    assert policy_evaluate(dialogue)["policy_decision"] == "REJECT"
    hard = {"schema_version": OBSERVATION_SCHEMA_VERSION,
            "observation": body(flags={"title_card": True, "credits": False, "logo": False,
                                         "dominant_text": False, "black_or_empty": False,
                                         "corrupt_or_unusable": False})}
    assert policy_evaluate(hard)["policy_decision"] == "REJECT"


def test_cached_replay_is_free_and_does_not_touch_canonical_event_store(tmp_path: Path):
    input_dir = tmp_path / "input" / "movie"; input_dir.mkdir(parents=True)
    run = tmp_path / "runs" / "movie"; run.mkdir(parents=True)
    item = event(); store = {"events": [item]}
    store_path = run / "visual_event_segments_v1.json"; write_json(store_path, store)
    persist_observation(run, item, body(), observation_fingerprint_value="fp", evidence_catalog={},
                        provider_provenance={"request_count": 0, "usage": {}, "cost_usd": 0.0})
    before = store_path.read_bytes()
    replay = evaluate_cached_observations(input_dir)
    assert replay["provider_requests"] == 0 and replay["api_cost_usd"] == 0
    assert replay["counts"]["KEEP"] == 1
    assert store_path.read_bytes() == before


def test_migration_uses_saved_checkpoint_without_provider_or_canonical_mutation(tmp_path: Path):
    input_dir = tmp_path / "input" / "movie"; input_dir.mkdir(parents=True)
    run = tmp_path / "runs" / "movie"; (run / "semantic_checkpoints").mkdir(parents=True)
    item = event(); store_path = run / "visual_event_segments_v1.json"; write_json(store_path, {"events": [item]})
    response = {"visual": {"actions": ["caminar"], "visible_interactions": [], "visible_emotions": [],
                           "people": [], "shot_focus_plan": []},
                "editorial": {"visual_utility_kind": "concrete_action", "conversation_visual_signal": "none",
                              "action_or_moment_complete": "true", "decision": "REJECT"}}
    write_json(run / "semantic_checkpoints" / "BRC_0001.json", {"provider": "openai", "model": "test", "response": response})
    before = store_path.read_bytes(); result = migrate_existing_observations(input_dir)
    assert result["provider_requests"] == 0 and result["created"] == 1
    assert store_path.read_bytes() == before
    assert observation_path(run, item["visual_event_id"]).is_file()


def test_observation_cache_resumes_only_missing_events(tmp_path: Path):
    input_dir = tmp_path / "input" / "movie"; input_dir.mkdir(parents=True)
    run = tmp_path / "runs" / "movie"; run.mkdir(parents=True)
    first, second = event("VE_000001"), event("VE_000002")
    second["candidate_id"] = "BRC_0002"

    class Provider:
        identifier = "test"; model = "test-observer"
        def __init__(self): self.calls = []
        def generate(self, prompt, context, jpeg):
            self.calls.append(context["visual_event_id"])
            event_id = context["visual_event_id"]
            return SemanticResponse(body(event_id), {"prompt_tokens": 1, "cached_tokens": 0,
                                                      "response_tokens": 1, "thinking_tokens": 0,
                                                      "total_tokens": 2}, self.identifier, self.model)
    provider = Provider()
    context = lambda item, catalog: {"visual_event_id": item["visual_event_id"], "narrative": {}}
    sheet = lambda item, evidence: b"fixed-contact-sheet"
    first_run = observe_missing_events(input_dir, [first, second], movie_sha256="movie", provider=provider,
                                       make_contact_sheet=sheet, make_context=context)
    assert first_run["requests"] == 2 and provider.calls == ["VE_000001", "VE_000002"]
    resumed = observe_missing_events(input_dir, [first, second], movie_sha256="movie", provider=provider,
                                    make_contact_sheet=sheet, make_context=context)
    assert resumed["requests"] == 0 and resumed["reused"] == 2
    observation_path(run, second["visual_event_id"]).unlink()
    one_missing = observe_missing_events(input_dir, [first, second], movie_sha256="movie", provider=provider,
                                         make_contact_sheet=sheet, make_context=context)
    assert one_missing["requests"] == 0 and one_missing["reused"] == 2
    assert provider.calls == ["VE_000001", "VE_000002"]


import pytest
from movie_broll.semantic_observations import validate_observation


@pytest.mark.parametrize('identity,expected', [('VE_000001', []), ('invented', ['event_id']),
                                               (None, ['missing:event_id', 'event_id'])])
def test_identity_contract(identity, expected):
    payload = body(identity)
    if identity is None:
        payload.pop('event_id')
    assert validate_observation({'schema_version': OBSERVATION_SCHEMA_VERSION,
                                 'observation': payload}, event()) == expected


@pytest.mark.parametrize('missing', [False, True])
def test_failed_observation_preserves_evidence_and_previous_work(tmp_path, missing):
    input_dir = tmp_path / 'input' / 'movie'; input_dir.mkdir(parents=True)
    run = tmp_path / 'runs' / 'movie'
    first, second = event(), event('VE_000002')
    class Provider:
        identifier = 'test'; model = 'test'
        def __init__(self): self.calls = []
        def generate(self, prompt, context, jpeg):
            self.calls.append(context)
            payload = body(context['expected_event_id'])
            if context['expected_event_id'] == second['visual_event_id']:
                payload['event_id'] = 'invented'
                if missing: payload.pop('event_id')
            return SemanticResponse(payload, {}, 'test', 'test')
    provider = Provider()
    kwargs = dict(movie_sha256='movie', provider=provider, make_contact_sheet=lambda *args: b'sheet',
                  make_context=lambda *args: {})
    downstream = []
    with pytest.raises(ValueError, match='expected_event_id'):
        observe_missing_events(input_dir, [first, second], **kwargs)
        downstream.append('finalize')
    assert not downstream
    assert len(provider.calls) == 3
    assert provider.calls[-1]['validation_feedback']['returned_event_id'] == (None if missing else 'invented')
    previous = observation_path(run, first['visual_event_id']).read_bytes()
    assert not observation_path(run, second['visual_event_id']).exists()
    evidence = list((run / 'semantic_observations/v1/responses' / second['visual_event_id']).glob('*/*.json'))
    assert len(evidence) == 2
    assert json.loads(evidence[0].read_text())['diagnostic']['event_id_present'] is not missing
    class Offline:
        def generate(self, *args): raise AssertionError('completed observation must be free')
    kwargs['provider'] = Offline()
    assert observe_missing_events(input_dir, [first], **kwargs)['reused'] == 1
    assert observation_path(run, first['visual_event_id']).read_bytes() == previous
    # Simulate interruption after response evidence was saved but before canonical persistence.
    observation_path(run, first['visual_event_id']).unlink()
    assert observe_missing_events(input_dir, [first], **kwargs)['reused'] == 1


def test_identity_retry_can_succeed_without_replacing_identity(tmp_path):
    input_dir = tmp_path / 'input' / 'movie'; input_dir.mkdir(parents=True)
    class Provider:
        identifier = 'test'; model = 'test'
        def __init__(self): self.calls = 0
        def generate(self, prompt, context, jpeg):
            self.calls += 1
            return SemanticResponse(body('wrong' if self.calls == 1 else context['expected_event_id']),
                                    {'total_tokens': 2}, 'test', 'test')
    provider = Provider()
    result = observe_missing_events(input_dir, [event()], movie_sha256='movie', provider=provider,
                                   make_contact_sheet=lambda *args: b'sheet', make_context=lambda *args: {})
    assert result['requests'] == 2
    record = json.loads(observation_path(tmp_path / 'runs/movie', event()['visual_event_id']).read_text())
    assert record['observation']['event_id'] == event()['visual_event_id']
    assert record['provider_provenance']['request_count'] == 2
