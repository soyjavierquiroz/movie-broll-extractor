"""Visual reuse decisions use synthetic temporal evidence, never a provider."""
import copy
import json
from pathlib import Path

import pytest

from movie_broll import broll_policy_v2 as policy
from movie_broll import semantic_observations as old
from movie_broll.temporal_evidence import PROFILE, sample_plan


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError('network forbidden in policy tests')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)


def fixture(duration=10, shots=1, utility='useful_state'):
    end = duration * 24
    event = {'visual_event_id': 'VE_SYNTHETIC', 'start_frame': 0, 'end_frame_exclusive': end,
             'source_shot_ids': [f'SHOT_{i}' for i in range(shots)],
             'technical_shots': [{'shot_id': f'SHOT_{i}', 'start_frame': i * end // shots,
                                  'end_frame_exclusive': (i + 1) * end // shots} for i in range(shots)]}
    evidence = {'fps': 24, 'evidence_profile': PROFILE, 'samples': sample_plan(event, 24), 'technical_shots': []}
    moment = {'moment_id': 'M1', 'utility_class': utility, 'visual_strength': 'useful',
              'moment_kind': 'sustained_state', 'action_completeness': 'false',
              'moment_usability': 'usable', 'asset_window_complete': 'true',
              'visual_context_dependency': 'low', 'start_frame': 0, 'end_frame_exclusive': end,
              'beginning_sample_id': None, 'development_sample_ids': [], 'completion_sample_id': None,
              'sample_ids': [s['sample_id'] for s in evidence['samples']], 'continuity': 'supported',
              'reuse_key': 'specific visual need'}
    body = {'event_id': event['visual_event_id'], 'represented_shot_ids': event['source_shot_ids'],
            'people_count': '0', 'visible_person_ids': [], 'action_evidence_ids': [], 'object_evidence_ids': [],
            'visible_states': ['visible sustained posture'], 'movement': 'present', 'physical_interactions': [],
            'visible_reactions': [], 'conversation_present': 'true', 'conversation_visual_signal': 'generic_dialogue_only',
            'visual_utility_kind': 'useful_state', 'action_or_moment_complete': 'false',
            'context_dependency': 'high', 'technical_observations': dict.fromkeys(old.HARD_REJECT_FLAGS, False),
            'shot_focus_plan': [{'shot_id': sid, 'focus_subject': 'environment', 'focus_reason': 'visible setting',
                                 'preserve_secondary_subject': False, 'interaction_requirement': 'none',
                                 'focus_position': 'center', 'target_person_ids': [], 'target_binding_confidence': 'unclear'}
                                for sid in event['source_shot_ids']],
            'moment_status': 'reusable_state', 'temporal_support_sample_ids': moment['sample_ids'],
            'visual_actions': [], 'observed_actions': [], 'visual_moments': [moment],
            'visual_assessment_complete': True, 'narrative_context_dependency': 'high'}
    return event, {'schema_version': policy.OBSERVATION_SCHEMA, 'observation': body, 'input_evidence': evidence}


@pytest.mark.parametrize('utility', policy.UTILITY_CLASSES)
def test_all_visual_utilities_can_keep_with_false_action_completeness(utility):
    event, record = fixture(utility=utility)
    original = copy.deepcopy(record)
    result = policy.evaluate(record, event)
    assert result['policy_decision'] == 'KEEP', result
    assert result['asset_windows'][0]['action_completeness'] == 'false'
    assert record == original
    assert result['provider_requests'] == 0


def test_flat_weak_talking_head_rejects_after_explicit_visual_assessment():
    event, record = fixture(utility='conversation_scene')
    record['observation']['visual_moments'][0]['visual_strength'] = 'weak'
    assert policy.evaluate(record, event)['policy_decision'] == 'REJECT'


def test_legacy_generic_dialogue_needs_assessment_not_automatic_reject():
    event, record = fixture()
    body = record['observation']
    for key in ('visual_moments', 'visual_assessment_complete', 'narrative_context_dependency'):
        body.pop(key)
    body['visual_utility_kind'] = 'generic_dialogue_only'
    body['moment_status'] = 'unclear'
    record['schema_version'] = 'semantic_observation_v2'
    result = policy.evaluate(record, event)
    assert result['policy_decision'] == 'REVIEW'
    assert result['targeted_reobservation']


def test_visual_context_high_reviews_but_narrative_context_high_can_keep():
    event, record = fixture()
    assert policy.evaluate(record, event)['policy_decision'] == 'KEEP'
    record['observation']['visual_moments'][0]['visual_context_dependency'] = 'high'
    assert policy.evaluate(record, event)['policy_decision'] == 'REVIEW'


@pytest.mark.parametrize('flag', old.HARD_REJECT_FLAGS)
def test_hard_exclusions_override_every_utility(flag):
    event, record = fixture()
    record['observation']['technical_observations'][flag] = True
    assert policy.evaluate(record, event)['policy_decision'] == 'REJECT'


def test_local_action_inside_dialogue_event_uses_actual_range():
    event, record = fixture(duration=20)
    # Add grounded local begin/development/completion to authoritative sample plan.
    event['event_type_hint'] = 'action'
    evidence = record['input_evidence']
    evidence['samples'] = sample_plan(event, 24)
    rows = evidence['samples'][1:-1]
    m = record['observation']['visual_moments'][0]
    m.update(moment_kind='complete_action', utility_class='concrete_action', action_completeness='true',
             start_frame=rows[0]['frame'], end_frame_exclusive=rows[-1]['frame'] + 1,
             sample_ids=[s['sample_id'] for s in rows], beginning_sample_id=rows[0]['sample_id'],
             development_sample_ids=[rows[1]['sample_id']], completion_sample_id=rows[-1]['sample_id'])
    b = record['observation']; b['temporal_support_sample_ids'] = [s['sample_id'] for s in evidence['samples']]
    b['moment_status'] = 'complete_action'; b['action_or_moment_complete'] = 'true'
    assert not policy.validate_visual_reuse(b, event, evidence)
    result = policy.evaluate(record, event)
    assert result['policy_decision'] == 'KEEP', result
    window = result['asset_windows'][0]
    assert window['start_frame'] > event['start_frame']
    assert window['end_frame_exclusive'] < event['end_frame_exclusive']


@pytest.mark.parametrize('seconds', [5, 10, 20])
def test_target_window_preserves_whole_moment(seconds):
    event, record = fixture(duration=seconds)
    w = policy.evaluate(record, event)['asset_windows'][0]
    assert w['duration_seconds'] == seconds and not w['duration_exception']


def test_long_state_uses_one_central_coherent_window():
    event, record = fixture(duration=60, shots=3)
    w = policy.evaluate(record, event)['asset_windows']
    assert len(w) == 1
    assert (w[0]['start_seconds'], w[0]['end_seconds']) == (20, 40)


def test_multishot_coherent_window_and_unproven_cut():
    event, record = fixture(duration=12, shots=3)
    assert len(policy.evaluate(record, event)['asset_windows'][0]['source_shot_ids']) == 3
    record['observation']['visual_moments'][0]['continuity'] = 'unclear'
    assert policy.evaluate(record, event)['policy_decision'] == 'REVIEW'


def test_complete_action_not_arbitrarily_cut_at_twenty_seconds():
    event, record = fixture(duration=30)
    m = record['observation']['visual_moments'][0]; rows = record['input_evidence']['samples']
    m.update(moment_kind='complete_action', action_completeness='true', beginning_sample_id=rows[0]['sample_id'],
             development_sample_ids=[rows[1]['sample_id']], completion_sample_id=rows[-1]['sample_id'])
    w = policy.evaluate(record, event)['asset_windows'][0]
    assert w['duration_seconds'] == 30 and w['duration_exception']


def test_short_moment_not_padded_or_extended():
    event, record = fixture(duration=3)
    w = policy.evaluate(record, event)['asset_windows'][0]
    assert w['duration_seconds'] == 3 and w['duration_exception']


def test_duplicate_and_overlap_scarcity_across_events():
    event, record = fixture()
    first = policy.evaluate(record, event)['asset_windows']
    second = copy.deepcopy(record['observation']['visual_moments'][0]); second['moment_id'] = 'M2'
    second['reuse_key'] = 'different label but same frames'
    record['observation']['visual_moments'].append(second)
    assert len(policy.evaluate(record, event)['asset_windows']) == 1
    assert policy.evaluate(record, event, selected=first)['policy_decision'] == 'REJECT'


def test_distinct_moments_can_yield_two_assets():
    event, record = fixture(duration=20)
    event['event_type_hint'] = 'action'
    ev = record['input_evidence']; ev['samples'] = sample_plan(event, 24)
    rows = ev['samples']; b = record['observation']; b['temporal_support_sample_ids'] = [s['sample_id'] for s in rows]
    first = b['visual_moments'][0]
    first.update(end_frame_exclusive=rows[1]['frame'] + 1, sample_ids=[s['sample_id'] for s in rows[:2]])
    second = copy.deepcopy(first)
    second.update(moment_id='M2', reuse_key='distinct visual need', start_frame=rows[-2]['frame'],
                  end_frame_exclusive=rows[-1]['frame'] + 1, sample_ids=[s['sample_id'] for s in rows[-2:]])
    b['visual_moments'] = [first, second]
    assert len(policy.evaluate(record, event)['asset_windows']) == 2


def test_forged_or_out_of_bounds_moment_reviews():
    event, record = fixture()
    record['observation']['visual_moments'][0]['start_frame'] = -1
    assert policy.evaluate(record, event)['policy_decision'] == 'REVIEW'


def test_v1_unchanged_and_explicit_dispatch():
    event, record = fixture()
    assert old.policy_evaluate(record, policy_version=policy.POLICY_VERSION, event=event)['policy_decision'] == 'KEEP'
    assert old.POLICY_VERSION == 'broll_policy_v1'
    b = copy.deepcopy(record['observation'])
    b['visual_utility_kind'] = 'generic_dialogue_only'
    assert old.policy_evaluate(b)['policy_decision'] == 'REJECT'


def test_read_only_projection_and_provider_quarantine(tmp_path):
    source = tmp_path / 'input' / 'unrelated-film'; source.mkdir(parents=True)
    run = tmp_path / 'runs' / source.name
    run.mkdir(parents=True)
    event, _ = fixture()
    (run / 'visual_event_segments_v1.json').write_text(json.dumps({'events': [event]}))
    terminal_dir = run / 'semantic_observations/v2/terminal' / event['visual_event_id'] / 'PROVIDER_BLOCKED'
    terminal_dir.mkdir(parents=True)
    (terminal_dir / 'test.json').write_text(json.dumps({'event_id': event['visual_event_id'],
        'status': 'PROVIDER_BLOCKED', 'event_input_identity': policy.temporal.event_identity(event)}))
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    report = old.evaluate_cached_observations(source, policy_version=policy.POLICY_VERSION)
    assert report['counts']['PROVIDER_BLOCKED'] == 1
    assert report['additional_provider_calls_expected'] == 0
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert report['provider_requests'] == 0


def test_contract_contains_no_episode_specific_rule():
    text = Path(policy.__file__).read_text()
    assert 'mi-otra' not in text and 's03e03' not in text
    assert 'subtitles alone' in policy.PROMPT


def test_new_revision_cache_reusable_and_old_artifacts_not_rewritten(tmp_path):
    source = tmp_path / 'input' / 'generic-film'; source.mkdir(parents=True)
    run = tmp_path / 'runs' / source.name; run.mkdir(parents=True)
    event, record = fixture()
    (run / 'visual_event_segments_v1.json').write_text(json.dumps({'events': [event]}))
    (run / 'source_fingerprint.json').write_text(json.dumps({'movie_sha256': 'synthetic-sha'}))
    saved = policy.persist_visual_reuse(run, event, record['observation'], record['input_evidence'],
                                        source_movie_sha256='synthetic-sha', provenance={})
    assert policy.persist_visual_reuse(run, event, record['observation'], record['input_evidence'],
                                       source_movie_sha256='synthetic-sha', provenance={}) == saved
    assert policy.latest_visual_reuse_record(run, event) == saved
    assert policy.offline_projection(source)['counts']['KEEP'] == 1
    (run / 'source_fingerprint.json').write_text(json.dumps({'movie_sha256': 'changed-source'}))
    assert policy.latest_visual_reuse_record(run, event) is None


@pytest.mark.parametrize('kind,utility', [('reaction', 'reaction'), ('composition', 'environment_or_composition')])
def test_reactions_and_compositions_use_separate_moment_contract(kind, utility):
    event, record = fixture(utility=utility)
    record['observation']['visual_moments'][0]['moment_kind'] = kind
    assert policy.evaluate(record, event)['policy_decision'] == 'KEEP'


def test_unclear_action_completion_can_keep_a_valid_state():
    event, record = fixture(utility='walking_or_movement')
    record['observation']['action_or_moment_complete'] = 'unclear'
    record['observation']['visual_moments'][0]['action_completeness'] = 'unclear'
    assert policy.evaluate(record, event)['policy_decision'] == 'KEEP'


def test_unsupported_exact_bounds_and_unknown_samples_need_evidence():
    event, record = fixture()
    m = record['observation']['visual_moments'][0]
    m['sample_ids'] = m['sample_ids'][1:]
    assert policy.evaluate(record, event)['policy_decision'] == 'REVIEW'
    m['sample_ids'].append('INVENTED')
    assert policy.evaluate(record, event)['policy_decision'] == 'REVIEW'


def test_old_projection_never_used_as_raw_action_completeness():
    event, record = fixture()
    b = record['observation']
    for key in ('visual_moments', 'visual_assessment_complete', 'narrative_context_dependency'):
        b.pop(key)
    b['context_dependency'] = 'low'
    record['schema_version'] = policy.temporal.SCHEMA
    assert policy.evaluate(record, event)['policy_decision'] == 'KEEP'
    assert b['action_or_moment_complete'] == 'false'
    projected = policy.temporal.policy_projection({**record, 'evidence_catalog': {'actions': []}})
    assert policy.evaluate(projected, event)['policy_decision'] == 'REVIEW'


def test_legacy_production_application_cannot_silently_ignore_v2_windows(tmp_path):
    with pytest.raises(ValueError, match='asset-window migration'):
        old.apply_cached_policy(tmp_path / 'input' / 'film', [], policy_version=policy.POLICY_VERSION)


def historical(utility='useful_state', status='reusable_state', shots=1):
    event, record = fixture(shots=shots)
    for key in ('visual_moments', 'visual_assessment_complete', 'narrative_context_dependency'):
        record['observation'].pop(key)
    record['schema_version'] = policy.temporal.SCHEMA
    record['observation'].update(visual_utility_kind=utility, moment_status=status, context_dependency='medium')
    return event, record


def test_old_conversation_maps_category_without_inventing_strength():
    from movie_broll.policy_compatibility import project
    event, record = historical('generic_dialogue_only', 'unclear')
    projection = project(record, event)
    assert projection['fields']['utility_class']['value'] == 'conversation_scene'
    assert 'missing_action_or_sustained_state_assessment' in projection['missing_requirements']
    assert policy.evaluate(record, event)['policy_decision'] == 'REVIEW'


def test_old_physical_interaction_maps_with_completed_temporal_action():
    from movie_broll.policy_compatibility import project
    event, record = historical('physical_interaction', 'complete_action')
    record['observation']['action_or_moment_complete'] = 'true'
    before = copy.deepcopy(record)
    assert project(record, event)['fields']['utility_class']['value'] == 'physical_interaction'
    assert policy.evaluate(record, event)['policy_decision'] == 'KEEP'
    assert before == record


def test_old_useful_state_and_provenance():
    from movie_broll.policy_compatibility import project
    event, record = historical()
    before = copy.deepcopy(record)
    projected = project(record, event)
    assert projected['fields']['moment_usability']['derived_from'] == 'temporal_v2'
    assert projected['fields']['asset_window_eligibility']['derived_from'] == 'deterministic_event_evidence'
    record['schema_version'] = old.OBSERVATION_SCHEMA_VERSION
    projected = project(record, event)
    assert projected['fields']['utility_class']['derived_from'] == 'existing_observation'
    assert projected['visual_moments'] == []
    record['schema_version'] = before['schema_version']
    assert record == before


@pytest.mark.parametrize('change', [{'action_or_moment_complete': 'true'}, {'visible_states': []},
                                  {'visual_utility_kind': 'unclear'}, {'context_dependency': 'high'}])
def test_contradictory_or_missing_old_state_not_projected(change):
    from movie_broll.policy_compatibility import project
    event, record = historical()
    record['observation'].update(change)
    assert not project(record, event)['visual_moments']
    assert policy.evaluate(record, event)['policy_decision'] == 'REVIEW'


def test_state_with_unproven_cuts_selects_one_cited_shot():
    event, record = historical(shots=2)
    before = copy.deepcopy(record)
    result = policy.evaluate(record, event)
    assert result['policy_decision'] == 'KEEP'
    assert len(result['asset_windows']) == 1
    assert len(result['asset_windows'][0]['source_shot_ids']) == 1
    assert record == before


def test_window_handoff_resume_ids_and_close_registry(tmp_path):
    from movie_broll.asset_window_handoff import producer_candidates, registry_matches
    from movie_broll.finalization import asset_identity
    event, record = fixture(duration=20)
    event.update(timeline_ordinal=42, visual={'shot_focus_plan': record['observation']['shot_focus_plan']})
    rows = record['input_evidence']['samples']
    first = record['observation']['visual_moments'][0]
    first.update(end_frame_exclusive=rows[1]['frame']+1, sample_ids=[s['sample_id'] for s in rows[:2]])
    second = copy.deepcopy(first)
    second.update(moment_id='second', reuse_key='different moment', start_frame=rows[-2]['frame'],
                  end_frame_exclusive=rows[-1]['frame']+1, sample_ids=[s['sample_id'] for s in rows[-2:]])
    record['observation']['visual_moments'] = [first, second]
    decision = policy.evaluate(record, event)
    original = copy.deepcopy(event)
    candidates = producer_candidates(event, decision)
    assert len(candidates) == 2 and event == original
    assert len({c['visual_event_id'] for c in candidates}) == 2
    ids = [asset_identity(tmp_path, 'test-film', c) for c in candidates]
    assert ids[0] != ids[1]
    assert [asset_identity(tmp_path, 'test-film', c) for c in reversed(candidates)] == list(reversed(ids))
    registry = json.loads((tmp_path/'asset_registry.json').read_text())
    for (aid, slug), candidate in zip(ids, candidates):
        assert registry_matches(registry, aid, event['visual_event_id'], slug, candidate['producer_window_id'])
        assert not registry_matches(registry, aid, event['visual_event_id'], slug)
    duplicate = {**decision, 'asset_windows': decision['asset_windows']*2}
    assert producer_candidates(event, duplicate) == candidates
    assert producer_candidates(event, {'policy_decision': 'REVIEW'}) == []
    bad = copy.deepcopy(candidates[0]); bad['start_frame'] += 1
    with pytest.raises(ValueError, match='identity mismatch'):
        asset_identity(tmp_path, 'test-film', bad)


def test_compatibility_forged_citations_do_not_establish_support():
    from movie_broll.policy_compatibility import project
    event, record = historical()
    record['observation']['temporal_support_sample_ids'].append('FORGED')
    assert not project(record, event)['visual_moments']


def test_compatibility_moment_provenance_and_generic_implementation():
    from movie_broll import policy_compatibility, asset_window_handoff
    event, record = historical()
    record['observation_fingerprint'] = 'source-fingerprint'
    projection = policy_compatibility.project(record, event)
    origin = projection['moment_provenance'][0]
    assert origin['observation_fingerprint'] == 'source-fingerprint'
    assert origin['derived_from'] == ['temporal_v2', 'deterministic_event_evidence']
    for module in (policy_compatibility, asset_window_handoff):
        source = Path(module.__file__).read_text()
        assert 'mi-otra' not in source and 's03e03' not in source


def test_window_review_reconciliation_uses_own_ledger(tmp_path):
    from movie_broll.finalization import _persisted_review_plan, _persisted_review_qa
    source = 'VE_SOURCE'
    window = 'abc123'
    data = {'source_timeline': {'visual_event_id': source, 'producer_window_id': window}}
    ledger = {'events': {source: {'stages': {'vertical_reframe': {'plan': ['wrong']}}},
                        source+':window:'+window: {'stages': {'vertical_reframe': {'plan': ['window']},
                            'vertical_validation': {'validation': {'status': 'PASS'}}}}}}
    (tmp_path/'processing_ledger.json').write_text(json.dumps(ledger))
    assert _persisted_review_plan(tmp_path, data) == ['window']
    assert _persisted_review_qa(tmp_path, data) is not None


def test_subwindow_does_not_assert_uncited_eventwide_semantics():
    from movie_broll.asset_window_handoff import producer_candidates
    event, record = fixture(duration=30)
    event.update(visual={'actions': ['outside selected moment'], 'summary_es': 'whole event description',
                         'shot_focus_plan': record['observation']['shot_focus_plan']})
    candidate = producer_candidates(event, policy.evaluate(record, event))[0]
    assert 'actions' not in candidate['visual'] and 'summary_es' not in candidate['visual']
    assert candidate['source_event_visual_evidence']['actions'] == event['visual']['actions']
