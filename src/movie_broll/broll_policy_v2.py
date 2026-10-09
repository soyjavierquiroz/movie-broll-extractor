"""Visual reuse policy and coherent window planning. Pure, offline, opt-in.

Historical observation contracts are immutable. V2 replay never promotes prose,
subtitle hints, or legacy projected completeness to temporal evidence.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal
import math

from pydantic import Field

from . import semantic_observations as old
from . import temporal_semantics as temporal
from .processing_ledger import fingerprint
from .utils import write_json

POLICY_VERSION = 'broll_policy_v2'
OBSERVATION_SCHEMA = 'semantic_observation_v2_visual_reuse'
PROMPT_VERSION = 'semantic_observation_prompt_v2_visual_reuse'
UTILITY_CLASSES = (
    'conversation_scene', 'therapy_or_session', 'discussion_or_argument',
    'meditation_or_stillness', 'reaction', 'sustained_emotional_state',
    'physical_interaction', 'nonverbal_interaction', 'walking_or_movement',
    'concrete_action', 'object_interaction', 'useful_state',
    'environment_or_composition', 'establishing_visual',
)
ALIASES = {'clear_reaction': 'reaction', 'movement': 'walking_or_movement',
           'object_activity': 'object_interaction', 'environment': 'environment_or_composition',
           'strong_nonverbal_interaction': 'nonverbal_interaction'}
Utility = Literal[*UTILITY_CLASSES]


class VisualMoment(old._StrictObservation):
    moment_id: str
    utility_class: Utility
    visual_strength: Literal['useful', 'weak', 'unclear']
    moment_kind: Literal['complete_action', 'sustained_state', 'reaction', 'composition', 'unclear']
    # Separate raw action completeness from usefulness and window completeness.
    action_completeness: Literal['true', 'false', 'unclear']
    moment_usability: Literal['usable', 'unusable', 'unclear']
    asset_window_complete: Literal['true', 'false', 'unclear']
    visual_context_dependency: Literal['low', 'medium', 'high', 'unclear']
    start_frame: int
    end_frame_exclusive: int
    beginning_sample_id: str | None
    development_sample_ids: list[str]
    completion_sample_id: str | None
    sample_ids: list[str]
    continuity: Literal['supported', 'unsupported', 'unclear']
    # Same key means same visual need, even across distinct event identities.
    reuse_key: str = Field(min_length=1)


class VisualReuseObservation(temporal.TemporalObservationResult):
    """Additive revision: retain existing ontology and explicitly assess moments."""
    visual_moments: list[VisualMoment]
    visual_assessment_complete: bool
    narrative_context_dependency: Literal['low', 'medium', 'high', 'unclear']


RESPONSE_SCHEMA = VisualReuseObservation.model_json_schema()
PROMPT = old.OBSERVATION_PROMPT.replace('semantic_observation_v1', OBSERVATION_SCHEMA) + '''
Use actual temporal images, never subtitles alone, to assess visual reuse.
Explicitly assess conversation/session composition, discussion/argument, stillness,
listening, sustained emotions, reactions, physical contact, object activity,
walking/movement, environmental/architectural composition and local actions.
Retain existing visual_utility_kind ontology; visual_moments adds distinctions.
Dialogue is allowed; flat weak talking heads may be visually weak. Story context
is separate from visual_context_dependency (whether the IMAGE is understandable).
Do not claim raw action completeness for a sustained state. For the event summary
use reusable_state with useful_state and raw completeness false/unclear when apt.
Cite supplied sample IDs only. A local complete action needs actual beginning,
development and completion samples inside its own temporal range, NOT event
endpoints. Specify exact frame bounds within the event; end is exclusive.
For every distinct reusable moment provide visual strength, moment usability,
asset-window completeness and continuity across every included technical cut.
Sparse samples do not prove unseen transitions or exact completion. Use unclear
where coverage cannot establish those facts. Do not invent actions to meet length.
A coherent sustained state can supply a representative 5–20 second window.
Never truncate a complete action or subdivide one moment just to meet duration.
Provide stable reuse_key describing the specific visual need and composition;
near-duplicates share a key. Multiple moments require genuinely distinct needs.
Empty moments are valid for weak footage; visual_assessment_complete=true means
all the listed visual signals were assessed, not that any useful one was found.
Return action_evidence_ids=[]; IDs are assigned locally after grounding.
'''


def validate_visual_reuse(body, event, evidence):
    """Revision-specific validation; leaves historical V2 validators unchanged."""
    try:
        parsed = VisualReuseObservation.model_validate(body)
    except ValueError as error:
        return [str(error)]
    base = parsed.model_dump(exclude={'visual_moments', 'visual_assessment_complete',
                                      'narrative_context_dependency'})
    # Preserve all historical identity/catalog checks except the event-wide action
    # endpoint rule. Local complete actions are validated separately below.
    errors = temporal.validate_response(base, event, evidence)
    errors = [e for e in errors if e != 'complete_action_requires_begin_development_end']
    if base['moment_status'] == 'complete_action' and not any(
            m.moment_kind == 'complete_action' for m in parsed.visual_moments):
        errors.append('complete_action_requires_local_moment')
    ids = [m.moment_id for m in parsed.visual_moments]
    if len(ids) != len(set(ids)):
        errors.append('duplicate_moment_id')
    for moment in parsed.visual_moments:
        errors.extend(validate_moment(moment.model_dump(), event, evidence))
    return errors


def validate_moment(moment, event, evidence):
    errors = []
    try:
        m = VisualMoment.model_validate(moment).model_dump()
        samples = {s['sample_id']: s for s in evidence['samples']}
        start, end = m['start_frame'], m['end_frame_exclusive']
        if not event['start_frame'] <= start < end <= event['end_frame_exclusive']:
            errors.append('moment_outside_event')
        ids = m['sample_ids']
        if len(ids) != len(set(ids)) or any(i not in samples for i in ids):
            return errors + ['invalid_moment_sample_ids']
        rows = [samples[i] for i in ids]
        if len({s['frame'] for s in rows}) < 2 or any(not start <= s['frame'] < end for s in rows):
            errors.append('insufficient_local_temporal_support')
        if rows and (min(s['frame'] for s in rows) > start or max(s['frame'] for s in rows) < end - 1):
            errors.append('moment_bounds_without_visual_support')
        if m['moment_kind'] == 'complete_action':
            begin, finish = m['beginning_sample_id'], m['completion_sample_id']
            middle = m['development_sample_ids']
            if (m['action_completeness'] != 'true' or begin not in ids or finish not in ids
                    or not middle or any(i not in ids for i in middle)):
                errors.append('local_action_requires_begin_development_completion')
            elif not (samples[begin]['frame'] == start and samples[finish]['frame'] == end - 1
                      and all(start < samples[i]['frame'] < end - 1 for i in middle)):
                errors.append('local_action_temporal_order')
        elif m['moment_kind'] == 'sustained_state' and m['action_completeness'] == 'true':
            errors.append('sustained_state_cannot_claim_action_completion')
        shots = _shots(event, start, end, float(evidence['fps']))
        if set(shots) - {s['shot_id'] for s in rows}:
            errors.append('missing_moment_shot_support')
        if len(shots) > 1 and m['continuity'] != 'supported':
            errors.append('multishot_continuity_unproven')
    except (KeyError, TypeError, ValueError):
        errors.append('invalid_visual_moment')
    return errors


def _shots(event, start, end, fps):
    shots = event.get('technical_shots', [])
    if not shots and len(event['source_shot_ids']) == 1:
        return list(event['source_shot_ids'])
    return [s['shot_id'] for s in shots
            if (s['start_frame'] if 'start_frame' in s else round(s['start_seconds'] * fps)) < end
            and (s['end_frame_exclusive'] if 'end_frame_exclusive' in s else round(s['end_seconds'] * fps)) > start]


def plan_asset_windows(event, moments, evidence, *, selected=()):
    """One window per distinct moment; never fill a quota or tile a long event.

    Complete actions are preserved whole (even outside target, with an explicit
    duration exception). Long states use a central 20s window only when the entire
    cited span and all cuts are supported. Short moments stay whole as exceptions.
    A same-key window or >=80% overlap is suppressed, within/across events.
    """
    fps = float(evidence['fps'])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('positive finite fps required')
    accepted = []
    review = []
    suppressed = []
    for moment in moments:
        errors = validate_moment(moment, event, evidence)
        if errors:
            review.append({'moment_id': moment.get('moment_id'), 'reasons': errors}); continue
        m = moment
        if m['visual_strength'] == 'weak' or m['moment_usability'] == 'unusable':
            continue
        if (m['visual_strength'] != 'useful' or m['moment_usability'] != 'usable'
                or m['asset_window_complete'] != 'true' or m['moment_kind'] == 'unclear'
                or m['visual_context_dependency'] in {'high', 'unclear'}):
            review.append({'moment_id': m['moment_id'], 'reasons': ['visual_window_not_established']}); continue
        start, end = m['start_frame'], m['end_frame_exclusive']
        duration = (end - start) / fps
        if duration > 20 and m['moment_kind'] in {'sustained_state', 'composition'}:
            length = max(1, math.floor(20 * fps))
            start += (end - start - length) // 2
            end = start + length
        elif duration > 20 and m['moment_kind'] != 'complete_action':
            review.append({'moment_id': m['moment_id'], 'reasons': ['long_moment_requires_supported_boundary']}); continue
        from .asset_window_handoff import window_identity
        window = {'producer_window_id': window_identity(event['visual_event_id'], start, end), 'event_id': event['visual_event_id'], 'moment_id': m['moment_id'],
                  'utility_class': m['utility_class'], 'reuse_key': m['reuse_key'],
                  'start_frame': start, 'end_frame_exclusive': end,
                  'start_seconds': start / fps, 'end_seconds': end / fps,
                  'duration_seconds': (end - start) / fps, 'source_shot_ids': _shots(event, start, end, fps),
                  'action_completeness': m['action_completeness'],
                  'moment_usability': m['moment_usability'], 'asset_window_complete': m['asset_window_complete'],
                  'duration_exception': not 5 <= (end - start) / fps <= 20}
        duplicate = False
        for prior in [*selected, *accepted]:
            overlap = max(0, min(end, prior['end_frame_exclusive']) - max(start, prior['start_frame']))
            fraction = overlap / min(end - start, prior['end_frame_exclusive'] - prior['start_frame'])
            if prior['reuse_key'] == window['reuse_key'] or fraction >= .8:
                duplicate = True; break
        if duplicate:
            suppressed.append({'moment_id': m['moment_id'], 'reason': 'duplicate_or_near_duplicate'}); continue
        accepted.append(window)
    return {'windows': accepted, 'review': review, 'suppressed': suppressed}


def replay_moments(record, event):
    """Conservative structural adapter. Never search descriptive prose for labels."""
    b = record['observation']
    if 'visual_moments' in b:
        return b['visual_moments'], []
    if record.get('schema_version') != temporal.SCHEMA:
        return [], ['missing_temporal_visual_moments']
    samples = {s['sample_id']: s for s in record['input_evidence']['samples']}
    rows = [samples[i] for i in b['temporal_support_sample_ids'] if i in samples]
    utility = ALIASES.get(b['visual_utility_kind'], b['visual_utility_kind'])
    if utility not in UTILITY_CLASSES:
        return [], ['legacy_generic_class_did_not_assess_visual_reuse']
    if b['context_dependency'] in {'high', 'unclear'}:
        return [], ['visual_vs_narrative_context_unresolved']
    # Explicit useful-state judgment establishes state usability. Mere action
    # truncation does not establish a reusable state or local action boundaries.
    status = b['moment_status']
    if status not in {'reusable_state', 'complete_action'}:
        return [], ['local_moment_or_state_usability_unresolved']
    if len({s['frame'] for s in rows}) < 2:
        return [], ['insufficient_temporal_visual_support']
    start, end = min(s['frame'] for s in rows), max(s['frame'] for s in rows) + 1
    shots = _shots(event, start, end, float(record['input_evidence']['fps']))
    # Historical event merging is not proof of visually continuous asset utility.
    if len(shots) > 1:
        pairs = event.get('continuity', {}).get('adjacent_pairs', [])
        relevant = [p for p in pairs if p.get('from_shot_id') in shots and p.get('to_shot_id') in shots]
        if len(relevant) != len(shots) - 1 or any(not any(p.get('evidence', {}).get(k) is True
                for k in ('same_action', 'same_interaction', 'same_establishing_sequence',
                          'validated_shot_reverse_shot')) for p in relevant):
            return [], ['asset_cut_continuity_requires_visual_assessment']
    ordered = sorted(rows, key=lambda s: s['frame'])
    m = {'moment_id': event['visual_event_id'] + ':saved_moment', 'utility_class': utility,
         'visual_strength': 'useful', 'moment_kind': 'sustained_state' if status == 'reusable_state' else 'complete_action',
         'action_completeness': b['action_or_moment_complete'], 'moment_usability': 'usable',
         'asset_window_complete': 'true', 'visual_context_dependency': b['context_dependency'],
         'start_frame': start, 'end_frame_exclusive': end,
         'beginning_sample_id': ordered[0]['sample_id'] if status == 'complete_action' else None,
         'completion_sample_id': ordered[-1]['sample_id'] if status == 'complete_action' else None,
         'development_sample_ids': [s['sample_id'] for s in ordered[1:-1]] if status == 'complete_action' else [],
         'sample_ids': [s['sample_id'] for s in rows], 'continuity': 'supported',
         # Cannot establish cross-event equivalence from old observations. Use
         # exact structured state identity; temporal overlap is also checked.
         'reuse_key': repr((event['visual_event_id'], utility, b['visible_states'], b['physical_interactions'], b['visible_reactions']))}
    return [m], []


def evaluate(record, event=None, *, selected=(), compatibility=True):
    def result(decision, reasons, **extra):
        return {'policy_version': POLICY_VERSION, 'policy_decision': decision,
                'policy_reasons': reasons, 'policy_cost_usd': 0.0, 'provider_requests': 0, **extra}
    b = record.get('observation', {})
    flags = b.get('technical_observations', {})
    hard = [f for f in old.HARD_REJECT_FLAGS if flags.get(f) is True]
    if hard:
        return result('REJECT', ['hard_exclusion:' + f for f in hard], asset_windows=[])
    if any(type(flags.get(f)) is not bool for f in old.HARD_REJECT_FLAGS) or event is None:
        return result('REVIEW', ['observation_or_event_missing'], targeted_reobservation=True)
    if b.get('event_id') != event['visual_event_id']:
        return result('REVIEW', ['observation_event_mismatch'], targeted_reobservation=True)
    if 'visual_moments' in b:
        errors = validate_visual_reuse(b, event, record.get('input_evidence', {}))
        if errors:
            return result('REVIEW', errors, targeted_reobservation=True)
    projection = None
    if compatibility and 'visual_moments' not in b:
        from .policy_compatibility import project
        projection = project(record, event)
        moments, reasons = projection['visual_moments'], projection['missing_requirements']
    else:
        moments, reasons = replay_moments(record, event)
    if reasons:
        return result('REVIEW', reasons, targeted_reobservation=True, compatibility_projection=projection)
    plan = plan_asset_windows(event, moments, record['input_evidence'], selected=selected)
    if plan['windows']:
        return result('KEEP', ['visually_reusable_moment'], asset_windows=plan['windows'],
                      window_review=plan['review'], suppressed=plan['suppressed'], compatibility_projection=projection,
                      targeted_reobservation=bool(plan['review']))
    if plan['review'] or (not plan['suppressed'] and not b.get('visual_assessment_complete', False)):
        return result('REVIEW', ['visual_window_not_established'], window_review=plan['review'],
                      targeted_reobservation=True)
    return result('REJECT', ['duplicate_or_weak_visual_utility'], asset_windows=[], suppressed=plan['suppressed'])



def persist_visual_reuse(run, event, body, evidence, *, source_movie_sha256, provenance):
    """Explicit persistence boundary for the new observation revision only.

    No provider or production runner is constructed here. Callers supply an
    already received observation; old records, response archives and events stay
    immutable. Prompt/schema identities keep this cache separate from old V2.
    """
    errors = validate_visual_reuse(body, event, evidence)
    if errors:
        raise ValueError('invalid visual reuse observation: ' + '; '.join(errors))
    identity = {'schema_version': OBSERVATION_SCHEMA, 'prompt_version': PROMPT_VERSION,
                'event_input_identity': temporal.event_identity(event),
                'source_movie_sha256': source_movie_sha256, 'input_evidence': evidence,
                'observation': body}
    observation_fp = fingerprint(identity)
    eid = event['visual_event_id']
    if Path(eid).name != eid or eid in {'', '.', '..'}:
        raise ValueError('invalid event identity path')
    path = run / 'semantic_observations/v2_visual_reuse/events' / eid / (observation_fp + '.json')
    record = {**identity, 'event_id': eid, 'observation_fingerprint': observation_fp,
              'provider_provenance': provenance, 'created_at': old._utc()}
    if path.exists():
        existing = old._read(path)
        if any(existing.get(k) != v for k, v in identity.items()):
            raise ValueError('immutable visual reuse cache conflict')
        return existing
    write_json(path, record)
    return record


def latest_visual_reuse_record(run, event):
    source = run / 'source_fingerprint.json'
    source_sha = old._read(source).get('movie_sha256') if source.is_file() else None
    eid = event['visual_event_id']
    if Path(eid).name != eid or eid in {'', '.', '..'}:
        raise ValueError('invalid event identity path')
    records = []
    for path in (run / 'semantic_observations/v2_visual_reuse/events' / eid).glob('*.json'):
        try:
            record = old._read(path)
            identity = {k: record[k] for k in ('schema_version', 'prompt_version', 'event_input_identity',
                       'source_movie_sha256', 'input_evidence', 'observation')}
            if (record['schema_version'] == OBSERVATION_SCHEMA and record['prompt_version'] == PROMPT_VERSION
                    and record['event_id'] == eid and isinstance(record['created_at'], str)
                    and record['event_input_identity'] == temporal.event_identity(event)
                    and (source_sha is None or record['source_movie_sha256'] == source_sha)
                    and record['observation_fingerprint'] == fingerprint(identity)
                    and not validate_visual_reuse(record['observation'], event, record['input_evidence'])):
                records.append(record)
        except (OSError, KeyError, TypeError, ValueError):
            continue
    return max(records, key=lambda r: (r['created_at'], r['observation_fingerprint']), default=None)

def offline_projection(input_dir: Path, *, compatibility=True):
    """Read only. No provider construction, production, sidecars, or rendering."""
    run = old.run_dir(input_dir)
    events = old._read(run / 'visual_event_segments_v1.json')['events']
    counts = Counter(); decisions = {}; candidates = []; selected = []; cached = 0
    terminals = {}
    for path in (run / 'semantic_observations/v2').rglob('*.json'):
        if 'terminal' not in path.parts and 'historical_provider_blocks' not in path.parts:
            continue
        try:
            terminal = old._read(path)
            if terminal.get('status') == 'PROVIDER_BLOCKED':
                terminals[terminal['event_id']] = terminal
        except (OSError, ValueError, KeyError):
            continue
    for event in events:
        eid = event['visual_event_id']
        from .visual_utility_resolution import latest_record as latest_utility_record, policy_record
        utility_record = latest_utility_record(run, event)
        record = policy_record(utility_record) if utility_record else latest_visual_reuse_record(run, event) or temporal.latest_record(run, event)
        if record is None:
            try:
                record = old._read(old.observation_path(run, eid))
                if old.validate_observation(record, event):
                    record = None
            except (OSError, ValueError):
                record = None
        if record is None and eid in terminals and (terminals[eid].get('event_input_identity') or terminals[eid].get('input_identity', {}).get('event_input_identity')) == temporal.event_identity(event):
            decision = {'policy_version': POLICY_VERSION, 'policy_decision': 'PROVIDER_BLOCKED',
                        'policy_reasons': ['provider_input_quarantine'], 'targeted_reobservation': False}
        elif record is None:
            decision = {'policy_version': POLICY_VERSION, 'policy_decision': 'REVIEW',
                        'policy_reasons': ['missing_valid_observation'], 'targeted_reobservation': True}
        else:
            cached += 1
            decision = evaluate(record, event, selected=selected, compatibility=compatibility)
            if utility_record:
                decision['targeted_reobservation'] = False
                if utility_record['observation']['visually_reusable'] == 'unclear':
                    decision.update(policy_decision='REVIEW', policy_reasons=['genuine_visual_utility_ambiguity'], asset_windows=[])
        block = terminals.get(eid, {})
        block_identity = block.get('event_input_identity') or block.get('input_identity', {}).get('event_input_identity')
        if decision.get('targeted_reobservation') and block_identity == temporal.event_identity(event):
            decision = {**decision, 'policy_decision': 'PROVIDER_BLOCKED',
                        'policy_reasons': ['provider_input_quarantine', *decision['policy_reasons']],
                        'targeted_reobservation': False}
        decisions[eid] = decision; counts[decision['policy_decision']] += 1
        selected.extend(decision.get('asset_windows', []))
        if decision.get('targeted_reobservation'):
            candidates.append({'event_id': eid, 'reasons': decision['policy_reasons']})
    return {'schema_version': 'broll_policy_v2_offline_projection', 'policy_version': POLICY_VERSION,
            'total_events': len(events), 'cached': cached, 'missing': len(events) - cached,
            'counts': {k: counts[k] for k in ('KEEP', 'REVIEW', 'REJECT', 'PROVIDER_BLOCKED')},
            'asset_window_count': len(selected), 'multi_window_event_count': sum(len(d.get('asset_windows', [])) > 1 for d in decisions.values()), 'decisions': decisions,
            'targeted_reobservation': candidates, 'additional_provider_calls_expected': len(candidates),
            'additional_provider_calls_validation_budget': 2 * len(candidates),
            'projection_is_provisional': bool(candidates), 'provider_requests': 0,
            'api_cost_usd': 0.0, 'artifacts_modified': False}
