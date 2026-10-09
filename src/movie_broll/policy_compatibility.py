"""Read-only, partial evidence projection. Samples are coverage, not semantics."""
from __future__ import annotations

from .processing_ledger import fingerprint

VERSION = 'policy_v2_compatibility_v1'


def project(record, event):
    from . import broll_policy_v2 as p
    b = record.get('observation', {})
    temporal = record.get('schema_version') == p.temporal.SCHEMA
    origin = 'temporal_v2' if temporal else 'existing_observation'
    fields = {}
    def put(key, value, paths, source=origin):
        fields[key] = {'value': value, 'derived_from': source, 'source_fields': paths,
                       'observation_fingerprint': record.get('observation_fingerprint')}
    utility = p.ALIASES.get(b.get('visual_utility_kind'), b.get('visual_utility_kind'))
    # Conversation presence establishes a category, never visual strength or duration.
    if utility == 'generic_dialogue_only' and b.get('conversation_present') == 'true':
        utility = 'conversation_scene'
    if utility in p.UTILITY_CLASSES:
        put('utility_class', utility, ['visual_utility_kind', 'conversation_present'])
    if b.get('context_dependency') in {'low', 'medium'}:
        put('visual_context_dependency', b['context_dependency'], ['context_dependency'])
    evidence = record.get('input_evidence', {})
    ids = b.get('temporal_support_sample_ids', [])
    rows = [s for s in evidence.get('samples', []) if s['sample_id'] in ids]
    samples = {s['sample_id']: s for s in evidence.get('samples', [])}
    if len(ids) == len(set(ids)) and all(i in samples for i in ids) and len({s['frame'] for s in rows}) >= 2:
        put('temporal_support', ids, ['temporal_support_sample_ids', 'input_evidence.samples'])
    status = b.get('moment_status')
    if temporal and status == 'reusable_state' and b.get('visual_utility_kind') == 'useful_state' and b.get('visible_states') and b.get('action_or_moment_complete') in {'false', 'unclear'}:
        put('moment_usability', 'usable', ['moment_status', 'visible_states', 'visual_utility_kind'])
        put('moment_kind', 'sustained_state', ['moment_status'])
    elif temporal and status == 'complete_action' and b.get('action_or_moment_complete') == 'true':
        put('moment_usability', 'usable', ['moment_status'])
        put('moment_kind', 'complete_action', ['moment_status'])
    missing = []
    for key, reason in [('utility_class', 'missing_moment_utility_category'),
                        ('temporal_support', 'missing_temporal_support'),
                        ('visual_context_dependency', 'missing_context_dependency_interpretation'),
                        ('moment_usability', 'missing_action_or_sustained_state_assessment')]:
        if key not in fields:
            missing.append(reason)
    moments = []
    if not missing:
        moments, reasons = p.replay_moments(record, event)
        if reasons == ['asset_cut_continuity_requires_visual_assessment'] and status == 'reusable_state':
            # Preserve one state identity: choose the longest cited within-shot span.
            # Never turn every camera angle of the same state into another asset.
            groups = [[s for s in rows if s['shot_id'] == sid] for sid in event['source_shot_ids']]
            groups = [g for g in groups if len({s['frame'] for s in g}) >= 2]
            if groups:
                span = max(groups, key=lambda g: (max(s['frame'] for s in g)-min(s['frame'] for s in g), -min(s['frame'] for s in g)))
                start, end = min(s['frame'] for s in span), max(s['frame'] for s in span)+1
                m = {'moment_id': event['visual_event_id'] + ':saved_state', 'utility_class': utility,
                     'visual_strength': 'useful', 'moment_kind': 'sustained_state',
                     'action_completeness': b['action_or_moment_complete'], 'moment_usability': 'usable',
                     'asset_window_complete': 'true', 'visual_context_dependency': b['context_dependency'],
                     'start_frame': start, 'end_frame_exclusive': end, 'beginning_sample_id': None,
                     'completion_sample_id': None, 'development_sample_ids': [],
                     'sample_ids': [s['sample_id'] for s in span], 'continuity': 'supported',
                     'reuse_key': fingerprint({'event': event['visual_event_id'], 'state': b['visible_states']})}
                if not p.validate_moment(m, event, evidence):
                    moments, reasons = [m], []
        missing.extend(reasons)
    if moments:
        put('asset_window_eligibility', True, ['event.technical_shots', 'temporal_support_sample_ids'], 'deterministic_event_evidence')
    return {'moment_provenance': [{'moment_id': m['moment_id'], 'derived_from': ['temporal_v2', 'deterministic_event_evidence'], 'observation_fingerprint': record.get('observation_fingerprint'), 'sample_ids': m['sample_ids'], 'event_input_identity': p.temporal.event_identity(event)} for m in moments], 'version': VERSION, 'fields': fields, 'visual_moments': moments,
            'missing_requirements': missing, 'provider_requests': 0}
