"""Pure producer handoff: distinct windows own ledger and package identities."""
from __future__ import annotations
import copy
from .processing_ledger import fingerprint


def window_identity(event_id, start, end):
    if not isinstance(event_id, str) or not event_id or type(start) is not int or type(end) is not int or start >= end:
        raise ValueError('invalid window identity')
    return fingerprint({'version': 'producer_window_v1', 'source_event_id': event_id,
                        'start_frame': start, 'end_frame_exclusive': end})[:24]


def producer_candidates(event, decision):
    """Materialize reviewable render inputs without writing or rendering anything.

    IDs depend on source identity and final frame range, never selected-list order.
    Semantics and focus directives are reused; missing focus stays a local error.
    """
    if decision.get('policy_decision') != 'KEEP':
        return []
    candidates, seen = [], set()
    for window in decision.get('asset_windows', []):
        start, end = window['start_frame'], window['end_frame_exclusive']
        if window['event_id'] != event['visual_event_id'] or not event['start_frame'] <= start < end <= event['end_frame_exclusive']:
            raise ValueError('window/source mismatch')
        key = window_identity(event['visual_event_id'], start, end)
        if key in seen:
            continue
        seen.add(key)
        candidate = copy.deepcopy(event)
        candidate.update(visual_event_id=event['visual_event_id'] + ':window:' + key,
                         candidate_id='WINDOW_' + key, source_visual_event_id=event['visual_event_id'],
                         producer_window_id=key, start_frame=start, end_frame_exclusive=end,
                         start_seconds=window['start_seconds'], end_seconds=window['end_seconds'],
                         duration_seconds=window['duration_seconds'], source_shot_ids=list(window['source_shot_ids']))
        candidate['technical_shots'] = []
        for shot in event.get('technical_shots', []):
            if shot['shot_id'] not in window['source_shot_ids']:
                continue
            fps = (end-start)/(window['end_seconds']-window['start_seconds'])
            left = shot.get('start_frame', round(shot.get('start_seconds', start/fps)*fps))
            right = shot.get('end_frame_exclusive', round(shot.get('end_seconds', end/fps)*fps))
            left, right = max(start, left), min(end, right)
            candidate['technical_shots'].append({**shot, 'start_frame': left, 'end_frame_exclusive': right,
                                                'start_seconds': left/fps, 'end_seconds': right/fps})
        candidate['source_event_visual_evidence'] = copy.deepcopy(candidate.get('visual', {}))
        if start != event['start_frame'] or end != event['end_frame_exclusive']:
            # Event-wide prose/actions are not assertions about a selected subrange.
            candidate['visual'] = {'shot_focus_plan': candidate.get('visual', {}).get('shot_focus_plan', [])}
            for field in ('people', 'semantic_people', 'relationships'):
                candidate.pop(field, None)
            candidate.get('editorial', {}).pop('standalone_meaning_es', None)
        visual = candidate.setdefault('visual', {})
        visual['shot_focus_plan'] = [d for d in visual.get('shot_focus_plan', []) if d['shot_id'] in window['source_shot_ids']]
        continuity = candidate.setdefault('continuity', {})
        continuity['adjacent_pairs'] = [p for p in continuity.get('adjacent_pairs', [])
            if p['from_shot_id'] in window['source_shot_ids'] and p['to_shot_id'] in window['source_shot_ids']]
        candidate['editorial'] = {**candidate.get('editorial', {}), 'decision': 'KEEP', 'status': 'VALIDATED',
                                  'policy_version': decision['policy_version'], 'asset_window': copy.deepcopy(window)}
        candidates.append(candidate)
    return candidates


def registry_matches(registry, asset_id, source_event_id, slug, window_id=None):
    matches = [(key, entry) for key, entry in registry.get('events', {}).items() if entry.get('asset_id') == asset_id]
    return (len(matches) == 1 and matches[0][1].get('source_visual_event_id', matches[0][0]) == source_event_id
            and matches[0][1].get('slug') == slug and matches[0][1].get('producer_window_id') == window_id)
