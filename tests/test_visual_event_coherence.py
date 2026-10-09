import numpy as np

from movie_broll.broll_pilot import candidates, prepublication_event_coherence


def _shot(identifier, start, duration=4, *, narrative_ids=None, subtitle=.4, **extra):
    return {
        'shot_id': identifier,
        'start_seconds': start,
        'end_seconds': start + duration,
        'start_frame': round(start * 24),
        'end_frame_exclusive': round((start + duration) * 24),
        'duration_seconds': duration,
        'brightness_mean': 100,
        'sharpness_score': 100,
        'motion_score': 3,
        'near_black_fraction': 0,
        'subtitle_occupancy_ratio': subtitle,
        'narrative_segment_ids': narrative_ids or ['N1'],
        '_hist': np.ones((16, 16), np.float32),
        **extra,
    }


def test_camera_cut_conversation_without_scene_metadata_stays_one_event():
    events = candidates([_shot('speaker', 0), _shot('listener', 4)])
    assert [event['source_shot_ids'] for event in events] == [['speaker', 'listener']]
    assert 'camera_cut_not_treated_as_asset_boundary' in events[0]['grouping_reason']


def test_narrative_boundary_is_soft_and_event_carries_all_overlaps():
    events = candidates([
        _shot('speaker', 0, narrative_ids=['N1']),
        _shot('listener', 4, narrative_ids=['N2']),
    ])
    assert [event['source_shot_ids'] for event in events] == [['speaker', 'listener']]
    assert events[0]['narrative_segment_ids'] == ['N1', 'N2']


def test_photo_insert_to_unrelated_seated_scene_is_a_hard_split():
    events = candidates([
        _shot('family_tree', 0, subtitle=0, boundary_scene_change_score=.91,
              composition_signature='photo_wall', primary_subject_id='photo'),
        _shot('seated_man', 4, subtitle=0, boundary_scene_change_score=.91,
              composition_signature='seated_person', primary_subject_id='P9'),
    ])
    assert [event['source_shot_ids'] for event in events] == [['family_tree'], ['seated_man']]


def test_strong_visual_discontinuity_without_dialogue_or_interaction_splits():
    events = candidates([
        _shot('letter', 0, subtitle=0, _hist=np.arange(256, dtype=np.float32).reshape(16, 16)),
        _shot('bicycle', 4, subtitle=0, _hist=-np.arange(256, dtype=np.float32).reshape(16, 16)),
    ])
    assert [event['source_shot_ids'] for event in events] == [['letter'], ['bicycle']]


def test_short_orphan_is_rescued_only_into_its_immediate_coherent_neighbour():
    events = candidates([
        _shot('speaker', 0, 3.96),
        _shot('listener_reaction', 3.96, 3.0),
    ])
    assert [event['source_shot_ids'] for event in events] == [['speaker', 'listener_reaction']]
    assert events[0]['duration_seconds'] == 6.96


def test_autonomous_ten_second_event_is_not_absorbed_by_short_neighbour():
    events = candidates([
        _shot('reflection', 0, 10, subtitle=0, autonomous_broll_value=True),
        _shot('reaction', 10, 3.5, subtitle=.4),
    ])
    assert [event['source_shot_ids'] for event in events] == [['reflection'], ['reaction']]


def test_orphan_rescue_never_crosses_a_hard_break():
    events = candidates([
        _shot('photo_insert', 0, 4, subtitle=0, boundary_scene_change_score=.9),
        _shot('seated_man', 4, 3.5, subtitle=.4, boundary_scene_change_score=.9),
    ])
    assert [event['source_shot_ids'] for event in events] == [['photo_insert'], ['seated_man']]


def test_normal_event_ceiling_remains_eighteen_seconds():
    events = candidates([_shot(f'shot_{index}', index * 4) for index in range(5)])
    assert max(event['duration_seconds'] for event in events) <= 18


def test_prepublication_rejects_only_persisted_hard_breaks():
    event = candidates([_shot('a', 0), _shot('b', 4)])[0]
    assert prepublication_event_coherence(event)['status'] == 'PASS'
    event['continuity']['adjacent_pairs'][0]['merge'] = False
    assert prepublication_event_coherence(event)['status'] == 'SPLIT_REQUIRED'
