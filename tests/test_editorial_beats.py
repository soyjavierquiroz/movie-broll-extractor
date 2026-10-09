import numpy as np

from movie_broll.broll_pilot import technical_candidates
from movie_broll.editorial_beats import build_editorial_beats, scene_runs


def _shot(identifier, start, duration, *, action="reflecting", setting="therapy_room",
          context="emotional_exchange", **extra):
    return {
        "shot_id": identifier,
        "start_seconds": start,
        "end_seconds": start + duration,
        "start_frame": round(start * 24),
        "end_frame_exclusive": round((start + duration) * 24),
        "duration_seconds": duration,
        "brightness_mean": 100,
        "sharpness_score": 100,
        "motion_score": 3,
        "near_black_fraction": 0,
        "subtitle_occupancy_ratio": 0.4,
        "narrative_segment_ids": ["N1"],
        "_hist": np.ones((16, 16), np.float32),
        **extra,
    }


def _semantic(shots):
    items = technical_candidates(shots)
    for item, shot in zip(items, shots):
        item["visual"] = {"setting": shot.pop("semantic_setting", "therapy_room"),
                          "actions": [shot.pop("semantic_action", "reflecting")],
                          "visible_interactions": []}
        item["narrative"] = {"interaction_context": [shot.pop("semantic_context", "emotional_exchange")]}
        item["editorial"] = {"decision": "KEEP", "status": "VALIDATED"}
    return items


def _groups(items, shots):
    return [x["source_shot_ids"] for x in build_editorial_beats(items, shots)]


def test_short_speaker_and_listener_reaction_rescued_into_one_beat():
    shots = [
        _shot("speaker", 0, 3.9, semantic_action="speaking"),
        _shot("listener", 3.9, 3.0, semantic_action="listening"),
    ]
    assert _groups(_semantic(shots), shots) == [["speaker", "listener"]]


def test_complete_ten_second_reflective_shot_stays_autonomous():
    shots = [
        _shot("speaker", 0, 3.9, semantic_action="speaking"),
        _shot("listener", 3.9, 3.0, semantic_action="listening"),
        _shot("reflective", 6.9, 10.1, semantic_action="reflecting"),
    ]
    assert _groups(_semantic(shots), shots) == [["speaker", "listener"], ["reflective"]]


def test_speaker_plus_short_listener_reaction_may_merge():
    shots = [
        _shot("speaker", 0, 6.1, semantic_action="speaking"),
        _shot("listener", 6.1, 2.5, semantic_action="listening"),
    ]
    assert _groups(_semantic(shots), shots) == [["speaker", "listener"]]


def test_camera_change_inside_conversation_is_one_scene_run():
    scene = {"conversation_id": "therapy", "setting_id": "office", "participant_set_id": "A_B"}
    shots = [_shot("wide", 0, 4, scene_continuity=scene), _shot("close", 4, 4, scene_continuity=scene)]
    assert len(scene_runs(_semantic(shots), shots)) == 1


def test_unrelated_location_or_situation_always_splits_scene_runs():
    shots = [
        _shot("therapy", 0, 4, scene_continuity={"location_id": "office"}),
        _shot("park", 4, 4, scene_continuity={"location_id": "park", "temporal_jump_before": True}),
    ]
    assert len(scene_runs(_semantic(shots), shots)) == 2


def test_full_therapy_scene_partitions_into_multiple_bounded_beats():
    shots = [
        _shot("a", 0, 3.9, semantic_action="speaking"),
        _shot("b", 3.9, 3.0, semantic_action="listening"),
        _shot("c", 6.9, 10.1, semantic_action="reflecting"),
        _shot("d", 17.0, 6.2, semantic_action="speaking"),
        _shot("e", 23.2, 12.0, semantic_action="listening"),
    ]
    beats = build_editorial_beats(_semantic(shots), shots)
    assert len(beats) == 3
    assert [x["source_shot_ids"] for x in beats] == [["a", "b"], ["c"], ["d", "e"]]
    assert max(x["duration_seconds"] for x in beats) <= 20


def test_continuous_therapy_scene_crosses_narrative_boundaries_without_fragmenting_run():
    shots = [
        _shot("a", 0, 3.9, narrative_segment_ids=["NARR_001"], semantic_action="speaking"),
        _shot("b", 3.9, 3.0, narrative_segment_ids=["NARR_001"], semantic_action="listening"),
        _shot("c", 6.9, 10.1, narrative_segment_ids=["NARR_002"], semantic_action="reflecting"),
        _shot("d", 17.0, 6.2, narrative_segment_ids=["NARR_002"], semantic_action="speaking"),
        _shot("e", 23.2, 12.0, narrative_segment_ids=["NARR_003"], semantic_action="listening"),
    ]
    items = _semantic(shots)

    runs = scene_runs(items, shots)
    beats = build_editorial_beats(items, shots)

    assert len(runs) == 1
    assert runs[0]["narrative_segment_ids"] == ["NARR_001", "NARR_002", "NARR_003"]
    assert [x["source_shot_ids"] for x in beats] == [["a", "b"], ["c"], ["d", "e"]]
    assert beats[2]["narrative_segment_ids"] == ["NARR_002", "NARR_003"]
    assert beats[2]["scene_run"]["narrative_segment_ids"] == ["NARR_001", "NARR_002", "NARR_003"]


def test_narrative_boundary_with_actual_location_or_time_change_still_splits_scene_run():
    shots = [
        _shot("office", 0, 4, narrative_segment_ids=["NARR_001"],
              scene_continuity={"location_id": "office"}),
        _shot("street", 4, 4, narrative_segment_ids=["NARR_002"],
              scene_continuity={"location_id": "street", "temporal_jump_before": True}),
    ]

    assert len(scene_runs(_semantic(shots), shots)) == 2


def test_short_visually_valuable_autonomous_shot_can_remain_standalone():
    shots = [
        _shot("gesture", 0, 3.5, semantic_action="gesture", autonomous_broll_value=True),
        _shot("reflection", 3.5, 9.0, semantic_action="reflecting"),
    ]
    assert _groups(_semantic(shots), shots) == [["gesture"], ["reflection"]]
