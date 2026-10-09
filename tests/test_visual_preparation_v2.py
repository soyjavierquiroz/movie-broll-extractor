import json

import cv2
import numpy as np
import pytest

from movie_broll import broll_pilot, production


def _shot(shot_id, start, end):
    return {'shot_id': shot_id, 'start_frame': start, 'end_frame_exclusive': end,
            'start_seconds': start / 24, 'end_seconds': end / 24,
            'duration_seconds': (end - start) / 24}


def test_long_shots_become_balanced_frame_exact_analysis_units_and_keep_parent_identity():
    long_213 = _shot('FULL_SHOT_0392', 52605, 57722)
    long_71 = _shot('FULL_SHOT_0394', 57764, 59469)
    units = production.derive_visual_analysis_units([long_213, long_71], 24)
    by_parent = {parent: [x for x in units if x['parent_shot_id'] == parent]
                 for parent in ('FULL_SHOT_0392', 'FULL_SHOT_0394')}
    assert [len(by_parent['FULL_SHOT_0392']), len(by_parent['FULL_SHOT_0394'])] == [12, 4]
    for parent, original in ((long_213, by_parent['FULL_SHOT_0392']), (long_71, by_parent['FULL_SHOT_0394'])):
        assert original[0]['start_frame'] == parent['start_frame']
        assert original[-1]['end_frame_exclusive'] == parent['end_frame_exclusive']
        assert all(x['duration_seconds'] <= 18 for x in original)
        assert all(x['derived_from_long_shot'] and x['shot_id'] == parent['shot_id'] for x in original)
        assert all(left['end_frame_exclusive'] == right['start_frame'] for left, right in zip(original, original[1:]))
        lengths = [x['end_frame_exclusive'] - x['start_frame'] for x in original]
        assert max(lengths) - min(lengths) <= 1  # no tiny tail remainder


def test_candidate_from_subrange_preserves_one_canonical_technical_shot():
    units = production.derive_visual_analysis_units([_shot('S_LONG', 0, 1705)], 24)
    for unit in units:
        unit.update(brightness_mean=100, sharpness_score=100, motion_score=10,
                    near_black_fraction=0, subtitle_occupancy_ratio=0,
                    narrative_segment_ids=[], _hist=np.ones((16, 16), dtype=np.float32))
    event = broll_pilot._candidate_from_part(units[:1])
    assert event['source_shot_ids'] == ['S_LONG']
    assert event['technical_shots'] == [{'shot_id': 'S_LONG',
                                         'start_seconds': units[0]['start_seconds'],
                                         'end_seconds': units[0]['end_seconds']}]
    assert event['duration_seconds'] <= 18


class _NoDecodeCapture:
    def __init__(self, *_):
        self.released = False
    def get(self, property):
        return 100 if property == cv2.CAP_PROP_FRAME_COUNT else 0
    def set(self, *_):
        pass
    def read(self):
        return False, None
    def release(self):
        self.released = True


class _DecodeCapture(_NoDecodeCapture):
    def read(self):
        return True, np.full((2, 2, 3), 40, dtype=np.uint8)


def test_terminal_sliver_is_auditable_but_other_decode_failures_remain_fatal(monkeypatch, tmp_path):
    monkeypatch.setattr(cv2, 'VideoCapture', _NoDecodeCapture)
    terminal = _shot('EOF', 98, 100) | {'is_final_technical_shot': True}
    value = broll_pilot.visual_signals(tmp_path / 'movie.mp4', [terminal])[0]
    assert value == {'status': 'TERMINAL_SLIVER_SKIPPED', 'reason': 'eof_microshot_no_decodable_sample'}
    with pytest.raises(RuntimeError, match='cannot decode'):
        broll_pilot.visual_signals(tmp_path / 'movie.mp4', [_shot('NOT_EOF', 50, 52) | {'is_final_technical_shot': False}])
    with pytest.raises(RuntimeError, match='cannot decode'):
        broll_pilot.visual_signals(tmp_path / 'movie.mp4', [_shot('LONG_EOF', 76, 100) | {'is_final_technical_shot': True}])
    monkeypatch.setattr(cv2, 'VideoCapture', _DecodeCapture)
    assert broll_pilot.visual_signals(tmp_path / 'movie.mp4', [terminal])[0]['status'] == 'COMPLETE'


def test_signal_cache_checkpoints_then_reuses_only_missing_units(monkeypatch, tmp_path):
    movie = tmp_path / 'movie.mp4'; movie.write_bytes(b'movie')
    run = tmp_path / 'run'; run.mkdir()
    info = {'movie': movie, 'run': run, 'movie_sha256': 'a' * 64}
    units = production.derive_visual_analysis_units([_shot('S1', 0, 72), _shot('S2', 72, 144), _shot('S3', 144, 216)], 24)
    calls = []
    def first(_movie, missing, on_complete=None, **_kwargs):
        for index, unit in enumerate(missing):
            calls.append(unit['analysis_unit_id'])
            if index == 2:
                raise RuntimeError('simulated final decode failure')
            on_complete(unit, {'status': 'COMPLETE', 'brightness_mean': 1, 'brightness_std': 0,
                               'sharpness_score': 1, 'motion_score': 0, 'near_black_fraction': 0,
                               '_hist': np.zeros((16, 16), dtype=np.float32)})
    monkeypatch.setattr(broll_pilot, 'visual_signals', first)
    with pytest.raises(RuntimeError, match='simulated'):
        production._cached_visual_signals(info, units, lambda _: None)
    cache = json.loads((run / 'visual_signals_v2.json').read_text())
    assert list(cache['signals']) == [units[0]['analysis_unit_id'], units[1]['analysis_unit_id']]
    def second(_movie, missing, on_complete=None, **_kwargs):
        assert [x['analysis_unit_id'] for x in missing] == [units[2]['analysis_unit_id']]
        on_complete(missing[0], {'status': 'COMPLETE', 'brightness_mean': 1, 'brightness_std': 0,
                                 'sharpness_score': 1, 'motion_score': 0, 'near_black_fraction': 0,
                                 '_hist': np.zeros((16, 16), dtype=np.float32)})
    monkeypatch.setattr(broll_pilot, 'visual_signals', second)
    assert len(production._cached_visual_signals(info, units, lambda _: None)) == 3
    info_changed = {**info, 'movie_sha256': 'b' * 64}
    invalidated = []
    def recompute(_movie, missing, on_complete=None, **_kwargs):
        invalidated.extend(x['analysis_unit_id'] for x in missing)
        for unit in missing:
            on_complete(unit, {'status': 'COMPLETE', 'brightness_mean': 1, 'brightness_std': 0,
                               'sharpness_score': 1, 'motion_score': 0, 'near_black_fraction': 0,
                               '_hist': np.zeros((16, 16), dtype=np.float32)})
    monkeypatch.setattr(broll_pilot, 'visual_signals', recompute)
    production._cached_visual_signals(info_changed, units, lambda _: None)
    assert invalidated == [x['analysis_unit_id'] for x in units]
    changed_units = production.derive_visual_analysis_units([_shot('S1', 0, 73), _shot('S2', 73, 144), _shot('S3', 144, 216)], 24)
    invalidated.clear()
    production._cached_visual_signals(info, changed_units, lambda _: None)
    assert invalidated == [x['analysis_unit_id'] for x in changed_units]
