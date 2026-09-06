import json
import signal
from pathlib import Path

import pytest

from movie_broll.broll_pilot import visual_signals
from movie_broll.supervisor import Supervisor


class Clock:
    def __init__(self): self.value=0.
    def __call__(self): return self.value
    def sleep(self, seconds): self.value += seconds


class Process:
    def __init__(self, pid, polls): self.pid=pid; self.polls=list(polls); self.last=None
    def poll(self):
        if self.polls: self.last=self.polls.pop(0)
        return self.last


def setup_run(tmp_path, summary=None):
    source=tmp_path/'input'/'film'; source.mkdir(parents=True)
    run=tmp_path/'runs'/'film'; run.mkdir(parents=True)
    if summary is not None: (run/'progress_summary.json').write_text(json.dumps(summary))
    (run/'progress.jsonl').write_text('{}\n')
    return source,run


def test_restart_and_backoff_cap(tmp_path):
    source,run=setup_run(tmp_path, {'status':'RUNNING','segments_complete':1,'segments_total':2})
    clock=Clock(); children=[Process(10,[1]),Process(11,[1]),Process(12,[1]),Process(13,[1]),Process(14,[1])]
    calls=[]
    def factory(*args, **kwargs):
        calls.append(kwargs); child=children.pop(0)
        if not children: (run/'progress_summary.json').write_text(json.dumps({'status':'COMPLETE','segments_complete':2,'segments_total':2}))
        return child
    assert Supervisor(source, clock=clock, sleeper=clock.sleep, process_factory=factory, poll_interval_seconds=0).run()==0
    log=(run/'supervisor.log').read_text()
    assert 'BACKOFF seconds=30' in log and 'BACKOFF seconds=60' in log and 'BACKOFF seconds=120' in log and 'BACKOFF seconds=300' in log
    assert len(calls)==5 and all(x['start_new_session'] is True for x in calls)


def test_exit_zero_incomplete_restarts_then_complete(tmp_path):
    source,run=setup_run(tmp_path, {'status':'RUNNING','segments_complete':1,'segments_total':2})
    clock=Clock(); children=[Process(10,[0]),Process(11,[0])]
    def factory(*args, **kwargs):
        child=children.pop(0)
        if not children: (run/'progress_summary.json').write_text(json.dumps({'status':'COMPLETE','segments_complete':2,'segments_total':2}))
        return child
    assert Supervisor(source, clock=clock, sleeper=clock.sleep, process_factory=factory, poll_interval_seconds=0).run()==0
    assert 'RESTART generation=2' in (run/'supervisor.log').read_text()


def test_watchdog_kills_group_then_restarts(tmp_path):
    source,run=setup_run(tmp_path, {'status':'RUNNING','segments_complete':0,'segments_total':1})
    clock=Clock(); first=Process(42,[None,None,None,None]); second=Process(43,[0]); killed=[]
    def kill(pgid, sig):
        killed.append((pgid,sig))
        if sig == signal.SIGKILL: first.last=-9; first.polls=[]
    children=[first,second]
    def factory(*args, **kwargs):
        child=children.pop(0)
        if child is second: (run/'progress_summary.json').write_text(json.dumps({'status':'COMPLETE','segments_complete':1,'segments_total':1}))
        return child
    assert Supervisor(source, stale_timeout_seconds=2, grace_period_seconds=1, clock=clock, sleeper=clock.sleep, process_factory=factory, killer=kill, poll_interval_seconds=1).run()==0
    assert (42,signal.SIGTERM) in killed and (42,signal.SIGKILL) in killed
    assert 'WATCHDOG_STALE' in (run/'supervisor.log').read_text()


def test_signal_stops_child_without_restart(tmp_path):
    source,_=setup_run(tmp_path)
    child=Process(55,[None]); killed=[]
    supervisor=Supervisor(source, clock=Clock(), sleeper=lambda _: None, process_factory=lambda *a,**k: child, killer=lambda pgid,sig: killed.append((pgid,sig)), grace_period_seconds=0)
    supervisor.child=child; supervisor.request_shutdown(signal.SIGTERM)
    assert supervisor.shutdown_requested and killed == [(55,signal.SIGTERM),(55,signal.SIGKILL)]


def test_lock_prevents_second_supervisor(tmp_path):
    source,_=setup_run(tmp_path)
    import fcntl
    lock=(tmp_path/'runs'/'film'/'supervisor.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    with pytest.raises(RuntimeError, match='another supervisor'):
        Supervisor(source).run()
    lock.close()


def _batch(frames):
    import cv2, numpy as np
    gray=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in frames]; values=np.concatenate([x.ravel() for x in gray])
    hist=cv2.calcHist([cv2.cvtColor(frames[len(frames)//2],cv2.COLOR_BGR2HSV)],[0,1],None,[16,16],[0,180,0,256]); cv2.normalize(hist,hist)
    return {'brightness_mean':float(values.mean()),'brightness_std':float(values.std()),'sharpness_score':float(np.mean([cv2.Laplacian(x,cv2.CV_64F).var() for x in gray])),'motion_score':float(np.mean([np.mean(cv2.absdiff(a,b)) for a,b in zip(gray,gray[1:])])),'near_black_fraction':float((values<20).mean()),'_hist':hist}


def test_visual_signals_matches_batch_and_releases(monkeypatch, tmp_path):
    import cv2, numpy as np
    frames=[np.full((3,4,3), value, dtype=np.uint8) for value in (0,30,80,160)]
    class Capture:
        instances=[]
        def __init__(self, *_): self.index=0; self.released=False; Capture.instances.append(self)
        def set(self, _property, milliseconds): self.index=min(round(milliseconds/1000),len(frames)-1)
        def read(self): return True,frames[self.index].copy()
        def release(self): self.released=True
    monkeypatch.setattr(cv2,'VideoCapture',Capture)
    result=visual_signals(tmp_path/'fake.mp4',[{'shot_id':'s','start_seconds':0,'end_seconds':4}],sample_fps=1)[0]
    expected=_batch(frames)
    for key in ('brightness_mean','brightness_std','sharpness_score','motion_score','near_black_fraction'):
        assert result[key] == pytest.approx(expected[key], abs=1e-10)
    assert np.allclose(result['_hist'],expected['_hist'],atol=1e-7)
    assert Capture.instances[0].released
    class Broken(Capture):
        def read(self): raise RuntimeError('decode failure')
    monkeypatch.setattr(cv2,'VideoCapture',Broken)
    with pytest.raises(RuntimeError): visual_signals(tmp_path/'fake.mp4',[{'shot_id':'s','start_seconds':0,'end_seconds':1}],1)
    assert Broken.instances[-1].released
