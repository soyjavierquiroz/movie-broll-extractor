import json
import os

import pytest

from movie_broll import production_preflight, production_run
from movie_broll import production_ownership as ownership
from movie_broll import supervisor
from movie_broll.utils import write_json


def _input(tmp_path):
    input_dir = tmp_path / "input" / "film"
    input_dir.mkdir(parents=True)
    return input_dir, tmp_path / "runs" / "film"


def _package(directory, base, *, complete=True):
    directory.mkdir(parents=True, exist_ok=True)
    for name in (f"{base}.json", f"{base}.mp4", f"v{base}.mp4", f"{base}.jpg", f"v{base}.jpg")[:5 if complete else 3]:
        (directory / name).write_bytes(b"x")


def test_recorded_stale_or_reused_pid_is_not_an_owner(monkeypatch, tmp_path):
    input_dir, _ = _input(tmp_path)
    expected = input_dir.resolve()
    monkeypatch.setattr(ownership, "_candidate_pids", lambda _state: iter((404,)))
    monkeypatch.setattr(ownership, "_process_command", lambda _pid: (["sleep", "999"], tmp_path))
    assert ownership.active_title_owner(input_dir, {"owner_pid": 404}) is None
    # A real PID is only an owner when both movie-broll command and exact title match.
    monkeypatch.setattr(ownership, "_process_command", lambda _pid: (
        ["python", "-m", "movie_broll.cli", "process", str(expected)], tmp_path))
    assert "pid 404" in ownership.active_title_owner(input_dir, {"owner_pid": 404})


def test_owner_predicate_excludes_current_run_and_preflight_uses_it(monkeypatch, tmp_path):
    input_dir, run = _input(tmp_path)
    write_json(run / "supervisor_state.json", {"status": "RUNNING", "owner_pid": os.getpid()})
    # Supply just enough unrelated preflight input state to reach runtime owner evaluation.
    (input_dir / "movie.mp4").write_bytes(b"movie")
    (input_dir / "subtitles.srt").write_bytes(b"srt")
    write_json(run / "source-v1" / "source_manifest.json", {"source": {"movie": {}}})
    write_json(run / "narrative-v2" / "narrative_run.json", {})
    write_json(run / "narrative-v2" / "narrative_map.json", {"segments": [{"segment_id": "N"}], "analysis": {}})
    (run / "narrative-v2" / "chunks").mkdir(parents=True, exist_ok=True)
    (run / "narrative-v2" / "chunks" / "NCHUNK_0001.input.json").write_text("{}")
    (run / "narrative-v2" / "maps").mkdir(parents=True, exist_ok=True)
    (run / "narrative-v2" / "maps" / "NCHUNK_0001.narrative_map.json").write_text("{}")
    seen = []
    monkeypatch.setattr(production_preflight, "active_title_owner", lambda *_args, **kwargs: seen.append(kwargs.get("exclude_pid")) or None)
    report = production_preflight.preflight(input_dir, exclude_owner_pid=os.getpid())
    assert seen == [os.getpid()]
    assert report["runtime"]["active_run_conflict"] is False


def test_preflight_and_supervisor_share_the_canonical_owner_predicate():
    assert production_preflight.active_title_owner is supervisor.active_title_owner is ownership.active_title_owner


def test_supervisor_refuses_genuine_matching_owner(monkeypatch, tmp_path):
    input_dir, run = _input(tmp_path)
    write_json(run / "progress_summary.json", {"status": "RUNNING", "run_state": "RUNNING"})
    monkeypatch.setattr(supervisor, "active_title_owner", lambda *_args, **_kwargs: "pid 123: movie-broll process film")
    with pytest.raises(RuntimeError, match="active production process"):
        supervisor.Supervisor(input_dir).run()


def test_supervisor_excludes_its_own_owner_check(monkeypatch, tmp_path):
    input_dir, run = _input(tmp_path)
    write_json(run / "progress_summary.json", {"status": "RUNNING", "run_state": "RUNNING"})
    seen = []
    monkeypatch.setattr(supervisor, "active_title_owner", lambda *_args, **kwargs: seen.append(kwargs.get("exclude_pid")) or None)

    class CompleteChild:
        pid = 1
        def poll(self): return 0

    def factory(*_args, **_kwargs):
        write_json(run / "progress_summary.json", {"status": "COMPLETE", "segments_complete": 1, "segments_total": 1})
        return CompleteChild()

    assert supervisor.Supervisor(input_dir, process_factory=factory).run() == 0
    assert seen == [os.getpid()]


def test_status_uses_event_store_and_complete_package_units_only(tmp_path):
    input_dir, run = _input(tmp_path)
    write_json(run / "progress_summary.json", {
        "status": "INTERRUPTED", "technical_shots": 395,
        "editorial": {"KEEP": 38, "REJECT": 40, "REVIEW": 92},
        "semantic": {"failed": 0}, "finalization": {"assets": 0, "review": 0},
    })
    events = [
        {"visual_event_id": "VE_KEEP", "editorial": {"status": "VALIDATED", "decision": "KEEP"}},
        {"visual_event_id": "VE_REJECT", "editorial": {"status": "VALIDATED", "decision": "REJECT"}},
        {"visual_event_id": "VE_REVIEW", "editorial": {"status": "VALIDATED", "decision": "REVIEW"}},
        {"visual_event_id": "VE_PROVISIONAL", "editorial": {"status": "PROVISIONAL", "decision": "KEEP"}},
        {"visual_event_id": "VE_FAILURE", "editorial": {"status": "SEMANTIC_INCOMPLETE", "decision": "REVIEW"}},
    ]
    write_json(run / "visual_event_segments_v1.json", {"production_status": "RUNNING", "events": events,
        "batches": {"PBATCH_0001": {"status": "COMPLETE", "event_ids": ["VE_KEEP"]},
                    "PBATCH_0002": {"status": "PENDING", "event_ids": ["VE_FAILURE"]}}})
    write_json(run / "semantic_failures" / "VE_FAILURE.json", {"visual_event_id": "VE_FAILURE", "failure_stage": "semantic_validation"})
    _package(run / "assets", "A1")
    _package(run / "review", "R1")
    _package(run / "assets", "partial", complete=False)
    before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    status = production_run.read_status(input_dir)
    assert status["status"] == "RUNNING"
    assert (status["validated"], status["keep"], status["reject"], status["review"]) == (3, 1, 1, 1)
    assert status["provisional"] == 1 and status["semantic_incomplete"] == status["semantic_failures"] == 1
    assert status["assets"] == status["review_packages"] == 1
    assert {path: path.read_bytes() for path in run.rglob("*") if path.is_file()} == before


def test_status_counts_artifact_failure_without_event_and_no_transient_ledger(tmp_path):
    input_dir, run = _input(tmp_path)
    write_json(run / "visual_event_segments_v1.json", {"events": [], "batches": {}})
    write_json(run / "semantic_failures" / "VE_ARTIFACT.json", {"visual_event_id": "VE_ARTIFACT", "failure_stage": "semantic_validation"})
    status = production_run.read_status(input_dir)
    assert status["semantic_failures"] == 1
