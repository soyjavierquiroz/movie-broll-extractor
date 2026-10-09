import pytest
import json

from movie_broll import production_run
from movie_broll.utils import write_json, write_jsonl


def _input(tmp_path):
    source = tmp_path / "input" / "film"
    source.mkdir(parents=True)
    run = tmp_path / "runs" / "film" / "source-v1"
    run.mkdir(parents=True)
    write_jsonl(run / "srt_cues.jsonl", [{"cue_id": "SRT_000001", "source_index": 1,
        "start_seconds": 0.0, "end_seconds": 1.0, "duration_seconds": 1.0, "text": "Hola"}])
    write_json(run / "source_manifest.json", {"source": {"movie": {"filename": "movie.mp4"}}})
    return source


def test_first_run_prepares_external_handoff_without_provider_or_pipeline_reset(tmp_path):
    source = _input(tmp_path)
    lines = []
    assert not production_run.ensure_narrative(source, lines.append)
    narrative = tmp_path / "runs" / "film" / "narrative-v2"
    assert len(list((narrative / "chunks").glob("NCHUNK_*.input.json"))) == 1
    assert (narrative / "narrative_run.json").is_file()
    assert "STATUS: WAITING_EXTERNAL" in lines
    assert any(str(narrative / "external-v3-inbox") in line for line in lines)
    assert any("movie-broll run" in line for line in lines)


def test_run_bootstraps_source_v1_only_when_absent(monkeypatch, tmp_path):
    source = tmp_path / "input" / "film"
    source.mkdir(parents=True)
    (source / "movie.mp4").write_bytes(b"movie")
    (source / "subtitles.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHola\n")
    monkeypatch.setattr(production_run, "inspect_movie", lambda _: {"duration_seconds": 1,
        "video": {"width": 160, "height": 90, "fps": 24}})
    assert not production_run.ensure_source(source)
    manifest = tmp_path / "runs" / "film" / "source-v1" / "source_manifest.json"
    before = manifest.read_bytes()
    assert production_run.ensure_source(source)
    assert manifest.read_bytes() == before


def test_incomplete_external_responses_stay_waiting_and_preserve_chunks(tmp_path):
    source = _input(tmp_path)
    initial = []
    production_run.ensure_narrative(source, initial.append)
    narrative = tmp_path / "runs" / "film" / "narrative-v2"
    before = {path: path.read_bytes() for path in (narrative / "chunks").glob("*")}
    lines = []
    assert not production_run.ensure_narrative(source, lines.append)
    assert any(line.startswith("MISSING RESPONSES: NCHUNK_0001") for line in lines)
    assert {path: path.read_bytes() for path in (narrative / "chunks").glob("*")} == before


def test_completed_narrative_is_reused_without_writing(tmp_path):
    source = _input(tmp_path)
    narrative = tmp_path / "runs" / "film" / "narrative-v2"
    write_json(narrative / "narrative_map.json", {"segments": [{"segment_id": "NARR_000001"}]})
    write_json(narrative / "reconciliation_report.json", {"status": "PASS"})
    before = (narrative / "narrative_map.json").read_bytes()
    lines = []
    assert production_run.ensure_narrative(source, lines.append)
    assert (narrative / "narrative_map.json").read_bytes() == before
    assert lines == ["[movie-broll] narrative-v2: REUSED"]


def test_complete_external_inbox_is_finalized_and_marked_complete(monkeypatch, tmp_path):
    source = _input(tmp_path)
    production_run.ensure_narrative(source, lambda _: None)
    narrative = tmp_path / "runs" / "film" / "narrative-v2"
    (narrative / "external-v3-inbox").mkdir()
    (narrative / "external-v3-inbox" / "NCHUNK_0001.external-v3.json").write_text("{}")
    def finalize(_input, output):
        write_json(narrative / "narrative_map.json", {"segments": [{"segment_id": "NARR_000001"}]})
        write_json(narrative / "reconciliation_report.json", {"status": "PASS"})
        return {"status": "PASS", "segments": 1}
    monkeypatch.setattr(production_run, "finalize_external", finalize)
    assert production_run.ensure_narrative(source, lambda _: None)
    assert json.loads((narrative / "narrative_run.json").read_text())["status"] == "COMPLETE"


def test_status_is_read_only_and_final_summary_uses_persisted_counters(tmp_path):
    source = _input(tmp_path)
    run = tmp_path / "runs" / "film"
    write_json(run / "progress_summary.json", {"movie_id": "film", "status": "COMPLETE", "stage": "complete",
        "technical_shots": 8, "visual_events": 3, "production_batches_complete": 3,
        "production_batches_total": 3, "editorial": {"KEEP": 1, "REJECT": 2, "REVIEW": 0},
        "semantic": {"failed": 1, "retryable": 0}, "finalization": {"assets": 1, "review": 0}, "resume_safe": True})
    write_json(run / "visual_event_segments_v1.json", {"events": [{"visual_event_id": "VE_1"}], "batches": {
        "PBATCH_0001": {"status": "COMPLETE", "event_ids": ["VE_1"]},
        "PBATCH_0002": {"status": "PENDING", "event_ids": ["VE_2"]}}})
    before = (run / "progress_summary.json").read_bytes()
    status = production_run.read_status(source)
    lines = []
    production_run.print_status(status, lines.append)
    production_run.print_final_summary(source, lines.append)
    assert (run / "progress_summary.json").read_bytes() == before
    assert "PRODUCTION BATCHES: 1/2" in lines
    assert "CURRENT: PBATCH_0002 / VE_2" in lines
    assert "STATUS: COMPLETE" in lines
    assert "SEMANTIC KEEP: 0  SEMANTIC REJECT: 0  SEMANTIC REVIEW: 0" in lines
    assert "AUTO-PUBLISHED ASSETS: 0" in lines
    assert "HUMAN REVIEW REQUIRED: 0" in lines
    assert "SOFT-WARNING ASSETS: 0" in lines
    assert "RESUME SAFE: YES" in lines


def test_run_reuses_prior_work_and_prints_final_summary(monkeypatch, tmp_path):
    source = _input(tmp_path)
    run = tmp_path / "runs" / "film"
    write_json(run / "progress_summary.json", {"movie_id": "film", "status": "PARTIAL", "technical_shots": 4,
        "production_batches_complete": 1, "production_batches_total": 2, "finalization": {"assets": 1}})
    monkeypatch.setattr(production_run, "ensure_source", lambda _: True)
    monkeypatch.setattr(production_run, "ensure_narrative", lambda *_: True)
    monkeypatch.setattr(production_run, "preflight", lambda *_args, **_kwargs: {"ready": True, "blockers": []})

    class CompleteSupervisor:
        def __init__(self, _input, **kwargs): self.before_start = kwargs["before_start"]
        def run(self):
            self.before_start()
            write_json(run / "progress_summary.json", {"movie_id": "film", "status": "COMPLETE", "stage": "complete",
                "technical_shots": 4, "visual_events": 2, "production_batches_complete": 2,
                "production_batches_total": 2, "editorial": {"KEEP": 1, "REJECT": 1, "REVIEW": 0},
                "semantic": {"failed": 0, "retryable": 0}, "finalization": {"assets": 1, "review": 0}, "resume_safe": True})
            return 0

    lines = []
    assert production_run.run(source, lines.append, CompleteSupervisor) == 0
    assert "[movie-broll] resume: reusing compatible persisted work" in lines
    assert "STATUS: COMPLETE" in lines


def test_run_waiting_external_returns_zero_without_starting_child(monkeypatch, tmp_path):
    source = _input(tmp_path)
    monkeypatch.setattr(production_run, "ensure_source", lambda _: True)
    monkeypatch.setattr(production_run, "ensure_narrative", lambda *_: False)

    class ChildMustNotStart:
        def __init__(self, _input, **kwargs): self.before_start = kwargs["before_start"]
        def run(self): self.before_start(); raise AssertionError("child must not start")

    # The fake supervisor represents the lock-held callback boundary; a real
    # supervisor raises WaitingExternal before it can spawn a child.
    assert production_run.run(source, supervisor_factory=ChildMustNotStart) == 0


@pytest.fixture(autouse=True)
def external_transport_for_legacy_tests(monkeypatch):
    monkeypatch.setenv("NARRATIVE_PROVIDER_MODE", "external")
