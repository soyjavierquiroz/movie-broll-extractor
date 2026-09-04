import json
from pathlib import Path

import pytest

from movie_broll import production
from movie_broll.finalization import _source_movie_sha256


def fixture(tmp_path: Path):
    source = tmp_path / "input" / "film"; source.mkdir(parents=True)
    (source / "movie.mp4").write_bytes(b"movie")
    (source / "subtitles.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHola\n")
    run = tmp_path / "runs" / "film" / "narrative-v2"; run.mkdir(parents=True)
    (run / "narrative_map.json").write_text(json.dumps({"segments": [{"segment_id": "N1", "start_seconds": 0, "end_seconds": 1}]}))
    return source


def test_preflight_requires_canonical_inputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="movie.mp4"):
        production.preflight(tmp_path / "input" / "missing")


def test_process_orchestrates_full_movie_and_reuses_same_source(monkeypatch, tmp_path):
    source = fixture(tmp_path)
    info = {"movie": source / "movie.mp4", "srt": source / "subtitles.srt", "run": tmp_path / "runs" / "film",
            "narrative": tmp_path / "runs" / "film" / "narrative-v2" / "narrative_map.json", "movie_sha256": "a" * 64,
            "srt_sha256": "b" * 64, "metadata": {"video": {"fps": 24, "width": 160, "height": 120}, "duration_seconds": 1}}
    monkeypatch.setattr(production, "preflight", lambda _: info)
    shots = [{"shot_id": "FULL_S_1", "start_seconds": 0, "end_seconds": 1, "start_frame": 0, "end_frame_exclusive": 24}]
    event = {"candidate_id": "BRC_0001", "visual_event_id": "VE_stable", "start_seconds": 0, "end_seconds": 1,
             "start_frame": 0, "end_frame_exclusive": 24, "source_shot_ids": ["FULL_S_1"], "editorial": {"decision": "KEEP", "status": "VALIDATED"}}
    monkeypatch.setattr(production, "_technical_shots", lambda _: shots)
    monkeypatch.setattr(production, "_segment_events", lambda *_: [event])
    calls = []
    def semantics(items, *args, **kwargs):
        calls.append(kwargs["preserve_event_ids"])
        return {"quota_exhausted": False, "status": "COMPLETE"}
    monkeypatch.setattr(production, "semantic_validate", semantics)
    monkeypatch.setattr(production, "apply_semantic_scarcity", lambda _: None)
    monkeypatch.setattr(production, "person_detector_preflight", lambda: {})
    monkeypatch.setattr(production, "finalize_pilot", lambda *a, **k: {"status": "COMPLETE", "completed": 1, "review": 0, "failed_final": 0})
    lines=[]; result = production.process(source, provider=object(), reporter=lines.append)
    assert result["status"] == "COMPLETE" and calls == [True]
    assert (info["run"] / "progress_summary.json").is_file()
    assert any("building Visual Events" in line for line in lines)
    assert "PRODUCTION_STARTED" in (info["run"] / "progress.jsonl").read_text()
    assert production.process(source, provider=object())["status"] == "COMPLETE"
    assert calls == [True]  # completed segment skips Gemini on resume


def test_heartbeat_is_visible_and_logged(monkeypatch, tmp_path):
    run = tmp_path / "runs" / "film"; run.mkdir(parents=True)
    from movie_broll.processing_ledger import ProcessingLedger
    ledger = ProcessingLedger(run, "film", {})
    lines = []
    monkeypatch.setattr(production, "HEARTBEAT_SECONDS", .001)
    def slow():
        import time; time.sleep(.01); return "done"
    assert production._bounded_operation("test", slow, lines.append, ledger) == "done"
    assert any("working" in line for line in lines)
    assert "PRODUCTION_HEARTBEAT" in (run / "progress.jsonl").read_text()


def test_source_change_retires_only_source_artifacts(tmp_path):
    run = tmp_path / "runs" / "film"; (run / "assets").mkdir(parents=True); (run / "review").mkdir(); (run / ".work").mkdir()
    for path in (run / "technical_shots.json", run / "visual_events.json", run / "assets" / "old.mp4", run / "review" / "old.mp4"):
        path.write_bytes(b"x")
    narrative = run / "narrative-v2"; narrative.mkdir(); (narrative / "narrative_map.json").write_text("{}")
    production._invalidate_source_media(run, "old", "new")
    assert not (run / "technical_shots.json").exists() and not list((run / "assets").iterdir())
    assert (narrative / "narrative_map.json").exists()


def test_production_fingerprint_beats_legacy_source_manifest(tmp_path):
    movie = tmp_path / "movie.mp4"; movie.write_bytes(b"current")
    run = tmp_path / "runs" / "film"; (run / "source-v1").mkdir(parents=True)
    (run / "source-v1" / "source_manifest.json").write_text(json.dumps({"source": {"movie": {"sha256": "a" * 64}}}))
    (run / "source_fingerprint.json").write_text(json.dumps({"movie_sha256": "b" * 64}))
    assert _source_movie_sha256(run, movie) == "b" * 64
