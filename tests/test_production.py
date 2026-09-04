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



def test_process_builds_provider_once_for_all_segments(monkeypatch, tmp_path):
    source = fixture(tmp_path)

    narrative = (
        tmp_path
        / "runs"
        / "film"
        / "narrative-v2"
        / "narrative_map.json"
    )
    narrative.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "segment_id": "N1",
                        "start_seconds": 0,
                        "end_seconds": 1,
                    },
                    {
                        "segment_id": "N2",
                        "start_seconds": 1,
                        "end_seconds": 2,
                    },
                ]
            }
        )
    )

    info = {
        "movie": source / "movie.mp4",
        "srt": source / "subtitles.srt",
        "run": tmp_path / "runs" / "film",
        "narrative": narrative,
        "movie_sha256": "a" * 64,
        "srt_sha256": "b" * 64,
        "metadata": {
            "video": {
                "fps": 24,
                "width": 160,
                "height": 120,
            },
            "duration_seconds": 2,
        },
    }

    monkeypatch.setattr(
        production,
        "preflight",
        lambda _: info,
    )

    shots = [
        {
            "shot_id": "FULL_S_1",
            "start_seconds": 0,
            "end_seconds": 2,
            "start_frame": 0,
            "end_frame_exclusive": 48,
        }
    ]

    monkeypatch.setattr(
        production,
        "_technical_shots",
        lambda _: shots,
    )

    def event_for_segment(_info, segment, _shots):
        sid = segment["segment_id"]
        return [
            {
                "candidate_id": f"BRC_{sid}",
                "visual_event_id": f"VE_{sid}",
                "start_seconds": segment["start_seconds"],
                "end_seconds": segment["end_seconds"],
                "start_frame": 0,
                "end_frame_exclusive": 24,
                "source_shot_ids": ["FULL_S_1"],
                "editorial": {
                    "decision": "KEEP",
                    "status": "VALIDATED",
                },
            }
        ]

    monkeypatch.setattr(
        production,
        "_segment_events",
        event_for_segment,
    )

    pool = object()
    build_calls = []

    def build(*args, **kwargs):
        build_calls.append((args, kwargs))
        return pool

    monkeypatch.setattr(
        production,
        "build_gemini_provider_from_env",
        build,
    )

    semantic_providers = []

    def semantics(items, *args, **kwargs):
        semantic_providers.append(kwargs["provider"])
        return {
            "status": "COMPLETE",
            "quota_exhausted": False,
            "provider_unavailable": False,
        }

    monkeypatch.setattr(
        production,
        "semantic_validate",
        semantics,
    )
    monkeypatch.setattr(
        production,
        "apply_semantic_scarcity",
        lambda _: None,
    )
    monkeypatch.setattr(
        production,
        "person_detector_preflight",
        lambda: {},
    )
    monkeypatch.setattr(
        production,
        "finalize_pilot",
        lambda *args, **kwargs: {
            "status": "COMPLETE",
            "completed": 0,
            "review": 0,
            "failed_final": 0,
        },
    )

    result = production.process(
        source,
        reporter=lambda _: None,
    )

    assert result["status"] == "COMPLETE"
    assert len(build_calls) == 1
    assert semantic_providers == [pool, pool]


def test_partial_quota_survives_production_with_resume_failure(
    monkeypatch,
    tmp_path,
):
    source = fixture(tmp_path)

    info = {
        "movie": source / "movie.mp4",
        "srt": source / "subtitles.srt",
        "run": tmp_path / "runs" / "film",
        "narrative": (
            tmp_path
            / "runs"
            / "film"
            / "narrative-v2"
            / "narrative_map.json"
        ),
        "movie_sha256": "a" * 64,
        "srt_sha256": "b" * 64,
        "metadata": {
            "video": {
                "fps": 24,
                "width": 160,
                "height": 120,
            },
            "duration_seconds": 1,
        },
    }

    monkeypatch.setattr(
        production,
        "preflight",
        lambda _: info,
    )

    shots = [
        {
            "shot_id": "FULL_S_1",
            "start_seconds": 0,
            "end_seconds": 1,
            "start_frame": 0,
            "end_frame_exclusive": 24,
        }
    ]

    event = {
        "candidate_id": "BRC_0009",
        "visual_event_id": "VE_QUOTA",
        "start_seconds": 0,
        "end_seconds": 1,
        "start_frame": 0,
        "end_frame_exclusive": 24,
        "source_shot_ids": ["FULL_S_1"],
        "editorial": {
            "decision": "REVIEW",
            "status": "SEMANTIC_INCOMPLETE",
        },
    }

    monkeypatch.setattr(
        production,
        "_technical_shots",
        lambda _: shots,
    )
    monkeypatch.setattr(
        production,
        "_segment_events",
        lambda *_: [event],
    )

    def semantics(*args, **kwargs):
        return {
            "status": "PARTIAL_QUOTA",
            "quota_exhausted": True,
            "provider_unavailable": False,
            "failure": {
                "failure_stage": "semantic",
                "failed_segment": "N1",
                "failed_event_id": "VE_QUOTA",
                "candidate_id": "BRC_0009",
                "provider": "gemini-primary-3",
                "model": "gemini-3.6-flash",
                "http_status": 429,
                "reason": "quota_exceeded",
                "retryable": True,
                "retry_after_seconds": 17.25,
                "checkpoint_saved": False,
                "ledger_saved": True,
                "resume_safe": True,
                "message": "quota exceeded",
            },
        }

    monkeypatch.setattr(
        production,
        "semantic_validate",
        semantics,
    )

    # Must never reach finalization after operational semantic stop.
    monkeypatch.setattr(
        production,
        "finalize_pilot",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("finalization must not run")
        ),
    )

    lines = []

    result = production.process(
        source,
        provider=object(),
        reporter=lines.append,
    )

    assert result["status"] == "PARTIAL_QUOTA"

    failure = result["failure"]

    assert failure["failed_segment"] == "N1"
    assert failure["segment_index"] == 1
    assert failure["segments_total"] == 1
    assert failure["failed_event_id"] == "VE_QUOTA"
    assert failure["provider"] == "gemini-primary-3"
    assert failure["http_status"] == 429
    assert failure["reason"] == "quota_exceeded"
    assert failure["retryable"] is True
    assert failure["retry_after_seconds"] == 17.25
    assert failure["resume_safe"] is True

    assert result["summary"]["status"] == "PARTIAL_QUOTA"
    assert result["summary"]["failure"]["reason"] == "quota_exceeded"

    assert any(
        line.startswith("[provider-error]")
        and "http_status=429" in line
        and "reason=quota_exceeded" in line
        and "event=VE_QUOTA" in line
        for line in lines
    )

    store = json.loads(
        (
            info["run"]
            / "production_segments.json"
        ).read_text()
    )

    segment = store["segments"]["N1"]

    assert segment["status"] == "FAILED_RETRYABLE"
    assert segment["failure"]["failed_event_id"] == "VE_QUOTA"
    assert segment["failure"]["reason"] == "quota_exceeded"

    summary = json.loads(
        (
            info["run"]
            / "progress_summary.json"
        ).read_text()
    )

    assert summary["status"] == "PARTIAL_QUOTA"
    assert summary["failure"]["reason"] == "quota_exceeded"

    log = (
        info["run"]
        / "progress.jsonl"
    ).read_text()

    assert "PRODUCTION_PARTIAL_QUOTA" in log



def test_failed_retryable_counts_as_remaining_work(tmp_path):
    from movie_broll.processing_ledger import ProcessingLedger

    run = tmp_path / "runs" / "film"
    run.mkdir(parents=True)

    ledger = ProcessingLedger(
        run,
        "film",
        {},
    )

    ledger.data["events"]["VE_RETRY"] = {
        "visual_event_id": "VE_RETRY",
        "stages": {
            "semantic": {
                "status": "FAILED_RETRYABLE",
            }
        },
    }

    info = {
        "run": run,
        "movie_sha256": "a" * 64,
        "srt_sha256": "b" * 64,
    }

    summary = production._summary(
        info,
        ledger,
        [],
        "PARTIAL_QUOTA",
    )

    assert summary["semantic"]["retryable"] == 1
    assert summary["semantic"]["remaining"] == 1
