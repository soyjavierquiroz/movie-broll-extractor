import json

import pytest

from movie_broll.cli import main
from movie_broll.narrative import narrative_input
from movie_broll.narrative_finalize import finalize_external
from movie_broll.production_preflight import preflight, print_preflight
from movie_broll.srt import Cue
from movie_broll.utils import write_json, write_jsonl
from movie_broll.visual_event_audit import summarize_events


def _external(input_data):
    assertion = lambda value: {"value": value, "source": "srt_llm", "confidence": .8}
    return {"chunk_summary": assertion("Resumen."), "segments": [{
        "first_cue_id": input_data["cues"][0]["cue_id"], "last_cue_id": input_data["cues"][-1]["cue_id"],
        "segment_type": assertion("conversation"), "situation": assertion("Situación."),
        "participants": assertion("Interlocutores."), "interaction_action": assertion("Conversan."),
        "location_context": assertion("No inferible."), "continuity": {"previous": "outside_chunk", "next": "outside_chunk"},
        "continuity_rationale": assertion("Bloque completo."), "narrative_tone": assertion("neutral"),
        "narrative_function": assertion("conversation"), "context_dependency": assertion("medium"),
        "transition_reason_start": "chunk_start", "boundary_reason_end": "chunk_end",
        "possible_visual_opportunities": [],
    }]}


def _prepared(tmp_path, monkeypatch, count=3):
    monkeypatch.chdir(tmp_path)
    movie_id = "film"
    input_dir = tmp_path / "input" / movie_id
    input_dir.mkdir(parents=True)
    root = tmp_path / "runs" / movie_id
    cues = [Cue(f"SRT_{number:06d}", number, float(number), float(number) + .5, f"cue {number}") for number in range(1, count * 2 + 1)]
    (root / "source-v1").mkdir(parents=True)
    write_jsonl(root / "source-v1" / "srt_cues.jsonl", [cue.as_dict() for cue in cues])
    run = root / "narrative-v2"
    write_json(run / "narrative_run.json", {"schema_version": "narrative_run_v2", "movie_id": movie_id,
        "provider": "external_llm", "model": "external_unspecified", "prompt_version": "srt_narrative_mapper_v3",
        "window_seconds": 600, "overlap_seconds": 60})
    for number in range(count):
        selected = cues[number * 2:number * 2 + 2]
        chunk = type("Chunk", (), {"chunk_id": f"NCHUNK_{number + 1:04d}", "start_seconds": selected[0].start_seconds,
                                     "end_seconds": selected[-1].end_seconds, "cues": selected})()
        payload = narrative_input(movie_id, chunk, 600, 60)
        write_json(run / "chunks" / f"{chunk.chunk_id}.input.json", payload)
        write_json(run / "external-v3-inbox" / f"{chunk.chunk_id}.external-v3.json", _external(payload))
    return input_dir, root, run


def test_finalize_external_discovers_arbitrary_chunks_imports_consolidates_and_is_idempotent(tmp_path, monkeypatch):
    input_dir, _, run = _prepared(tmp_path, monkeypatch, count=3)
    result = finalize_external(input_dir, output=lambda _: None)
    assert result["status"] == "PASS" and result["chunks"] == result["external_responses"] == result["imported_maps"] == 3
    final = json.loads((run / "narrative_map.json").read_text())
    assert final["analysis"]["provider"] == "external_llm"
    first = (run / "narrative_map.json").read_bytes()
    rerun = finalize_external(input_dir, output=lambda _: None)
    assert rerun["maps_created"] == 0 and (run / "narrative_map.json").read_bytes() == first


@pytest.mark.parametrize("kind", ["missing", "orphan"])
def test_finalize_external_stops_before_import_for_non_matching_responses(tmp_path, monkeypatch, kind):
    input_dir, _, run = _prepared(tmp_path, monkeypatch, count=2)
    if kind == "missing":
        (run / "external-v3-inbox" / "NCHUNK_0002.external-v3.json").unlink()
    else:
        (run / "external-v3-inbox" / "NCHUNK_9999.external-v3.json").write_text("{}")
    with pytest.raises(ValueError, match="missing external responses|orphan external responses"):
        finalize_external(input_dir, output=lambda _: None)
    assert not (run / "maps").exists()


def test_preflight_pass_missing_map_and_no_secret_output(tmp_path, monkeypatch, capsys):
    input_dir, root, run = _prepared(tmp_path, monkeypatch, count=1)
    (input_dir / "movie.mp4").write_bytes(b"movie")
    (input_dir / "subtitles.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nHola\n")
    write_json(root / "source-v1" / "source_manifest.json", {"source": {"movie": {"filename": "movie.mp4"}}})
    finalize_external(input_dir, output=lambda _: None)
    monkeypatch.setenv("GEMINI_API_KEY_99", "do-not-print-this-secret")
    monkeypatch.setenv("SEMANTIC_PROVIDER", "gemini")
    report = preflight(input_dir)
    assert report["ready"] and report["runtime"]["gemini_credentials"] >= 1
    print_preflight(report)
    assert "do-not-print-this-secret" not in capsys.readouterr().out
    (run / "narrative_map.json").unlink()
    assert not preflight(input_dir)["ready"]


def test_preflight_reports_recoverable_stale_state_without_writing(tmp_path, monkeypatch):
    input_dir, root, _ = _prepared(tmp_path, monkeypatch, count=1)
    (input_dir / "movie.mp4").write_bytes(b"movie")
    (input_dir / "subtitles.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nHola\n")
    write_json(root / "source-v1" / "source_manifest.json", {"source": {"movie": {"filename": "movie.mp4"}}})
    finalize_external(input_dir, output=lambda _: None)
    monkeypatch.setenv("SEMANTIC_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    summary = root / "progress_summary.json"
    state = root / "supervisor_state.json"
    write_json(summary, {"status":"RUNNING", "run_state":"RUNNING", "stage":"visual-signals"})
    write_json(state, {"status":"RUNNING", "run_state":"RUNNING"})
    before = {path: path.read_bytes() for path in (summary, state)}
    report = preflight(input_dir)
    assert report["ready"] and report["runtime"]["recoverable_stale_state"] is True
    assert "recoverable stale runtime state" in report["warnings"][0]
    assert {path: path.read_bytes() for path in (summary, state)} == before


def test_visual_event_audit_duration_buckets_and_shot_grouping():
    events = [
        {"start_seconds": 0, "end_seconds": 3, "source_shot_ids": ["S1"]},
        {"start_seconds": 0, "end_seconds": 4, "source_shot_ids": ["S2", "S3"]},
        {"start_seconds": 0, "end_seconds": 10, "source_shot_ids": ["S4", "S5"]},
        {"start_seconds": 0, "end_seconds": 18, "source_shot_ids": ["S6"]},
        {"start_seconds": 0, "end_seconds": 20, "source_shot_ids": ["S7", "S8"]},
        {"start_seconds": 0, "end_seconds": 21, "source_shot_ids": ["S9"]},
    ]
    report = summarize_events(events, technical_shots=9)
    assert report["technical_shots"] == 9 and report["single_shot_events"] == 3 and report["multi_shot_events"] == 3
    assert {key: value["count"] for key, value in report["buckets"].items()} == {"<4s": 1, "4-10s": 1, "10-18s": 1, "18-20s": 2, ">20s": 1}
    assert report["buckets"]["<4s"]["percentage"] == pytest.approx(16.7)


def test_new_cli_surfaces(tmp_path, monkeypatch, capsys):
    input_dir, root, _ = _prepared(tmp_path, monkeypatch, count=1)
    assert main(["narrative", "finalize-external", str(input_dir)]) == 0
    write_json(root / "visual_events.json", {"events": [{"start_seconds": 1, "end_seconds": 7, "source_shot_ids": ["S1", "S2"]}]})
    assert main(["visual", "audit-events", str(input_dir)]) == 0
    assert "VISUAL EVENT AUDIT" in capsys.readouterr().out
