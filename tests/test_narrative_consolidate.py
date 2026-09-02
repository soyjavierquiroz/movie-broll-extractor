import json
from pathlib import Path

from movie_broll.narrative import normalize_llm_v2_response
from movie_broll.narrative_consolidate import consolidate_narrative, validate_consolidated_map
from movie_broll.utils import write_json, write_jsonl


def _cue(number):
    return {"cue_id": f"SRT_{number:06d}", "source_index": number, "start_seconds": float(number), "end_seconds": float(number) + .5, "text": f"cue {number}"}


def _semantic(spec):
    first, last, kind, function, previous, following, summary = spec
    return {"first_cue_id": f"SRT_{first:06d}", "last_cue_id": f"SRT_{last:06d}", "segment_type": kind, "narrative_summary_es": summary, "narrative_tone": "serious", "narrative_function": function, "context_dependency": "medium", "continuity_previous": previous, "continuity_next": following, "possible_visual_opportunities": ["conversation"]}


def _case(tmp_path, monkeypatch, left, right):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "input" / "pilot"; input_dir.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "runs" / "pilot"; cues = [_cue(index) for index in range(1, 21)]
    (root / "source-inspect-v1").mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "source-inspect-v1" / "srt_cues.jsonl", cues)
    run = root / "narrative-v2"; write_json(run / "narrative_run.json", {"provider": "fixture", "model": "fixture", "prompt_version": "fixture", "window_seconds": 10, "overlap_seconds": 4})
    for chunk_id, selected, specs, start, end in (("NCHUNK_0001", cues[:10], left, 0, 10), ("NCHUNK_0002", cues[6:], right, 6, 20)):
        data = {"schema_version": "srt_narrative_input_v1", "movie_id": "pilot", "source": {"type": "external_srt", "literal_transcription": False, "timing_reliability": "good", "language": "es"}, "chunk": {"chunk_id": chunk_id, "start_seconds": start, "end_seconds": end, "target_window_seconds": 10, "overlap_seconds": 4}, "cues": selected}
        write_json(run / "chunks" / f"{chunk_id}.input.json", data)
        response = {"schema_version": "narrative_mapper_llm_v2", "chunk_summary_es": "Resumen.", "segments": [_semantic(spec) for spec in specs]}
        write_json(run / "maps" / f"{chunk_id}.narrative_map.json", normalize_llm_v2_response(data, response))
    return consolidate_narrative(input_dir, output=lambda _: None), root / "source-inspect-v1" / "srt_cues.jsonl", run


def test_exact_duplicate_and_global_timestamp_and_ids(tmp_path, monkeypatch):
    report, cues, run = _case(tmp_path, monkeypatch, [(7, 8, "conversation", "conversation", "unknown", "same_interaction", "left")], [(7, 8, "conversation", "conversation", "same_interaction", "unknown", "right")])
    result = json.loads((run / "narrative_map.json").read_text())
    assert report["status"] == "PASS" and len(report["exact_duplicates"]) == 1
    assert result["segments"][0]["segment_id"] == "NARR_000001"
    assert result["segments"][0]["cue_ids"] == [f"SRT_{index:06d}" for index in range(7, 9)]
    assert (result["segments"][0]["start_seconds"], result["segments"][0]["end_seconds"]) == (7.0, 8.5)
    assert validate_consolidated_map(result, cues) == []


def test_contained_near_duplicate_and_deterministic_rerun(tmp_path, monkeypatch):
    report, _, run = _case(tmp_path, monkeypatch, [(1, 10, "conversation", "conversation", "unknown", "same_interaction", "edge")], [(7, 10, "conversation", "conversation", "same_interaction", "unknown", "interior")])
    first = (run / "narrative_map.json").read_bytes()
    rerun = consolidate_narrative(tmp_path / "input" / "pilot", output=lambda _: None)
    assert report["status"] == rerun["status"] == "PASS" and len(report["near_duplicates"]) == 1
    assert (run / "narrative_map.json").read_bytes() == first


def test_safe_cross_boundary_continuation_unions_ranges(tmp_path, monkeypatch):
    report, _, run = _case(tmp_path, monkeypatch, [(1, 9, "conversation", "conversation", "unknown", "same_interaction", "left")], [(7, 15, "conversation", "conversation", "likely_same_interaction", "unknown", "right")])
    segment = json.loads((run / "narrative_map.json").read_text())["segments"][0]
    assert report["status"] == "PASS" and len(report["cross_boundary_merges"]) == 1
    assert (segment["first_cue_id"], segment["last_cue_id"]) == ("SRT_000001", "SRT_000015")


def test_contradictory_new_interaction_is_not_merged(tmp_path, monkeypatch):
    report, _, _ = _case(tmp_path, monkeypatch, [(1, 9, "conversation", "conversation", "unknown", "new_interaction", "left")], [(7, 15, "conversation", "conversation", "same_interaction", "unknown", "right")])
    assert report["status"] == "NEEDS_REVIEW" and not report["cross_boundary_merges"] and report["ambiguous_overlaps"]


def test_one_to_many_and_many_to_one_are_ambiguous_not_overmerged(tmp_path, monkeypatch):
    report, _, _ = _case(tmp_path, monkeypatch, [(1, 10, "conversation", "conversation", "unknown", "same_interaction", "one")], [(7, 8, "conversation", "conversation", "same_interaction", "new_interaction", "a"), (9, 15, "conversation", "conversation", "new_interaction", "unknown", "b")])
    assert report["status"] == "NEEDS_REVIEW" and not report["cross_boundary_merges"]
    report, _, _ = _case(tmp_path, monkeypatch, [(1, 7, "conversation", "conversation", "unknown", "new_interaction", "a"), (8, 10, "conversation", "conversation", "new_interaction", "same_interaction", "b")], [(7, 15, "conversation", "conversation", "same_interaction", "unknown", "one")])
    assert report["status"] == "NEEDS_REVIEW" and not report["cross_boundary_merges"]


def test_semantic_payload_prefers_more_context_and_stale_inputs_change_checksum(tmp_path, monkeypatch):
    _, _, run = _case(tmp_path, monkeypatch, [(1, 10, "conversation", "conversation", "unknown", "same_interaction", "edge")], [(7, 10, "conversation", "conversation", "same_interaction", "unknown", "interior")])
    result = json.loads((run / "narrative_map.json").read_text())
    assert result["segments"][0]["narrative_summary"]["value"] == "interior"
    map_path = run / "maps" / "NCHUNK_0002.narrative_map.json"; data = json.loads(map_path.read_text()); data["segments"][0]["narrative_summary"]["value"] = "changed"; write_json(map_path, data)
    consolidate_narrative(tmp_path / "input" / "pilot", output=lambda _: None)
    updated = json.loads((run / "narrative_map.json").read_text())
    assert updated["provenance"]["checksums"]["NCHUNK_0002_map_sha256"] != result["provenance"]["checksums"]["NCHUNK_0002_map_sha256"]
