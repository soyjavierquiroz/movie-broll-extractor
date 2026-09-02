import inspect
import json
from pathlib import Path

from movie_broll.narrative import normalize_llm_v2_response
from movie_broll.narrative_consolidate import Segment, _choose_seam, _reconcile_boundary, consolidate_narrative, validate_consolidated_map
from movie_broll.utils import write_json, write_jsonl


def _cue(number, start=None):
    start = float(number) if start is None else start
    return type("Cue", (), {"cue_id": f"SRT_{number:06d}", "start_seconds": start, "end_seconds": start + .5})()


def _segment(chunk, ident, first, last, chunk_first=0, chunk_last=19, continuity=("unknown", "unknown"), summary="summary"):
    payload = {"segment_type": {"value": "conversation"}, "narrative_function": {"value": "conversation"}, "narrative_tone": {"value": "serious"}, "narrative_summary": {"value": summary, "confidence": .5}, "continuity": {"previous": continuity[0], "next": continuity[1]}}
    return Segment(chunk, ident, first, last, payload, chunk_first, chunk_last)


def _case(tmp_path, monkeypatch, left, right):
    monkeypatch.chdir(tmp_path); input_dir = tmp_path / "input" / "pilot"; input_dir.mkdir(parents=True)
    root = tmp_path / "runs" / "pilot"; raw_cues = [{"cue_id": f"SRT_{i:06d}", "source_index": i, "start_seconds": float(i), "end_seconds": float(i) + .5, "text": f"cue {i}"} for i in range(1, 21)]
    (root / "source-inspect-v1").mkdir(parents=True); write_jsonl(root / "source-inspect-v1" / "srt_cues.jsonl", raw_cues)
    run = root / "narrative-v2"; write_json(run / "narrative_run.json", {"provider": "fixture", "model": "fixture", "prompt_version": "fixture", "window_seconds": 10, "overlap_seconds": 4})
    for chunk_id, selected, specs, start, end in (("NCHUNK_0001", raw_cues[:10], left, 0, 10), ("NCHUNK_0002", raw_cues[6:], right, 6, 20)):
        data = {"schema_version": "srt_narrative_input_v1", "movie_id": "pilot", "source": {"type": "external_srt", "literal_transcription": False, "timing_reliability": "good", "language": "es"}, "chunk": {"chunk_id": chunk_id, "start_seconds": start, "end_seconds": end, "target_window_seconds": 10, "overlap_seconds": 4}, "cues": selected}
        write_json(run / "chunks" / f"{chunk_id}.input.json", data)
        semantic = lambda row: {"first_cue_id": f"SRT_{row[0]:06d}", "last_cue_id": f"SRT_{row[1]:06d}", "segment_type": "conversation", "narrative_summary_es": row[4], "narrative_tone": "serious", "narrative_function": "conversation", "context_dependency": "medium", "continuity_previous": row[2], "continuity_next": row[3], "possible_visual_opportunities": ["conversation"]}
        write_json(run / "maps" / f"{chunk_id}.narrative_map.json", normalize_llm_v2_response(data, {"schema_version": "narrative_mapper_llm_v2", "chunk_summary_es": "Resumen.", "segments": [semantic(row) for row in specs]}))
    return consolidate_narrative(input_dir, output=lambda _: None), root / "source-inspect-v1" / "srt_cues.jsonl", run


def test_seam_preference_consensus_near_fallback_new_interaction_and_gap():
    cues = [_cue(i) for i in range(20)]
    left, right = [_segment("L", "a", 1, 8), _segment("L", "b", 9, 18)], [_segment("R", "a", 1, 8), _segment("R", "b", 9, 18)]
    seam, report = _choose_seam(cues, left, right, 4, 14); assert seam == 8 and "consensus" in report["chosen_seam_reason"]
    seam, report = _choose_seam(cues, [_segment("L", "a", 1, 8)], [_segment("R", "a", 1, 9)], 4, 14); assert seam in {8, 9} and "near_consensus" in report["chosen_seam_reason"]
    seam, report = _choose_seam(cues, [], [], 4, 14); assert seam == 8 and report["fallback"]
    seam, report = _choose_seam(cues, [_segment("L", "a", 1, 8, continuity=("unknown", "new_interaction"))], [_segment("R", "a", 1, 10)], 4, 14); assert seam == 8 and "new_interaction" in report["chosen_seam_reason"]
    gapped = [_cue(i, float(i) + (10 if i >= 10 else 0)) for i in range(20)]; seam, _ = _choose_seam(gapped, [_segment("L", "a", 1, 9)], [_segment("R", "a", 1, 11)], 4, 14); assert seam == 9


def test_bridge_winners_and_opposing_suppression_are_deterministic():
    left = _segment("L", "left", 3, 12, 0, 19); right = _segment("R", "right", 7, 16, 6, 19)
    result, evidence = _reconcile_boundary([left], [right], 9); assert evidence["bridge_winner"] == "L:left" and len(result) == 1 and evidence["suppressed_segments"]
    left = _segment("L", "left", 7, 12, 0, 12); right = _segment("R", "right", 7, 16, 6, 19)
    _, evidence = _reconcile_boundary([left], [right], 9); assert evidence["bridge_winner"] == "R:right"
    left = _segment("L", "left", 5, 12, 0, 17); right = _segment("R", "right", 5, 12, 0, 17)
    _, evidence = _reconcile_boundary([left], [right], 9); assert evidence["bridge_winner"] == "L:left"


def test_one_to_many_does_not_union_and_gaps_are_allowed(tmp_path, monkeypatch):
    report, cues, run = _case(tmp_path, monkeypatch, [(1, 10, "unknown", "same_interaction", "one")], [(7, 8, "same_interaction", "new_interaction", "a"), (9, 15, "new_interaction", "unknown", "b")])
    result = json.loads((run / "narrative_map.json").read_text())
    assert report["status"] == "PASS" and all(item["semantic_source"]["segment_id"] != "NARR_0001_001" or item["last_cue_id"] != "SRT_000015" for item in result["segments"])
    assert validate_consolidated_map(result, cues) == [] and report["source_cue_coverage"]["unassigned_cue_count"] > 0


def test_provenance_ids_and_rerun_are_stable(tmp_path, monkeypatch):
    report, cues, run = _case(tmp_path, monkeypatch, [(1, 9, "unknown", "same_interaction", "left")], [(7, 15, "same_interaction", "unknown", "right")])
    first = (run / "narrative_map.json").read_bytes(); rerun = consolidate_narrative(tmp_path / "input" / "pilot", output=lambda _: None); result = json.loads(first)
    assert report["status"] == rerun["status"] == "PASS" and (run / "narrative_map.json").read_bytes() == first
    assert [x["segment_id"] for x in result["segments"]] == [f"NARR_{i:06d}" for i in range(1, len(result["segments"]) + 1)]
    assert all(x["semantic_source"] == x["source_segments"][0] for x in result["segments"]) and validate_consolidated_map(result, cues) == []


def test_no_movie_specific_logic():
    source = inspect.getsource(__import__("movie_broll.narrative_consolidate", fromlist=["x"]))
    assert "romper-el-circulo" not in source and "AMBIGUOUS_OVERLAP" not in source
