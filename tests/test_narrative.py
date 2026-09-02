import json

from movie_broll.narrative import clean_llm_text, chunk_cues, prepare_narrative_inputs, validate_narrative_map
from movie_broll.srt import Cue


def cue(number, start, end, text="text"):
    return Cue(f"SRT_{number:06d}", number, start, end, text)


def test_clean_llm_text_is_conservative():
    assert clean_llm_text(" <b>Hola!</b>\n <i>¿Qué tal?</i> ") == "Hola! ¿Qué tal?"
    assert clean_llm_text("¿Hola, mundo?!") == "¿Hola, mundo?!"


def test_chunking_windows_overlap_crossing_and_final_partial():
    cues = [cue(1, 10, 20), cue(2, 599, 601), cue(3, 600, 602), cue(4, 1140, 1150)]
    chunks = chunk_cues(cues, 600, 60)
    assert [(chunk.chunk_id, chunk.start_seconds, chunk.end_seconds) for chunk in chunks] == [("NCHUNK_0001", 0.0, 600.0), ("NCHUNK_0002", 540.0, 1140.0), ("NCHUNK_0003", 1080.0, 1150)]
    assert [item.cue_id for item in chunks[0].cues] == ["SRT_000001", "SRT_000002"]
    assert [item.cue_id for item in chunks[1].cues] == ["SRT_000002", "SRT_000003"]
    assert [item.cue_id for item in chunks[2].cues] == ["SRT_000004"]


def valid_map(input_data):
    first, last = input_data["cues"][0], input_data["cues"][-1]
    assertion = lambda value: {"value": value, "source": "srt_llm", "confidence": 0.5}
    return {"schema_version": "narrative_map_chunk_v1", "movie_id": input_data["movie_id"], "chunk": {key: input_data["chunk"][key] for key in ("chunk_id", "start_seconds", "end_seconds")}, "source": {"type": "external_srt", "literal_transcription": False}, "chunk_summary": assertion("Resumen."), "segments": [{"segment_id": "NARR_0001_001", "start_seconds": first["start_seconds"], "end_seconds": last["end_seconds"], "cue_ids": [cue["cue_id"] for cue in input_data["cues"]], "segment_type": assertion("conversation"), "narrative_summary": assertion("Resumen."), "dialogue_density": assertion("medium"), "narrative_tone": assertion("neutral"), "narrative_function": assertion("conversation"), "continuity": {"previous": "unknown", "next": "outside_chunk"}, "possible_visual_opportunities": [{"value": "reaction", "source": "srt_llm_hint", "confidence": 0.4}], "context_dependency": assertion("medium"), "boundary": {"start_confidence": 0.5, "end_confidence": 0.5}}]}


def test_prepare_and_validator_contract(tmp_path):
    cues = [cue(1, 1, 2, "<b>Hola</b>"), cue(2, 3, 4)]
    source = tmp_path / "srt_cues.jsonl"
    source.write_text("".join(json.dumps(item.as_dict()) + "\n" for item in cues), encoding="utf-8")
    paths = prepare_narrative_inputs(source, "pilot", tmp_path / "exchange", 600, 60)
    input_data = json.loads(paths[0].read_text())
    assert input_data["cues"][0] == {"cue_id": "SRT_000001", "source_index": 1, "start_seconds": 1, "end_seconds": 2, "text": "Hola"}
    map_path = tmp_path / "map.json"; good = valid_map(input_data)
    map_path.write_text(json.dumps(good), encoding="utf-8")
    assert validate_narrative_map(paths[0], map_path) == []
    bad = json.loads(json.dumps(good)); bad["segments"][0]["cue_ids"][0] = "SRT_999999"; map_path.write_text(json.dumps(bad), encoding="utf-8")
    assert any("unknown cue SRT_999999" in error for error in validate_narrative_map(paths[0], map_path))
    bad = json.loads(json.dumps(good)); bad["segments"][0]["start_seconds"] = 99; map_path.write_text(json.dumps(bad), encoding="utf-8")
    assert any("start_seconds" in error for error in validate_narrative_map(paths[0], map_path))
    bad = json.loads(json.dumps(good)); bad["segments"][0]["segment_type"]["value"] = "bad"; map_path.write_text(json.dumps(bad), encoding="utf-8")
    assert any("allowed enum" in error for error in validate_narrative_map(paths[0], map_path))
    bad = json.loads(json.dumps(good)); bad["segments"][0]["narrative_tone"]["source"] = "wrong"; map_path.write_text(json.dumps(bad), encoding="utf-8")
    assert any("narrative_tone.source" in error for error in validate_narrative_map(paths[0], map_path))
    bad = json.loads(json.dumps(good)); bad["segments"][0]["segment_type"]["confidence"] = 2; map_path.write_text(json.dumps(bad), encoding="utf-8")
    assert any("confidence must be between" in error for error in validate_narrative_map(paths[0], map_path))
    bad = json.loads(json.dumps(good)); bad["segments"].append(json.loads(json.dumps(good["segments"][0]))); map_path.write_text(json.dumps(bad), encoding="utf-8")
    assert any("duplicate segment_id" in error for error in validate_narrative_map(paths[0], map_path))
    bad = json.loads(json.dumps(good)); bad["movie_id"] = "other"; bad["chunk"]["chunk_id"] = "NCHUNK_9999"; map_path.write_text(json.dumps(bad), encoding="utf-8")
    errors = validate_narrative_map(paths[0], map_path)
    assert any("movie_id" in error for error in errors) and any("chunk.chunk_id" in error for error in errors)
