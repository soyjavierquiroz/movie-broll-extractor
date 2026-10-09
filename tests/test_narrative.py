import json
from pathlib import Path

from movie_broll.cli import main
from movie_broll.narrative import (BOUNDARY_REASON_ALIASES, BOUNDARY_REASONS,
                                   clean_llm_text, chunk_cues,
                                   import_external_v3_map,
                                   normalize_llm_v3_response,
                                   prepare_narrative_inputs,
                                   validate_llm_v3_response,
                                   validate_narrative_map)
from movie_broll.narrative_provider import GEMINI_RESPONSE_SCHEMA
from movie_broll.narrative_runner import narrative_artifacts_compatible
from movie_broll.srt import Cue


def cue(number, start, end, text="text"):
    return Cue(f"SRT_{number:06d}", number, start, end, text)


def test_clean_llm_text_is_conservative():
    assert clean_llm_text(" <b>Hola!</b>\n <i>¿Qué tal?</i> ") == "Hola! ¿Qué tal?"
    assert clean_llm_text("¿Hola, mundo?!") == "¿Hola, mundo?!"


def test_chunking_windows_overlap_crossing_and_tiny_tail_absorption():
    cues = [cue(1, 10, 20), cue(2, 599, 601), cue(3, 600, 602), cue(4, 1140, 1150)]
    chunks = chunk_cues(cues, 600, 60)
    assert [(chunk.chunk_id, chunk.start_seconds, chunk.end_seconds) for chunk in chunks] == [("NCHUNK_0001", 0.0, 600.0), ("NCHUNK_0002", 540.0, 1150)]
    assert [item.cue_id for item in chunks[0].cues] == ["SRT_000001", "SRT_000002"]
    assert [item.cue_id for item in chunks[1].cues] == ["SRT_000002", "SRT_000003", "SRT_000004"]


def test_production_defaults_skip_empty_windows_but_keep_real_canonical_gaps():
    chunks = chunk_cues([cue(1, 1, 2), cue(2, 1199, 1201), cue(3, 2310, 2312)])
    assert [(item.start_seconds, item.end_seconds) for item in chunks] == [(0.0, 600.0), (1080.0, 1680.0), (2160.0, 2312.0)]


def semantic_response(input_data):
    return {"schema_version": "narrative_mapper_llm_v3", "chunk_summary_es": "Resumen.", "segments": [{"first_cue_id": "SRT_000001", "last_cue_id": "SRT_000002", "segment_type": "conversation", "narrative_summary_es": "Conversan.", "situation_es": "Conversación continuada.", "participants_es": "Interlocutores no identificados.", "interaction_action_es": "Conversan.", "location_context_es": "No inferible por los subtítulos.", "narrative_tone": "serious", "narrative_function": "conversation", "context_dependency": "medium", "continuity_previous": "unknown", "continuity_next": "outside_chunk", "continuity_rationale_es": "El intercambio mantiene su propósito.", "transition_reason_start": "chunk_start", "boundary_reason_end": "chunk_end", "long_segment_reason": None, "possible_visual_opportunities": ["conversation", "reaction"]}]}


def test_v3_schema_enums_and_local_normalization(tmp_path):
    cues = [cue(1, 1, 2), cue(2, 8, 10), cue(3, 11, 12)]
    source = tmp_path / "cues.jsonl"; source.write_text("".join(json.dumps(item.as_dict()) + "\n" for item in cues), encoding="utf-8")
    input_data = json.loads(prepare_narrative_inputs(source, "pilot", tmp_path / "exchange")[0].read_text())
    semantic = semantic_response(input_data)
    assert validate_llm_v3_response(input_data, semantic) == []
    canonical = normalize_llm_v3_response(input_data, semantic)
    segment = canonical["segments"][0]
    assert segment["segment_id"] == "NARR_0001_001" and segment["cue_ids"] == ["SRT_000001", "SRT_000002"]
    assert (segment["start_seconds"], segment["end_seconds"], segment["dialogue_density"]["source"]) == (1, 10, "derived_timeline")
    for field, expected in (("segment_type", {"conversation", "unknown"}), ("narrative_tone", {"serious", "unclear"}), ("narrative_function", {"conflict", "unknown"}), ("context_dependency", {"low", "high"}), ("continuity_previous", {"same_interaction", "unknown"}), ("transition_reason_start", {"conflict_start", "unknown"})):
        schema = GEMINI_RESPONSE_SCHEMA["properties"]["segments"]["items"]["properties"][field]
        assert "enum" in schema and expected <= set(schema["enum"])
    visual = GEMINI_RESPONSE_SCHEMA["properties"]["segments"]["items"]["properties"]["possible_visual_opportunities"]["items"]
    assert "enum" in visual and {"conversation", "unknown"} <= set(visual["enum"])


def test_boundary_reason_contract_is_synced_for_provider_schema_and_prompt():
    provider_values = set(GEMINI_RESPONSE_SCHEMA["properties"]["segments"]["items"]["properties"]["transition_reason_start"]["enum"])
    schema = json.loads((Path(__file__).parents[1] / "schemas" / "narrative_mapper_llm_v3.schema.json").read_text())
    file_values = set(schema["properties"]["segments"]["items"]["properties"]["transition_reason_start"]["enum"])
    prompt = (Path(__file__).parents[1] / "config" / "prompts" / "srt_narrative_mapper_v3.md").read_text()
    assert provider_values == file_values == BOUNDARY_REASONS
    assert all(f"`{value}`" in prompt for value in BOUNDARY_REASONS)
    assert not (set(BOUNDARY_REASON_ALIASES) & provider_values)


def test_v3_rejects_bad_ranges(tmp_path):
    cues = [cue(1, 1, 2), cue(2, 3, 4), cue(3, 5, 6)]
    source = tmp_path / "cues.jsonl"; source.write_text("".join(json.dumps(item.as_dict()) + "\n" for item in cues), encoding="utf-8")
    input_data = json.loads(prepare_narrative_inputs(source, "pilot", tmp_path / "exchange")[0].read_text())
    bad = semantic_response(input_data); bad["segments"][0]["first_cue_id"] = "SRT_999999"
    assert any("first_cue_id" in value for value in validate_llm_v3_response(input_data, bad))
    bad = semantic_response(input_data); bad["segments"][0]["last_cue_id"] = "SRT_999999"
    assert any("last_cue_id" in value for value in validate_llm_v3_response(input_data, bad))
    bad = semantic_response(input_data); bad["segments"][0]["first_cue_id"], bad["segments"][0]["last_cue_id"] = "SRT_000003", "SRT_000001"
    assert any("reversed" in value for value in validate_llm_v3_response(input_data, bad))
    bad = semantic_response(input_data); extra = dict(bad["segments"][0]); extra["first_cue_id"] = "SRT_000002"; extra["last_cue_id"] = "SRT_000003"; bad["segments"].append(extra)
    assert any("overlaps" in value for value in validate_llm_v3_response(input_data, bad))
    bad = semantic_response(input_data); bad["segments"][0]["first_cue_id"] = "SRT_000003"; bad["segments"][0]["last_cue_id"] = "SRT_000003"; extra = semantic_response(input_data)["segments"][0]; bad["segments"].append(extra)
    assert any("timeline order" in value for value in validate_llm_v3_response(input_data, bad))


def valid_map(input_data):
    first, last = input_data["cues"][0], input_data["cues"][-1]
    assertion = lambda value: {"value": value, "source": "srt_llm", "confidence": 0.5}
    return {"schema_version": "narrative_map_chunk_v1", "movie_id": input_data["movie_id"], "chunk": {key: input_data["chunk"][key] for key in ("chunk_id", "start_seconds", "end_seconds")}, "source": {"type": "external_srt", "literal_transcription": False}, "chunk_summary": assertion("Resumen."), "segments": [{"segment_id": "NARR_0001_001", "start_seconds": first["start_seconds"], "end_seconds": last["end_seconds"], "cue_ids": [cue["cue_id"] for cue in input_data["cues"]], "segment_type": assertion("conversation"), "narrative_summary": assertion("Resumen."), "situation": assertion("Situación continuada."), "participants": assertion("No identificados."), "interaction_action": assertion("Conversan."), "location_context": assertion("No inferible por los subtítulos."), "continuity_rationale": assertion("Sin cambio material."), "dialogue_density": assertion("medium"), "narrative_tone": assertion("neutral"), "narrative_function": assertion("conversation"), "continuity": {"previous": "unknown", "next": "outside_chunk"}, "transition_reason_start": "chunk_start", "boundary_reason_end": "chunk_end", "long_segment_reason": None, "possible_visual_opportunities": [{"value": "reaction", "source": "srt_llm_hint", "confidence": 0.4}], "context_dependency": assertion("medium"), "boundary": {"start_confidence": 0.5, "end_confidence": 0.5}}]}


def test_prepare_and_validator_contract(tmp_path):
    cues = [cue(1, 1, 2, "<b>Hola</b>"), cue(2, 3, 4)]
    source = tmp_path / "srt_cues.jsonl"
    source.write_text("".join(json.dumps(item.as_dict()) + "\n" for item in cues), encoding="utf-8")
    paths = prepare_narrative_inputs(source, "pilot", tmp_path / "exchange", 600, 60)
    assert paths == [tmp_path / "exchange" / "chunks" / "NCHUNK_0001.input.json"]
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


def test_coherent_therapy_exchange_can_exceed_120_seconds_with_justification(tmp_path):
    cues = [cue(1, 0, 10), cue(2, 130, 135)]
    source = tmp_path / "cues.jsonl"; source.write_text("".join(json.dumps(item.as_dict()) + "\n" for item in cues), encoding="utf-8")
    input_data = json.loads(prepare_narrative_inputs(source, "pilot", tmp_path / "exchange")[0].read_text())
    semantic = semantic_response(input_data)
    semantic["segments"][0].update(last_cue_id="SRT_000002", situation_es="Una misma fase de terapia.", interaction_action_es="La paciente desarrolla el mismo asunto con la terapeuta.", long_segment_reason="continuous_therapy_exchange")
    assert validate_llm_v3_response(input_data, semantic) == []
    assert normalize_llm_v3_response(input_data, semantic)["segments"][0]["long_segment_reason"] == "continuous_therapy_exchange"


def test_multisituation_block_requires_explicit_boundaries_even_when_long(tmp_path):
    cues = [cue(1, 0, 10), cue(2, 65, 70), cue(3, 130, 135)]
    source = tmp_path / "cues.jsonl"; source.write_text("".join(json.dumps(item.as_dict()) + "\n" for item in cues), encoding="utf-8")
    input_data = json.loads(prepare_narrative_inputs(source, "pilot", tmp_path / "exchange")[0].read_text())
    semantic = semantic_response(input_data)
    semantic["segments"][0]["last_cue_id"] = "SRT_000003"
    semantic["segments"][0]["long_segment_reason"] = None
    assert any("requires long_segment_reason" in item for item in validate_llm_v3_response(input_data, semantic))
    first = semantic_response(input_data)["segments"][0]
    first.update(last_cue_id="SRT_000001", boundary_reason_end="conflict_resolves", continuity_next="new_interaction")
    second = semantic_response(input_data)["segments"][0]
    second.update(first_cue_id="SRT_000002", last_cue_id="SRT_000002", transition_reason_start="conflict_resolves", boundary_reason_end="revelation_changes_situation", continuity_previous="new_interaction", continuity_next="new_interaction", situation_es="Relajación después del conflicto.")
    third = semantic_response(input_data)["segments"][0]
    third.update(first_cue_id="SRT_000003", last_cue_id="SRT_000003", transition_reason_start="revelation_changes_situation", continuity_previous="new_interaction", situation_es="Nueva situación tras la revelación.")
    semantic["segments"] = [first, second, third]
    assert validate_llm_v3_response(input_data, semantic) == []


def test_conflict_relaxation_flirting_and_revelation_are_machine_readable_boundaries(tmp_path):
    cues = [cue(1, 0, 10), cue(2, 20, 30), cue(3, 40, 50), cue(4, 60, 70)]
    source = tmp_path / "cues.jsonl"; source.write_text("".join(json.dumps(item.as_dict()) + "\n" for item in cues), encoding="utf-8")
    input_data = json.loads(prepare_narrative_inputs(source, "pilot", tmp_path / "exchange")[0].read_text())
    base = semantic_response(input_data)["segments"][0]
    ranges = [("SRT_000001", "SRT_000001", "chunk_start", "conflict_resolves"), ("SRT_000002", "SRT_000002", "conflict_resolves", "interaction_purpose_change"), ("SRT_000003", "SRT_000003", "interaction_purpose_change", "revelation_changes_situation"), ("SRT_000004", "SRT_000004", "revelation_changes_situation", "chunk_end")]
    segments = []
    for first, last, start_reason, end_reason in ranges:
        item = dict(base); item.update(first_cue_id=first, last_cue_id=last, transition_reason_start=start_reason, boundary_reason_end=end_reason, continuity_previous="new_interaction" if start_reason != "chunk_start" else "unknown", continuity_next="new_interaction" if end_reason != "chunk_end" else "outside_chunk")
        segments.append(item)
    semantic = {"schema_version": "narrative_mapper_llm_v3", "chunk_summary_es": "Cambios de situación.", "segments": segments}
    canonical = normalize_llm_v3_response(input_data, semantic)
    assert [item["transition_reason_start"] for item in canonical["segments"]] == ["chunk_start", "conflict_end", "interaction_change", "revelation"]


def test_import_external_v3_archives_raw_reconstructs_timing_and_rejects_overlap(tmp_path):
    cues = [cue(1, 1, 2), cue(2, 8, 10), cue(3, 11, 12)]
    source = tmp_path / "cues.jsonl"; source.write_text("".join(json.dumps(item.as_dict()) + "\n" for item in cues), encoding="utf-8")
    input_path = prepare_narrative_inputs(source, "pilot", tmp_path / "narrative-v2")[0]
    maps = tmp_path / "narrative-v2" / "maps"; maps.mkdir(parents=True)
    assertion = lambda value: {"value": value, "source": "srt_llm", "confidence": .8}
    external = {"chunk_summary": assertion("Resumen."), "segments": [{
        "first_cue_id": "SRT_000001", "last_cue_id": "SRT_000002",
        "segment_type": assertion("conversation"), "situation": assertion("Situación."),
        "participants": assertion(["A", "B"]), "interaction_action": assertion("Conversan."),
        "location_context": assertion("No inferible por los subtítulos."),
        "continuity": {"previous": "outside_chunk", "next": "new_interaction"},
        "continuity_rationale": assertion("Cambio de situación."),
        "narrative_tone": assertion("neutral"), "narrative_function": assertion("conversation"),
        "context_dependency": assertion("medium"), "transition_reason_start": "chunk_start",
        "boundary_reason_end": "interaction_change",
        "possible_visual_opportunities": [{"value": "conversation", "source": "srt_llm_hint", "confidence": .6}],
    }, {
        "first_cue_id": "SRT_000003", "last_cue_id": "SRT_000003",
        "segment_type": assertion("conversation"), "situation": assertion("Nueva situación."),
        "participants": assertion(["A"]), "interaction_action": assertion("Continúa."),
        "location_context": assertion("No inferible por los subtítulos."),
        "continuity": {"previous": "new_interaction", "next": "outside_chunk"},
        "continuity_rationale": assertion("Final del chunk."), "narrative_tone": assertion("neutral"),
        "narrative_function": assertion("conversation"), "context_dependency": assertion("medium"),
        "transition_reason_start": "interaction_change", "boundary_reason_end": "chunk_end",
        "possible_visual_opportunities": [],
    }]}
    map_path = maps / "NCHUNK_0001.narrative_map.json"; raw = json.dumps(external, ensure_ascii=False, indent=2).encode(); map_path.write_bytes(raw)
    import_external_v3_map(input_path, map_path)
    canonical = json.loads(map_path.read_text())
    assert (canonical["segments"][0]["start_seconds"], canonical["segments"][0]["end_seconds"]) == (1, 10)
    assert canonical["segments"][0]["narrative_summary"] == canonical["segments"][0]["situation"]
    assert "boundary" not in canonical["segments"][0]
    assert (tmp_path / "narrative-v2" / "responses" / "NCHUNK_0001.external-v3.import-source.json").read_bytes() == raw
    assert validate_narrative_map(input_path, map_path) == []
    external["segments"][1]["first_cue_id"] = "SRT_000002"
    bad_path = tmp_path / "badmaps" / "bad.json"; bad_path.parent.mkdir(); bad_path.write_text(json.dumps(external), encoding="utf-8")
    try:
        import_external_v3_map(input_path, bad_path)
    except ValueError as error:
        assert "overlaps" in str(error)
    else:
        raise AssertionError("overlapping ownership must fail")


def test_external_prepare_import_and_consolidate_use_canonical_run_layout(tmp_path, monkeypatch):
    """The external workflow needs no root-level chunk-file handoff."""
    monkeypatch.chdir(tmp_path)
    movie_id = "pilot"
    cues = [cue(1, 1, 2), cue(2, 8, 10)]
    source = tmp_path / "runs" / movie_id / "source-v1" / "srt_cues.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("".join(json.dumps(item.as_dict()) + "\n" for item in cues), encoding="utf-8")
    run = tmp_path / "runs" / movie_id / "narrative-v2"

    assert main(["narrative", "prepare", "--srt-cues", str(source), "--movie-id", movie_id,
                 "--output-dir", str(run), "--window-seconds", "600", "--overlap-seconds", "60"]) == 0
    input_path = run / "chunks" / "NCHUNK_0001.input.json"
    assert input_path.is_file()
    assert not (run / "NCHUNK_0001.input.json").exists()
    manifest = json.loads((run / "narrative_run.json").read_text())
    assert {key: manifest[key] for key in ("provider", "model", "prompt_version")} == {
        "provider": "external_llm", "model": "external_unspecified", "prompt_version": "srt_narrative_mapper_v3",
    }

    assertion = lambda value: {"value": value, "source": "srt_llm", "confidence": .8}
    external = {"chunk_summary": assertion("Resumen."), "segments": [{
        "first_cue_id": "SRT_000001", "last_cue_id": "SRT_000002",
        "segment_type": assertion("conversation"), "situation": assertion("Situación."),
        "participants": assertion(["A", "B"]), "interaction_action": assertion("Conversan."),
        "location_context": assertion("No inferible por los subtítulos."),
        "continuity": {"previous": "outside_chunk", "next": "outside_chunk"},
        "continuity_rationale": assertion("Un único bloque."), "narrative_tone": assertion("neutral"),
        "narrative_function": assertion("conversation"), "context_dependency": assertion("medium"),
        "transition_reason_start": "chunk_start", "boundary_reason_end": "chunk_end",
        "possible_visual_opportunities": [],
    }]}
    inbox = run / "external-v3-inbox" / "NCHUNK_0001.external-v3.json"
    inbox.parent.mkdir()
    raw = json.dumps(external, ensure_ascii=False, indent=2).encode()
    inbox.write_bytes(raw)
    map_path = run / "maps" / "NCHUNK_0001.narrative_map.json"
    assert main(["narrative", "import-external", "--input", str(input_path), "--map", str(inbox),
                 "--output", str(map_path)]) == 0
    assert map_path.is_file() and inbox.read_bytes() == raw

    assert main(["narrative", "consolidate", str(tmp_path / "input" / movie_id)]) == 0
    assert (run / "narrative_map.json").is_file()
    assert narrative_artifacts_compatible(run)

    prior_input, prior_map = input_path.read_bytes(), map_path.read_bytes()
    assert main(["narrative", "prepare", "--srt-cues", str(source), "--movie-id", movie_id,
                 "--output-dir", str(run), "--window-seconds", "600", "--overlap-seconds", "60", "--force"]) == 0
    assert input_path.read_bytes() == prior_input
    assert map_path.read_bytes() == prior_map and inbox.read_bytes() == raw


def test_speaker_or_camera_change_alone_does_not_require_a_narrative_boundary(tmp_path):
    cues = [cue(1, 0, 10, "Habla una persona."), cue(2, 15, 25, "Responde otra persona.")]
    source = tmp_path / "cues.jsonl"; source.write_text("".join(json.dumps(item.as_dict()) + "\n" for item in cues), encoding="utf-8")
    input_data = json.loads(prepare_narrative_inputs(source, "pilot", tmp_path / "exchange")[0].read_text())
    semantic = semantic_response(input_data)
    semantic["segments"][0]["continuity_rationale_es"] = "Cambio de hablante sin cambio de propósito; la cámara no es evidencia narrativa."
    assert validate_llm_v3_response(input_data, semantic) == []
