"""Deterministic SRT narrative-mapper interchange preparation and validation."""
from __future__ import annotations

import json
import math
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .srt import Cue
from .utils import write_json
from .production_profile import narrative as production_narrative_profile

SCHEMA_VERSION = "srt_narrative_input_v1"
MAP_SCHEMA_VERSION = "narrative_map_chunk_v1"
PRODUCTION_NARRATIVE_PROFILE = production_narrative_profile()
TARGET_WINDOW_SECONDS = PRODUCTION_NARRATIVE_PROFILE["window_seconds"]
OVERLAP_SECONDS = PRODUCTION_NARRATIVE_PROFILE["overlap_seconds"]
TINY_TAIL_MAX_EXTENSION_SECONDS = 120.0
_PRESENTATION_TAG = re.compile(r"</?(?:b|i|u|s|strike|font)(?:\s+[^>]*)?>", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")

BOUNDARY_REASONS = frozenset({
    # Short names are the v3 interchange vocabulary.  They name the same
    # semantic event at either edge of a segment.
    "chunk_start", "chunk_end", "continues_prior_situation",
    "conflict_start", "conflict_end", "interaction_change",
    "participant_change", "phase_change", "subject_change",
    "context_change", "new_action", "arrival", "departure", "call",
    "time_jump", "revelation", "unknown",
    # This old broad value is retained only because it cannot be losslessly
    # reduced to call, arrival, or departure.
    "phone_call_arrival_departure",
})
# Earlier generated v3 responses used these longer spellings.  They are
# explicit compatibility aliases, not a semantic guess by the importer.
BOUNDARY_REASON_ALIASES = {
    "conflict_begins": "conflict_start",
    "conflict_resolves": "conflict_end",
    "interaction_purpose_change": "interaction_change",
    "subject_or_situation_change": "subject_change",
    "revelation_changes_situation": "revelation",
    "location_or_context_change": "context_change",
    "participant_group_change": "participant_change",
    "therapy_phase_change": "phase_change",
}

ENUMS = {
    "segment_type": {"conversation", "monologue", "narration", "sparse_dialogue", "transition", "unknown"},
    "dialogue_density": {"none", "low", "medium", "high"},
    "narrative_tone": {"neutral", "serious", "tense", "sad", "warm", "affectionate", "angry", "anxious", "humorous", "hopeful", "fearful", "reflective", "celebratory", "mixed", "unclear"},
    "narrative_function": {"exposition", "conversation", "conflict", "decision", "revelation", "setup", "transition", "resolution", "emotional_exchange", "everyday_interaction", "unknown"},
    "context_dependency": {"low", "medium", "high"},
    "continuity": {"same_interaction", "likely_same_interaction", "new_interaction", "outside_chunk", "unknown"},
    "boundary_reason": set(BOUNDARY_REASONS),
    "long_segment_reason": {"long_continuous_conversation", "continuous_action", "continuous_therapy_exchange"},
    "possible_visual_opportunities": {"conversation", "listening", "reaction", "pause", "gesture", "movement", "object_interaction", "physical_interaction", "establishing", "transition", "unknown"},
}
_VISUAL_FACT_KEYS = {"people_count", "setting", "visible_emotions", "visual_summary", "objects", "visual_actions"}
LLM_V3_SCHEMA_VERSION = "narrative_mapper_llm_v3"


def normalize_boundary_reason(value: Any) -> str | None:
    """Return the canonical v3 boundary reason, or ``None`` if unknown."""
    if not isinstance(value, str):
        return None
    value = BOUNDARY_REASON_ALIASES.get(value, value)
    return value if value in BOUNDARY_REASONS else None


def clean_llm_text(text: str) -> str:
    """Remove only simple display tags and normalize whitespace for LLM input."""
    return _WHITESPACE.sub(" ", _PRESENTATION_TAG.sub("", text)).strip()


def load_canonical_cues(path: Path) -> list[Cue]:
    cues: list[Cue] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                cue = Cue(str(row["cue_id"]), int(row["source_index"]), float(row["start_seconds"]), float(row["end_seconds"]), str(row["text"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: invalid canonical cue: {error}") from error
            if not re.fullmatch(r"SRT_\d{6}", cue.cue_id):
                raise ValueError(f"{path}:{line_number}: non-canonical cue_id {cue.cue_id!r}")
            cues.append(cue)
    if not cues:
        raise ValueError(f"{path}: no cues")
    if len({cue.cue_id for cue in cues}) != len(cues):
        raise ValueError(f"{path}: duplicate cue_id")
    if any(next_cue.start_seconds < cue.start_seconds for cue, next_cue in zip(cues, cues[1:])):
        raise ValueError(f"{path}: cues are not in source timeline order")
    return cues


@dataclass(frozen=True)
class NarrativeChunk:
    chunk_id: str
    start_seconds: float
    end_seconds: float
    cues: list[Cue]


def chunk_cues(cues: list[Cue], window_seconds: float = TARGET_WINDOW_SECONDS, overlap_seconds: float = OVERLAP_SECONDS) -> list[NarrativeChunk]:
    """Use half-open temporal intersection: cue.end > start and cue.start < end.

    Windows advance by ``window_seconds - overlap_seconds``. Empty temporal
    windows are omitted: a subtitle gap is not an LLM request. A one-cue tail
    close to the preceding window is absorbed, avoiding a semantically empty
    request for a final music marker.
    """
    if window_seconds <= 0 or overlap_seconds < 0 or overlap_seconds >= window_seconds:
        raise ValueError("window_seconds must be positive and overlap_seconds must be in [0, window_seconds)")
    if not cues:
        return []
    step = window_seconds - overlap_seconds
    last_end = max(cue.end_seconds for cue in cues)
    chunks: list[NarrativeChunk] = []
    start = 0.0
    sequence = 1
    while start < last_end:
        end = min(start + window_seconds, last_end)
        selected = [cue for cue in cues if cue.end_seconds > start and cue.start_seconds < end]
        if selected:
            chunks.append(NarrativeChunk(f"NCHUNK_{sequence:04d}", start, end, selected))
            sequence += 1
        start += step
    if len(chunks) >= 2 and len(chunks[-1].cues) == 1:
        previous, tail = chunks[-2], chunks[-1]
        extension = tail.end_seconds - previous.end_seconds
        if extension <= max(TINY_TAIL_MAX_EXTENSION_SECONDS, overlap_seconds):
            selected = [cue for cue in cues if cue.end_seconds > previous.start_seconds and cue.start_seconds < tail.end_seconds]
            chunks[-2] = NarrativeChunk(previous.chunk_id, previous.start_seconds, tail.end_seconds, selected)
            chunks.pop()
    return chunks


def narrative_input(movie_id: str, chunk: NarrativeChunk, window_seconds: float, overlap_seconds: float) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "movie_id": movie_id,
        "source": {"type": "external_srt", "literal_transcription": False, "timing_reliability": "good", "language": "es"},
        "chunk": {"chunk_id": chunk.chunk_id, "start_seconds": chunk.start_seconds, "end_seconds": chunk.end_seconds, "target_window_seconds": window_seconds, "overlap_seconds": overlap_seconds},
        "cues": [{"cue_id": cue.cue_id, "source_index": cue.source_index, "start_seconds": cue.start_seconds, "end_seconds": cue.end_seconds, "text": clean_llm_text(cue.text)} for cue in chunk.cues],
    }


def prepare_narrative_inputs(cues_path: Path, movie_id: str, output_dir: Path, window_seconds: float = TARGET_WINDOW_SECONDS, overlap_seconds: float = OVERLAP_SECONDS, force: bool = False) -> list[Path]:
    chunks = chunk_cues(load_canonical_cues(cues_path), window_seconds, overlap_seconds)
    # ``output_dir`` is always the narrative-v2 run root.  Keeping this
    # invariant here (rather than asking callers to append ``chunks``) makes
    # the externally prepared files match consolidate's canonical layout.
    chunks_dir = output_dir / "chunks"
    paths = [chunks_dir / f"{chunk.chunk_id}.input.json" for chunk in chunks]
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to overwrite existing input file(s): {', '.join(str(path) for path in existing)}; use --force")
    for chunk, path in zip(chunks, paths):
        write_json(path, narrative_input(movie_id, chunk, window_seconds, overlap_seconds))
    return paths


def validate_llm_v3_response(chunk_input: dict[str, Any], response: Any) -> list[str]:
    """Validate Gemini's semantic-only, situation-boundary response contract."""
    errors: list[str] = []
    if not isinstance(response, dict):
        return ["response root must be an object"]
    if response.get("schema_version") != LLM_V3_SCHEMA_VERSION:
        errors.append(f"schema_version must be {LLM_V3_SCHEMA_VERSION}")
    summary = response.get("chunk_summary_es")
    if not isinstance(summary, str) or not summary.strip():
        errors.append("chunk_summary_es must be a non-empty string")
    cues = chunk_input.get("cues", [])
    positions = {cue.get("cue_id"): index for index, cue in enumerate(cues) if isinstance(cue, dict)}
    segments = response.get("segments")
    if not isinstance(segments, list):
        return errors + ["segments must be an array"]
    prior_end = -1
    for index, segment in enumerate(segments, 1):
        label = f"segment {index}"
        if not isinstance(segment, dict):
            errors.append(f"{label} must be an object")
            continue
        first, last = segment.get("first_cue_id"), segment.get("last_cue_id")
        if first not in positions:
            errors.append(f"{label} first_cue_id references unknown cue {first}")
            continue
        if last not in positions:
            errors.append(f"{label} last_cue_id references unknown cue {last}")
            continue
        first_position, last_position = positions[first], positions[last]
        if first_position > last_position:
            errors.append(f"{label} cue range is reversed")
        if first_position <= prior_end:
            errors.append(f"{label} cue range overlaps or is out of timeline order")
        prior_end = max(prior_end, last_position)
        for field in ("segment_type", "narrative_tone", "narrative_function", "context_dependency"):
            if segment.get(field) not in ENUMS[field]:
                errors.append(f"{label} {field} is not an allowed enum")
        for field in ("narrative_summary_es", "situation_es", "participants_es", "interaction_action_es", "location_context_es", "continuity_rationale_es"):
            if not isinstance(segment.get(field), str) or not segment[field].strip():
                errors.append(f"{label} {field} must be a non-empty string")
        for field in ("continuity_previous", "continuity_next"):
            if segment.get(field) not in ENUMS["continuity"]:
                errors.append(f"{label} {field} is not an allowed enum")
        for field in ("transition_reason_start", "boundary_reason_end"):
            if normalize_boundary_reason(segment.get(field)) is None:
                errors.append(f"{label} {field} is not an allowed enum")
        long_reason = segment.get("long_segment_reason")
        if long_reason is not None and long_reason not in ENUMS["long_segment_reason"]:
            errors.append(f"{label} long_segment_reason is not an allowed enum or null")
        if first in positions and last in positions:
            duration = float(cues[positions[last]]["end_seconds"]) - float(cues[positions[first]]["start_seconds"])
            if duration > 120.0 and long_reason not in ENUMS["long_segment_reason"]:
                errors.append(f"{label} exceeds 120 seconds ({duration:.3f}s, {first}–{last}) and requires long_segment_reason")
        opportunities = segment.get("possible_visual_opportunities")
        if not isinstance(opportunities, list):
            errors.append(f"{label} possible_visual_opportunities must be an array")
        elif any(value not in ENUMS["possible_visual_opportunities"] for value in opportunities):
            errors.append(f"{label} possible_visual_opportunities contains an invalid enum")
    return errors


def _dialogue_density(cues: list[dict[str, Any]]) -> str:
    """Subtitle occupancy heuristic: cue-covered seconds / enclosing segment span."""
    if not cues:
        return "none"
    start, end = float(cues[0]["start_seconds"]), float(cues[-1]["end_seconds"])
    span = max(end - start, 0.001)
    occupancy = sum(max(0.0, float(cue["end_seconds"]) - float(cue["start_seconds"])) for cue in cues) / span
    if occupancy < .10: return "low"
    if occupancy < .35: return "medium"
    return "high"


def normalize_llm_v3_response(chunk_input: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Expand validated semantic ranges into the immutable canonical map contract."""
    errors = validate_llm_v3_response(chunk_input, response)
    if errors:
        raise ValueError("validation: " + "; ".join(errors[:3]))
    cues = chunk_input["cues"]
    positions = {cue["cue_id"]: index for index, cue in enumerate(cues)}
    assertion = lambda value: {"value": value, "source": "srt_llm", "confidence": .5}
    hint = lambda value: {"value": value, "source": "srt_llm_hint", "confidence": .5}
    suffix = chunk_input["chunk"]["chunk_id"].removeprefix("NCHUNK_")
    segments = []
    for index, semantic in enumerate(response["segments"], 1):
        selected = cues[positions[semantic["first_cue_id"]]:positions[semantic["last_cue_id"]] + 1]
        segments.append({
            "segment_id": f"NARR_{suffix}_{index:03d}",
            "start_seconds": selected[0]["start_seconds"], "end_seconds": selected[-1]["end_seconds"],
            "cue_ids": [cue["cue_id"] for cue in selected],
            "segment_type": assertion(semantic["segment_type"]),
            "narrative_summary": assertion(semantic["narrative_summary_es"]),
            "situation": assertion(semantic["situation_es"]),
            "participants": assertion(semantic["participants_es"]),
            "interaction_action": assertion(semantic["interaction_action_es"]),
            "location_context": assertion(semantic["location_context_es"]),
            "continuity_rationale": assertion(semantic["continuity_rationale_es"]),
            "dialogue_density": {"value": _dialogue_density(selected), "source": "derived_timeline", "confidence": 1.0},
            "narrative_tone": assertion(semantic["narrative_tone"]),
            "narrative_function": assertion(semantic["narrative_function"]),
            "continuity": {"previous": semantic["continuity_previous"], "next": semantic["continuity_next"]},
            "transition_reason_start": normalize_boundary_reason(semantic["transition_reason_start"]),
            "boundary_reason_end": normalize_boundary_reason(semantic["boundary_reason_end"]),
            "long_segment_reason": semantic["long_segment_reason"],
            "possible_visual_opportunities": [hint(value) for value in semantic["possible_visual_opportunities"]],
            "context_dependency": assertion(semantic["context_dependency"]),
        })
    return {"schema_version": MAP_SCHEMA_VERSION, "movie_id": chunk_input["movie_id"],
            "chunk": {key: chunk_input["chunk"][key] for key in ("chunk_id", "start_seconds", "end_seconds")},
            "source": {"type": "external_srt", "literal_transcription": False},
            "chunk_summary": assertion(response["chunk_summary_es"]), "segments": segments}


def _external_value(value: Any, label: str, allowed: set[str] | None = None,
                    source: str = "srt_llm") -> dict[str, Any]:
    """Preserve external assertion evidence, wrapping plain v3 values once."""
    assertion = deepcopy(value) if isinstance(value, dict) else {
        "value": value, "source": source, "confidence": .5,
    }
    errors: list[str] = []
    _check_assertion(assertion, label, allowed, source, errors)
    if errors:
        raise ValueError("external v3 validation: " + "; ".join(errors))
    return assertion


def _external_field(segment: dict[str, Any], name: str, spanish_name: str | None = None) -> Any:
    """Accept the archived external form and the structured-provider v3 form."""
    if name in segment:
        return segment[name]
    if spanish_name and spanish_name in segment:
        return segment[spanish_name]
    raise ValueError(f"external v3 validation: missing {name}")


def normalize_external_v3_response(chunk_input: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Import semantic-only external v3 JSON into ``narrative_map_chunk_v1``.

    Cue ranges are the only boundary decisions accepted from the external map.
    IDs, range expansion, timestamps, envelope metadata, and dialogue density
    are reconstructed from the authoritative chunk input.
    """
    if not isinstance(response, dict):
        raise ValueError("external v3 validation: response root must be an object")
    raw_segments = response.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("external v3 validation: segments must be a non-empty array")
    cues = chunk_input.get("cues")
    if not isinstance(cues, list) or not cues:
        raise ValueError("external v3 validation: authoritative input has no cues")
    positions = {cue.get("cue_id"): index for index, cue in enumerate(cues) if isinstance(cue, dict)}
    if len(positions) != len(cues):
        raise ValueError("external v3 validation: authoritative input has duplicate or invalid cue IDs")
    suffix = str(chunk_input.get("chunk", {}).get("chunk_id", "")).removeprefix("NCHUNK_")
    if not suffix:
        raise ValueError("external v3 validation: authoritative input has no chunk_id")

    summary_raw = response.get("chunk_summary", response.get("chunk_summary_es"))
    if summary_raw is None:
        raise ValueError("external v3 validation: missing chunk_summary")
    summary = _external_value(summary_raw, "chunk_summary")
    segments: list[dict[str, Any]] = []
    prior_end = -1
    for index, raw in enumerate(raw_segments, 1):
        label = f"segment {index}"
        if not isinstance(raw, dict):
            raise ValueError(f"external v3 validation: {label} must be an object")
        first, last = raw.get("first_cue_id"), raw.get("last_cue_id")
        if first not in positions:
            raise ValueError(f"external v3 validation: {label} first_cue_id references unknown cue {first}")
        if last not in positions:
            raise ValueError(f"external v3 validation: {label} last_cue_id references unknown cue {last}")
        first_position, last_position = positions[first], positions[last]
        if first_position > last_position:
            raise ValueError(f"external v3 validation: {label} cue range is reversed")
        if first_position <= prior_end:
            raise ValueError(f"external v3 validation: {label} cue range overlaps or is out of timeline order")
        prior_end = last_position
        selected = cues[first_position:last_position + 1]
        start_reason = normalize_boundary_reason(raw.get("transition_reason_start"))
        end_reason = normalize_boundary_reason(raw.get("boundary_reason_end"))
        if start_reason is None or end_reason is None:
            raise ValueError(f"external v3 validation: {label} has an unsupported boundary reason")
        continuity = raw.get("continuity")
        if not isinstance(continuity, dict):
            continuity = {"previous": raw.get("continuity_previous"), "next": raw.get("continuity_next")}
        if continuity.get("previous") not in ENUMS["continuity"] or continuity.get("next") not in ENUMS["continuity"]:
            raise ValueError(f"external v3 validation: {label} continuity is not an allowed enum")
        long_reason = raw.get("long_segment_reason")
        if long_reason is not None and long_reason not in ENUMS["long_segment_reason"]:
            raise ValueError(f"external v3 validation: {label} long_segment_reason is not an allowed enum or null")
        duration = float(selected[-1]["end_seconds"]) - float(selected[0]["start_seconds"])
        if duration > 120.0 and long_reason not in ENUMS["long_segment_reason"]:
            raise ValueError(f"external v3 validation: {label} exceeds 120 seconds and requires long_segment_reason")
        opportunities_raw = _external_field(raw, "possible_visual_opportunities")
        if not isinstance(opportunities_raw, list):
            raise ValueError(f"external v3 validation: {label} possible_visual_opportunities must be an array")
        opportunities = [_external_value(value, f"{label}.possible_visual_opportunities", ENUMS["possible_visual_opportunities"], "srt_llm_hint") for value in opportunities_raw]
        situation = _external_value(_external_field(raw, "situation", "situation_es"), f"{label}.situation")
        segments.append({
            "segment_id": f"NARR_{suffix}_{index:03d}",
            "start_seconds": selected[0]["start_seconds"],
            "end_seconds": selected[-1]["end_seconds"],
            "cue_ids": [cue["cue_id"] for cue in selected],
            "segment_type": _external_value(_external_field(raw, "segment_type"), f"{label}.segment_type", ENUMS["segment_type"]),
            # The old canonical summary is not an LLM request: situation is
            # the direct source value and carries its original evidence.
            "narrative_summary": deepcopy(situation),
            "situation": situation,
            "participants": _external_value(_external_field(raw, "participants", "participants_es"), f"{label}.participants"),
            "interaction_action": _external_value(_external_field(raw, "interaction_action", "interaction_action_es"), f"{label}.interaction_action"),
            "location_context": _external_value(_external_field(raw, "location_context", "location_context_es"), f"{label}.location_context"),
            "continuity_rationale": _external_value(_external_field(raw, "continuity_rationale", "continuity_rationale_es"), f"{label}.continuity_rationale"),
            "dialogue_density": {"value": _dialogue_density(selected), "source": "derived_timeline", "confidence": 1.0},
            "narrative_tone": _external_value(_external_field(raw, "narrative_tone"), f"{label}.narrative_tone", ENUMS["narrative_tone"]),
            "narrative_function": _external_value(_external_field(raw, "narrative_function"), f"{label}.narrative_function", ENUMS["narrative_function"]),
            "continuity": {"previous": continuity["previous"], "next": continuity["next"]},
            "transition_reason_start": start_reason,
            "boundary_reason_end": end_reason,
            "long_segment_reason": long_reason,
            "possible_visual_opportunities": opportunities,
            "context_dependency": _external_value(_external_field(raw, "context_dependency"), f"{label}.context_dependency", ENUMS["context_dependency"]),
        })
    return {
        "schema_version": MAP_SCHEMA_VERSION,
        "movie_id": chunk_input.get("movie_id"),
        "chunk": {key: chunk_input["chunk"][key] for key in ("chunk_id", "start_seconds", "end_seconds")},
        "source": {"type": "external_srt", "literal_transcription": False},
        "chunk_summary": summary,
        "segments": segments,
    }


def import_external_v3_map(input_path: Path, map_path: Path, output_path: Path | None = None) -> Path:
    """Archive an external v3 map verbatim and atomically write its canonical import."""
    output_path = output_path or map_path
    try:
        chunk_input = json.loads(input_path.read_text(encoding="utf-8"))
        response = json.loads(map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON: {error}") from error
    if response.get("schema_version") == MAP_SCHEMA_VERSION:
        errors = validate_narrative_map(input_path, map_path)
        if errors:
            raise ValueError("canonical map validation: " + "; ".join(errors[:3]))
        return map_path
    canonical = normalize_external_v3_response(chunk_input, response)
    archive = map_path.parent.parent / "responses" / f"{chunk_input['chunk']['chunk_id']}.external-v3.import-source.json"
    if archive.exists() and archive.read_bytes() != map_path.read_bytes():
        raise ValueError(f"refusing to replace preserved external source: {archive}")
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(map_path, archive)
    write_json(output_path, canonical)
    errors = validate_narrative_map(input_path, output_path)
    if errors:
        raise RuntimeError("importer produced invalid canonical map: " + "; ".join(errors[:3]))
    return output_path


# The aliases avoid an import-time break for external integrations while the
# automatic runner deliberately uses only the v3 contract.
validate_llm_v2_response = validate_llm_v3_response
normalize_llm_v2_response = normalize_llm_v3_response


def _number(value: Any, label: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        errors.append(f"{label} must be a finite number")
        return None
    return float(value)


def _assert_close(actual: Any, expected: Any, label: str, errors: list[str], tolerance: float = 1e-6) -> None:
    number = _number(actual, label, errors)
    if number is not None and abs(number - float(expected)) > tolerance:
        errors.append(f"{label} {number} does not match expected {expected}")


def _check_assertion(value: Any, label: str, allowed: set[str] | None, source: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return
    if allowed is not None and value.get("value") not in allowed:
        errors.append(f"{label}.value {value.get('value')!r} is not an allowed enum")
    if value.get("source") != source:
        errors.append(f"{label}.source must be {source}")
    confidence = _number(value.get("confidence"), f"{label}.confidence", errors)
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        errors.append(f"{label}.confidence must be between 0.0 and 1.0")


def _find_visual_contamination(value: Any, label: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_label = f"{label}.{key}"
            if key in _VISUAL_FACT_KEYS:
                errors.append(f"{child_label} is not permitted in narrative maps")
            _find_visual_contamination(child, child_label, errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _find_visual_contamination(child, f"{label}[{index}]", errors)


def validate_narrative_map(input_path: Path, map_path: Path) -> list[str]:
    """Return deterministic, human-readable contract violations (empty is valid)."""
    errors: list[str] = []
    try:
        input_data, map_data = json.loads(input_path.read_text(encoding="utf-8")), json.loads(map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"cannot read JSON: {error}"]
    if not isinstance(input_data, dict) or not isinstance(map_data, dict):
        return ["input and map roots must be JSON objects"]
    _find_visual_contamination(map_data, "map", errors)
    if input_data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"input.schema_version must be {SCHEMA_VERSION}")
    if map_data.get("schema_version") != MAP_SCHEMA_VERSION:
        errors.append(f"schema_version must be {MAP_SCHEMA_VERSION}")
    if map_data.get("movie_id") != input_data.get("movie_id"):
        errors.append("movie_id does not match input")
    expected_chunk = input_data.get("chunk", {})
    actual_chunk = map_data.get("chunk")
    if not isinstance(actual_chunk, dict):
        errors.append("chunk must be an object")
    else:
        for key in ("chunk_id",):
            if actual_chunk.get(key) != expected_chunk.get(key): errors.append(f"chunk.{key} does not match input")
        for key in ("start_seconds", "end_seconds"):
            _assert_close(actual_chunk.get(key), expected_chunk.get(key), f"chunk.{key}", errors)
    source = map_data.get("source")
    if not isinstance(source, dict): errors.append("source must be an object")
    else:
        if source.get("type") != "external_srt": errors.append("source.type must be external_srt")
        if source.get("literal_transcription") is not False: errors.append("source.literal_transcription must be false")
    _check_assertion(map_data.get("chunk_summary"), "chunk_summary", None, "srt_llm", errors)
    input_cues = input_data.get("cues", [])
    if not isinstance(input_cues, list):
        errors.append("input.cues must be an array")
        input_cues = []
    cue_positions = {cue.get("cue_id"): index for index, cue in enumerate(input_cues) if isinstance(cue, dict)}
    cue_by_id = {cue.get("cue_id"): cue for cue in input_cues if isinstance(cue, dict)}
    segments = map_data.get("segments")
    if not isinstance(segments, list): return errors + ["segments must be an array"]
    chunk_id = expected_chunk.get("chunk_id", "")
    suffix = str(chunk_id).removeprefix("NCHUNK_")
    seen_ids: set[str] = set()
    prior_range_end = -1
    for index, segment in enumerate(segments, 1):
        label = f"segment {index}"
        if not isinstance(segment, dict): errors.append(f"{label} must be an object"); continue
        segment_id = segment.get("segment_id")
        expected_id = f"NARR_{suffix}_{index:03d}"
        if segment_id in seen_ids: errors.append(f"duplicate segment_id {segment_id}")
        seen_ids.add(segment_id)
        if segment_id != expected_id: errors.append(f"{label} segment_id must be {expected_id}")
        cue_ids = segment.get("cue_ids")
        if not isinstance(cue_ids, list) or not cue_ids: errors.append(f"segment {segment_id} cue_ids must be a non-empty array"); cue_ids = []
        prior = -1
        for cue_id in cue_ids:
            if cue_id not in cue_by_id: errors.append(f"segment {segment_id} references unknown cue {cue_id}"); continue
            if cue_positions[cue_id] <= prior: errors.append(f"segment {segment_id} cue_ids are not in source timeline order")
            prior = cue_positions[cue_id]
        if cue_ids and cue_ids[0] in cue_positions and cue_ids[-1] in cue_positions:
            first_position, last_position = cue_positions[cue_ids[0]], cue_positions[cue_ids[-1]]
            expected_range = [cue["cue_id"] for cue in input_cues[first_position:last_position + 1]]
            if cue_ids != expected_range:
                errors.append(f"segment {segment_id} cue_ids must be an inclusive authoritative cue range")
            if first_position <= prior_range_end:
                errors.append(f"segment {segment_id} cue range overlaps or is out of timeline order")
            prior_range_end = max(prior_range_end, last_position)
        if cue_ids and cue_ids[0] in cue_by_id: _assert_close(segment.get("start_seconds"), cue_by_id[cue_ids[0]].get("start_seconds"), f"segment {segment_id}.start_seconds", errors)
        if cue_ids and cue_ids[-1] in cue_by_id: _assert_close(segment.get("end_seconds"), cue_by_id[cue_ids[-1]].get("end_seconds"), f"segment {segment_id}.end_seconds", errors)
        for field in ("segment_type", "narrative_tone", "narrative_function", "context_dependency"):
            _check_assertion(segment.get(field), f"segment {segment_id}.{field}", ENUMS[field], "srt_llm", errors)
        for field in ("situation", "participants", "interaction_action", "location_context", "continuity_rationale"):
            _check_assertion(segment.get(field), f"segment {segment_id}.{field}", None, "srt_llm", errors)
        density_source = segment.get("dialogue_density", {}).get("source") if isinstance(segment.get("dialogue_density"), dict) else None
        _check_assertion(segment.get("dialogue_density"), f"segment {segment_id}.dialogue_density", ENUMS["dialogue_density"], density_source if density_source in {"srt_llm", "derived_timeline"} else "derived_timeline", errors)
        _check_assertion(segment.get("narrative_summary"), f"segment {segment_id}.narrative_summary", None, "srt_llm", errors)
        continuity = segment.get("continuity")
        if not isinstance(continuity, dict): errors.append(f"segment {segment_id}.continuity must be an object")
        else:
            for field in ("previous", "next"):
                if continuity.get(field) not in ENUMS["continuity"]: errors.append(f"segment {segment_id}.continuity.{field} is not an allowed enum")
        for field in ("transition_reason_start", "boundary_reason_end"):
            if segment.get(field) not in BOUNDARY_REASONS:
                errors.append(f"segment {segment_id}.{field} is not an allowed enum")
        long_reason = segment.get("long_segment_reason")
        if long_reason is not None and long_reason not in ENUMS["long_segment_reason"]:
            errors.append(f"segment {segment_id}.long_segment_reason is not an allowed enum or null")
        if cue_ids and cue_ids[0] in cue_by_id and cue_ids[-1] in cue_by_id:
            duration = float(cue_by_id[cue_ids[-1]]["end_seconds"]) - float(cue_by_id[cue_ids[0]]["start_seconds"])
            if duration > 120.0 and long_reason not in ENUMS["long_segment_reason"]:
                errors.append(f"segment {segment_id} exceeds 120 seconds and requires long_segment_reason")
        opportunities = segment.get("possible_visual_opportunities")
        if not isinstance(opportunities, list): errors.append(f"segment {segment_id}.possible_visual_opportunities must be an array")
        else:
            for opportunity_index, opportunity in enumerate(opportunities, 1): _check_assertion(opportunity, f"segment {segment_id}.possible_visual_opportunities[{opportunity_index}]", ENUMS["possible_visual_opportunities"], "srt_llm_hint", errors)
        boundary = segment.get("boundary")
        # Boundary confidence had no evidence source in the external-v3
        # contract and consolidate discards it.  It remains valid when older
        # maps carry it, but is intentionally optional for normalized maps.
        if boundary is not None and not isinstance(boundary, dict): errors.append(f"segment {segment_id}.boundary must be an object")
        elif isinstance(boundary, dict):
            for field in ("start_confidence", "end_confidence"):
                confidence = _number(boundary.get(field), f"segment {segment_id}.boundary.{field}", errors)
                if confidence is not None and not 0 <= confidence <= 1: errors.append(f"segment {segment_id}.boundary.{field} must be between 0.0 and 1.0")
    return errors
