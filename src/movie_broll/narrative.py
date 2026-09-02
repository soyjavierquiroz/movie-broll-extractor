"""Deterministic SRT narrative-mapper interchange preparation and validation."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .srt import Cue
from .utils import write_json

SCHEMA_VERSION = "srt_narrative_input_v1"
MAP_SCHEMA_VERSION = "narrative_map_chunk_v1"
TARGET_WINDOW_SECONDS = 600
OVERLAP_SECONDS = 60
_PRESENTATION_TAG = re.compile(r"</?(?:b|i|u|s|strike|font)(?:\s+[^>]*)?>", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")

ENUMS = {
    "segment_type": {"conversation", "monologue", "narration", "sparse_dialogue", "transition", "unknown"},
    "dialogue_density": {"none", "low", "medium", "high"},
    "narrative_tone": {"neutral", "serious", "tense", "sad", "warm", "affectionate", "angry", "anxious", "humorous", "hopeful", "fearful", "reflective", "celebratory", "mixed", "unclear"},
    "narrative_function": {"exposition", "conversation", "conflict", "decision", "revelation", "setup", "transition", "resolution", "emotional_exchange", "everyday_interaction", "unknown"},
    "context_dependency": {"low", "medium", "high"},
    "continuity": {"same_interaction", "likely_same_interaction", "new_interaction", "outside_chunk", "unknown"},
    "possible_visual_opportunities": {"conversation", "listening", "reaction", "pause", "gesture", "movement", "object_interaction", "physical_interaction", "establishing", "transition", "unknown"},
}
_VISUAL_FACT_KEYS = {"people_count", "setting", "visible_emotions", "visual_summary", "objects", "visual_actions"}


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

    Windows advance by ``window_seconds - overlap_seconds``.  The final window
    ends at the last cue end, so it may be a partial nominal window.
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
        chunks.append(NarrativeChunk(f"NCHUNK_{sequence:04d}", start, end, selected))
        start += step
        sequence += 1
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
    paths = [output_dir / f"{chunk.chunk_id}.input.json" for chunk in chunks]
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to overwrite existing input file(s): {', '.join(str(path) for path in existing)}; use --force")
    for chunk, path in zip(chunks, paths):
        write_json(path, narrative_input(movie_id, chunk, window_seconds, overlap_seconds))
    return paths


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
        if cue_ids and cue_ids[0] in cue_by_id: _assert_close(segment.get("start_seconds"), cue_by_id[cue_ids[0]].get("start_seconds"), f"segment {segment_id}.start_seconds", errors)
        if cue_ids and cue_ids[-1] in cue_by_id: _assert_close(segment.get("end_seconds"), cue_by_id[cue_ids[-1]].get("end_seconds"), f"segment {segment_id}.end_seconds", errors)
        for field in ("segment_type", "dialogue_density", "narrative_tone", "narrative_function", "context_dependency"):
            _check_assertion(segment.get(field), f"segment {segment_id}.{field}", ENUMS[field], "srt_llm", errors)
        _check_assertion(segment.get("narrative_summary"), f"segment {segment_id}.narrative_summary", None, "srt_llm", errors)
        continuity = segment.get("continuity")
        if not isinstance(continuity, dict): errors.append(f"segment {segment_id}.continuity must be an object")
        else:
            for field in ("previous", "next"):
                if continuity.get(field) not in ENUMS["continuity"]: errors.append(f"segment {segment_id}.continuity.{field} is not an allowed enum")
        opportunities = segment.get("possible_visual_opportunities")
        if not isinstance(opportunities, list): errors.append(f"segment {segment_id}.possible_visual_opportunities must be an array")
        else:
            for opportunity_index, opportunity in enumerate(opportunities, 1): _check_assertion(opportunity, f"segment {segment_id}.possible_visual_opportunities[{opportunity_index}]", ENUMS["possible_visual_opportunities"], "srt_llm_hint", errors)
        boundary = segment.get("boundary")
        if not isinstance(boundary, dict): errors.append(f"segment {segment_id}.boundary must be an object")
        else:
            for field in ("start_confidence", "end_confidence"):
                confidence = _number(boundary.get(field), f"segment {segment_id}.boundary.{field}", errors)
                if confidence is not None and not 0 <= confidence <= 1: errors.append(f"segment {segment_id}.boundary.{field} must be between 0.0 and 1.0")
    return errors
