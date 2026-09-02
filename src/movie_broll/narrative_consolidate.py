"""Offline, deterministic reconciliation of validated narrative V2 chunks."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .narrative import load_canonical_cues, validate_narrative_map
from .utils import sha256_file, write_json

MERGE_PROFILE = "deterministic_overlap_v1"
NEAR_DUPLICATE_OVERLAP_COEFFICIENT = 0.70
_CONTINUOUS = {"same_interaction", "likely_same_interaction"}


@dataclass
class Segment:
    chunk_id: str
    source_segment_id: str
    first: int
    last: int
    payload: dict[str, Any]
    chunk_start: float
    chunk_end: float
    sources: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.sources:
            self.sources = [{"chunk_id": self.chunk_id, "segment_id": self.source_segment_id}]

    @property
    def key(self) -> str:
        return f"{self.chunk_id}:{self.source_segment_id}"

    @property
    def length(self) -> int:
        return self.last - self.first + 1


def _value(segment: Segment, name: str) -> Any:
    value = segment.payload.get(name)
    return value.get("value") if isinstance(value, dict) else value


def _continuity(segment: Segment, edge: str) -> str:
    continuity = segment.payload.get("continuity", {})
    return continuity.get(edge, "unknown") if isinstance(continuity, dict) else "unknown"


def _metrics(left: Segment, right: Segment) -> dict[str, Any]:
    intersection = max(0, min(left.last, right.last) - max(left.first, right.first) + 1)
    union = max(left.last, right.last) - min(left.first, right.first) + 1
    return {
        "left_segment_id": left.source_segment_id, "right_segment_id": right.source_segment_id,
        "first_cue_id": None, "last_cue_id": None,
        "cue_intersection_count": intersection, "cue_union_count": union,
        "jaccard": round(intersection / union, 6) if union else 0.0,
        "overlap_coefficient": round(intersection / min(left.length, right.length), 6) if intersection else 0.0,
        "left": {"segment_type": _value(left, "segment_type"), "narrative_function": _value(left, "narrative_function"), "continuity_previous": _continuity(left, "previous"), "continuity_next": _continuity(left, "next")},
        "right": {"segment_type": _value(right, "segment_type"), "narrative_function": _value(right, "narrative_function"), "continuity_previous": _continuity(right, "previous"), "continuity_next": _continuity(right, "next")},
    }


def _compatible(left: Segment, right: Segment) -> bool:
    return _value(left, "segment_type") == _value(right, "segment_type") and _value(left, "narrative_function") == _value(right, "narrative_function")


def _contains(left: Segment, right: Segment) -> bool:
    return left.first <= right.first and left.last >= right.last or right.first <= left.first and right.last >= left.last


def _support(segment: Segment, cues: list[Any]) -> tuple[float, float, str, str]:
    """Higher is more context; stable source IDs resolve exact ties."""
    start, end = cues[segment.first].start_seconds, cues[segment.last].end_seconds
    summary = segment.payload.get("narrative_summary", {})
    confidence = summary.get("confidence", 0.0) if isinstance(summary, dict) else 0.0
    return (min(start - segment.chunk_start, segment.chunk_end - end), confidence, segment.chunk_id, segment.source_segment_id)


def _semantic_winner(members: list[Segment], cues: list[Any]) -> Segment:
    # ``min`` with negated support makes better context win, then source order.
    return min(members, key=lambda item: (-_support(item, cues)[0], -_support(item, cues)[1], item.chunk_id, item.source_segment_id))


def _expanded(cues: list[Any], first: int, last: int) -> list[str]:
    return [cue.cue_id for cue in cues[first:last + 1]]


def validate_consolidated_map(map_data: dict[str, Any], cues_path: Path) -> list[str]:
    """Strict deterministic checks for the global map contract."""
    errors: list[str] = []
    cues = load_canonical_cues(cues_path)
    by_id = {cue.cue_id: index for index, cue in enumerate(cues)}
    if map_data.get("schema_version") != "narrative_map_v1": errors.append("schema_version must be narrative_map_v1")
    source = map_data.get("source", {})
    if source.get("type") != "external_srt" or source.get("literal_transcription") is not False: errors.append("source must preserve external_srt and literal_transcription=false")
    seen_ids: set[str] = set(); seen_ranges: set[tuple[str, str]] = set(); prior_last = -1
    for number, segment in enumerate(map_data.get("segments", []), 1):
        label = f"segment {number}"
        if segment.get("segment_id") in seen_ids: errors.append(f"{label}: duplicate segment_id")
        seen_ids.add(segment.get("segment_id"))
        first_id, last_id = segment.get("first_cue_id"), segment.get("last_cue_id")
        if first_id not in by_id or last_id not in by_id: errors.append(f"{label}: unknown cue range"); continue
        first, last = by_id[first_id], by_id[last_id]
        if first > last: errors.append(f"{label}: reversed cue range")
        if first <= prior_last: errors.append(f"{label}: unexplained overlap or out-of-order segment")
        prior_last = max(prior_last, last)
        if (first_id, last_id) in seen_ranges: errors.append(f"{label}: duplicate cue range")
        seen_ranges.add((first_id, last_id))
        if segment.get("cue_ids") != _expanded(cues, first, last): errors.append(f"{label}: cue_ids are not contiguous canonical range")
        if segment.get("start_seconds") != cues[first].start_seconds or segment.get("end_seconds") != cues[last].end_seconds: errors.append(f"{label}: timestamps do not match canonical cues")
        if not segment.get("source_segments"): errors.append(f"{label}: missing source provenance")
    return errors


def _load(run_dir: Path, cues_path: Path) -> tuple[list[Any], list[Segment], list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    cues = load_canonical_cues(cues_path); ordinal = {cue.cue_id: index for index, cue in enumerate(cues)}
    inputs = sorted((run_dir / "chunks").glob("NCHUNK_*.input.json")); maps = sorted((run_dir / "maps").glob("NCHUNK_*.narrative_map.json"))
    if not inputs or len(inputs) != len(maps): raise ValueError("validated chunk inputs and maps must exist one-for-one")
    segments: list[Segment] = []; chunks: list[dict[str, Any]] = []; checksums = {"canonical_srt_sha256": sha256_file(cues_path)}
    for input_path in inputs:
        chunk_id = input_path.name.removesuffix(".input.json"); map_path = run_dir / "maps" / f"{chunk_id}.narrative_map.json"
        if not map_path.is_file(): raise ValueError(f"missing map for {chunk_id}")
        errors = validate_narrative_map(input_path, map_path)
        if errors: raise ValueError(f"invalid map {chunk_id}: {errors[0]}")
        source, mapped = json.loads(input_path.read_text(encoding="utf-8")), json.loads(map_path.read_text(encoding="utf-8"))
        chunk = source["chunk"]; chunks.append(chunk); checksums[f"{chunk_id}_map_sha256"] = sha256_file(map_path)
        for item in mapped["segments"]:
            cue_ids = item["cue_ids"]
            if any(cue_id not in ordinal for cue_id in cue_ids): raise ValueError(f"{chunk_id}:{item['segment_id']} references a noncanonical cue")
            first, last = ordinal[cue_ids[0]], ordinal[cue_ids[-1]]
            if cue_ids != _expanded(cues, first, last): raise ValueError(f"{chunk_id}:{item['segment_id']} cue_ids are not contiguous")
            segments.append(Segment(chunk_id, item["segment_id"], first, last, item, float(chunk["start_seconds"]), float(chunk["end_seconds"])))
    manifest = json.loads((run_dir / "narrative_run.json").read_text(encoding="utf-8"))
    return cues, segments, chunks, checksums, manifest


def consolidate_narrative(input_dir: Path, output: callable = print) -> dict[str, Any]:
    movie_id = input_dir.name; root = Path("runs") / movie_id; run_dir = root / "narrative-v2"; cues_path = root / "source-inspect-v1" / "srt_cues.jsonl"
    if not cues_path.is_file(): raise ValueError(f"canonical SRT cues missing: {cues_path}")
    cues, segments, chunks, checksums, manifest = _load(run_dir, cues_path)
    by_chunk: dict[str, list[Segment]] = {}
    for item in segments: by_chunk.setdefault(item.chunk_id, []).append(item)
    decisions: list[dict[str, Any]] = []; pairs_report: list[dict[str, Any]] = []; selected: list[tuple[Segment, Segment, str]] = []
    for left_chunk, right_chunk in zip(chunks, chunks[1:]):
        left_items, right_items = by_chunk[left_chunk["chunk_id"]], by_chunk[right_chunk["chunk_id"]]
        overlap_start, overlap_end = max(float(left_chunk["start_seconds"]), float(right_chunk["start_seconds"])), min(float(left_chunk["end_seconds"]), float(right_chunk["end_seconds"]))
        overlap_ord = [index for index, cue in enumerate(cues) if cue.end_seconds > overlap_start and cue.start_seconds < overlap_end]
        candidates: list[tuple[Segment, Segment, dict[str, Any]]] = []
        for left in left_items:
            for right in right_items:
                metric = _metrics(left, right)
                if metric["cue_intersection_count"]:
                    metric["first_cue_id"] = cues[max(left.first, right.first)].cue_id; metric["last_cue_id"] = cues[min(left.last, right.last)].cue_id
                    metric["classification_compatible"] = _compatible(left, right)
                    candidates.append((left, right, metric))
        pair_report = {"left_chunk_id": left_chunk["chunk_id"], "right_chunk_id": right_chunk["chunk_id"], "nominal_overlap_seconds": overlap_end - overlap_start, "overlap_cue_count": len(overlap_ord), "left_segments_intersecting_overlap": [item.source_segment_id for item in left_items if item.last >= (overlap_ord[0] if overlap_ord else len(cues)) and item.first <= (overlap_ord[-1] if overlap_ord else -1)], "right_segments_intersecting_overlap": [item.source_segment_id for item in right_items if item.last >= (overlap_ord[0] if overlap_ord else len(cues)) and item.first <= (overlap_ord[-1] if overlap_ord else -1)], "candidate_pairs": [metric for _, _, metric in candidates]}
        pairs_report.append(pair_report)
        eligible: list[tuple[Segment, Segment, str, dict[str, Any]]] = []
        for left, right, metric in candidates:
            exact = left.first == right.first and left.last == right.last
            near = _contains(left, right) and metric["overlap_coefficient"] >= NEAR_DUPLICATE_OVERLAP_COEFFICIENT and _compatible(left, right)
            continuation = (left.first < (overlap_ord[0] if overlap_ord else left.first) and right.last > (overlap_ord[-1] if overlap_ord else right.last) and _compatible(left, right) and _continuity(left, "next") in _CONTINUOUS and _continuity(right, "previous") in _CONTINUOUS and _continuity(left, "next") != "new_interaction" and _continuity(right, "previous") != "new_interaction")
            if exact: eligible.append((left, right, "exact_duplicate", metric))
            elif near: eligible.append((left, right, "near_duplicate", metric))
            elif continuation: eligible.append((left, right, "cross_boundary_continuation", metric))
        for left, right, kind, metric in eligible:
            conflicts = [entry for entry in eligible if entry[0] is left or entry[1] is right]
            if len(conflicts) == 1:
                selected.append((left, right, kind)); decisions.append({"kind": kind, "left": left.key, "right": right.key, "evidence": metric})
    parent = {item.key: item.key for item in segments}
    def find(key: str) -> str:
        while parent[key] != key: parent[key] = parent[parent[key]]; key = parent[key]
        return key
    for left, right, _ in selected: parent[find(right.key)] = find(left.key)
    groups: dict[str, list[Segment]] = {}
    for item in segments: groups.setdefault(find(item.key), []).append(item)
    final: list[dict[str, Any]] = []; payload_choices: list[dict[str, Any]] = []
    for members in groups.values():
        winner = _semantic_winner(members, cues); first, last = min(item.first for item in members), max(item.last for item in members)
        payload = dict(winner.payload); payload.update({"first_cue_id": cues[first].cue_id, "last_cue_id": cues[last].cue_id, "cue_ids": _expanded(cues, first, last), "start_seconds": cues[first].start_seconds, "end_seconds": cues[last].end_seconds, "source_segments": sorted(sum((item.sources for item in members), []), key=lambda row: (row["chunk_id"], row["segment_id"]))})
        payload.pop("boundary", None); payload.pop("dialogue_density", None)
        final.append(payload); payload_choices.append({"selected": winner.key, "contributors": payload["source_segments"]})
    final.sort(key=lambda item: (item["start_seconds"], item["end_seconds"], item["source_segments"][0]["chunk_id"]))
    for index, item in enumerate(final, 1): item["segment_id"] = f"NARR_{index:06d}"
    final_map = {"schema_version": "narrative_map_v1", "source": {"movie_id": movie_id, "type": "external_srt", "literal_transcription": False, "timing_reliability": "good", "language": "es"}, "analysis": {"provider": manifest.get("provider"), "model": manifest.get("model"), "prompt_version": manifest.get("prompt_version"), "chunk_profile": f"{manifest.get('window_seconds')}s_{manifest.get('overlap_seconds')}s_overlap", "merge_profile": MERGE_PROFILE}, "provenance": {"checksums": checksums}, "segments": final}
    errors = validate_consolidated_map(final_map, cues_path)
    assigned = {cue_id for item in final for cue_id in item["cue_ids"]}
    unresolved = []
    for left, right in zip(final, final[1:]):
        if left["last_cue_id"] >= right["first_cue_id"]: unresolved.append({"code": "AMBIGUOUS_OVERLAP", "left_segment_id": left["segment_id"], "right_segment_id": right["segment_id"], "left_range": [left["first_cue_id"], left["last_cue_id"]], "right_range": [right["first_cue_id"], right["last_cue_id"]]})
    if unresolved and not any("unexplained overlap" in error for error in errors): errors.append("unexplained overlapping canonical narrative segments")
    report = {"schema_version": "narrative_reconciliation_report_v1", "merge_profile": MERGE_PROFILE, "status": "PASS" if not errors else "NEEDS_REVIEW", "checksums": checksums, "chunk_pairs": pairs_report, "exact_duplicates": [item for item in decisions if item["kind"] == "exact_duplicate"], "near_duplicates": [item for item in decisions if item["kind"] == "near_duplicate"], "cross_boundary_merges": [item for item in decisions if item["kind"] == "cross_boundary_continuation"], "semantic_payload_choices": payload_choices, "ambiguous_overlaps": unresolved, "segments_before": len(segments), "segments_after": len(final), "source_cue_coverage": {"assigned_cue_count": len(assigned), "unassigned_cue_count": len(cues) - len(assigned)}, "validation_errors": errors}
    write_json(run_dir / "narrative_map.json", final_map); write_json(run_dir / "reconciliation_report.json", report)
    output(f"[narrative] consolidate status={report['status']} segments={len(final)} ambiguous={len(unresolved)}")
    return report
