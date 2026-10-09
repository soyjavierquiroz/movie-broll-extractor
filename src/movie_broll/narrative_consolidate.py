"""Deterministic seam reconciliation for overlapping narrative chunk maps."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .narrative import load_canonical_cues, validate_narrative_map
from .utils import sha256_file, write_json

MERGE_PROFILE = "deterministic_seam_v2_semantic_boundaries"


@dataclass(frozen=True)
class Segment:
    chunk_id: str; source_segment_id: str; first: int; last: int
    payload: dict[str, Any]; chunk_first: int; chunk_last: int
    @property
    def key(self) -> str: return f"{self.chunk_id}:{self.source_segment_id}"
    @property
    def length(self) -> int: return self.last - self.first + 1


def _value(item: Segment, name: str) -> Any:
    value = item.payload.get(name)
    return value.get("value") if isinstance(value, dict) else value


def _continuity(item: Segment, edge: str) -> str:
    value = item.payload.get("continuity", {})
    return value.get(edge, "unknown") if isinstance(value, dict) else "unknown"


def _boundary_reason(item: Segment, edge: str) -> str:
    key = "transition_reason_start" if edge == "start" else "boundary_reason_end"
    value = item.payload.get(key)
    return value if isinstance(value, str) else "unknown"


def _expanded(cues: list[Any], first: int, last: int) -> list[str]: return [cue.cue_id for cue in cues[first:last + 1]]
def _overlaps(a: Segment, b: Segment) -> bool: return a.first <= b.last and b.first <= a.last
def _crosses(item: Segment, seam: int) -> bool: return item.first <= seam < item.last


def _clip(item: Segment, first: int, last: int) -> Segment:
    """Assign overlap ownership without inventing a merged narrative range."""
    return replace(item, first=first, last=last)


def _boundaries(items: list[Segment], low: int, high: int) -> set[int]:
    return {boundary for item in items for boundary in (item.first - 1, item.last) if low <= boundary < high}


def _new_interaction(items: list[Segment], boundary: int) -> bool:
    non_transition = {"chunk_start", "chunk_end", "continues_prior_situation", "unknown"}
    return any(
        (item.last == boundary and (_continuity(item, "next") == "new_interaction" or _boundary_reason(item, "end") not in non_transition))
        or (item.first == boundary + 1 and (_continuity(item, "previous") == "new_interaction" or _boundary_reason(item, "start") not in non_transition))
        for item in items
    )


def _choose_seam(cues: list[Any], left: list[Segment], right: list[Segment], low: int, high: int) -> tuple[int, dict[str, Any]]:
    lb, rb = _boundaries(left, low, high), _boundaries(right, low, high)
    narrative = sorted(lb | rb); fallback = not narrative; candidates = narrative or list(range(low, high))
    if not candidates: raise ValueError("adjacent chunks have no canonical cue boundary in their overlap")
    midpoint = (cues[low].start_seconds + cues[high].end_seconds) / 2
    def score(boundary: int) -> tuple[int, int, int, float, float, int]:
        consensus = int(boundary in lb and boundary in rb)
        near = int(not consensus and (boundary in lb and min((abs(boundary - x) for x in rb), default=9999) <= 1 or boundary in rb and min((abs(boundary - x) for x in lb), default=9999) <= 1))
        transition = int(_new_interaction(left, boundary) or _new_interaction(right, boundary))
        gap = max(0.0, cues[boundary + 1].start_seconds - cues[boundary].end_seconds)
        centrality = -abs((cues[boundary].end_seconds + cues[boundary + 1].start_seconds) / 2 - midpoint)
        # A declared situation change is stronger evidence than two chunk
        # renderings agreeing on a broad boundary: overlap reconciliation may
        # remove duplicate coverage, never flatten an explicit transition.
        return transition, consensus, near, gap, centrality, -boundary
    chosen = max(candidates, key=score); values = score(chosen)
    reason = "midpoint_fallback" if fallback else "consensus_boundary" if values[1] else "near_consensus_boundary" if values[2] else "narrative_boundary"
    if values[0]: reason += "+new_interaction"
    return chosen, {"candidate_seams": [{"last_left_owned_cue_id": cues[x].cue_id, "first_right_owned_cue_id": cues[x + 1].cue_id} for x in candidates], "chosen_seam": {"last_left_owned_cue_id": cues[chosen].cue_id, "first_right_owned_cue_id": cues[chosen + 1].cue_id}, "chosen_seam_reason": reason, "fallback": fallback}


def _known(item: Segment) -> int:
    return int(all(_value(item, name) not in {None, "unknown", "unclear"} for name in ("segment_type", "narrative_function", "narrative_tone")))


def _bridge_key(item: Segment) -> tuple[float, int, float, int, str, str]:
    summary = item.payload.get("narrative_summary", {})
    confidence = float(summary.get("confidence", 0)) if isinstance(summary, dict) else 0.0
    margin = min(item.first - item.chunk_first, item.chunk_last - item.last)
    return -margin, -_known(item), -confidence, -item.length, item.chunk_id, item.source_segment_id


def _reconcile_boundary(left: list[Segment], right: list[Segment], seam: int) -> tuple[list[Segment], dict[str, Any]]:
    left_cross, right_cross = [x for x in left if _crosses(x, seam)], [x for x in right if _crosses(x, seam)]
    protected = _new_interaction(left, seam) or _new_interaction(right, seam)
    # A crossing map range is allowed to bridge duplicate overlap only. At an
    # explicit mapper boundary, ownership is split at the seam instead, so a
    # broad response from the adjacent chunk cannot erase that boundary.
    bridge = None if protected else (min(left_cross + right_cross, key=_bridge_key) if left_cross or right_cross else None)
    retained = [x for x in left if x.last <= seam] + [x for x in right if x.first > seam]
    clipped: list[dict[str, str]] = []
    if protected:
        # Retain each side of an explicit boundary by clipping only duplicate
        # overlap ownership. This is not a semantic merge: each retained piece
        # keeps its original mapper provenance and its situation metadata.
        for item in left_cross:
            retained.append(_clip(item, item.first, seam))
            clipped.append({"chunk_id": item.chunk_id, "segment_id": item.source_segment_id, "owned_side": "left"})
        for item in right_cross:
            retained.append(_clip(item, seam + 1, item.last))
            clipped.append({"chunk_id": item.chunk_id, "segment_id": item.source_segment_id, "owned_side": "right"})
    suppressed: list[dict[str, str]] = []
    if bridge:
        for item in left + right:
            if item != bridge and _overlaps(item, bridge): suppressed.append({"chunk_id": item.chunk_id, "segment_id": item.source_segment_id})
        retained = [x for x in retained if not _overlaps(x, bridge)] + [bridge]
    unique = {item.key: item for item in retained}
    retained = sorted(unique.values(), key=lambda x: (x.first, x.last, x.chunk_id, x.source_segment_id))
    if any(_overlaps(a, b) for a, b in zip(retained, retained[1:])): raise ValueError("seam selection produced overlapping segments")
    return retained, {"crossing_left_segment": left_cross[0].key if left_cross else None, "crossing_right_segment": right_cross[0].key if right_cross else None, "bridge_winner": bridge.key if bridge else None, "bridge_reason": "duplicate_overlap_context_margin_then_semantics_confidence_coverage_source_order" if bridge else None, "preserved_mapper_boundary": protected, "clipped_for_overlap_ownership": clipped, "suppressed_segments": suppressed}


def _concatenate_canonical_gap(left: list[Segment], right: list[Segment]) -> tuple[list[Segment], dict[str, Any]]:
    """Join disjoint canonical ranges without treating a subtitle gap as ambiguity."""
    if any(_overlaps(a, b) for a in left for b in right):
        raise ValueError("conflicting overlapping cue coverage cannot be concatenated as a canonical gap")
    combined = sorted({item.key: item for item in left + right}.values(), key=lambda x: (x.first, x.last, x.chunk_id, x.source_segment_id))
    if any(_overlaps(a, b) for a, b in zip(combined, combined[1:])):
        raise ValueError("canonical-gap concatenation produced overlapping segments")
    return combined, {"bridge_winner": None, "bridge_reason": None, "suppressed_segments": [],
                      "crossing_left_segment": None, "crossing_right_segment": None}


def validate_consolidated_map(data: dict[str, Any], cues_path: Path) -> list[str]:
    errors: list[str] = []; cues = load_canonical_cues(cues_path); positions = {cue.cue_id: i for i, cue in enumerate(cues)}
    if data.get("schema_version") != "narrative_map_v1": errors.append("schema_version must be narrative_map_v1")
    if data.get("source", {}).get("type") != "external_srt" or data.get("source", {}).get("literal_transcription") is not False: errors.append("source must preserve external_srt and literal_transcription=false")
    prior, ids, ranges = -1, set(), set()
    for number, segment in enumerate(data.get("segments", []), 1):
        label = f"segment {number}"; first_id, last_id = segment.get("first_cue_id"), segment.get("last_cue_id")
        if segment.get("segment_id") in ids: errors.append(f"{label}: duplicate segment_id")
        ids.add(segment.get("segment_id"))
        if first_id not in positions or last_id not in positions: errors.append(f"{label}: unknown cue range"); continue
        first, last = positions[first_id], positions[last_id]
        if first > last: errors.append(f"{label}: reversed cue range")
        if first <= prior: errors.append(f"{label}: unexplained overlap or out-of-order segment")
        prior = max(prior, last)
        if (first_id, last_id) in ranges: errors.append(f"{label}: duplicate cue range")
        ranges.add((first_id, last_id))
        if segment.get("cue_ids") != _expanded(cues, first, last): errors.append(f"{label}: cue_ids are not contiguous canonical range")
        if segment.get("start_seconds") != cues[first].start_seconds or segment.get("end_seconds") != cues[last].end_seconds: errors.append(f"{label}: timestamps do not match canonical cues")
        source = segment.get("semantic_source", {})
        if not isinstance(source, dict) or not source.get("chunk_id") or not source.get("segment_id"): errors.append(f"{label}: missing semantic provenance")
    return errors


def _load(run: Path, cues_path: Path):
    cues = load_canonical_cues(cues_path); ordinal = {cue.cue_id: i for i, cue in enumerate(cues)}; inputs = sorted((run / "chunks").glob("NCHUNK_*.input.json")); maps = sorted((run / "maps").glob("NCHUNK_*.narrative_map.json"))
    if not inputs or len(inputs) != len(maps): raise ValueError("validated chunk inputs and maps must exist one-for-one")
    chunks, by_chunk, checksums = [], {}, {"canonical_srt_sha256": sha256_file(cues_path)}
    for path in inputs:
        chunk_id = path.name.removesuffix(".input.json"); map_path = run / "maps" / f"{chunk_id}.narrative_map.json"; errors = validate_narrative_map(path, map_path)
        if errors: raise ValueError(f"invalid map {chunk_id}: {errors[0]}")
        source, mapped = json.loads(path.read_text()), json.loads(map_path.read_text()); chunk = source["chunk"]; ordinals = [ordinal[x["cue_id"]] for x in source["cues"]]
        chunks.append(chunk); checksums[f"{chunk_id}_map_sha256"] = sha256_file(map_path); by_chunk[chunk_id] = []
        for payload in mapped["segments"]:
            first, last = ordinal[payload["cue_ids"][0]], ordinal[payload["cue_ids"][-1]]
            if payload["cue_ids"] != _expanded(cues, first, last): raise ValueError(f"{chunk_id}:{payload['segment_id']} cue_ids are not contiguous")
            by_chunk[chunk_id].append(Segment(chunk_id, payload["segment_id"], first, last, payload, min(ordinals), max(ordinals)))
    return cues, by_chunk, chunks, checksums, json.loads((run / "narrative_run.json").read_text())


def _payload(item: Segment, cues: list[Any]) -> dict[str, Any]:
    result = dict(item.payload); result.update({"first_cue_id": cues[item.first].cue_id, "last_cue_id": cues[item.last].cue_id, "cue_ids": _expanded(cues, item.first, item.last), "start_seconds": cues[item.first].start_seconds, "end_seconds": cues[item.last].end_seconds, "semantic_source": {"chunk_id": item.chunk_id, "segment_id": item.source_segment_id}, "source_segments": [{"chunk_id": item.chunk_id, "segment_id": item.source_segment_id}]}); result.pop("boundary", None); result.pop("dialogue_density", None); return result


def consolidate_narrative(input_dir: Path, output: callable = print) -> dict[str, Any]:
    movie_id = input_dir.name; root = input_dir.resolve().parents[1] / "runs" / movie_id; run = root / "narrative-v2"
    # Automatic narrative mapping owns source-v1; keep source-inspect-v1 as a
    # compatible older import location without requiring an operator bridge.
    cues_path = next((path for path in (root / "source-v1" / "srt_cues.jsonl", root / "source-inspect-v1" / "srt_cues.jsonl") if path.is_file()), root / "source-v1" / "srt_cues.jsonl")
    if not cues_path.is_file(): raise ValueError(f"canonical SRT cues missing: {cues_path}")
    old = run / "reconciliation_report.json"
    if old.is_file() and MERGE_PROFILE not in old.read_text(): shutil.copyfile(old, run / "reconciliation_report.semantic_merge_v1.json")
    cues, by_chunk, chunks, checksums, manifest = _load(run, cues_path); selected = list(by_chunk[chunks[0]["chunk_id"]]); reports = []
    for left_chunk, right_chunk in zip(chunks, chunks[1:]):
        left_id, right_id = left_chunk["chunk_id"], right_chunk["chunk_id"]
        left_items, right_items = by_chunk[left_id], by_chunk[right_id]
        if not left_items or not right_items:
            raise ValueError(f"{left_id}/{right_id}: chunk map has no semantic segments")
        left_first, left_last = min(x.chunk_first for x in left_items), max(x.chunk_last for x in left_items)
        right_first, right_last = min(x.chunk_first for x in right_items), max(x.chunk_last for x in right_items)
        shared = sorted(set(range(left_first, left_last + 1)) & set(range(right_first, right_last + 1)))
        if not shared:
            if left_last < right_first:
                selected, bridge = _concatenate_canonical_gap(selected, right_items)
                reports.append({"left_chunk": left_id, "right_chunk": right_id, "seam_type": "canonical_gap",
                                "left_last_cue_id": cues[left_last].cue_id, "right_first_cue_id": cues[right_first].cue_id,
                                **bridge, "resulting_overlap_count": 0})
                continue
            raise ValueError(f"{left_id}/{right_id}: conflicting overlapping cue coverage without a common canonical cue")
        if len(shared) == 1:
            seam = shared[0]
            evidence = {"seam_type": "single_cue_overlap", "candidate_seams": [{"last_left_owned_cue_id": cues[seam].cue_id}],
                        "chosen_seam": {"last_left_owned_cue_id": cues[seam].cue_id}, "chosen_seam_reason": "single_shared_cue_deterministic", "fallback": False}
        else:
            seam, evidence = _choose_seam(cues, left_items, right_items, shared[0], shared[-1])
            evidence["seam_type"] = "canonical_overlap"
        selected, bridge = _reconcile_boundary(selected, right_items, seam)
        reports.append({"left_chunk": left_id, "right_chunk": right_id, "overlap_start": cues[shared[0]].cue_id, "overlap_end": cues[shared[-1]].cue_id, **evidence, **bridge, "resulting_overlap_count": 0})
    final = [_payload(x, cues) for x in selected]; final.sort(key=lambda x: (x["start_seconds"], x["end_seconds"], x["semantic_source"]["chunk_id"])); [item.update(segment_id=f"NARR_{index:06d}") for index, item in enumerate(final, 1)]
    data = {"schema_version": "narrative_map_v1", "source": {"movie_id": movie_id, "type": "external_srt", "literal_transcription": False, "timing_reliability": "good", "language": "es"}, "analysis": {"provider": manifest.get("provider"), "model": manifest.get("model"), "prompt_version": manifest.get("prompt_version"), "chunk_profile": f"{float(manifest.get('window_seconds')):g}s_{float(manifest.get('overlap_seconds')):g}s_overlap", "merge_profile": MERGE_PROFILE}, "provenance": {"checksums": checksums}, "segments": final}
    errors = validate_consolidated_map(data, cues_path); assigned = {cue for item in final for cue in item["cue_ids"]}
    report = {"schema_version": "narrative_reconciliation_report_v2", "merge_profile": MERGE_PROFILE, "status": "PASS" if not errors else "FAIL", "checksums": checksums, "chunk_boundaries": reports, "segments_before": sum(map(len, by_chunk.values())), "segments_after": len(final), "source_cue_coverage": {"assigned_cue_count": len(assigned), "unassigned_cue_count": len(cues) - len(assigned)}, "validation_errors": errors, "unresolved_seam_ambiguities": 0}
    write_json(run / "narrative_map.json", data); write_json(run / "reconciliation_report.json", report); output(f"[narrative] consolidate status={report['status']} segments={len(final)} seams={len(reports)}"); return report
