"""Post-semantic Scene Run and Editorial Beat partitioning.

Technical cuts are useful input evidence, not asset boundaries.  This module is
intentionally between the first per-shot semantic pass and final multi-shot
semantic validation: it records the evidence used for every join, but never
copies per-shot semantics onto a newly formed asset.
"""
from __future__ import annotations

import re
from typing import Any

from .broll_pilot import _candidate_from_part, adjacent_continuity, dedupe

VERSION = "editorial_beat_builder_v2_cross_narrative_scene_runs"
MAX_EXTENDED_BEAT_SECONDS = 20.0


def _tokens(value: Any) -> set[str]:
    return {x for x in re.findall(r"[\wáéíóúñ]+", str(value).lower().replace("_", " ")) if len(x) > 2}


def _semantic_tokens(item: dict[str, Any]) -> set[str]:
    visual = item.get("visual", {})
    return set().union(
        _tokens(visual.get("actions", [])),
        _tokens(visual.get("visible_interactions", [])),
        _tokens(item.get("editorial", {}).get("standalone_meaning_es", "")),
    )


def _conversation_context(item: dict[str, Any]) -> bool:
    values = item.get("narrative", {}).get("interaction_context", [])
    values = values if isinstance(values, list) else [values]
    words = _tokens(values)
    return bool(words & {
        "conversation", "dialogue", "dialogo", "diálogo", "interaction",
        "interacción", "emotional", "exchange", "intercambio", "therapy",
        "terapia", "argument", "discusion", "discusión",
    })


def _role(tokens: set[str]) -> str | None:
    speaker = {"talking", "speaking", "habla", "hablando", "conversa", "conversando"}
    listener = {"listen", "listening", "escucha", "escuchando", "reaction", "reaccion", "reacción"}
    if tokens & speaker:
        return "speaker"
    if tokens & listener:
        return "listener"
    return None


def _explicit_break(edge: dict[str, Any]) -> bool:
    return any(reason != "no_positive_scene_or_interaction_continuity_evidence"
               for reason in edge.get("split_reasons", []))


def scene_edge(left: dict[str, Any], right: dict[str, Any],
               technical: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return recorded positive evidence that two adjacent shots share a scene.

    Explicit detector/tracker continuity remains authoritative.  When it is not
    present, the fallback uses *existing* semantic output plus synchronized
    narrative context.  Narrative segmentation describes story phases, not
    physical scenes, so a narrative boundary is soft evidence only: it can
    lower confidence but cannot force a scene split.  This is a labelled
    inference, not fabricated scene, person, or conversation IDs.
    """
    base = {"from_shot_id": left["source_shot_ids"][0], "to_shot_id": right["source_shot_ids"][0]}
    raw_left = technical.get(left["source_shot_ids"][0], left)
    raw_right = technical.get(right["source_shot_ids"][0], right)
    if abs(float(right["start_seconds"]) - float(left["end_seconds"])) > 0.1:
        return {**base, "merge": False, "merge_reasons": [], "split_reasons": ["temporal_gap_between_technical_shots"],
                "evidence": {"temporally_adjacent": False}, "evidence_source": "timeline"}
    raw = adjacent_continuity(raw_left, raw_right)
    if raw["merge"]:
        return {**base, **raw, "evidence_source": "technical_scene_metadata"}
    if _explicit_break(raw):
        return {**base, **raw, "evidence_source": "technical_scene_metadata"}

    left_segments = set(left.get("narrative_segment_ids", []))
    right_segments = set(right.get("narrative_segment_ids", []))
    left_setting = str(left.get("visual", {}).get("setting", "")).strip().lower()
    right_setting = str(right.get("visual", {}).get("setting", "")).strip().lower()
    left_tokens, right_tokens = _semantic_tokens(left), _semantic_tokens(right)
    left_role, right_role = _role(left_tokens), _role(right_tokens)
    speaker_listener = {left_role, right_role} == {"speaker", "listener"}
    same_segment = bool(left_segments & right_segments)
    same_setting = bool(left_setting and left_setting == right_setting)
    context = _conversation_context(left) and _conversation_context(right)
    # A semantic setting plus an ongoing interpersonal exchange is positive
    # scene evidence.  Do not require shared narrative IDs here: one continuous
    # therapy conversation can legitimately pass through several narrative
    # situations/phases.
    compatible = same_setting and (context or speaker_listener)
    narrative_boundary = bool(left_segments and right_segments and not same_segment)
    evidence = {
        "same_narrative_segment": same_segment,
        "narrative_boundary": narrative_boundary,
        "narrative_boundary_is_soft_evidence": narrative_boundary,
        "same_semantic_setting": same_setting,
        "semantic_conversation_context": context,
        "speaker_listener": speaker_listener,
        "left_role": left_role,
        "right_role": right_role,
    }
    if compatible:
        reasons = ["semantic_scene_context"]
        if context:
            reasons.append("semantic_conversation_context")
        if speaker_listener:
            reasons.append("speaker_listener")
        return {**base, "merge": True, "merge_reasons": reasons, "split_reasons": [],
                "evidence": evidence, "evidence_source": "semantic_neighbor_results",
                "merge_confidence": "medium" if narrative_boundary else "high"}
    return {**base, "merge": False, "merge_reasons": [],
            "split_reasons": ["insufficient_semantic_scene_continuity_evidence"],
            "evidence": evidence, "evidence_source": "semantic_neighbor_results"}


def scene_runs(items: list[dict[str, Any]], technical_shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(items, key=lambda x: (float(x["start_seconds"]), x["candidate_id"]))
    technical = {x["shot_id"]: x for x in technical_shots}
    runs: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for item in ordered:
        if not current:
            current = [item]
            continue
        edge = scene_edge(current[-1], item, technical)
        if edge["merge"]:
            current.append(item)
            pairs.append(edge)
        else:
            runs.append({"shots": current, "adjacent_pairs": pairs})
            current, pairs = [item], []
    if current:
        runs.append({"shots": current, "adjacent_pairs": pairs})
    for ordinal, run in enumerate(runs, 1):
        run["scene_run_id"] = f"SR_{ordinal:04d}"
        run["narrative_segment_ids"] = list(dict.fromkeys(
            segment_id for shot in run["shots"]
            for segment_id in shot.get("narrative_segment_ids", [])
        ))
    return runs


def _beat_score(part: list[dict[str, Any]], edges: list[dict[str, Any]]) -> float:
    duration = float(part[-1]["end_seconds"]) - float(part[0]["start_seconds"])
    def shot_duration(item: dict[str, Any]) -> float:
        return float(item.get("duration_seconds", float(item["end_seconds"]) - float(item["start_seconds"])))
    if duration < 4.5:
        score = -12.0
    elif 5 <= duration <= 9:
        score = 10.0
    elif 4 <= duration <= 12:
        score = 7.0
    elif 12 < duration <= 18:
        score = 3.0
    elif 18 < duration <= MAX_EXTENDED_BEAT_SECONDS:
        score = -2.0
    else:
        return -10_000.0
    short = any(shot_duration(x) < 4.5 for x in part)
    score += 3.0 * len(edges)
    tokens = [_semantic_tokens(x) for x in part]
    roles = [_role(x) for x in tokens]
    speaker_listener = any({a, b} == {"speaker", "listener"} for a, b in zip(roles, roles[1:]))
    if speaker_listener:
        score += 12.0
    if short and edges:
        score += 6.0  # short-shot rescue is only a reward for a contextual join
    if len(part) == 1 and 8 <= duration <= 12:
        score += 7.0  # complete sustained moments are autonomous beats
    if len(part) == 1 and duration < 4.5 and part[0].get("autonomous_broll_value") is True:
        score += 20.0
    if duration > 18 and not speaker_listener:
        score -= 8.0
    return score


def _partition(run: dict[str, Any]) -> list[tuple[int, int]]:
    shots, pair_edges = run["shots"], run["adjacent_pairs"]
    # Dynamic programming chooses the whole run's partition, instead of greedily
    # publishing a short technical shot before its useful neighbour is known.
    best: list[tuple[float, list[tuple[int, int]]]] = [(-10_000.0, []) for _ in range(len(shots) + 1)]
    best[0] = (0.0, [])
    for end in range(1, len(shots) + 1):
        for start in range(end):
            score = _beat_score(shots[start:end], pair_edges[start:end - 1])
            if score <= -10_000 or best[start][0] <= -10_000:
                continue
            proposal = best[start][0] + score
            if proposal > best[end][0]:
                best[end] = (proposal, best[start][1] + [(start, end)])
    return best[-1][1]


def build_editorial_beats(items: list[dict[str, Any]], technical_shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Partition semantic technical-shot results into final candidate beats."""
    result: list[dict[str, Any]] = []
    for run in scene_runs(items, technical_shots):
        for start, end in _partition(run):
            part = run["shots"][start:end]
            pairs = run["adjacent_pairs"][start:end - 1]
            reasons = sorted({reason for edge in pairs for reason in edge.get("merge_reasons", [])}) or ["single_technical_shot"]
            beat = _candidate_from_part(part, continuity={"merge_reasons": reasons, "adjacent_pairs": pairs})
            beat["scene_run"] = {
                "version": VERSION,
                "scene_run_id": run["scene_run_id"],
                "source_shot_count": len(run["shots"]),
                "narrative_segment_ids": run["narrative_segment_ids"],
                "adjacent_pairs": run["adjacent_pairs"],
            }
            beat["editorial_beat"] = {
                "version": VERSION,
                "partition": "global_dynamic_programming",
                "source_shot_range": [start, end],
                "score": round(_beat_score(part, pairs), 3),
                "duration_guidance": "ideal_5_9_useful_4_12_extended_20",
                "narrative_segment_ids": beat["narrative_segment_ids"],
            }
            result.append(beat)
    return dedupe(result)
