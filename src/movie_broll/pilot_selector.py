"""Narrative-first, deterministic selection of the next bounded B-roll pilot."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .utils import write_json

REGISTRY_SCHEMA = "pilot_windows_v1"
NOMINAL_SECONDS = 60.0
MINIMUM_SECONDS = 45.0
MAXIMUM_SECONDS = 75.0
_SPECIFIC_RELATIONSHIPS = {
    "mother_daughter": ("mother and daughter", "madre e hija", "madre e hija"),
    "father_daughter": ("father and daughter", "padre e hija"),
    "mother_son": ("mother and son", "madre e hijo"),
    "father_son": ("father and son", "padre e hijo"),
    "siblings": ("siblings", "hermanos", "hermanas"),
    "romantic_partner": ("romantic partner", "pareja", "novios"),
}
_NON_STORY_TERMS = ("credits", "créditos", "title card", "title sequence", "opening logo", "production logo")


def _root(input_dir: Path) -> Path:
    return input_dir.resolve().parents[1]


def registry_path(input_dir: Path) -> Path:
    return _root(input_dir) / "runs" / input_dir.name / "pilot_windows.json"


def _value(segment: dict[str, Any], name: str) -> Any:
    value = segment.get(name)
    return value.get("value") if isinstance(value, dict) else value


def _values(segment: dict[str, Any], name: str) -> list[str]:
    value = segment.get(name, [])
    if not isinstance(value, list):
        return []
    return [str(item.get("value")) for item in value if isinstance(item, dict) and item.get("value")]


def _overlap(a: float, b: float, c: float, d: float) -> float:
    return max(0.0, min(b, d) - max(a, c))


def _relation_hints(segment: dict[str, Any]) -> list[str]:
    """Only use relations explicitly stated by the canonical narrative text."""
    summary = str(_value(segment, "narrative_summary") or "").lower()
    return [name for name, phrases in _SPECIFIC_RELATIONSHIPS.items() if any(phrase in summary for phrase in phrases)]


def profile_for_range(narrative: dict[str, Any], start: float, end: float) -> dict[str, Any]:
    segments = [s for s in narrative.get("segments", []) if _overlap(start, end, float(s["start_seconds"]), float(s["end_seconds"])) > 0]
    functions = sorted({str(_value(s, "narrative_function")) for s in segments if _value(s, "narrative_function") not in {None, "unknown"}})
    tones = sorted({str(_value(s, "narrative_tone")) for s in segments if _value(s, "narrative_tone") not in {None, "unclear"}})
    hints = sorted({str(_value(s, "segment_type")) for s in segments if _value(s, "segment_type") not in {None, "unknown"}} | {x for s in segments for x in _values(s, "possible_visual_opportunities") if x != "unknown"})
    relations = sorted({x for s in segments for x in _relation_hints(s)})
    result: dict[str, Any] = {
        "narrative_segment_ids": [str(s["segment_id"]) for s in segments],
        "interaction_context": functions,
        "tone": tones,
        "content_hints": hints,
    }
    if relations:
        result["relationship_hints"] = relations
    return result


def _movie_duration(input_dir: Path, narrative: dict[str, Any]) -> float:
    source = _root(input_dir) / "runs" / input_dir.name / "source-inspect-v1" / "source_manifest.json"
    if source.is_file():
        try:
            value = json.loads(source.read_text(encoding="utf-8"))["source"]["movie"]["duration_seconds"]
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return max((float(x.get("end_seconds", 0)) for x in narrative.get("segments", [])), default=0.0)


def _window_for_segment(segment: dict[str, Any], duration: float) -> tuple[float, float, float]:
    left, right = float(segment["start_seconds"]), float(segment["end_seconds"])
    span = right - left
    length = min(MAXIMUM_SECONDS, max(MINIMUM_SECONDS, min(NOMINAL_SECONDS, duration)))
    # The coherent context itself bounds long segments. Short segments may use a
    # little adjacent timeline, but their coherence score makes that explicit.
    if span >= length:
        center = (left + right) / 2
        start = max(left, min(center - length / 2, right - length))
    else:
        start = max(0.0, min((left + right) / 2 - length / 2, max(0.0, duration - length)))
    end = min(duration, start + length)
    start = max(0.0, end - length)
    return round(start, 3), round(end, 3), min(1.0, _overlap(start, end, left, right) / max(end - start, 0.001))


def _next_id(windows: list[dict[str, Any]]) -> str:
    used = {int(m.group(1)) for row in windows if (m := re.fullmatch(r"SW_(\d+)", str(row.get("window_id", ""))))}
    number = 1
    while number in used:
        number += 1
    return f"SW_{number:02d}"


def _load_registry(input_dir: Path, narrative: dict[str, Any]) -> dict[str, Any]:
    path = registry_path(input_dir)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data.get("windows"), list):
            return data
    # Legacy visual windows are retained as records. A completed B-roll manifest
    # makes them TESTED; otherwise they remain historical selections, never
    # silently renumbered or overwritten.
    smoke = _root(input_dir) / "runs" / input_dir.name / "visual-smoke-v1" / "windows.json"
    windows: list[dict[str, Any]] = []
    if smoke.is_file():
        for row in json.loads(smoke.read_text(encoding="utf-8")).get("windows", []):
            ident = row.get("window_id")
            if not ident:
                continue
            output = _root(input_dir) / "runs" / input_dir.name / "broll-pilot-v1" / str(ident) / "candidates.json"
            start, end = float(row["start_seconds"]), float(row["end_seconds"])
            windows.append({"window_id": ident, "start_seconds": start, "end_seconds": end, "duration": round(end - start, 3), "narrative_segment_ids": row.get("source_narrative_segment_ids", []), "profile": profile_for_range(narrative, start, end), "status": "TESTED" if output.is_file() else "LEGACY"})
    return {"schema_version": REGISTRY_SCHEMA, "movie_id": input_dir.name, "windows": windows}


def _feature_difference(candidate: dict[str, Any], prior: dict[str, Any], field: str) -> float:
    a, b = set(candidate["profile"].get(field, [])), set(prior.get("profile", {}).get(field, []))
    if not a:
        return 0.0
    return 1.0 - len(a & b) / len(a | b) if a | b else 0.0


def _score(candidate: dict[str, Any], tested: list[dict[str, Any]], duration: float) -> tuple[float, list[str]]:
    if any(_overlap(candidate["start_seconds"], candidate["end_seconds"], float(x["start_seconds"]), float(x["end_seconds"])) > 0 for x in tested):
        return -1.0, ["overlaps_existing_tested_window"]
    if not tested:
        return round(.45 + .25 * candidate["coherence"], 4), ["coherent_narrative_context"]
    narrative = min(_feature_difference(candidate, x, "narrative_segment_ids") for x in tested)
    interaction = min(_feature_difference(candidate, x, "interaction_context") for x in tested)
    tone = min(_feature_difference(candidate, x, "tone") for x in tested)
    content = min(_feature_difference(candidate, x, "content_hints") for x in tested)
    midpoint = (candidate["start_seconds"] + candidate["end_seconds"]) / 2
    temporal = min(1.0, min(abs(midpoint - (float(x["start_seconds"]) + float(x["end_seconds"])) / 2) / max(duration * .25, 1.0) for x in tested))
    score = .28 * narrative + .22 * interaction + .16 * tone + .14 * content + .10 * temporal + .10 * candidate["coherence"]
    reasons = ["different_narrative_segment"]
    if interaction >= .5: reasons.append("different_interaction_context")
    if tone >= .5: reasons.append("different_tone")
    if temporal >= .5: reasons.append("different_movie_region")
    if candidate["coherence"] >= .75: reasons.append("coherent_narrative_context")
    return round(score, 4), reasons


def _story_candidate(segment: dict[str, Any], start: float, end: float, duration: float) -> bool:
    """Cheap timeline/text guards; this is deliberately not visual classification."""
    if start < 30.0 or end > duration - 30.0:
        return False
    summary = str(_value(segment, "narrative_summary") or "").lower()
    return not any(term in summary for term in _NON_STORY_TERMS)


def select_next(input_dir: Path) -> dict[str, Any]:
    narrative_path = _root(input_dir) / "runs" / input_dir.name / "narrative-v2" / "narrative_map.json"
    if not narrative_path.is_file():
        raise FileNotFoundError(f"required narrative map does not exist: {narrative_path}")
    narrative = json.loads(narrative_path.read_text(encoding="utf-8"))
    duration = _movie_duration(input_dir, narrative)
    if duration <= 0 or not narrative.get("segments"):
        raise ValueError("narrative map requires ordered segments and a positive duration")
    registry = _load_registry(input_dir, narrative)
    selected = [x for x in registry["windows"] if x.get("status") == "SELECTED"]
    if selected:  # Idempotent until an explicit pilot attempt changes lifecycle.
        write_json(registry_path(input_dir), registry)
        return selected[0]
    tested = [x for x in registry["windows"] if x.get("status") in {"TESTED", "LEGACY"}]
    options = []
    for segment in narrative["segments"]:
        start, end, coherence = _window_for_segment(segment, duration)
        if not _story_candidate(segment, start, end, duration):
            continue
        candidate = {"start_seconds": start, "end_seconds": end, "duration": round(end - start, 3), "narrative_segment_ids": [segment["segment_id"]], "profile": profile_for_range(narrative, start, end), "coherence": coherence}
        score, reasons = _score(candidate, tested, duration)
        if score >= 0:
            candidate.update(diversity_score=score, selection_reason=reasons)
            options.append(candidate)
    if not options:
        raise ValueError("no non-overlapping coherent narrative pilot window is available")
    winner = max(options, key=lambda x: (x["diversity_score"], x["coherence"], -x["start_seconds"], str(x["narrative_segment_ids"][0])))
    winner.pop("coherence")
    winner.update(window_id=_next_id(registry["windows"]), status="SELECTED", selected=True)
    registry["windows"].append(winner)
    write_json(registry_path(input_dir), registry)
    return winner


def mark_attempted(input_dir: Path, window_id: str, pilot_status: str) -> None:
    """A partial quota run is still the same selected window, never a new one."""
    path = registry_path(input_dir)
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    for window in data.get("windows", []):
        if window.get("window_id") == window_id:
            window["status"] = "TESTED"
            window["pilot_status"] = pilot_status
            write_json(path, data)
            return
