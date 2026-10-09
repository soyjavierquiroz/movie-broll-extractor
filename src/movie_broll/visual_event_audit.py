"""Cheap health reporting for already-persisted Visual Event manifests."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("events"), list):
        raise ValueError(f"Visual Event manifest has no events array: {path}")
    return value


def find_event_manifest(input_dir: Path) -> Path:
    run = Path("runs") / input_dir.name
    for path in (run / "visual_events.json", run / "visual_event_segments_v1.json"):
        if path.is_file():
            return path
    pilots = sorted((run / "broll-pilot-v1").glob("*/visual_events_manifest.json"))
    if len(pilots) == 1:
        return pilots[0]
    if len(pilots) > 1:
        raise ValueError("multiple pilot Visual Event manifests found; use a title with one persisted manifest")
    raise FileNotFoundError(f"no persisted Visual Event manifest found under {run}")


def summarize_events(events: list[dict[str, Any]], technical_shots: int | None = None) -> dict[str, Any]:
    durations = [float(event.get("duration_seconds", float(event["end_seconds"]) - float(event["start_seconds"]))) for event in events]
    buckets = {"<4s": 0, "4-10s": 0, "10-18s": 0, "18-20s": 0, ">20s": 0}
    for duration in durations:
        if duration < 4:
            buckets["<4s"] += 1
        elif duration < 10:
            buckets["4-10s"] += 1
        elif duration < 18:
            buckets["10-18s"] += 1
        elif duration <= 20:
            buckets["18-20s"] += 1
        else:
            buckets[">20s"] += 1
    single = sum(len(event.get("source_shot_ids", [])) == 1 for event in events)
    multi = sum(len(event.get("source_shot_ids", [])) > 1 for event in events)
    total = len(events)
    return {"technical_shots": technical_shots, "visual_events": total,
            "technical_shots_per_visual_event": round(technical_shots / total, 2) if total else 0.0,
            "single_shot_events": single, "single_shot_event_percentage": round(100 * single / total, 1) if total else 0.0,
            "multi_shot_events": multi, "multi_shot_event_percentage": round(100 * multi / total, 1) if total else 0.0,
            "duration": {"min": min(durations) if durations else 0.0,
            "median": statistics.median(durations) if durations else 0.0, "mean": statistics.mean(durations) if durations else 0.0,
            "max": max(durations) if durations else 0.0}, "buckets": {key: {"count": count,
            "percentage": round(100 * count / total, 1) if total else 0.0} for key, count in buckets.items()}}


def audit_events(input_dir: Path) -> dict[str, Any]:
    manifest_path = find_event_manifest(input_dir)
    data = _load(manifest_path)
    run = Path("runs") / input_dir.name
    technical_path = run / "technical_shots.json"
    technical_shots = None
    if technical_path.is_file():
        technical = json.loads(technical_path.read_text(encoding="utf-8"))
        technical_shots = len(technical.get("shots", [])) if isinstance(technical, dict) else None
    if technical_shots is None:
        technical_shots = len({shot for event in data["events"] for shot in event.get("source_shot_ids", [])})
    return {"manifest": str(manifest_path), **summarize_events(data["events"], technical_shots)}


def print_audit(report: dict[str, Any], output: callable = print) -> None:
    output("=== VISUAL EVENT AUDIT ===")
    output(f"technical_shots: {report['technical_shots']}")
    output(f"visual_events: {report['visual_events']}")
    output(f"technical_shots_per_visual_event: {report['technical_shots_per_visual_event']:.2f}")
    output(f"single_shot_events: {report['single_shot_events']} ({report['single_shot_event_percentage']:.1f}%)")
    output(f"multi_shot_events: {report['multi_shot_events']} ({report['multi_shot_event_percentage']:.1f}%)")
    output("duration:")
    for key in ("min", "median", "mean", "max"):
        output(f"  {key}: {report['duration'][key]:.2f}s")
    output("buckets:")
    for key, value in report["buckets"].items():
        output(f"  {key}: {value['count']} ({value['percentage']:.1f}%)")
