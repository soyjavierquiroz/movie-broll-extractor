"""Bounded visual inputs; never infer actions from text or change event ranges."""
from __future__ import annotations

import math
from typing import Any

PROFILE = "temporal_evidence_v2"
MAX_FRAMES = 16
TILE_WIDTH = 480


def sample_plan(event: dict[str, Any], fps: float) -> list[dict[str, Any]]:
    """Cover every shot midpoint and event endpoints, then bounded shot endpoints.

    Additional quarter samples require an existing action/interaction hint or
    motion >= 12. No subtitle/narrative fields participate in this predicate.
    Events with more shots than the budget fail locally rather than omit shots.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("temporal evidence requires positive finite fps")
    shots = event.get("technical_shots") or []
    if not shots and len(event.get("source_shot_ids", [])) == 1:
        shots = [{**event, "shot_id": event["source_shot_ids"][0]}]
    if not shots:
        raise ValueError("temporal evidence requires technical-shot ranges")
    ranges = []
    for shot in shots:
        start = int(shot["start_frame"]) if "start_frame" in shot else round(float(shot["start_seconds"]) * fps)
        end = int(shot["end_frame_exclusive"]) if "end_frame_exclusive" in shot else round(float(shot["end_seconds"]) * fps)
        start = max(start, int(event["start_frame"]))
        end = min(end, int(event["end_frame_exclusive"]))
        if end <= start:
            raise ValueError("empty temporal shot range")
        ranges.append((str(shot["shot_id"]), start, end))
    ranges.sort(key=lambda row: (row[1], row[0]))
    if len({sid for sid, _, _ in ranges}) != len(ranges):
        raise ValueError("duplicate technical-shot range")
    selected: dict[tuple[str, int], set[str]] = {}

    def add(row, frame, role):
        key = (row[0], frame)
        if key in selected:
            selected[key].add(role)
        elif len(selected) < MAX_FRAMES:
            selected[key] = {role}
        else:
            return False
        return True

    for row in ranges:
        if not add(row, (row[1] + row[2] - 1) // 2, "middle"):
            raise ValueError("technical-shot coverage exceeds temporal frame cap")
    # Endpoints are mandatory, including when midpoint coverage uses the cap.
    for row, frame, role in [(ranges[0], ranges[0][1], "begin"),
                             (ranges[-1], ranges[-1][2] - 1, "end")]:
        if not add(row, frame, role):
            raise ValueError("event endpoints exceed temporal frame cap")
    for role in ("begin", "end"):
        for row in ranges:
            add(row, row[1] if role == "begin" else row[2] - 1, role)
    extra = (event.get("event_type_hint") in {"action", "interaction", "movement", "activity"}
             or float(event.get("signals", {}).get("motion", 0)) >= 12)
    if extra:
        for row in sorted(ranges, key=lambda row: (-(row[2] - row[1]), row[1])):
            for fraction, role in [(1 / 4, "development"), (3 / 4, "development")]:
                add(row, row[1] + int((row[2] - row[1] - 1) * fraction), role)
    order = {"begin": 0, "middle": 1, "end": 2, "development": 3}
    return [{"sample_id": f"SAMPLE_{i:02d}", "shot_id": sid, "frame": frame,
             "timestamp_seconds": frame / fps,
             "roles": sorted(roles, key=order.__getitem__),
             "role": "+".join(sorted(roles, key=order.__getitem__))}
            for i, ((sid, frame), roles) in enumerate(sorted(selected.items(), key=lambda x: (x[0][1], x[0][0])), 1)]
