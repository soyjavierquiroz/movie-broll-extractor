"""Single, small production contract shared by inexpensive quality gates."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def load() -> dict[str, Any]:
    return json.loads((Path(__file__).resolve().parents[2] / "config" / "production_v1.json").read_text())


def narrative() -> dict[str, float | str]:
    """Return the single canonical automatic Narrative Mapper profile."""
    profile = load()
    settings = profile.get("narrative")
    if not isinstance(settings, dict):
        raise ValueError("production_v1 narrative configuration is missing")
    window = settings.get("target_chunk_duration_seconds")
    overlap = settings.get("overlap_seconds")
    if isinstance(window, bool) or not isinstance(window, (int, float)) or window <= 0:
        raise ValueError("production_v1 narrative target_chunk_duration_seconds must be positive")
    if isinstance(overlap, bool) or not isinstance(overlap, (int, float)) or overlap < 0 or overlap >= window:
        raise ValueError("production_v1 narrative overlap_seconds must be in [0, window)")
    return {"profile_version": str(profile.get("profile_version", "production_v1")),
            "window_seconds": float(window), "overlap_seconds": float(overlap)}
