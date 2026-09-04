"""One-command, durable full-movie production orchestration.

This module intentionally owns orchestration only.  Shot detection, editorial
grouping, semantics, reframing, and Asset Hub metadata remain implemented by
their already-validated modules.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .broll_pilot import (add_context, apply_semantic_scarcity, candidates,
                          semantic_validate, visual_signals)
from .finalization import finalize_pilot, person_detector_preflight, safe_cleanup
from .inspect_source import inspect_movie
from .processing_ledger import ProcessingLedger, fingerprint
from .srt import parse_srt_file
from .utils import sha256_file, write_json
from .visual import Window, build_shots, detect_cuts

TECHNICAL_VERSION = "full_movie_technical_shots_v1"
EVENT_VERSION = "full_movie_visual_events_v1"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _root(input_dir: Path) -> Path:
    return input_dir.resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def preflight(input_dir: Path) -> dict[str, Any]:
    """Validate inexpensive inputs before any detector or provider work."""
    movie, srt = input_dir / "movie.mp4", input_dir / "subtitles.srt"
    if not movie.is_file():
        raise FileNotFoundError(f"canonical movie.mp4 does not exist: {movie}")
    if not srt.is_file():
        raise FileNotFoundError(f"canonical subtitles.srt does not exist: {srt}")
    metadata = inspect_movie(movie)
    if not metadata.get("video", {}).get("width"):
        raise ValueError("movie has no readable video stream")
    parsed = parse_srt_file(srt)
    if not parsed.cues:
        raise ValueError("SRT contains no usable cues")
    run = _root(input_dir) / "runs" / input_dir.name
    run.mkdir(parents=True, exist_ok=True)
    for path in (run, run / ".work", run / "assets", run / "review"):
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise OSError(f"required directory is not writable: {path}")
    free = shutil.disk_usage(run).free
    if free <= 0:
        raise OSError("no output disk space available")
    narrative = run / "narrative-v2" / "narrative_map.json"
    if not narrative.is_file():
        raise FileNotFoundError(f"canonical Narrative Map does not exist: {narrative}")
    # Narrative compatibility is intentionally SRT-scoped, not movie-pixel-scoped.
    # Its own structural schema is validated by the earlier narrative stage.
    if not isinstance(_read_json(narrative).get("segments"), list):
        raise ValueError("Narrative Map has no segments")
    return {"movie": movie, "srt": srt, "run": run, "narrative": narrative,
            "movie_sha256": sha256_file(movie), "srt_sha256": sha256_file(srt),
            "metadata": metadata, "free_bytes": free, "cue_count": len(parsed.cues)}


def _technical_shots(info: dict[str, Any]) -> list[dict[str, Any]]:
    path = info["run"] / "technical_shots.json"
    md, sha = info["metadata"], info["movie_sha256"]
    existing = _read_json(path) if path.exists() else {}
    if existing.get("source_movie_sha256") == sha and existing.get("version") == TECHNICAL_VERSION:
        return existing["shots"]
    fps = float(md["video"]["fps"])
    duration = float(md["duration_seconds"])
    window = Window("FULL", 0.0, duration, "production", [])
    cuts = detect_cuts(info["movie"], 0, round(duration * fps), 24.0)
    shots = build_shots(window, fps, cuts, 24.0)
    write_json(path, {"schema_version": "technical_shots_v1", "version": TECHNICAL_VERSION,
                      "source_movie_sha256": sha, "fps": fps, "shots": shots})
    return shots


def _events(info: dict[str, Any], shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = info["run"] / "visual_events.json"
    event_fp = fingerprint({"source_movie_sha256": info["movie_sha256"],
                            "technical": TECHNICAL_VERSION, "grouping": EVENT_VERSION})
    existing = _read_json(path) if path.exists() else {}
    if existing.get("fingerprint") == event_fp:
        return existing["events"]
    add_context(shots, info["srt"], info["narrative"])
    for signal, shot in zip(visual_signals(info["movie"], shots), shots):
        shot.update(signal)
    events = candidates(shots)
    # Identity comes from source-frame authority, rather than a window ordinal.
    # Thus it is stable for the same full movie and cannot collide with pilots.
    for event in events:
        event["visual_event_id"] = "VE_" + fingerprint({"start": event["start_frame"],
                                                            "end": event["end_frame_exclusive"],
                                                            "shots": event["source_shot_ids"]})[:16]
    write_json(path, {"schema_version": "visual_events_v1", "version": EVENT_VERSION,
                      "fingerprint": event_fp, "events": events})
    return events


def _invalidate_source_media(run: Path, old_sha: str | None, current_sha: str) -> None:
    """Retire only source-pixel artifacts; Narrative Map deliberately survives."""
    if old_sha == current_sha:
        return
    for name in ("technical_shots.json", "visual_events.json"):
        (run / name).unlink(missing_ok=True)
    for directory in (run / "assets", run / "review", run / "semantic_checkpoints"):
        # Explicitly guard the only two non-work directories this runner owns.
        if directory.parent.resolve() != run.resolve() or directory.name not in {"assets", "review", "semantic_checkpoints"}:
            raise ValueError("refusing source invalidation outside owned run directories")
        if directory.exists():
            for child in directory.iterdir():
                if child.is_dir(): shutil.rmtree(child)
                else: child.unlink()
    safe_cleanup(run / ".work")


def _summary(info: dict[str, Any], ledger: ProcessingLedger, events: list[dict[str, Any]], state: str,
             final: dict[str, Any] | None = None) -> dict[str, Any]:
    records = ledger.data["events"].values()
    semantic = [x.get("stages", {}).get("semantic", {}) for x in records]
    editorial = {key: sum(1 for x in events if x.get("editorial", {}).get("decision") == key)
                 for key in ("KEEP", "REVIEW", "REJECT")}
    run = info["run"]
    value = {"schema_version": "production_progress_summary_v1", "movie_id": run.name,
             "source": {"movie_sha256": info["movie_sha256"], "srt_sha256": info["srt_sha256"]},
             "run_state": state, "updated_at": _utc(), "technical_shots": len(_technical_shots(info)),
             "visual_events": len(events),
             "semantic": {"complete": sum(x.get("status") == "COMPLETE" for x in semantic),
                          "retryable": sum(x.get("status") == "FAILED_RETRYABLE" for x in semantic),
                          "failed": sum(x.get("status") == "FAILED_FINAL" for x in semantic),
                          "remaining": sum(x.get("status") in {"PENDING", "STALE", "RUNNING"} for x in semantic)},
             "editorial": editorial,
             "finalization": {"assets": (final or {}).get("completed", 0), "review": (final or {}).get("review", 0),
                              "failed": (final or {}).get("failed_final", 0)},
             "disk": {"temporary_bytes": sum(x.stat().st_size for x in (run / ".work").rglob("*") if x.is_file()),
                      "final_bytes": sum(x.stat().st_size for x in (run / "assets").glob("*") if x.is_file())}}
    ledger.summary(**value)
    return value


def process(input_dir: Path, provider: Any = None, model: str = "gemini-3.6-flash") -> dict[str, Any]:
    info = preflight(input_dir)
    # A supplied test/provider implementation is already configured.  The CLI
    # path must fail before video traversal when Gemini credentials are absent.
    if provider is None:
        from dotenv import load_dotenv
        load_dotenv(_root(input_dir) / ".env")
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("provider configuration missing: GEMINI_API_KEY")
    source_path = info["run"] / "source_fingerprint.json"
    # Missing production provenance is deliberately incompatible: legacy pixel
    # artifacts must never be promoted merely because names/frame IDs coincide.
    old = _read_json(source_path).get("movie_sha256") if source_path.exists() else None
    _invalidate_source_media(info["run"], old, info["movie_sha256"])
    write_json(source_path, {"movie_sha256": info["movie_sha256"], "srt_sha256": info["srt_sha256"], "updated_at": _utc()})
    ledger = ProcessingLedger(info["run"], input_dir.name, {"movie_sha256": info["movie_sha256"],
                              "srt_sha256": info["srt_sha256"], "narrative_sha256": sha256_file(info["narrative"]),
                              "orchestrator_version": "production_v1", "model": model})
    shots = _technical_shots(info)
    events = _events(info, shots)
    # semantic_validate is the existing one-request-per-Visual-Event engine.
    semantic = semantic_validate(events, info["movie"], info["srt"], info["narrative"],
                                 info["run"] / "semantic_checkpoints", float(info["metadata"]["video"]["fps"]),
                                 "FULL_MOVIE", provider=provider, model=model, preserve_event_ids=True)
    if semantic.get("quota_exhausted"):
        summary = _summary(info, ledger, events, "PARTIAL_PROVIDER")
        return {"status": "PARTIAL_PROVIDER", "summary": summary, "semantic": semantic}
    apply_semantic_scarcity(events)
    write_json(info["run"] / "visual_events.json", {"schema_version": "visual_events_v1", "version": EVENT_VERSION,
               "fingerprint": fingerprint({"source_movie_sha256": info["movie_sha256"], "technical": TECHNICAL_VERSION, "grouping": EVENT_VERSION}), "events": events})
    # The detector is required only when a validated KEEP needs vertical geometry.
    if any(x.get("editorial", {}).get("decision") == "KEEP" and x.get("editorial", {}).get("status") == "VALIDATED" for x in events):
        person_detector_preflight()
    final = finalize_pilot(input_dir, "FULL_MOVIE", candidates=events,
                           shots={x["shot_id"]: x for x in shots})
    state = "COMPLETE" if final["status"] == "COMPLETE" else "PARTIAL"
    return {"status": state, "summary": _summary(info, ledger, events, state, final), "semantic": semantic, "finalization": final}
