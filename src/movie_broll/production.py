"""One-command, durable full-movie production orchestration.

This module intentionally owns orchestration only.  Shot detection, editorial
grouping, semantics, reframing, and Asset Hub metadata remain implemented by
their already-validated modules.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .broll_pilot import (add_context, apply_semantic_scarcity, candidates,
                          semantic_validate, visual_signals)
from .broll_semantics import build_gemini_provider_from_env
from .finalization import finalize_pilot, person_detector_preflight, safe_cleanup
from .inspect_source import inspect_movie
from .processing_ledger import ProcessingLedger, fingerprint
from .srt import parse_srt_file
from .utils import sha256_file, write_json
from .visual import Window, build_shots, detect_cuts

TECHNICAL_VERSION = "full_movie_technical_shots_v1"
EVENT_VERSION = "full_movie_visual_events_v1"
HEARTBEAT_SECONDS = 30


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _root(input_dir: Path) -> Path:
    return input_dir.resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bounded_operation(label: str, action: Any, report: Any, ledger: ProcessingLedger) -> Any:
    """Keep tmux-visible liveness during decoder work without frame spam."""
    began, done = time.monotonic(), threading.Event()
    def heartbeat() -> None:
        while not done.wait(HEARTBEAT_SECONDS):
            elapsed = int(time.monotonic() - began)
            report(f"[movie-broll] {label}: working (elapsed {elapsed//60:02d}:{elapsed%60:02d})")
            ledger.log("PRODUCTION_HEARTBEAT", mode="production", stage=label, elapsed_seconds=elapsed)
            try:
                summary = _read_json(ledger.summary_path) if ledger.summary_path.exists() else {}
                summary.update(updated_at=_utc(), stage=label, heartbeat_elapsed_seconds=elapsed, mode="production", status="RUNNING")
                write_json(ledger.summary_path, summary)
            except OSError:
                pass
    worker = threading.Thread(target=heartbeat, daemon=True); worker.start()
    try:
        return action()
    finally:
        done.set(); worker.join(timeout=1)


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


def _segment_store(info: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = info["run"] / "production_segments.json"
    fp = fingerprint({"source_movie_sha256": info["movie_sha256"], "events": EVENT_VERSION})
    data = _read_json(path) if path.exists() else {}
    if data.get("fingerprint") != fp:
        data = {"schema_version": "production_segments_v1", "fingerprint": fp, "segments": {}}
    return path, data


def _segment_events(info: dict[str, Any], segment: dict[str, Any], shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bound video decoding to one narrative segment and persist before Gemini."""
    start, end = float(segment["start_seconds"]), float(segment["end_seconds"])
    selected = [dict(x) for x in shots if float(x["start_seconds"]) >= start and float(x["end_seconds"]) <= end]
    if not selected:
        return []
    add_context(selected, info["srt"], info["narrative"])
    for signal, shot in zip(visual_signals(info["movie"], selected), selected):
        shot.update(signal)
    events = candidates(selected)
    for event in events:
        event["visual_event_id"] = "VE_" + fingerprint({"segment": segment["segment_id"], "start": event["start_frame"],
                                                            "end": event["end_frame_exclusive"], "shots": event["source_shot_ids"]})[:16]
    return events


def _response_validation_retries(ledger: ProcessingLedger, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only schema/content failures receive the existing second same-event try."""
    selected=[]
    for event in events:
        stage=ledger.data.get("events",{}).get(event["visual_event_id"],{}).get("stages",{}).get("semantic",{})
        if stage.get("status")=="FAILED_RETRYABLE" and stage.get("failure_kind")=="provider_response_validation": selected.append(event)
    return selected


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
             final: dict[str, Any] | None = None, *, stage: str = "production", segments_total: int = 0,
             segments_complete: int = 0, timings: dict[str, float] | None = None,
             technical_shot_count: int = 0) -> dict[str, Any]:
    records = ledger.data["events"].values()
    semantic = [x.get("stages", {}).get("semantic", {}) for x in records]
    editorial = {key: sum(1 for x in events if x.get("editorial", {}).get("decision") == key)
                 for key in ("KEEP", "REVIEW", "REJECT")}
    run = info["run"]
    value = {"schema_version": "production_progress_summary_v1", "movie_id": run.name, "mode": "production", "status": state,
             "source": {"movie_sha256": info["movie_sha256"], "srt_sha256": info["srt_sha256"]},
             "run_state": state, "stage": stage, "updated_at": _utc(), "technical_shots": technical_shot_count,
             "segments_total": segments_total, "segments_complete": segments_complete, "timings_seconds": timings or {},
             "visual_events": len(events),
             "semantic": {"complete": sum(x.get("status") == "COMPLETE" for x in semantic),
                          "retryable": sum(x.get("status") == "FAILED_RETRYABLE" for x in semantic),
                          "failed": sum(x.get("status") == "FAILED_FINAL" for x in semantic),
                          "remaining": sum(x.get("status") in {"PENDING", "STALE", "RUNNING", "FAILED_RETRYABLE"} for x in semantic)},
             "editorial": editorial,
             "finalization": {"assets": (final or {}).get("completed", 0), "review": (final or {}).get("review", 0),
                              "failed": (final or {}).get("failed_final", 0)},
             "disk": {"temporary_bytes": sum(x.stat().st_size for x in (run / ".work").rglob("*") if x.is_file()),
                      "final_bytes": sum(x.stat().st_size for x in (run / "assets").glob("*") if x.is_file())}}
    ledger.summary(**value)
    return value


def _semantic_stop_status(semantic: dict[str, Any]) -> str | None:
    """Preserve the semantic operational stop cause."""
    status = semantic.get("status")

    if status in {"PARTIAL_QUOTA", "PARTIAL_PROVIDER"}:
        return status

    if semantic.get("quota_exhausted"):
        return "PARTIAL_QUOTA"

    if semantic.get("provider_unavailable"):
        return "PARTIAL_PROVIDER"

    return None


def _semantic_stop_result(
    info: dict[str, Any],
    ledger: ProcessingLedger,
    events: list[dict[str, Any]],
    record: dict[str, Any],
    store_path: Path,
    store: dict[str, Any],
    semantic: dict[str, Any],
    segment_id: str,
    segment_index: int,
    segments_total: int,
    shots: list[dict[str, Any]],
    report: Any,
) -> dict[str, Any] | None:
    status = _semantic_stop_status(semantic)

    if status is None:
        return None

    failure = dict(semantic.get("failure") or {})

    failure.update(
        failure_stage="semantic",
        failed_segment=segment_id,
        segment_index=segment_index,
        segments_total=segments_total,
        resume_safe=True,
    )

    if not failure.get("reason"):
        failure["reason"] = (
            "quota_exceeded"
            if status == "PARTIAL_QUOTA"
            else "provider_unavailable"
        )

    record["status"] = (
        "FAILED_FINAL"
        if failure.get("retryable") is False
        else "FAILED_RETRYABLE"
    )
    record["failure"] = failure

    write_json(store_path, store)

    ledger.log(
        f"PRODUCTION_{status}",
        mode="production",
        segment_id=segment_id,
        segment_index=segment_index,
        segments_total=segments_total,
        failure=failure,
    )

    report(
        "[provider-error] "
        f"stage=semantic "
        f"segment={segment_index}/{segments_total} "
        f"event={failure.get('failed_event_id')} "
        f"provider={failure.get('provider')} "
        f"model={failure.get('model')} "
        f"http_status={failure.get('http_status')} "
        f"reason={failure.get('reason')} "
        f"retryable={failure.get('retryable')} "
        f"retry_after_seconds={failure.get('retry_after_seconds')} "
        f"checkpoint_saved={failure.get('checkpoint_saved')} "
        "resume_safe=true"
    )

    summary = _summary(
        info,
        ledger,
        events,
        status,
        stage="semantic",
        segments_total=segments_total,
        segments_complete=segment_index - 1,
        technical_shot_count=len(shots),
    )

    summary["failure"] = failure
    ledger.summary(**summary)

    return {
        "status": status,
        "summary": summary,
        "semantic": semantic,
        "failure": failure,
    }


def process(input_dir: Path, provider: Any = None, model: str = "gemini-3.6-flash", reporter: Any = None) -> dict[str, Any]:
    report = reporter or (lambda message: None)
    started = time.monotonic()
    info = preflight(input_dir)
    # One provider/pool instance for the complete production invocation.
    # Round-robin cursor and cooldown state survive across segments.
    active_provider = provider

    if active_provider is None:
        from dotenv import load_dotenv

        load_dotenv(_root(input_dir) / ".env")

        active_provider = build_gemini_provider_from_env(
            model,
            reporter=report,
        )

        if active_provider is None:
            raise RuntimeError(
                "provider configuration missing: "
                "GEMINI_API_KEY_1/2/3, GEMINI_API_KEY_BACKUP, "
                "or legacy GEMINI_API_KEY"
            )
    report(f"[movie-broll] movie: {input_dir.name}")
    report(f"[movie-broll] source: {info['movie_sha256'][:12]}...")
    video = info["metadata"]["video"]
    report(f"[movie-broll] media: {video['width']}x{video['height']} @ {video['fps']}")
    report("[movie-broll] Narrative Map: REUSED")
    source_path = info["run"] / "source_fingerprint.json"
    # Missing production provenance is deliberately incompatible: legacy pixel
    # artifacts must never be promoted merely because names/frame IDs coincide.
    old = _read_json(source_path).get("movie_sha256") if source_path.exists() else None
    _invalidate_source_media(info["run"], old, info["movie_sha256"])
    write_json(source_path, {"movie_sha256": info["movie_sha256"], "srt_sha256": info["srt_sha256"], "updated_at": _utc()})
    ledger = ProcessingLedger(info["run"], input_dir.name, {"movie_sha256": info["movie_sha256"],
                              "srt_sha256": info["srt_sha256"], "narrative_sha256": sha256_file(info["narrative"]),
                              "orchestrator_version": "production_v1", "model": model})
    narrative_segments = sorted(_read_json(info["narrative"])["segments"], key=lambda x: (float(x["start_seconds"]), x["segment_id"]))
    _summary(info, ledger, [], "RUNNING", stage="preflight", segments_total=len(narrative_segments), timings={"preflight_seconds": time.monotonic()-started})
    ledger.log("PRODUCTION_STARTED", mode="production")
    ledger.log("PREFLIGHT_COMPLETE", mode="production")
    technical_started = time.monotonic()
    shots = _technical_shots(info)
    report(f"[movie-broll] technical shots: REUSED {len(shots)}")
    ledger.log("TECHNICAL_SHOTS_REUSED", mode="production", count=len(shots))
    _summary(info, ledger, [], "RUNNING", stage="technical_shots", segments_total=len(narrative_segments),
             timings={"technical_shots_seconds": time.monotonic()-technical_started}, technical_shot_count=len(shots))
    store_path, store = _segment_store(info); all_events=[]; finals=[]; semantic_reports=[]
    try:
        for index, segment in enumerate(narrative_segments, 1):
            sid = segment["segment_id"]; record = store["segments"].setdefault(sid, {"status": "PENDING", "segment": segment})
            if record.get("status") == "COMPLETE":
                all_events.extend(record.get("events", [])); continue
            record["status"] = "RUNNING"; record.pop("failure", None); write_json(store_path, store)
            ledger.log("SEGMENT_RUNNING", mode="production", segment_id=sid, segment_index=index, segments_total=len(narrative_segments))
            report(f"[movie-broll] building Visual Events: segment {index}/{len(narrative_segments)}")
            event_started=time.monotonic()
            if record.get("events"):
                events=record["events"]; ledger.log("VISUAL_EVENTS_REUSED", mode="production", segment_id=sid, events=len(events))
            else:
                events=_bounded_operation(f"visual-events segment {index}/{len(narrative_segments)}",lambda: _segment_events(info, segment, shots),report,ledger)
                record["events"] = events; record["shots"] = [x["shot_id"] for x in shots if x["shot_id"] in {y for e in events for y in e["source_shot_ids"]}]
                write_json(store_path, store); ledger.log("VISUAL_EVENTS_COMPLETE", mode="production", segment_id=sid, events=len(events))
            all_events.extend(events)
            report(f"[movie-broll] segment {index}/{len(narrative_segments)}: shots={len(record['shots'])} events={len(events)}")
            _summary(info, ledger, all_events, "RUNNING", stage="semantic", segments_total=len(narrative_segments), segments_complete=index-1, timings={"technical_shots_seconds":time.monotonic()-technical_started,"segment_visual_events_seconds":time.monotonic()-event_started}, technical_shot_count=len(shots))
            report(f"[movie-broll] semantic: segment {index}/{len(narrative_segments)} events={len(events)}")
            ledger.log("SEMANTIC_RUNNING", mode="production", segment_id=sid, events=len(events))
            semantic=_bounded_operation(f"semantic segment {index}/{len(narrative_segments)}",lambda: semantic_validate(events,info["movie"],info["srt"],info["narrative"],info["run"] / "semantic_checkpoints",float(video["fps"]),sid,provider=active_provider,model=model,preserve_event_ids=True),report,ledger); semantic_reports.append(semantic)
            stop_result=_semantic_stop_result(info,ledger,all_events,record,store_path,store,semantic,sid,index,len(narrative_segments),shots,report)
            if stop_result is not None:
                return stop_result
            retry_events=_response_validation_retries(ledger,events)
            if retry_events:
                ledger.log("SEMANTIC_RESPONSE_VALIDATION_RETRY",mode="production",segment_id=sid,events=len(retry_events))
                semantic=_bounded_operation(f"semantic validation retry segment {index}/{len(narrative_segments)}",lambda: semantic_validate(retry_events,info["movie"],info["srt"],info["narrative"],info["run"] / "semantic_checkpoints",float(video["fps"]),sid,provider=active_provider,model=model,preserve_event_ids=True),report,ledger); semantic_reports.append(semantic)
                stop_result=_semantic_stop_result(info,ledger,all_events,record,store_path,store,semantic,sid,index,len(narrative_segments),shots,report)
                if stop_result is not None:
                    return stop_result
            apply_semantic_scarcity(events)
            ledger.log("SEMANTIC_COMPLETE", mode="production", segment_id=sid, complete=semantic.get("complete", 0))
            if any(x.get("editorial",{}).get("decision")=="KEEP" and x.get("editorial",{}).get("status")=="VALIDATED" for x in events): person_detector_preflight()
            final=finalize_pilot(input_dir,sid,candidates=events,shots={x["shot_id"]:x for x in shots}); finals.append(final)
            if final.get("completed", 0): ledger.log("ASSET_COMPLETE", mode="production", segment_id=sid, assets=final["completed"])
            record.update(status="COMPLETE", completed_at=_utc(), finalization=final); write_json(store_path,store)
            ledger.log("SEGMENT_COMPLETE",mode="production",segment_id=sid,events=len(events)); _summary(info,ledger,all_events,"RUNNING",final,stage="segment_complete",segments_total=len(narrative_segments),segments_complete=index,technical_shot_count=len(shots))
        write_json(info["run"] / "visual_events.json", {"schema_version":"visual_events_v1","version":EVENT_VERSION,"events":all_events})
        ledger.log("PRODUCTION_COMPLETE",mode="production")
        return {"status":"COMPLETE","summary":_summary(info,ledger,all_events,"COMPLETE",stage="complete",segments_total=len(narrative_segments),segments_complete=len(narrative_segments),timings={"total_seconds":time.monotonic()-started},technical_shot_count=len(shots)),"semantic":semantic_reports,"finalization":finals}
    except KeyboardInterrupt:
        ledger.log("PRODUCTION_INTERRUPTED",mode="production")
        return {"status":"PARTIAL","summary":_summary(info,ledger,all_events,"PARTIAL",stage="interrupted",segments_total=len(narrative_segments),segments_complete=sum(x.get("status")=="COMPLETE" for x in store["segments"].values()),technical_shot_count=len(shots))}
