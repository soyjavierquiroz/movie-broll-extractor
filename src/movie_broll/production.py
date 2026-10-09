"""One-command, durable full-movie production orchestration.

This module intentionally owns orchestration only.  Shot detection, editorial
grouping, semantics, reframing, and producer-package metadata remain implemented by
their already-validated modules.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np

from .broll_pilot import (add_context, apply_semantic_scarcity, candidates,
                          semantic_validate, visual_signals, _semantic_checkpoint,
                          candidate_contact_sheet, semantic_request_context)
from .broll_semantics import build_semantic_provider_from_env, DEFAULT_OPENAI_MODEL
from .finalization import finalize_pilot, person_detector_preflight, reconcile_review_packages, safe_cleanup
from .inspect_source import inspect_movie
from .processing_ledger import ProcessingLedger, fingerprint
from .srt import parse_srt_file
from .utils import sha256_file, write_json
from .visual import Window, build_shots, detect_cuts
from .production_profile import load as load_production_profile
from .active_picture import load_or_detect

TECHNICAL_VERSION = "full_movie_technical_shots_v1"
EVENT_VERSION = "full_movie_visual_events_v1"
EVENT_STORE_VERSION = "production_visual_events_v1_global_soft_narrative"
VISUAL_SIGNAL_CACHE_VERSION = "visual_signals_v3"
VISUAL_SIGNAL_ALGORITHM_VERSION = "single_pass_active_picture_v3"
MAX_VISUAL_ANALYSIS_SECONDS = 18.0
HEARTBEAT_SECONDS = 30
# Production is intentionally event-granular: a completed semantic event must
# reach scarcity/finalization before a later provider call can block it.
SEMANTIC_PRODUCTION_BATCH_SIZE = 1


class ProductionFailure(RuntimeError):
    """A process-level failure with a supervisor-safe restart classification."""
    def __init__(self, message: str, classification: str = "DETERMINISTIC") -> None:
        super().__init__(message)
        self.outcome_classification = classification


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _root(input_dir: Path) -> Path:
    return input_dir.resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_workspace(run: Path) -> Path | None:
    """Return the active versioned checkpoint workspace, if canon selected one."""
    try:
        store = _read_json(run / "visual_event_segments_v1.json")
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if store.get("canonical_semantic_contract") != "V9":
        return None
    relative = store.get("canonical_semantic_workspace")
    if not isinstance(relative, str) or not relative:
        return None
    candidate = run / relative
    # A canonical manifest must never use a path outside its own movie run.
    try:
        candidate.resolve().relative_to(run.resolve())
    except ValueError:
        raise RuntimeError("canonical semantic workspace escapes movie run")
    return candidate


def _semantic_checkpoint_dir(run: Path) -> Path:
    workspace = _semantic_workspace(run)
    return workspace / "checkpoints" if workspace is not None else run / "semantic_checkpoints"


def _semantic_failure_dir(run: Path) -> Path:
    workspace = _semantic_workspace(run)
    return workspace / "events" if workspace is not None else run / "semantic_failures"


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
    from .narrative_runner import narrative_artifacts_compatible
    if not narrative_artifacts_compatible(narrative.parent, srt):
        raise ProductionFailure(
            "canonical narrative-v2 input is required; run 'movie-broll run "
            f"{input_dir}' to prepare or finalize the external Narrative Mapper handoff"
        )
    # Narrative compatibility is intentionally SRT-scoped, not movie-pixel-scoped.
    # Its own structural schema is validated by the earlier narrative stage.
    if not isinstance(_read_json(narrative).get("segments"), list):
        raise ValueError("Narrative Map has no segments")
    profile=load_production_profile()
    active_picture=load_or_detect(movie,run,profile["active_picture"])
    return {"movie": movie, "srt": srt, "run": run, "narrative": narrative,
            "movie_sha256": sha256_file(movie), "srt_sha256": sha256_file(srt),
            "metadata": metadata, "free_bytes": free, "cue_count": len(parsed.cues),
            "active_picture": active_picture, "production_profile": profile}


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


def derive_visual_analysis_units(shots: list[dict[str, Any]], fps: float) -> list[dict[str, Any]]:
    """Split only long technical ranges into balanced, frame-exact subranges.

    The returned records are analysis work items, not technical shots.  Their
    ``shot_id`` stays canonical for all downstream semantic/finalization paths;
    ``analysis_unit_id`` is audit/cache-only provenance.
    """
    max_frames=max(1,int(math.floor(MAX_VISUAL_ANALYSIS_SECONDS*fps+1e-9)))
    result=[]
    ordered=sorted(shots,key=lambda x:(int(x['start_frame']),str(x['shot_id'])))
    for position,shot in enumerate(ordered):
        start,end=int(shot['start_frame']),int(shot['end_frame_exclusive'])
        if end<=start: raise ValueError(f"invalid technical shot range: {shot['shot_id']}")
        length=end-start; parts=max(1,math.ceil(length/max_frames)); base,remainder=divmod(length,parts)
        cursor=start
        for ordinal in range(parts):
            size=base+(1 if ordinal<remainder else 0); right=cursor+size
            unit=dict(shot)
            unit.update(analysis_unit_id=f"{shot['shot_id']}__VAU_{ordinal+1:03d}",
                        parent_shot_id=shot['shot_id'], shot_id=shot['shot_id'],
                        start_frame=cursor,end_frame_exclusive=right,
                        start_seconds=cursor/fps,end_seconds=right/fps,
                        duration_seconds=(right-cursor)/fps,
                        derived_from_long_shot=parts>1,
                        is_final_technical_shot=position==len(ordered)-1)
            result.append(unit); cursor=right
    return result


def _signal_cache_path(info: dict[str, Any]) -> Path:
    return info['run'] / 'visual_signals_v2.json'


def _unit_fingerprint(info: dict[str, Any], units: list[dict[str, Any]]) -> str:
    boundaries=[{key:unit[key] for key in ('analysis_unit_id','parent_shot_id','start_frame','end_frame_exclusive')}
                for unit in units]
    return fingerprint({'source_movie_sha256':info['movie_sha256'],
                        'algorithm_version':VISUAL_SIGNAL_ALGORITHM_VERSION,
                        'active_picture':_active_picture_fingerprint_fields(info.get('active_picture', {})),
                        'analysis_unit_boundaries':boundaries})


def _active_picture_fingerprint_fields(active_picture: dict[str, Any]) -> dict[str, Any]:
    """Only the active-picture identity fields that define content pixels."""
    return {key: active_picture.get(key) for key in (
        'x', 'y', 'width', 'height', 'source_width', 'source_height',
        'structural_bars', 'detection_profile',
    )}


def _active_picture(info: dict[str, Any]) -> dict[str, Any]:
    """Production preflight always supplies this; retain test/legacy call safety."""
    if isinstance(info.get('active_picture'), dict):
        return info['active_picture']
    video=info.get('metadata', {}).get('video', {})
    width, height=int(video.get('width', 0)), int(video.get('height', 0))
    return {'x':0,'y':0,'width':width,'height':height,'source_width':width,
            'source_height':height,'structural_bars':False,
            'detection_profile':'legacy_full_frame'}


def _json_signal(value: dict[str, Any]) -> dict[str, Any]:
    result={key:item for key,item in value.items() if key != '_hist'}
    if '_hist' in value: result['histogram_hsv_16x16']=np.asarray(value['_hist']).tolist()
    return result


def _runtime_signal(value: dict[str, Any]) -> dict[str, Any]:
    result=dict(value)
    histogram=result.pop('histogram_hsv_16x16',None)
    if histogram is not None: result['_hist']=np.asarray(histogram,dtype=np.float32)
    return result


def _visual_progress(info: dict[str, Any], total: int, complete: int, *, stage: str = 'visual-signals',
                     events: int | None = None) -> None:
    info['_visual_signal_units_total']=total; info['_visual_signal_units_complete']=complete
    ledger=info.get('_ledger')
    path=info['run']/'progress_summary.json'
    summary=_read_json(path) if path.exists() else {}
    summary.update(status='RUNNING',run_state='RUNNING',stage=stage,updated_at=_utc(),
                   visual_signal_units_total=total,visual_signal_units_complete=complete)
    if events is not None: summary['visual_events']=events
    write_json(path,summary)
    if ledger: ledger.log('VISUAL_SIGNAL_PROGRESS',mode='production',complete=complete,total=total,stage=stage)


def _cached_visual_signals(info: dict[str, Any], units: list[dict[str, Any]], report: Any) -> list[tuple[dict[str, Any],dict[str, Any]]]:
    """Return completed unit signals, checkpointing every newly decoded unit."""
    path=_signal_cache_path(info); cache=_read_json(path) if path.exists() else {}
    fp=_unit_fingerprint(info,units)
    if cache.get('fingerprint') != fp:
        cache={'schema_version':VISUAL_SIGNAL_CACHE_VERSION,'algorithm_version':VISUAL_SIGNAL_ALGORITHM_VERSION,
               'source_movie_sha256':info['movie_sha256'],'fingerprint':fp,
               'analysis_units':[{key:unit[key] for key in ('analysis_unit_id','parent_shot_id','start_frame','end_frame_exclusive','start_seconds','end_seconds','duration_seconds','derived_from_long_shot')} for unit in units],
               'signals':{}}
        write_json(path,cache)
    records=cache.setdefault('signals',{})
    missing=[unit for unit in units if unit['analysis_unit_id'] not in records]
    complete=len(units)-len(missing)
    _visual_progress(info,len(units),complete)
    if complete:
        report(f"[movie-broll] visual-signals: {complete}/{len(units)} REUSED")
    from .broll_pilot import visual_signals
    milestones=max(1,min(50,len(units)//4 or 1))
    def checkpoint(unit: dict[str, Any], signal: dict[str, Any]) -> None:
        nonlocal complete
        records[unit['analysis_unit_id']]=_json_signal(signal)
        write_json(path,cache)
        complete+=1; _visual_progress(info,len(units),complete)
        if complete==len(units) or complete%milestones==0:
            report(f"[movie-broll] visual-signals: {complete}/{len(units)}")
    if missing: visual_signals(info['movie'],missing,on_complete=checkpoint,
                               active_picture=info.get('active_picture'))
    return [(unit,_runtime_signal(records[unit['analysis_unit_id']])) for unit in units]


def _event_store(info: dict[str, Any], shots: list[dict[str, Any]] | None = None) -> tuple[Path, dict[str, Any]]:
    """Use a separate v1 store so aborted v5 draft state is forensic-only."""
    path = info["run"] / "visual_event_segments_v1.json"
    visual_signal_fingerprint=None
    if shots is not None:
        fps=float(info['metadata']['video']['fps'])
        visual_signal_fingerprint=_unit_fingerprint(info,derive_visual_analysis_units(shots,fps))
    fp = fingerprint({"source_movie_sha256": info["movie_sha256"], "events": EVENT_STORE_VERSION,
                      "visual_signal_fingerprint": visual_signal_fingerprint})
    data = _read_json(path) if path.exists() else {}
    if data.get("fingerprint") != fp:
        data = {"schema_version": "production_visual_events_v1", "fingerprint": fp,
                "status": "PENDING", "events": []}
    return path, data


def _visual_events(info: dict[str, Any], shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build v1 Visual Events across the continuous technical-shot timeline."""
    canonical = [dict(x) for x in sorted(shots, key=lambda x: (x["start_seconds"], x["shot_id"]))]
    if not canonical:
        return []
    fps=float(info['metadata']['video']['fps'])
    units=derive_visual_analysis_units(canonical,fps)
    report=info.get('_report',lambda _message: None)
    report(f"[movie-broll] visual-analysis-units: {len(units)}")
    add_context(units, info["srt"], info["narrative"])
    selected=[]
    for unit,signal in _cached_visual_signals(info,units,report):
        # EOF slivers remain in the cache/audit trail but cannot be candidates.
        if signal.get('status') == 'TERMINAL_SLIVER_SKIPPED':
            continue
        unit.update(signal); selected.append(unit)
    report('[movie-broll] grouping: working')
    events = candidates(selected)
    _visual_progress(info,len(units),len(units),stage='grouping',events=len(events))
    report(f"[movie-broll] grouping: {len(events)} Visual Events")
    profile=info.get("production_profile") or load_production_profile()
    if profile.get("intrusive_text",{}).get("enabled",True):
        from .intrusive_text import inspect_event
        settings=profile["intrusive_text"]
        for index,event in enumerate(events,1):
            gate=inspect_event(info["movie"],event,active_picture=_active_picture(info),**settings)
            event["intrusive_text_gate"]=gate
            if gate["decision"] == "REJECT":
                event["editorial"]={"decision":"REJECT","status":"VALIDATED","reason":gate["reason"]}
            if index==len(events) or index%max(1,min(50,len(events)//4 or 1))==0:
                report(f"[movie-broll] intrusive-text: {index}/{len(events)}")
    for event in events:
        event["visual_event_id"] = "VE_" + fingerprint({"start": event["start_frame"],
                                                            "end": event["end_frame_exclusive"],
                                                            "shots": event["source_shot_ids"],
                                                            "narrative_segment_ids": event["narrative_segment_ids"]})[:16]
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


def _production_queue(events: list[dict[str, Any]]) -> list[str]:
    """Return the stable work order without changing the canonical manifest order."""
    return [event["visual_event_id"] for event in sorted(
        events,
        key=lambda event: (-event.get("score", {}).get("total", 0),
                           event["start_seconds"], event["candidate_id"]),
    )]


def _assign_timeline_ordinals(events: list[dict[str, Any]]) -> bool:
    """Attach canonical timeline ordinals without changing the manifest order.

    These ordinals are production-identity data.  They deliberately use the
    global chronological manifest ordering, while ``_production_queue`` keeps
    the independent quality-first execution ordering.
    """
    changed=False
    for ordinal,event in enumerate(sorted(
            events,
            key=lambda event: (event['start_seconds'], event['end_seconds'], event['visual_event_id']),
    ),1):
        if event.get('timeline_ordinal') != ordinal:
            event['timeline_ordinal']=ordinal
            changed=True
    return changed


def _ensure_production_batches(store: dict[str, Any], events: list[dict[str, Any]]) -> None:
    """Persist deterministic batch membership separately from event discovery.

    Batch records are derived metadata.  A batch-size change therefore replaces
    only the queue/batch layout; it deliberately leaves the canonical event
    manifest (and its semantic state) untouched.  Semantic checkpoints are
    separate files keyed by candidate/event identity and are never involved in
    this migration.
    """
    queue = _production_queue(events)
    event_groups = [queue[index:index + SEMANTIC_PRODUCTION_BATCH_SIZE]
                    for index in range(0, len(queue), SEMANTIC_PRODUCTION_BATCH_SIZE)]
    existing = store.get("batches")
    if (store.get("production_queue") == queue
            and store.get("production_batch_size") == SEMANTIC_PRODUCTION_BATCH_SIZE
            and isinstance(existing, dict)
            and [batch.get("event_ids") for batch in existing.values()] == event_groups):
        return
    # The event-store fingerprint makes a changed event list a new store.  For a
    # compatible legacy store, retain its events and rebuild only the derived
    # production metadata before the next semantic request.
    store["production_queue"] = queue
    store["production_batch_size"] = SEMANTIC_PRODUCTION_BATCH_SIZE
    store["batches"] = {
        f"PBATCH_{index + 1:04d}": {
            "event_ids": event_ids,
            "semantic_status": "PENDING",
            "finalization_status": "PENDING",
            "status": "PENDING",
        }
        for index, event_ids in enumerate(event_groups)
    }


def _batch_events(events_by_id: dict[str, dict[str, Any]], batch: dict[str, Any]) -> list[dict[str, Any]]:
    return [events_by_id[event_id] for event_id in batch["event_ids"]]


def _accepted_higher_priority_keepers(events_by_id: dict[str, dict[str, Any]], queue: list[str], before: int) -> list[dict[str, Any]]:
    """The only scarcity context a batch needs: previously accepted priorities."""
    return [
        events_by_id[event_id] for event_id in queue[:before]
        if events_by_id[event_id].get("editorial", {}).get("decision") == "KEEP"
        and events_by_id[event_id].get("editorial", {}).get("status") == "VALIDATED"
        and events_by_id[event_id].get("semantic_redundancy", {}).get("status") != "SUPPRESSED"
    ]


def _batch_finalization_complete(batch_events: list[dict[str, Any]], ledger: ProcessingLedger) -> bool:
    """A batch is complete only when its accepted events need no finalization retry."""
    for event in batch_events:
        if event.get("editorial", {}).get("decision") != "KEEP" or event.get("editorial", {}).get("status") != "VALIDATED":
            continue
        stage = ledger.data.get("events", {}).get(event["visual_event_id"], {}).get("stages", {}).get("finalization", {})
        if stage and stage.get("status") in {"PENDING", "RUNNING", "FAILED_RETRYABLE", "STALE"}:
            return False
    return True


def _terminal_semantic_failure(run: Path, event_id: str) -> bool:
    """A semantic failure is terminal only with its canonical artifact."""
    workspace = _semantic_workspace(run)
    if workspace is not None:
        try:
            result = _read_json(workspace / "events" / f"{event_id}.json")
        except (OSError, json.JSONDecodeError, ValueError):
            return False
        return (result.get("event_id") == event_id
                and result.get("state") == "TERMINAL_FAILURE"
                and result.get("terminal_failure") is True)
    try:
        artifact = _read_json(run / "semantic_failures" / f"{event_id}.json")
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return (artifact.get("visual_event_id") == event_id
            and artifact.get("failure_stage") == "semantic_validation")


def _semantic_terminal_contract(run: Path, events: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Check the durable-success-or-durable-terminal-failure invariant."""
    invalid: list[str] = []
    for event in events:
        event_id = event["visual_event_id"]
        editorial = event.get("editorial", {})
        checkpoint = _semantic_checkpoint_dir(run) / f"{event['candidate_id']}.json"
        checkpoint_valid = _semantic_checkpoint(
            checkpoint, event, DEFAULT_OPENAI_MODEL, "FULL", None, None
        ) is not None
        # Observation-v1 is the durable success record for the normal pipeline.
        # Its policy result is local state, while the immutable observation is
        # the provider boundary; a legacy checkpoint is not required.
        try:
            from .semantic_observations import effective_observation, validate_observation
            observation = effective_observation(run, event)
            observation_valid = not validate_observation(observation, event)
        except (OSError, ValueError, json.JSONDecodeError):
            observation_valid = False
        if editorial.get("status") == "VALIDATED":
            if not checkpoint_valid and not observation_valid:
                invalid.append(event_id)
        elif editorial.get("status") == "SEMANTIC_INCOMPLETE":
            if not _terminal_semantic_failure(run, event_id):
                invalid.append(event_id)
        else:
            invalid.append(event_id)
    return not invalid, invalid


def _repair_resume_semantic_integrity(info: dict[str, Any], store: dict[str, Any],
                                      ledger: ProcessingLedger) -> list[str]:
    """Requeue only batches which claim completion without a terminal record."""
    events = {event["visual_event_id"]: event for event in store.get("events", [])}
    repaired: list[str] = []
    for batch_id, batch in store.get("batches", {}).items():
        batch_events = [events[event_id] for event_id in batch.get("event_ids", []) if event_id in events]
        valid, invalid = _semantic_terminal_contract(info["run"], batch_events)
        if (batch.get("semantic_status") == "COMPLETE" or batch.get("status") == "COMPLETE") and not valid:
            batch.update(semantic_status="PENDING", finalization_status="PENDING", status="PENDING",
                         resume_integrity_repair={"invalid_event_ids": invalid})
            repaired.append(batch_id)
            ledger.log("SEMANTIC_RESUME_INTEGRITY_REQUEUE", mode="production", segment_id=batch_id,
                       invalid_event_ids=invalid)
    if repaired:
        store.update(status="RUNNING", production_status="RUNNING")
        store.pop("completed_at", None)
    return repaired


def _batch_counts(store: dict[str, Any]) -> tuple[int, int]:
    batches = store.get("batches", {}).values()
    return len(batches), sum(batch.get("status") == "COMPLETE" for batch in batches)


def _summary(info: dict[str, Any], ledger: ProcessingLedger, events: list[dict[str, Any]], state: str,
             final: dict[str, Any] | None = None, *, stage: str = "production", segments_total: int = 0,
             segments_complete: int = 0, timings: dict[str, float] | None = None,
             technical_shot_count: int = 0, production_batches_total: int = 0,
             production_batches_complete: int = 0) -> dict[str, Any]:
    # Once a manifest exists, never let events from a superseded discovery
    # fingerprint inflate its progress counters.
    records = ([ledger.data["events"].get(event["visual_event_id"], {}) for event in events]
               if events else list(ledger.data["events"].values()))
    semantic = [x.get("stages", {}).get("semantic", {}) for x in records]
    editorial = {key: sum(1 for x in events if x.get("editorial", {}).get("decision") == key)
                 for key in ("KEEP", "REVIEW", "REJECT")}
    run = info["run"]
    finalized = [x.get("stages", {}).get("finalization", {}) for x in records]
    value = {"schema_version": "production_progress_summary_v1", "movie_id": run.name, "mode": "production", "status": state,
             "source": {"movie_sha256": info["movie_sha256"], "srt_sha256": info["srt_sha256"]},
             "run_state": state, "stage": stage, "updated_at": _utc(), "technical_shots": technical_shot_count,
             "segments_total": segments_total, "segments_complete": segments_complete, "timings_seconds": timings or {},
             "visual_events": len(events), "visual_events_total": len(events),
             "production_batches_total": production_batches_total,
             "production_batches_complete": production_batches_complete,
             "visual_signal_units_total": info.get('_visual_signal_units_total',0),
             "visual_signal_units_complete": info.get('_visual_signal_units_complete',0),
             "semantic": {"complete": sum(x.get("status") == "COMPLETE" for x in semantic),
                          "retryable": sum(x.get("status") == "FAILED_RETRYABLE" for x in semantic),
                          "failed": sum(x.get("status") == "FAILED_FINAL" for x in semantic),
                          "remaining": sum(x.get("status") in {"PENDING", "STALE", "RUNNING", "FAILED_RETRYABLE"} for x in semantic)},
             "editorial": editorial,
             "finalization": {"assets": sum(x.get("status") == "COMPLETE" and x.get("decision") == "PASS" for x in finalized),
                              "review": sum(x.get("status") == "COMPLETE" and x.get("decision") == "REVIEW_VERTICAL" for x in finalized),
                              "failed": sum(x.get("status") in {"FAILED_RETRYABLE", "FAILED_FINAL"} for x in finalized)},
             "disk": {"temporary_bytes": sum(x.stat().st_size for x in (run / ".work").rglob("*") if x.is_file()),
                      "final_bytes": sum(x.stat().st_size for x in (run / "assets").glob("*") if x.is_file())}}
    ledger.summary(**value)
    return value


def _semantic_stop_status(semantic: dict[str, Any]) -> str | None:
    """Preserve the semantic operational stop cause."""
    status = semantic.get("status")

    if status in {"PARTIAL_QUOTA", "PARTIAL_PROVIDER", "PROVIDER_DEFERRED", "BLOCKED_PROVIDER"}:
        return status

    if semantic.get("quota_exhausted"):
        return "PARTIAL_QUOTA"

    if semantic.get("provider_unavailable"):
        return "PROVIDER_DEFERRED"

    if semantic.get("blocked_provider"):
        return "BLOCKED_PROVIDER"

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
            else "blocked_provider"
            if status == "BLOCKED_PROVIDER"
            else "provider_deferred"
        )

    # Discovery is already complete and remains reusable.  The failed work is
    # production state, not a reason to discard the global manifest.
    record["semantic_status"] = status
    record["status"] = "PARTIAL"
    record["failure"] = failure
    store["event_discovery_status"] = "COMPLETE"
    store["production_status"] = status
    store["status"] = "RUNNING"

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


def process(input_dir: Path, provider: Any = None, model: str | None = None, reporter: Any = None, *, policy_version: str | None = None) -> dict[str, Any]:
    report = reporter or (lambda message: None)
    started = time.monotonic()
    info = preflight(input_dir)
    # A COMPLETE title must still receive cheap, metadata/media-only policy
    # reconciliation.  Do this before provider construction, source invalidation,
    # technical shots, or any semantic stage so a policy rollout cannot reopen
    # completed editorial work or require an API credential.
    completed_store_path=info['run']/'visual_event_segments_v1.json'
    completed_store=_read_json(completed_store_path) if completed_store_path.is_file() else {}
    chosen_policy = policy_version or completed_store.get('policy_version') or (
        'broll_policy_v1' if completed_store.get('events') or (provider is not None and not hasattr(provider, 'generate'))
        else 'broll_policy_v2')
    if chosen_policy not in {'broll_policy_v1', 'broll_policy_v2'}:
        raise ValueError('unsupported production policy')
    if chosen_policy == 'broll_policy_v2':
        from .visual_utility_production import process as process_visual_utility
        return process_visual_utility(input_dir, info, provider, model, report, started)
    source_record=info['run']/'source_fingerprint.json'
    recorded_source=_read_json(source_record).get('movie_sha256') if source_record.is_file() else None
    if (completed_store.get('production_status') == 'COMPLETE'
            and completed_store.get('production_batch_size') == SEMANTIC_PRODUCTION_BATCH_SIZE
            and recorded_source == info['movie_sha256']):
        # Historical completed titles (notably E02) are migrated from their
        # durable evidence and replayed locally.  This is deliberately before
        # provider construction: policy/finalization reconciliation is free.
        from .semantic_observations import migrate_existing_observations, evaluate_cached_observations
        migration = migrate_existing_observations(input_dir)
        policy_replay = evaluate_cached_observations(input_dir)
        events=list(completed_store.get('events',[]))
        total,complete=_batch_counts(completed_store)
        reconciliation=reconcile_review_packages(info['run'])
        ledger=ProcessingLedger(info['run'],input_dir.name,{
            'movie_sha256':info['movie_sha256'], 'srt_sha256':info['srt_sha256'],
            'orchestrator_version':'production_v1', 'model':model or 'not_invoked',
        })
        narrative_segments=_read_json(info['narrative']).get('segments',[])
        summary=_summary(info,ledger,events,'COMPLETE',stage='publication_reconciliation',
                         segments_total=len(narrative_segments),segments_complete=len(narrative_segments),
                         timings={'reconciliation_seconds':time.monotonic()-started},
                         production_batches_total=total,production_batches_complete=complete)
        report(f"[movie-broll] publication reconciliation: promoted={reconciliation['promoted']}; hard_review={reconciliation['hard_review']}; insufficient={reconciliation['insufficient_evidence']}")
        return {'status':'COMPLETE','summary':summary,'semantic':[],
                'observations': {'migration': migration, 'policy': policy_replay},
                'finalization':[{'reconciliation':reconciliation}]}
    # One provider/pool instance for the complete production invocation.
    # Round-robin cursor and cooldown state survive across segments.
    active_provider = provider

    if active_provider is None:
        from .temporal_semantics import TemporalObservationResult, RESPONSE_SCHEMA


        active_provider = build_semantic_provider_from_env(
            model,
            reporter=report,
            env_file=_root(input_dir) / ".env",
            response_model=TemporalObservationResult,
            response_schema=RESPONSE_SCHEMA,
        )

        if active_provider is None:
            raise RuntimeError(
                "provider configuration missing: OPENAI_API_KEY is not configured"
            )
    model = model or getattr(active_provider, "model", DEFAULT_OPENAI_MODEL)
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
    info['_ledger']=ledger; info['_report']=report
    narrative_segments = sorted(_read_json(info["narrative"])["segments"], key=lambda x: (float(x["start_seconds"]), x["segment_id"]))
    _summary(info, ledger, [], "RUNNING", stage="preflight", segments_total=len(narrative_segments), timings={"preflight_seconds": time.monotonic()-started})
    ledger.log("PRODUCTION_STARTED", mode="production")
    ledger.log("PREFLIGHT_COMPLETE", mode="production")
    technical_started = time.monotonic()
    shots = _bounded_operation("technical-shots", lambda: _technical_shots(info), report, ledger)
    report(f"[movie-broll] technical shots: REUSED {len(shots)}")
    ledger.log("TECHNICAL_SHOTS_REUSED", mode="production", count=len(shots))
    _summary(info, ledger, [], "RUNNING", stage="technical_shots", segments_total=len(narrative_segments),
             timings={"technical_shots_seconds": time.monotonic()-technical_started}, technical_shot_count=len(shots))
    store_path, store = _event_store(info, shots); all_events=[]; finals=[]; semantic_reports=[]; detector_preflight_result=None
    try:
        # v1 stores written before explicit discovery/production state already
        # hold the canonical global manifest.  Promote that fact in place so an
        # exhausted production resumes rather than rebuilding event boundaries.
        if store.get("events") and "event_discovery_status" not in store:
            store["event_discovery_status"] = "COMPLETE"
        # A completed pre-batching store is known to contain complete packages.
        # Migrate it without re-rendering.  Incomplete stores retain their global
        # discovery manifest and gain explicit production state below.
        if store.get("status") == "COMPLETE" and "production_status" not in store:
            store.update(event_discovery_status="COMPLETE", production_status="COMPLETE")
            write_json(store_path, store)
        # Migrate compatible global manifests in place.  The event fields and
        # order remain intact; only deterministic identity metadata is added.
        if store.get("events") and _assign_timeline_ordinals(store["events"]):
            write_json(store_path, store)
        # Completion flags are derived state.  Before accepting an interrupted
        # run as complete, verify that every semantic-complete batch has one
        # canonical durable outcome.  This is intentionally general and makes
        # historical parse-only records resumable without touching valid work.
        if store.get("events") and store.get("batches"):
            if _repair_resume_semantic_integrity(info, store, ledger):
                write_json(store_path, store)
        if (store.get("production_status") == "COMPLETE"
                and store.get("production_batch_size") == SEMANTIC_PRODUCTION_BATCH_SIZE):
            all_events = list(store.get("events", [])); total, complete = _batch_counts(store)
            write_json(info["run"] / "visual_events.json", {"schema_version":"visual_events_v1","version":EVENT_VERSION,"events":all_events})
            return {"status":"COMPLETE","summary":_summary(info,ledger,all_events,"COMPLETE",stage="complete",segments_total=len(narrative_segments),segments_complete=len(narrative_segments),timings={"total_seconds":time.monotonic()-started},technical_shot_count=len(shots),production_batches_total=total,production_batches_complete=complete),"semantic":semantic_reports,"finalization":finals}

        if store.get("event_discovery_status") == "COMPLETE" and store.get("events"):
            events = store["events"]
            report(f"[movie-broll] Visual Events: REUSED {len(events)}")
        else:
            store["status"]="RUNNING"; store["production_status"]="PENDING"; store.pop("failure",None)
            report("[movie-broll] building Visual Events: full technical timeline")
            events=_bounded_operation("visual-events full timeline",lambda: _visual_events(info,shots),report,ledger)
            _assign_timeline_ordinals(events)
            store.update(events=events,event_discovery_status="COMPLETE",production_status="PENDING")
            write_json(store_path,store)
            ledger.log("VISUAL_EVENTS_COMPLETE",mode="production",segment_id="FULL",events=len(events))
        all_events=events
        write_json(info["run"] / "visual_events.json", {"schema_version":"visual_events_v1","version":EVENT_VERSION,"events":events})
        _ensure_production_batches(store, events)
        store["status"]="RUNNING"; store["production_status"]="RUNNING"; store.pop("failure",None)
        write_json(store_path,store)
        events_by_id={event["visual_event_id"]: event for event in events}
        batch_total, _ = _batch_counts(store)
        observation_cues = parse_srt_file(info["srt"]).cues
        observation_segments = _read_json(info["narrative"]).get("segments", [])
        _summary(info,ledger,events,"RUNNING",stage="visual_events",segments_total=len(narrative_segments),timings={"technical_shots_seconds":time.monotonic()-technical_started},technical_shot_count=len(shots),production_batches_total=batch_total)

        for batch_index, (batch_id, batch) in enumerate(store["batches"].items(), 1):
            if batch.get("status") == "COMPLETE":
                continue
            batch_events=_batch_events(events_by_id,batch)
            finalization_complete=True
            current_event=batch_events[0]["visual_event_id"] if batch_events else "none"
            report(f"[movie-broll] semantic observations: {batch_id} ({len(batch_events)} Visual Events; current {current_event})")
            ledger.log("SEMANTIC_RUNNING",mode="production",segment_id=batch_id,events=len(batch_events))
            # Real configured providers expose ``generate``.  The legacy test
            # seam deliberately remains for old non-provider stand-ins only.
            if hasattr(active_provider, "generate"):
                from .semantic_observations import apply_cached_policy
                from .temporal_semantics import observe_events
                def contact(event: dict[str, Any], evidence: dict[str, Any]) -> bytes:
                    return candidate_contact_sheet(info["movie"], event, float(video["fps"]), evidence, _active_picture(info), evidence_profile="temporal_evidence_v2")
                def context(event: dict[str, Any], _catalog: dict[str, Any]) -> dict[str, Any]:
                    return semantic_request_context(event, observation_cues, observation_segments, {}, "FULL")
                observed = _bounded_operation(
                    f"observe {batch_id}",
                    lambda: observe_events(input_dir, batch_events, movie_sha256=info["movie_sha256"],
                                                   provider=active_provider, make_contact_sheet=contact,
                                                   make_context=context, active_picture=_active_picture(info)), report, ledger)
                policy = apply_cached_policy(input_dir, batch_events)
                semantic = {"status": "COMPLETE", "requests": observed["requests"],
                            "reused": observed["reused"], "usage": observed["usage"],
                            "estimated_cost_usd": observed["cost_usd"], "policy": policy,
                            "quota_exhausted": False, "provider_unavailable": False,
                            "blocked_provider": False, "failure": None}
            else:
                semantic=_bounded_operation(f"semantic {batch_id}",lambda: semantic_validate(batch_events,info["movie"],info["srt"],info["narrative"],_semantic_checkpoint_dir(info["run"]),float(video["fps"]),"FULL",provider=active_provider,model=model,preserve_event_ids=True,active_picture=_active_picture(info)),report,ledger)
            semantic_reports.append(semantic); batch["semantic_status"]=semantic.get("status","PARTIAL")
            # Preserve the existing bounded retry for malformed provider output.
            retry_events=[] if hasattr(active_provider, "generate") else _response_validation_retries(ledger,batch_events)
            if retry_events and _semantic_stop_status(semantic) is None:
                retry=_bounded_operation(f"semantic validation retry {batch_id}",lambda: semantic_validate(retry_events,info["movie"],info["srt"],info["narrative"],_semantic_checkpoint_dir(info["run"]),float(video["fps"]),"FULL",provider=active_provider,model=model,preserve_event_ids=True,active_picture=_active_picture(info)),report,ledger)
                semantic_reports.append(retry); semantic=retry; batch["semantic_status"]=retry.get("status","PARTIAL")

            # Provider/deferred stops are operational outcomes; do not turn
            # them into a generic semantic-integrity result before recording
            # their canonical resumable state.
            stop_status=_semantic_stop_status(semantic)
            if stop_status is not None:
                store["events"]=events
                batch["status"]="PARTIAL"
                write_json(store_path,store)
                stopped=_semantic_stop_result(info,ledger,events,batch,store_path,store,semantic,batch_id,batch_index,batch_total,shots,report)
                assert stopped is not None
                total, complete=_batch_counts(store)
                stopped["summary"]=_summary(info,ledger,events,stop_status,stage="semantic",segments_total=len(narrative_segments),segments_complete=complete,technical_shot_count=len(shots),production_batches_total=total,production_batches_complete=complete)
                stopped["summary"]["failure"]=stopped["failure"]; ledger.summary(**stopped["summary"])
                return stopped

            terminal, invalid_events = _semantic_terminal_contract(info["run"], batch_events)
            if not terminal:
                # Never allow downstream finalization (or a COMPLETE batch) to
                # mask an absent checkpoint/failure artifact.
                batch.update(semantic_status="PENDING", finalization_status="PENDING", status="PARTIAL",
                             resume_integrity_repair={"invalid_event_ids": invalid_events})
                store["events"] = events
                store["production_status"] = "PARTIAL"
                write_json(store_path, store)
                total, complete = _batch_counts(store)
                return {"status":"PARTIAL","summary":_summary(info,ledger,events,"PARTIAL",stage="semantic_integrity",segments_total=len(narrative_segments),segments_complete=complete,technical_shot_count=len(shots),production_batches_total=total,production_batches_complete=complete),"semantic":semantic_reports,"finalization":finals}

            # Scarcity sees exactly the already accepted higher-priority events
            # plus this batch's completed KEEP decisions.  This is equivalent to
            # one global pass because batches use the same stable priority order.
            first=store["production_queue"].index(batch["event_ids"][0]) if batch_events else 0
            prior=_accepted_higher_priority_keepers(events_by_id,store["production_queue"],first)
            eligible=[event for event in batch_events if event.get("editorial",{}).get("decision")=="KEEP" and event.get("editorial",{}).get("status")=="VALIDATED"]
            if eligible:
                apply_semantic_scarcity(prior + eligible)
            accepted=[event for event in batch_events if event.get("editorial",{}).get("decision")=="KEEP" and event.get("editorial",{}).get("status")=="VALIDATED"]
            if accepted:
                if detector_preflight_result is None:
                    detector_preflight_result=_bounded_operation("person-detector-preflight", person_detector_preflight, report, ledger)
                final=finalize_pilot(input_dir,batch_id,candidates=batch_events,shots={x["shot_id"]:x for x in shots},detector_preflight=detector_preflight_result)
                finals.append(final); batch["finalization_status"]=final.get("status","COMPLETE")
                batch["finalization"]=final
                finalization_complete=final.get("status", "COMPLETE") == "COMPLETE"
                if final.get("completed",0): ledger.log("ASSET_COMPLETE",mode="production",segment_id=batch_id,assets=final["completed"])
            else:
                batch["finalization_status"]="COMPLETE"

            store["events"]=events
            batch["status"]="COMPLETE" if finalization_complete and _batch_finalization_complete(batch_events,ledger) else "PARTIAL"
            if batch["status"] != "COMPLETE":
                store["events"]=events; store["production_status"]="PARTIAL"; write_json(store_path,store)
                total, complete=_batch_counts(store)
                return {"status":"PARTIAL","summary":_summary(info,ledger,events,"PARTIAL",stage="finalization",segments_total=len(narrative_segments),segments_complete=complete,technical_shot_count=len(shots),production_batches_total=total,production_batches_complete=complete),"semantic":semantic_reports,"finalization":finals}
            store["events"]=events; write_json(store_path,store)
            total, complete=_batch_counts(store)
            _summary(info,ledger,events,"RUNNING",stage="batch_complete",segments_total=len(narrative_segments),segments_complete=complete,technical_shot_count=len(shots),production_batches_total=total,production_batches_complete=complete)

        store.update(status="COMPLETE",production_status="COMPLETE",completed_at=_utc())
        write_json(store_path,store)
        ledger.log("PRODUCTION_COMPLETE",mode="production")
        total, complete=_batch_counts(store)
        return {"status":"COMPLETE","summary":_summary(info,ledger,events,"COMPLETE",stage="complete",segments_total=len(narrative_segments),segments_complete=len(narrative_segments),timings={"total_seconds":time.monotonic()-started},technical_shot_count=len(shots),production_batches_total=total,production_batches_complete=complete),"semantic":semantic_reports,"finalization":finals}
    except KeyboardInterrupt:
        ledger.log("PRODUCTION_INTERRUPTED",mode="production")
        return {"status":"PARTIAL","summary":_summary(info,ledger,all_events,"PARTIAL",stage="interrupted",segments_total=len(narrative_segments),segments_complete=0,technical_shot_count=len(shots))}
