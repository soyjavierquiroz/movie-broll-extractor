"""Versioned, sidecar semantic reclassification and explicit promotion.

This module deliberately does not reuse ``semantic_validate`` directly: that
function owns the canonical production ledger.  Reclassification uses the
same request, contact-sheet, and validation boundary, but persists every
outcome under a contract-specific workspace until an operator promotes it.
"""
from __future__ import annotations

import copy
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .active_picture import load_cached_active_picture
from .broll_pilot import (
    candidate_contact_sheet,
    semantic_request_context,
    shot_focus_diagnostics,
    target_binding_diagnostics,
)
from .broll_semantics import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_REASONING_EFFORT,
    PROMPT,
    SemanticProvider,
    build_openai_provider_from_env,
    classify_provider_error,
    estimate_openai_cost,
    redact_provider_error,
    validate_response,
)
from .finalization import finalize_pilot
from .inspect_source import inspect_movie
from .srt import parse_srt_file
from .utils import sha256_file, sha256_text, write_json
from .semantic_v9_1 import (
    OpenAISemanticV9_1StructuredResult, PROMPT_V9_1, SEMANTIC_CONTRACT_V9_1,
    SEMANTIC_PROMPT_V9_1, SEMANTIC_SCHEMA_VERSION_V9_1, effective_decision,
    validate_response_v9_1,
)


CONTRACT = "semantic-v9"
CONTRACT_VERSION = "V9"
WORKSPACE_SCHEMA = "semantic_reclassification_workspace_v1"
RESULT_SCHEMA = "semantic_reclassification_result_v1"
CANONICAL_CONTRACT_SCHEMA = "canonical_semantic_contract_v1"


@dataclass(frozen=True)
class ContractSpec:
    workspace_name: str
    version: str
    schema_version: str
    prompt_version: str
    prompt: str
    validator: Callable[[dict[str, Any]], list[str]]
    response_model: Any | None = None


V9_SPEC = ContractSpec(CONTRACT, CONTRACT_VERSION, "broll_semantics_v9", "broll_semantic_prompt_v9", PROMPT, validate_response)
V9_1_SPEC = ContractSpec("semantic-v9.1", "V9.1", SEMANTIC_SCHEMA_VERSION_V9_1, SEMANTIC_PROMPT_V9_1, PROMPT_V9_1, validate_response_v9_1, OpenAISemanticV9_1StructuredResult)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _root(input_dir: Path) -> Path:
    return input_dir.resolve().parents[1]


def _run(input_dir: Path) -> Path:
    return _root(input_dir) / "runs" / input_dir.name


def workspace_path(input_dir: Path, spec: ContractSpec = V9_SPEC) -> Path:
    return _run(input_dir) / "semantic_reclassifications" / spec.workspace_name


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _store(input_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = _run(input_dir) / "visual_event_segments_v1.json"
    if not path.is_file():
        raise FileNotFoundError(f"reclassification requires canonical Visual Events: {path}")
    store = _read(path)
    events = store.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("reclassification requires a non-empty canonical Visual Event set")
    if any(not isinstance(x, dict) or not isinstance(x.get("visual_event_id"), str) for x in events):
        raise ValueError("canonical Visual Event set has invalid event identities")
    return path, store


def _event_snapshot(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The immutable technical/narrative identity expected by this workspace."""
    required = ("visual_event_id", "candidate_id", "timeline_ordinal", "start_frame",
                "end_frame_exclusive", "start_seconds", "end_seconds", "source_shot_ids")
    snapshot = []
    for event in events:
        missing = [key for key in required if key not in event]
        if missing:
            raise ValueError(f"Visual Event {event.get('visual_event_id', '?')} missing {', '.join(missing)}")
        snapshot.append({key: event[key] for key in required})
    return sorted(snapshot, key=lambda x: (x["timeline_ordinal"], x["visual_event_id"]))


def _snapshot_digest(events: list[dict[str, Any]]) -> str:
    return sha256_text(json.dumps(_event_snapshot(events), ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _workspace_manifest(input_dir: Path, events: list[dict[str, Any]], active: Any | None, spec: ContractSpec = V9_SPEC) -> dict[str, Any]:
    return {
        "schema_version": WORKSPACE_SCHEMA,
        "contract": spec.workspace_name,
        "semantic_contract_version": spec.version,
        "semantic_contract_identity": SEMANTIC_CONTRACT_V9_1 if spec == V9_1_SPEC else "semantic_contract_v9",
        "semantic_schema_version": spec.schema_version,
        "semantic_prompt_version": spec.prompt_version,
        "created_at": _utc(),
        "updated_at": _utc(),
        "movie_id": input_dir.name,
        "canonical_event_count": len(events),
        "canonical_event_snapshot_sha256": _snapshot_digest(events),
        "event_ids": [x["visual_event_id"] for x in _event_snapshot(events)],
        "provider": getattr(active, "identifier", None),
        "model": getattr(active, "model", DEFAULT_OPENAI_MODEL),
        "reasoning_effort": getattr(active, "reasoning_effort", DEFAULT_OPENAI_REASONING_EFFORT),
        "state": "RUNNING",
        "promotion": {"state": "NOT_PROMOTED"},
    }


def _verify_workspace(manifest: dict[str, Any], input_dir: Path, events: list[dict[str, Any]], spec: ContractSpec = V9_SPEC) -> None:
    if manifest.get("contract") != spec.workspace_name or manifest.get("semantic_contract_version") != spec.version:
        raise ValueError(f"existing workspace is not a semantic {spec.version} workspace")
    if manifest.get("canonical_event_snapshot_sha256") != _snapshot_digest(events):
        raise RuntimeError("canonical Visual Events changed since V9 reclassification began; create a new migration workspace")


def _provider(provider: SemanticProvider | None, *, dry_run: bool, spec: ContractSpec = V9_SPEC) -> SemanticProvider | None:
    if dry_run:
        return None
    active = provider or build_openai_provider_from_env(DEFAULT_OPENAI_MODEL, response_model=spec.response_model) if spec.response_model else provider or build_openai_provider_from_env(DEFAULT_OPENAI_MODEL)
    if active is None:
        raise RuntimeError(f"OPENAI_API_KEY is required for semantic {spec.version} reclassification")
    if getattr(active, "identifier", None) != "openai" or getattr(active, "model", None) != DEFAULT_OPENAI_MODEL:
        raise ValueError(f"semantic {spec.version} reclassification requires openai/gpt-6-luna")
    if getattr(active, "reasoning_effort", DEFAULT_OPENAI_REASONING_EFFORT) != DEFAULT_OPENAI_REASONING_EFFORT:
        raise ValueError(f"semantic {spec.version} reclassification requires reasoning=none")
    return active


def _result_path(workspace: Path, event_id: str) -> Path:
    return workspace / "events" / f"{event_id}.json"


def _valid_result(record: dict[str, Any], spec: ContractSpec = V9_SPEC) -> bool:
    return (record.get("state") == "VALID"
            and record.get("semantic_contract_version") == spec.version
            and record.get("local_validation", {}).get("valid") is True
            and isinstance(record.get("response"), dict))


def _terminal_result(record: dict[str, Any]) -> bool:
    return record.get("state") == "TERMINAL_FAILURE" and record.get("terminal_failure") is True


def _load_result(workspace: Path, event_id: str) -> dict[str, Any] | None:
    try:
        record = _read(_result_path(workspace, event_id))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return record if record.get("event_id") == event_id else None


def _usage(raw: Any) -> dict[str, int]:
    keys = ("prompt_tokens", "cached_tokens", "response_tokens", "thinking_tokens", "total_tokens")
    return {key: int(raw.get(key, 0) or 0) if isinstance(raw, dict) else 0 for key in keys}


def _record_base(event: dict[str, Any], old_decision: Any, spec: ContractSpec = V9_SPEC, v9_decision: Any = None) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        "event_id": event["visual_event_id"],
        "candidate_id": event["candidate_id"],
        "timeline_ordinal": event["timeline_ordinal"],
        "old_decision": old_decision,
        "v9_decision": v9_decision,
        "semantic_contract_version": spec.version,
        "semantic_contract_identity": SEMANTIC_CONTRACT_V9_1 if spec == V9_1_SPEC else "semantic_contract_v9",
        "semantic_prompt_version": spec.prompt_version,
        "run_identity": spec.workspace_name,
        "state": "PENDING",
        "provider_requests_total": 0,
        "token_usage": _usage({}),
        "cost_usd": 0.0,
        "created_at": _utc(),
    }


def _add_usage(record: dict[str, Any], usage: dict[str, int]) -> None:
    total = _usage(record.get("token_usage"))
    for key, value in usage.items():
        total[key] += value
    record["token_usage"] = total
    record["cost_usd"] = estimate_openai_cost(total)


def run_reclassification(
    input_dir: Path, *, provider: SemanticProvider | None = None,
    contact_sheet: Callable[..., bytes] = candidate_contact_sheet,
    dry_run: bool = False, max_events: int | None = None, spec: ContractSpec = V9_SPEC,
) -> dict[str, Any]:
    """Run/resume V9 without touching canonical decisions, assets, or review."""
    input_dir = input_dir.resolve()
    store_path, store = _store(input_dir)
    del store_path
    events = [copy.deepcopy(x) for x in store["events"]]
    active = _provider(provider, dry_run=dry_run, spec=spec)
    workspace = workspace_path(input_dir, spec)
    manifest_path = workspace / "manifest.json"
    if manifest_path.is_file():
        manifest = _read(manifest_path)
        _verify_workspace(manifest, input_dir, events, spec)
    else:
        workspace.mkdir(parents=True, exist_ok=True)
        manifest = _workspace_manifest(input_dir, events, active, spec)
        write_json(manifest_path, manifest)

    movie, srt = input_dir / "movie.mp4", input_dir / "subtitles.srt"
    narrative = _run(input_dir) / "narrative-v2" / "narrative_map.json"
    for label, path in (("movie", movie), ("SRT", srt), ("narrative map", narrative)):
        if not path.is_file():
            raise FileNotFoundError(f"reclassification requires {label}: {path}")
    active_picture_path = _run(input_dir) / "active_picture.json"
    if not active_picture_path.is_file():
        raise FileNotFoundError(f"reclassification requires canonical active picture: {active_picture_path}")
    active_picture = load_cached_active_picture(active_picture_path)
    fps = float(inspect_movie(movie)["video"]["fps"])
    cues = parse_srt_file(srt).cues
    segments = _read(narrative).get("segments")
    if not isinstance(segments, list):
        raise ValueError("narrative map segments must be a list")

    processed = reused = 0
    for event in events:
        event_id = event["visual_event_id"]
        old = _load_result(workspace, event_id)
        if old and (_valid_result(old, spec) or _terminal_result(old)):
            reused += 1
            continue
        if max_events is not None and processed >= max_events:
            break
        v9_result = _load_result(workspace_path(input_dir, V9_SPEC), event_id) if spec == V9_1_SPEC else None
        v9_decision = (v9_result or {}).get("effective_decision") or ((v9_result or {}).get("response") or {}).get("editorial", {}).get("decision")
        record = old or _record_base(event, event.get("editorial", {}).get("decision"), spec, v9_decision)
        started = time.monotonic()
        evidence: dict[str, Any] = {}
        try:
            context = semantic_request_context(event, cues, segments, evidence, "FULL")
            sheet = contact_sheet(movie, event, fps, evidence, active_picture)
            context = semantic_request_context(event, cues, segments, evidence, "FULL")
            if dry_run:
                record.update(state="DRY_RUN", local_validation={"valid": False, "errors": ["dry_run"]}, updated_at=_utc())
            else:
                response = active.generate(spec.prompt, context, sheet)
                data = response.data
                usage = _usage(response.usage)
                record["provider_requests_total"] = int(record.get("provider_requests_total", 0)) + max(1, int(getattr(response, "attempts", 1) or 1))
                _add_usage(record, usage)
                focus = shot_focus_diagnostics(data, event)
                binding_errors = target_binding_diagnostics(data, event, evidence)
                errors = list(spec.validator(data))
                if focus["validation_reasons"]:
                    errors.append("incomplete or mismatched shot focus plan")
                if binding_errors:
                    errors.append("invalid or mismatched semantic target binding")
                if errors:
                    record.update(
                        state="TERMINAL_FAILURE", terminal_failure=True,
                        failure_stage="semantic_validation", error_message="; ".join(sorted(set(errors))),
                        response=data, provider=getattr(response, "provider", None) or active.identifier,
                        model=getattr(response, "model", None) or active.model,
                        local_validation={"valid": False, "errors": sorted(set(errors)), "focus": focus,
                                          "target_binding_errors": binding_errors},
                    )
                else:
                    local_validation = {"valid": True, "errors": [], "focus": focus,
                                        "target_binding_errors": []}
                    record.update(
                        state="VALID", terminal_failure=False, response=data,
                        provider=getattr(response, "provider", None) or active.identifier,
                        model=getattr(response, "model", None) or active.model,
                        provider_attempts=max(1, int(getattr(response, "attempts", 1) or 1)),
                        provider_trace=list(getattr(response, "provider_trace", []) or []),
                        local_validation=local_validation,
                    )
                    checkpoint = {
                        "semantic_contract_version": spec.version,
                        "semantic_contract_identity": SEMANTIC_CONTRACT_V9_1 if spec == V9_1_SPEC else "semantic_contract_v9",
                        "semantic_schema_version": spec.schema_version,
                        "semantic_prompt_version": spec.prompt_version,
                        "visual_event_id": event_id, "candidate_id": event["candidate_id"],
                        "window_id": "FULL", "start_frame": event["start_frame"],
                        "end_frame_exclusive": event["end_frame_exclusive"],
                        "candidate_identity": {"window_id": "FULL", "candidate_id": event["candidate_id"],
                                               "start_frame": event["start_frame"],
                                               "end_frame_exclusive": event["end_frame_exclusive"]},
                        "provider": record["provider"], "model": record["model"],
                        "timestamp": _utc(), "run_identity": spec.workspace_name,
                        "response": data, "usage": usage,
                    }
                    write_json(workspace / "checkpoints" / f"{event['candidate_id']}.json", checkpoint)
        except Exception as error:
            detail = classify_provider_error(error)
            record.update(
                state="PROVIDER_DEFERRED" if detail.get("retryable", True) else "TERMINAL_FAILURE",
                terminal_failure=not bool(detail.get("retryable", True)),
                failure_stage="provider_call", error_message=redact_provider_error(error),
                error_kind=detail.get("reason"), provider=detail.get("provider") or getattr(active, "identifier", None),
                model=detail.get("model") or getattr(active, "model", None), retryable=bool(detail.get("retryable", True)),
            )
            record["provider_requests_total"] = int(record.get("provider_requests_total", 0)) + 1
        model_decision = ((record.get("response") or {}).get("editorial") or {}).get("decision")
        record["model_decision"] = model_decision
        record["effective_decision"] = effective_decision(record.get("response"), record.get("local_validation")) if spec == V9_1_SPEC else (model_decision if record.get("local_validation", {}).get("valid") else "REVIEW")
        record["latency_seconds"] = round(time.monotonic() - started, 4)
        record["updated_at"] = _utc()
        write_json(_result_path(workspace, event_id), record)
        processed += 1

    report = reclassification_report(input_dir, spec=spec)
    manifest.update(updated_at=_utc(), state=("COMPLETE" if report["complete"] else "PARTIAL"),
                    totals=report["totals"], comparison=report["comparison"])
    write_json(manifest_path, manifest)
    write_json(workspace / "comparison_report.json", report)
    return {**report, "workspace": workspace, "processed": processed, "reused": reused,
            "dry_run": dry_run}


def reclassification_report(input_dir: Path, *, spec: ContractSpec = V9_SPEC) -> dict[str, Any]:
    """Read sidecar-only progress and comparison evidence."""
    input_dir = input_dir.resolve()
    _path, store = _store(input_dir)
    workspace = workspace_path(input_dir, spec)
    rows: list[dict[str, Any]] = []
    totals = {"KEEP": 0, "REJECT": 0, "REVIEW": 0, "failures": 0,
              "requests": 0, "prompt_tokens": 0, "cached_tokens": 0,
              "response_tokens": 0, "thinking_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
    transitions: dict[str, list[str]] = {}
    complete = True
    for event in store["events"]:
        old = str(event.get("editorial", {}).get("decision", "REVIEW"))
        result = _load_result(workspace, event["visual_event_id"])
        if _valid_result(result or {}, spec):
            model_decision = str(result["response"]["editorial"]["decision"])
            new = str(result.get("effective_decision") or model_decision)
        elif _terminal_result(result or {}):
            new = "REVIEW"
            totals["failures"] += 1
        else:
            complete = False
            new = None
        if new in ("KEEP", "REJECT", "REVIEW"):
            totals[new] += 1
            key = f"{old} -> {new}"
            transitions.setdefault(key, []).append(event["visual_event_id"])
        usage = _usage((result or {}).get("token_usage"))
        totals["requests"] += int((result or {}).get("provider_requests_total", 0) or 0)
        for name, value in usage.items():
            totals[name] += value
        rows.append({"event_id": event["visual_event_id"], "ordinal": event.get("timeline_ordinal"),
                     "old_decision": old, "model_decision": (result or {}).get("model_decision"), "effective_decision": new,
                     "v9_decision": (result or {}).get("v9_decision"),
                     "state": (result or {}).get("state", "MISSING")})
    totals["cost_usd"] = estimate_openai_cost({key: totals[key] for key in (
        "prompt_tokens", "cached_tokens", "response_tokens", "thinking_tokens", "total_tokens")})
    return {"schema_version": "semantic_reclassification_comparison_v1", "contract": spec.workspace_name,
            "complete": complete, "event_count": len(rows), "totals": totals,
            "comparison": {key: {"count": len(ids), "event_ids": ids} for key, ids in sorted(transitions.items())},
            "events": rows}


def _archive_package(directory: Path, base: str, destination: Path) -> bool:
    names = (f"{base}.mp4", f"v{base}.mp4", f"{base}.jpg", f"v{base}.jpg", f"{base}.json")
    present = [directory / name for name in names if (directory / name).exists()]
    if not present:
        return False
    destination.mkdir(parents=True, exist_ok=True)
    for source in present:
        shutil.move(str(source), str(destination / source.name))
    return True


def _archive_stale_packages(run: Path, events: list[dict[str, Any]], old: dict[str, str], new: dict[str, str], archive_name: str = "semantic-v8-to-v9") -> list[str]:
    try:
        registry = _read(run / "asset_registry.json").get("events", {})
    except (OSError, ValueError, json.JSONDecodeError):
        registry = {}
    archived: list[str] = []
    for event in events:
        eid = event["visual_event_id"]
        if old.get(eid) != "KEEP" or new.get(eid) != "REJECT":
            continue
        entry = registry.get(eid, {}) if isinstance(registry, dict) else {}
        if not isinstance(entry, dict) or not entry.get("asset_id") or not entry.get("slug"):
            continue
        base = f"{entry['asset_id']}-{entry['slug']}"
        destination = run / "superseded" / archive_name / eid
        moved = _archive_package(run / "assets", base, destination)
        moved = _archive_package(run / "review", base, destination / "review") or moved
        if moved:
            archived.append(eid)
    return archived


def promote_reclassification(input_dir: Path, *, finalize: bool = True, spec: ContractSpec = V9_SPEC) -> dict[str, Any]:
    """Atomically switch canonical editorial semantics after complete validation.

    The event manifest replacement is atomic.  Checkpoint/history preparation is
    completed first, so the canonical manifest never points at a partial V9 set.
    """
    input_dir = input_dir.resolve()
    store_path, store = _store(input_dir)
    events = store["events"]
    workspace = workspace_path(input_dir, spec)
    manifest = _read(workspace / "manifest.json")
    _verify_workspace(manifest, input_dir, events, spec)
    report = reclassification_report(input_dir, spec=spec)
    if not report["complete"]:
        raise RuntimeError(f"{spec.version} reclassification is incomplete; promotion is refused")
    new_by_id: dict[str, dict[str, Any]] = {}
    old_decisions: dict[str, str] = {}
    new_decisions: dict[str, str] = {}
    terminal_failures: list[dict[str, Any]] = []
    for event in events:
        eid = event["visual_event_id"]
        old_decisions[eid] = str(event.get("editorial", {}).get("decision", "REVIEW"))
        result = _load_result(workspace, eid)
        if _valid_result(result or {}, spec):
            response = result["response"]
            promoted = copy.deepcopy(event)
            promoted.update(visual=response["visual"], people=response["visual"].get("people", []),
                            relationships=response["relationships"],
                            editorial={**response["editorial"], "status": "VALIDATED"})
            new_decisions[eid] = str(result.get("effective_decision") or response["editorial"]["decision"])
        elif _terminal_result(result or {}):
            promoted = copy.deepcopy(event)
            promoted.update(visual={}, people=[], relationships=[], editorial={
                "decision": "REVIEW", "status": "SEMANTIC_INCOMPLETE",
                "reason": result.get("error_message", f"terminal {spec.version} semantic failure"),
            })
            terminal_failures.append(result)
            new_decisions[eid] = "REVIEW"
        else:  # protected by report; retain a hard guard against mixed state.
            raise RuntimeError(f"V9 result missing or invalid for {eid}")
        new_by_id[eid] = promoted

    # Preserve V8 evidence before activation.  V9 checkpoints deliberately
    # remain in their versioned workspace; the atomic event-store contract
    # pointer below selects them.  This avoids a window where a generic
    # ``semantic_checkpoints`` directory contains V9 results while the
    # canonical event manifest still says V8.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    history = _run(input_dir) / "semantic_history" / ("v8" if spec == V9_SPEC else "pre-v9.1") / stamp
    history.mkdir(parents=True, exist_ok=True)
    for name in ("semantic_checkpoints", "semantic_failures"):
        source = _run(input_dir) / name
        if source.exists():
            shutil.copytree(source, history / name, dirs_exist_ok=True)
    write_json(history / f"visual_event_segments_v1.before_{spec.workspace_name}.json", store)
    for event in events:
        result = _load_result(workspace, event["visual_event_id"])
        if _valid_result(result or {}, spec):
            source = workspace / "checkpoints" / f"{event['candidate_id']}.json"
            if not source.is_file():
                raise RuntimeError(f"{spec.version} checkpoint missing for {event['visual_event_id']}")

    promoted_store = copy.deepcopy(store)
    promoted_store["events"] = [new_by_id[event["visual_event_id"]] for event in events]
    promoted_store["canonical_semantic_contract"] = spec.version
    promoted_store["canonical_semantic_workspace"] = str(workspace.relative_to(_run(input_dir)))
    promoted_store["semantic_promoted_at"] = _utc()
    # Re-run ordinary production as a no-op after promotion: semantic batches
    # are complete, while new KEEPs are finalized immediately below.
    for batch in promoted_store.get("batches", {}).values():
        if isinstance(batch, dict):
            batch["semantic_status"] = "COMPLETE"
    write_json(store_path, promoted_store)
    write_json(_run(input_dir) / "canonical_semantic_contract.json", {
        "schema_version": CANONICAL_CONTRACT_SCHEMA, "active_contract": spec.version,
        "workspace": str(workspace.relative_to(_run(input_dir))), "promoted_at": _utc(),
        "provider_models": sorted({f"{_load_result(workspace, x['visual_event_id']).get('provider')}/{_load_result(workspace, x['visual_event_id']).get('model')}" for x in events}),
    })
    archived = _archive_stale_packages(_run(input_dir), events, old_decisions, new_decisions,
                                       "semantic-v8-to-v9" if spec == V9_SPEC else "semantic-v8-to-v9.1")
    newly_keep = [new_by_id[eid] for eid in new_by_id if old_decisions[eid] != "KEEP" and new_decisions[eid] == "KEEP"]
    finalization: dict[str, Any] | None = None
    if finalize and newly_keep:
        shots_path = _run(input_dir) / "technical_shots.json"
        shots_data = _read(shots_path).get("shots", []) if shots_path.is_file() else []
        shots = {x["shot_id"]: x for x in shots_data if isinstance(x, dict) and isinstance(x.get("shot_id"), str)}
        finalization = finalize_pilot(input_dir, spec.workspace_name, candidates=newly_keep, shots=shots)
    manifest.update(state="PROMOTED", promoted_at=_utc(), promotion={
        "state": "PROMOTED", "history": str(history.relative_to(_run(input_dir))),
        "archived_stale_event_ids": archived, "newly_finalized_event_ids": [x["visual_event_id"] for x in newly_keep],
    })
    write_json(workspace / "manifest.json", manifest)
    return {"status": "PROMOTED", "contract": spec.version, "report": report,
            "archived_stale_event_ids": archived, "newly_keep_event_ids": [x["visual_event_id"] for x in newly_keep],
            "finalization": finalization, "history": history}


def run_reclassification_v9_1(input_dir: Path, **kwargs: Any) -> dict[str, Any]:
    """Create/resume the isolated V9.1 sidecar; canonical events stay read-only."""
    return run_reclassification(input_dir, spec=V9_1_SPEC, **kwargs)


def promote_reclassification_v9_1(input_dir: Path, **kwargs: Any) -> dict[str, Any]:
    """Explicitly promote only a complete V9.1 workspace."""
    return promote_reclassification(input_dir, spec=V9_1_SPEC, **kwargs)
