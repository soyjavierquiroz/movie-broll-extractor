"""Read-only calibration runner for the V9 B-roll semantic contract.

This intentionally bypasses ``semantic_validate``: that production function
creates checkpoints and ledger entries.  The benchmark reads canonical events,
renders temporary in-memory contact sheets, and writes only its own evidence.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .broll_pilot import (
    candidate_contact_sheet,
    semantic_request_context,
    shot_focus_diagnostics,
    target_binding_diagnostics,
)
from .active_picture import load_cached_active_picture
from .broll_semantics import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_REASONING_EFFORT,
    PROMPT,
    SemanticProvider,
    build_openai_provider_from_env,
    estimate_openai_cost,
    validate_response,
)
from .srt import parse_srt_file
from .utils import sha256_file, write_json

BENCHMARK_SCHEMA_VERSION = "semantic_v9_read_only_benchmark_v1"
BENCHMARK_NAME = "semantic-v9"

# These controls are event identities supplied for E02 calibration.  Existing
# KEEP controls are deliberately selected from the canonical store below.
FALSE_NEGATIVE_IDS = (
    "VE_363308db3d63748a", "VE_384f21248c51722e", "VE_88e85094774cce23",
    "VE_36b2e31a201fd12a", "VE_d6003df7ccf57ab2", "VE_b2256b8c76ac23fd",
    "VE_d2296dd0be807b37", "VE_7d442d1de28df510", "VE_5ef3974601f4b22a",
    "VE_705822c98c105f3d",
)
HARD_NEGATIVE_IDS = (
    "VE_ae4a2fccbe627d9e", "VE_94b6aa8c3587bdac", "VE_d3ab2b1161e76d31",
    "VE_5663be42e6b58509", "VE_de9ccfc633c86f8d",
)
DIALOGUE_NEGATIVE_IDS = (
    "VE_f1a45e882662ff63", "VE_2cb31a038dff6f62", "VE_a74fa32a4007d061",
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _run_dir(input_dir: Path) -> Path:
    return input_dir.resolve().parents[1] / "runs" / input_dir.name


def _event_store(run: Path) -> list[dict[str, Any]]:
    path = run / "visual_event_segments_v1.json"
    events = _read_object(path).get("events")
    if not isinstance(events, list):
        raise ValueError(f"canonical event store has no events list: {path}")
    return [dict(event) for event in events if isinstance(event, dict)]


def _event_text(event: dict[str, Any]) -> str:
    visual = event.get("visual", {})
    editorial = event.get("editorial", {})
    return " ".join(str(value).casefold() for value in (
        visual.get("actions", []), visual.get("visible_interactions", []),
        visual.get("visible_emotions", []), editorial.get("standalone_meaning_es", ""),
    ))


def _select_positive_controls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive diverse controls from current canonical KEEPs, never fixed IDs."""
    keeps = sorted(
        (event for event in events if event.get("editorial", {}).get("decision") == "KEEP"),
        key=lambda event: (event.get("timeline_ordinal", 0), event.get("visual_event_id", "")),
    )
    rules = (
        ("movement", ("walking", "camina", "aleja")),
        ("object_activity", ("maquill", "makeup", "espejo", "mirror", "bebe", "copa", "cajón", "drawer")),
        ("interaction", ("touching", "hombro", "abraza", "acaricia", "face_to_face")),
        ("reaction", ("observa", "mirar", "looking", "listen", "escucha")),
        ("emotional_state", ("resting head", "apoya su cabeza")),
        ("action", ("carrying", "lleva comida", "gesticul", "lanzando besos")),
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for _category, terms in rules:
        candidate = next((event for event in keeps
                          if event.get("visual_event_id") not in selected_ids
                          and any(term in _event_text(event) for term in terms)), None)
        if candidate is not None:
            selected.append({**candidate, "_benchmark_category": f"positive_control:{_category}"})
            selected_ids.add(str(candidate["visual_event_id"]))
    for event in keeps:
        if len(selected) >= 6:
            break
        if event.get("visual_event_id") not in selected_ids:
            selected.append({**event, "_benchmark_category": "positive_control:canonical_keep"})
            selected_ids.add(str(event["visual_event_id"]))
    if len(selected) != 6:
        raise ValueError(f"benchmark requires six existing canonical KEEP controls; found {len(keeps)}")
    return selected


def select_benchmark_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return exactly 24 calibrated events, ordered by benchmark category."""
    by_id = {str(event.get("visual_event_id")): event for event in events}
    requested = (
        *(("false_negative_candidate", event_id) for event_id in FALSE_NEGATIVE_IDS),
        *(("negative_control:hard_reject", event_id) for event_id in HARD_NEGATIVE_IDS),
        *(("negative_control:plain_dialogue", event_id) for event_id in DIALOGUE_NEGATIVE_IDS),
    )
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for category, event_id in requested:
        event = by_id.get(event_id)
        if event is None:
            missing.append(event_id)
        else:
            selected.append({**event, "_benchmark_category": category})
    if missing:
        raise ValueError("benchmark controls absent from canonical event store: " + ", ".join(missing))
    selected.extend(_select_positive_controls(events))
    if len(selected) != 24 or len({event["visual_event_id"] for event in selected}) != 24:
        raise ValueError("semantic V9 benchmark must contain 24 unique events")
    return selected


def _old_provider(run: Path, event: dict[str, Any]) -> str | None:
    candidate_id = event.get("candidate_id")
    if not isinstance(candidate_id, str):
        return None
    path = run / "semantic_checkpoints" / f"{candidate_id}.json"
    try:
        checkpoint = _read_object(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    provider, model = checkpoint.get("provider"), checkpoint.get("model")
    if not isinstance(provider, str):
        return None
    return f"{provider}/{model}" if isinstance(model, str) and model else provider


def _protected_signature(run: Path) -> dict[str, Any]:
    """Cheap, deterministic proof that benchmark execution changed no canon."""
    files = ("active_picture.json", "visual_event_segments_v1.json", "visual_events.json", "processing_ledger.json",
             "progress_summary.json", "asset_registry.json")
    signature: dict[str, Any] = {"files": {}, "trees": {}}
    for name in files:
        path = run / name
        signature["files"][name] = sha256_file(path) if path.is_file() else None
    for name in ("semantic_checkpoints", "semantic_failures", "assets", "review"):
        directory = run / name
        signature["trees"][name] = sorted(
            (str(path.relative_to(run)), path.stat().st_size, path.stat().st_mtime_ns)
            for path in directory.rglob("*") if path.is_file()
        ) if directory.is_dir() else []
    return signature


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _output_directory(run: Path, run_id: str | None, benchmark_name: str = BENCHMARK_NAME) -> Path:
    output = run / "benchmarks" / benchmark_name / (run_id or _utc_run_id())
    if output.exists():
        raise FileExistsError(f"benchmark output already exists: {output}")
    return output


class BenchmarkSystemicFailure(RuntimeError):
    """All selected events failed before the benchmark made a provider request."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        dominant = report.get("dominant_failure") or "unknown"
        super().__init__(f"BENCHMARK FAILED: {dominant}")


_STAGES = ("input_preparation", "contact_sheet", "provider_call", "structured_output", "local_validation", "persistence")


def _stage_error(record: dict[str, Any], stage: str, error: Exception) -> None:
    record["failure_stage"] = stage
    record["error_type"] = type(error).__name__
    record["error_message"] = str(error)
    record["stages"][stage] = "FAILED"


def _dominant_failure(records: list[dict[str, Any]]) -> str | None:
    failures: dict[tuple[str, str, str], int] = {}
    for record in records:
        key = (record.get("failure_stage"), record.get("error_type"), record.get("error_message"))
        if all(key):
            failures[key] = failures.get(key, 0) + 1
    if not failures:
        return None
    (stage, kind, message), _count = max(failures.items(), key=lambda item: item[1])
    return f"{stage} / {kind}: {message}"


def run_semantic_v9_benchmark(
    input_dir: Path, *, provider: SemanticProvider | None = None, run_id: str | None = None,
    contact_sheet: Callable[..., bytes] = candidate_contact_sheet, dry_run: bool = False,
    selected_events: list[dict[str, Any]] | None = None, benchmark_name: str = BENCHMARK_NAME,
    prompt: str = PROMPT, validator: Callable[[dict[str, Any]], list[str]] = validate_response,
    response_model: Any | None = None, contract_version: str = "V9", benchmark_schema: str = BENCHMARK_SCHEMA_VERSION,
    contract_identity: str = "semantic_contract_v9",
    effective_decider: Callable[[dict[str, Any], dict[str, Any]], str | None] | None = None,
) -> dict[str, Any]:
    """Call V9/OpenAI for controls without creating production state or assets."""
    input_dir, run = input_dir.resolve(), _run_dir(input_dir)
    movie, srt = input_dir / "movie.mp4", input_dir / "subtitles.srt"
    narrative = run / "narrative-v2" / "narrative_map.json"
    for label, path in (("movie", movie), ("SRT", srt), ("narrative map", narrative)):
        if not path.is_file():
            raise FileNotFoundError(f"benchmark requires {label}: {path}")
    events = selected_events or select_benchmark_events(_event_store(run))
    active = None if dry_run else provider or (build_openai_provider_from_env(DEFAULT_OPENAI_MODEL, response_model=response_model) if response_model else build_openai_provider_from_env(DEFAULT_OPENAI_MODEL))
    if not dry_run:
        if active is None:
            raise RuntimeError(f"OPENAI_API_KEY is required for the semantic {contract_version} benchmark")
        if getattr(active, "identifier", None) != "openai" or getattr(active, "model", None) != DEFAULT_OPENAI_MODEL:
            raise ValueError(f"semantic {contract_version} benchmark requires openai/gpt-6-luna")
        if getattr(active, "reasoning_effort", DEFAULT_OPENAI_REASONING_EFFORT) != DEFAULT_OPENAI_REASONING_EFFORT:
            raise ValueError(f"semantic {contract_version} benchmark requires reasoning=none")

    from .inspect_source import inspect_movie
    fps = float(inspect_movie(movie)["video"]["fps"])
    cues = parse_srt_file(srt).cues
    narrative_segments = _read_object(narrative).get("segments", [])
    if not isinstance(narrative_segments, list):
        raise ValueError("narrative map segments must be a list")
    active_picture_path = run / "active_picture.json"
    if not active_picture_path.is_file():
        raise FileNotFoundError(f"benchmark requires canonical active picture: {active_picture_path}")
    active_picture = load_cached_active_picture(active_picture_path)
    before = _protected_signature(run)
    output = _output_directory(run, run_id, benchmark_name)
    output.mkdir(parents=True)
    totals = {"prompt_tokens": 0, "cached_tokens": 0, "response_tokens": 0,
              "thinking_tokens": 0, "total_tokens": 0}
    records: list[dict[str, Any]] = []
    provider_requests = 0
    semantic_results = 0

    for event in events:
        event_id = str(event["visual_event_id"])
        evidence: dict[str, Any] = {}
        started = time.monotonic()
        record: dict[str, Any] = {
            "event_id": event_id,
            "ordinal": event.get("timeline_ordinal"),
            "category": event["_benchmark_category"],
            "old_decision": event.get("editorial", {}).get("decision"),
            "old_provider": _old_provider(run, event),
            "v9_decision": event.get("_v9_decision"),
            "model_decision": None, "effective_decision": None,
            "visual_utility_kind": None, "conversation_visual_signal": None,
            "action_evidence": None,
            "reusable_intent": None,
            "action_or_moment_complete": None,
            "reusable_broll": None,
            "reason": None,
            "token_usage": {}, "cost_usd": 0.0, "latency_seconds": None,
            "local_validation": {"valid": False, "errors": ["not_run"]},
            "failure_stage": None, "error_type": None, "error_message": None,
            "stages": {stage: "PENDING" for stage in _STAGES},
        }
        try:
            context = semantic_request_context(event, cues, narrative_segments, evidence, "FULL")
            record["stages"]["input_preparation"] = "COMPLETE"
        except Exception as error:
            _stage_error(record, "input_preparation", error)
            context = None
        if context is not None:
            try:
                sheet = contact_sheet(movie, event, fps, evidence, active_picture)
                # Evidence is populated by the canonical contact sheet; rebuild
                # context only after it exists, just as production does.
                context = semantic_request_context(event, cues, narrative_segments, evidence, "FULL")
                record["stages"]["contact_sheet"] = "COMPLETE"
            except Exception as error:
                _stage_error(record, "contact_sheet", error)
                sheet = None
        else:
            sheet = None
        if sheet is not None and dry_run:
            record["stages"]["provider_call"] = "SKIPPED_DRY_RUN"
            record["stages"]["structured_output"] = "SKIPPED_DRY_RUN"
            record["stages"]["local_validation"] = "SKIPPED_DRY_RUN"
        elif sheet is not None:
            try:
                provider_requests += 1
                response = active.generate(prompt, context, sheet)
                record["stages"]["provider_call"] = "COMPLETE"
            except Exception as error:
                _stage_error(record, "provider_call", error)
                response = None
            if response is not None:
                try:
                    data = response.data
                    if not isinstance(data, dict):
                        raise ValueError("provider structured response must be an object")
                    editorial = data.get("editorial", {})
                    qualification = editorial.get("keep_qualification", {})
                    if not isinstance(editorial, dict) or not isinstance(qualification, dict):
                        raise ValueError("provider structured response has invalid editorial object")
                    usage = {key: int(value or 0) for key, value in response.usage.items() if key in totals}
                    for key in totals:
                        totals[key] += usage.get(key, 0)
                    semantic_results += 1
                    record["stages"]["structured_output"] = "COMPLETE"
                except Exception as error:
                    _stage_error(record, "structured_output", error)
                    data = None
                if data is not None:
                    try:
                        errors = validator(data)
                        focus = shot_focus_diagnostics(data, event)
                        if focus["validation_reasons"]:
                            errors.append("incomplete or mismatched shot focus plan")
                        binding_errors = target_binding_diagnostics(data, event, evidence)
                        if binding_errors:
                            errors.append("invalid or mismatched semantic target binding")
                        record.update({
                            "v9_decision": (record["v9_decision"] if contract_version == "V9.1"
                                            else editorial.get("decision")),
                            "model_decision": editorial.get("decision"),
                            "action_evidence": qualification.get("action_evidence_es"),
                            "reusable_intent": qualification.get("reusable_use_case_es"),
                            "action_or_moment_complete": editorial.get("action_or_moment_complete"),
                            "reusable_broll": editorial.get("reusable_broll"),
                            "reason": editorial.get("reason"), "token_usage": usage,
                            "cost_usd": estimate_openai_cost(usage),
                            "local_validation": {"valid": not errors, "errors": sorted(set(errors)), "focus": focus, "target_binding_errors": binding_errors},
                            "visual_utility_kind": editorial.get("visual_utility_kind"),
                            "conversation_visual_signal": editorial.get("conversation_visual_signal"),
                        })
                        record["effective_decision"] = (
                            effective_decider(data, record["local_validation"])
                            if effective_decider is not None
                            else editorial.get("decision") if not errors else "REVIEW"
                        )
                        record["stages"]["local_validation"] = "COMPLETE"
                    except Exception as error:
                        _stage_error(record, "local_validation", error)
        record["latency_seconds"] = round(time.monotonic() - started, 4)
        records.append(record)
        try:
            record["stages"]["persistence"] = "COMPLETE"
            write_json(output / "events" / f"{int(event.get('timeline_ordinal', 0)):03d}_{event_id}.json", record)
        except Exception as error:
            _stage_error(record, "persistence", error)
            raise

    after = _protected_signature(run)
    read_only_verified = before == after
    summary = {
        "schema_version": benchmark_schema, "benchmark": benchmark_name,
        "contract_version": contract_version, "contract_identity": contract_identity,
        "run_id": output.name, "dry_run": dry_run, "provider": getattr(active, "identifier", "dry-run" if dry_run else None),
        "model": getattr(active, "model", DEFAULT_OPENAI_MODEL),
        "reasoning_effort": getattr(active, "reasoning_effort", DEFAULT_OPENAI_REASONING_EFFORT),
        "image_detail": getattr(active, "image_detail", None), "event_count": len(records),
        "selected_events": len(events), "provider_requests": provider_requests,
        "semantic_results": semantic_results,
        "locally_valid_results": sum(bool(record["local_validation"]["valid"]) for record in records),
        "failed_events": sum(record.get("failure_stage") is not None for record in records),
        "stage_counts": {stage: sum(record["stages"].get(stage) == "COMPLETE" for record in records) for stage in _STAGES},
        "failures_by_stage": {stage: sum(record.get("failure_stage") == stage for record in records) for stage in _STAGES},
        "counts": {decision: sum(record.get("effective_decision") == decision for record in records) for decision in ("KEEP", "REVIEW", "REJECT")} if semantic_results else {},
        "usage": totals, "cost_usd": estimate_openai_cost(totals),
        "read_only_verified": read_only_verified,
        "events": [{key: record.get(key) for key in ("event_id", "ordinal", "category", "old_decision", "v9_decision", "model_decision", "effective_decision", "visual_utility_kind", "conversation_visual_signal")}
                   for record in records],
    }
    systemic = not dry_run and provider_requests == 0 and semantic_results == 0 and len(records) == len(events)
    if systemic:
        summary["dominant_failure"] = _dominant_failure(records)
    write_json(output / "summary.json", summary)
    if not read_only_verified:
        raise RuntimeError("semantic V9 benchmark changed protected canonical production state")
    report = {**summary, "output": output}
    if systemic:
        raise BenchmarkSystemicFailure(report)
    return report
