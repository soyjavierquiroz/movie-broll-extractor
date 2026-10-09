"""The operator-facing, resumable production command.

This layer intentionally coordinates existing durable stages.  It does not
invent a second cache, checkpoint, or recovery format.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .inspect_source import inspect_movie
from .narrative import OVERLAP_SECONDS, TARGET_WINDOW_SECONDS, prepare_narrative_inputs
from .narrative_finalize import EXTERNAL_PROVENANCE, finalize_external
from .production_preflight import preflight
from .srt import cue_statistics, parse_srt_file, validate_timeline
from .supervisor import Supervisor
from .utils import sha256_file, write_json, write_jsonl
from .broll_semantics import DEFAULT_OPENAI_MODEL, estimate_openai_cost


class WaitingExternal(RuntimeError):
    """A deliberate, successful pause for the manual Narrative Mapper handoff."""


def _root(input_dir: Path) -> Path:
    return input_dir.resolve().parents[1]


def _run_dir(input_dir: Path) -> Path:
    return _root(input_dir) / "runs" / input_dir.name


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _active_semantic_workspace(run: Path) -> Path | None:
    try:
        store = _read(run / "visual_event_segments_v1.json")
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if store.get("canonical_semantic_contract") != "V9":
        return None
    relative = store.get("canonical_semantic_workspace")
    if not isinstance(relative, str) or not relative:
        return None
    workspace = run / relative
    try:
        workspace.resolve().relative_to(run.resolve())
    except ValueError:
        return None
    return workspace


def _complete_packages(directory: Path) -> int:
    """Count only established, complete 5-file finalization package units."""
    if not directory.is_dir():
        return 0
    packages = 0
    for metadata in directory.glob("*.json"):
        base = metadata.stem
        required = (metadata, directory / f"{base}.mp4", directory / f"v{base}.mp4",
                    directory / f"{base}.jpg", directory / f"v{base}.jpg")
        if all(path.is_file() for path in required):
            packages += 1
    return packages


def _finalization_qa(directory: Path) -> tuple[int, dict[str, int]]:
    """Read persisted package QA without making status a mutating operation."""
    soft_assets = 0
    hard_reasons: Counter[str] = Counter()
    if not directory.is_dir():
        return soft_assets, {}
    for metadata in directory.glob("*.json"):
        try:
            final = _read(metadata).get("visual", {}).get("final_vertical", {})
        except (OSError, json.JSONDecodeError):
            continue
        if directory.name == "assets" and final.get("soft_warnings"):
            soft_assets += 1
        if directory.name == "review":
            hard_reasons.update(str(reason) for reason in final.get("hard_failures", []) if reason)
    return soft_assets, dict(hard_reasons.most_common())


def _legacy_insufficient_packages(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    count = 0
    for metadata in directory.glob("*.json"):
        base = metadata.stem
        required = (metadata, directory / f"{base}.mp4", directory / f"v{base}.mp4",
                    directory / f"{base}.jpg", directory / f"v{base}.jpg")
        if not all(path.is_file() for path in required):
            continue
        try:
            final = _read(metadata).get("visual", {}).get("final_vertical", {})
        except (OSError, json.JSONDecodeError):
            continue
        if final.get("review_reason") == "legacy_evidence_insufficient":
            count += 1
    return count


def _semantic_failure_ids(run: Path, events: list[dict[str, Any]]) -> set[str]:
    """Count only persisted terminal semantic-validation failures."""
    failures: set[str] = set()
    workspace = _active_semantic_workspace(run)
    if workspace is not None:
        for event in events:
            event_id = event.get("visual_event_id")
            if not isinstance(event_id, str):
                continue
            try:
                result = _read(workspace / "events" / f"{event_id}.json")
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if result.get("state") == "TERMINAL_FAILURE" and result.get("terminal_failure") is True:
                failures.add(event_id)
        return failures
    directory = run / "semantic_failures"
    if not directory.is_dir():
        return failures
    for artifact in directory.glob("*.json"):
        try:
            data = _read(artifact)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        event_id = data.get("visual_event_id")
        if (isinstance(event_id, str) and event_id
                and data.get("failure_stage") == "semantic_validation"):
            failures.add(event_id)
    return failures


def _technical_shot_count(run: Path, fallback: int = 0) -> int:
    """Use the durable technical-shot manifest when it is available."""
    path = run / "technical_shots.json"
    if not path.is_file():
        return fallback
    try:
        shots = _read(path).get("shots")
    except (OSError, ValueError, json.JSONDecodeError):
        return fallback
    return len(shots) if isinstance(shots, list) else fallback


def _semantic_usage(run: Path) -> dict[str, Any]:
    """Aggregate persisted checkpoint usage; historical checkpoints may lack it."""
    totals = {"prompt_tokens": 0, "cached_tokens": 0, "response_tokens": 0,
              "thinking_tokens": 0, "total_tokens": 0}
    provenance: dict[str, int] = {}
    requests = 0
    cost = 0.0
    workspace = _active_semantic_workspace(run)
    directory = workspace / "checkpoints" if workspace is not None else run / "semantic_checkpoints"
    for checkpoint in directory.glob("*.json"):
        try:
            item = _read(checkpoint)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        provider = item.get("provider")
        model = item.get("model")
        if isinstance(provider, str) and provider:
            label = provider + (f"/{model}" if isinstance(model, str) and model else "")
            provenance[label] = provenance.get(label, 0) + 1
        raw = item.get("usage")
        # API usage/cost is OpenAI-only.  A Gemini checkpoint can still carry
        # usage, but it must never be presented as an OpenAI request or cost.
        if provider != "openai" or not isinstance(raw, dict):
            continue
        requests += 1
        for key in totals:
            value = raw.get(key)
            if isinstance(value, int):
                totals[key] += value
        cost += estimate_openai_cost(raw)
    # Batch usage belongs to the HTTP request, never each sibling event.
    from .visual_utility_resolution import ROOT
    for archive in (run / ROOT / 'requests').glob('*.json'):
        try:
            item = _read(archive)
        except (OSError, ValueError):
            continue
        if item.get('status') != 'RECEIVED':
            continue
        label = str(item.get('provider') or item.get('identity', {}).get('provider', 'unknown'))
        label += '/' + str(item.get('model') or item.get('identity', {}).get('model', 'unknown'))
        provenance[label] = provenance.get(label, 0) + 1
        requests += int(item.get('http_attempts') or 1)
        for key in totals:
            raw = item.get('usage', {}).get(key)
            if isinstance(raw, int):
                totals[key] += raw
        cost += float(item.get('cost_usd') or 0.0)
    return {"requests": requests, "usage": totals, "estimated_cost_usd": cost, "provenance": provenance}


def _inconsistent_batches(run: Path, batches: dict[str, Any], events: list[dict[str, Any]]) -> list[str]:
    """Read-only status signal for completion claims lacking terminal evidence."""
    by_id = {event.get("visual_event_id"): event for event in events if isinstance(event, dict)}
    failures = _semantic_failure_ids(run, events)
    inconsistent: list[str] = []
    for batch_id, batch in batches.items():
        if batch.get("status") != "COMPLETE" and batch.get("semantic_status") != "COMPLETE":
            continue
        for event_id in batch.get("event_ids", []):
            event = by_id.get(event_id, {})
            editorial = event.get("editorial", {}) if isinstance(event, dict) else {}
            workspace = _active_semantic_workspace(run)
            checkpoint = (workspace / "checkpoints" if workspace is not None else run / "semantic_checkpoints") / f"{event.get('candidate_id')}.json"
            valid = editorial.get("status") == "VALIDATED" and checkpoint.is_file()
            terminal_failure = editorial.get("status") == "SEMANTIC_INCOMPLETE" and event_id in failures
            if not valid and not terminal_failure:
                inconsistent.append(batch_id)
                break
    return inconsistent


def _source_ready(source: Path) -> bool:
    return (source / "source_manifest.json").is_file() and (source / "srt_cues.jsonl").is_file()


def ensure_source(input_dir: Path) -> bool:
    """Create source-v1 only when it is absent; valid source artifacts remain immutable."""
    source = _run_dir(input_dir) / "source-v1"
    if _source_ready(source):
        return True
    movie, srt = input_dir / "movie.mp4", input_dir / "subtitles.srt"
    if not movie.is_file():
        raise FileNotFoundError(f"canonical movie.mp4 does not exist: {movie}")
    if not srt.is_file():
        raise FileNotFoundError(f"canonical subtitles.srt does not exist: {srt}")
    metadata, parsed = inspect_movie(movie), parse_srt_file(srt)
    if not metadata.get("video", {}).get("width"):
        raise ValueError("movie has no readable video stream")
    if not parsed.cues:
        raise ValueError("SRT contains no usable cues")
    timeline = validate_timeline(parsed.cues, metadata.get("duration_seconds") or 0)
    if timeline["errors"]:
        raise ValueError(f"SRT timeline is invalid: {timeline['errors'][0]}")
    stats = cue_statistics(parsed.cues)
    manifest = {
        "schema_version": "source_manifest_v1",
        "source": {"movie_id": input_dir.name,
                   "movie": {**metadata, "sha256": sha256_file(movie)},
                   "srt": {"filename": srt.name, "absolute_path": str(srt.resolve()),
                           "sha256": sha256_file(srt), "literal_transcription": False,
                           "timing_assumption": "synchronized_external_srt",
                           "cue_count": stats["cue_count"], "first_cue_start_seconds": stats["first_cue_start"],
                           "last_cue_end_seconds": stats["last_cue_end"], "statistics": stats}},
        "validation": {"movie_readable": True, "srt_readable": True,
                       "srt_timeline_status": timeline["status"],
                       "warnings": timeline["warnings"] + parsed.malformed, "errors": timeline["errors"]},
    }
    write_json(source / "source_manifest.json", manifest)
    write_jsonl(source / "srt_cues.jsonl", [cue.as_dict() for cue in parsed.cues])
    return False


def _write_narrative_manifest(input_dir: Path, count: int) -> None:
    write_json(_run_dir(input_dir) / "narrative-v2" / "narrative_run.json", {
        "schema_version": "narrative_run_v2", "narrative_profile": "narrative_v3",
        "movie_id": input_dir.name, **EXTERNAL_PROVENANCE,
        "window_seconds": TARGET_WINDOW_SECONDS, "overlap_seconds": OVERLAP_SECONDS,
        "status": "PREPARED", "chunk_count": count,
    })


def _waiting_message(input_dir: Path, expected: set[str], missing: set[str], output: Callable[[str], None]) -> None:
    run = _run_dir(input_dir) / "narrative-v2"
    output(f"MOVIE: {input_dir.name}")
    output("STATUS: WAITING_EXTERNAL")
    output(f"NARRATIVE CHUNKS: {len(expected)}")
    output(f"CHUNKS DIRECTORY: {run / 'chunks'}")
    output(f"EXTERNAL-V3 INBOX: {run / 'external-v3-inbox'}")
    if missing:
        output("MISSING RESPONSES: " + ", ".join(sorted(missing)))
    output("Process every chunk with the External LLM and place its External V3 response in the inbox.")
    output(f"Then run again: movie-broll run {input_dir}")


def ensure_narrative(input_dir: Path, output: Callable[[str], None]) -> bool:
    """Perform the one intentional external-AI handoff, without guessing data."""
    narrative = _run_dir(input_dir) / "narrative-v2"
    final, reconciliation = narrative / "narrative_map.json", narrative / "reconciliation_report.json"
    manifest = _read(narrative / "narrative_run.json") if (narrative / "narrative_run.json").is_file() else {}
    external_complete = (manifest.get("status") == "COMPLETE"
                         and all(manifest.get(key) == value for key, value in EXTERNAL_PROVENANCE.items()))
    if final.is_file() and (external_complete or (reconciliation.is_file() and _read(reconciliation).get("status") == "PASS")):
        output("[movie-broll] narrative-v2: REUSED")
        return True

    mode = os.getenv("NARRATIVE_PROVIDER_MODE", "api")
    if mode not in {"api", "external"}:
        raise ValueError("NARRATIVE_PROVIDER_MODE must be api or external")
    if mode == "api":
        from .narrative_provider import OpenAINarrativeProvider
        from .narrative_runner import run_narrative
        from .narrative_consolidate import consolidate_narrative
        provider_name = os.getenv("NARRATIVE_PROVIDER", "openai")
        if provider_name != "openai":
            raise ValueError("NARRATIVE_PROVIDER must be openai; no automatic fallback")
        model = os.getenv("NARRATIVE_MODEL", os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
        try:
            provider = OpenAINarrativeProvider(model)
        except RuntimeError:
            state = {"status": "WAITING_PROVIDER", "provider": provider_name, "model": model}
            write_json(narrative / "narrative_provider_state.json", state)
            if manifest:
                write_json(narrative / "narrative_run.json", {**manifest, **state})
            output("STATUS: WAITING_PROVIDER (configure OPENAI_API_KEY)")
            return False
        result = run_narrative(input_dir, model=model, provider=provider, output=output)
        write_json(narrative / "narrative_provider_state.json", {"status": result["status"], "provider": provider_name, "model": model})
        if result["status"] != "COMPLETE":
            output("STATUS: " + result["status"])
            return False
        report = consolidate_narrative(input_dir)
        if report["status"] != "PASS":
            raise RuntimeError("narrative consolidation did not pass")
        return True

    chunks = sorted((narrative / "chunks").glob("NCHUNK_*.input.json"))
    if not chunks:
        chunks = prepare_narrative_inputs(_run_dir(input_dir) / "source-v1" / "srt_cues.jsonl", input_dir.name, narrative)
        _write_narrative_manifest(input_dir, len(chunks))
        _waiting_message(input_dir, {path.stem.removesuffix(".input") for path in chunks},
                         {path.stem.removesuffix(".input") for path in chunks}, output)
        return False

    if not (narrative / "narrative_run.json").is_file():
        _write_narrative_manifest(input_dir, len(chunks))
    expected = {path.name.removesuffix(".input.json") for path in chunks}
    inbox = narrative / "external-v3-inbox"
    received = {path.name.removesuffix(".external-v3.json") for path in inbox.glob("NCHUNK_*.external-v3.json")}
    missing, orphaned = expected - received, received - expected
    if missing:
        _waiting_message(input_dir, expected, missing, output)
        return False
    if orphaned:
        raise ValueError("external-v3 inbox contains responses without canonical chunks: " + ", ".join(sorted(orphaned)))
    result = finalize_external(input_dir, output=lambda line: output(f"[movie-broll] {line}"))
    if result.get("status") != "PASS":
        raise RuntimeError("external Narrative Mapper consolidation did not pass")
    manifest_path = narrative / "narrative_run.json"
    manifest = _read(manifest_path)
    manifest["status"] = "COMPLETE"
    write_json(manifest_path, manifest)
    output(f"[movie-broll] narrative-v2: COMPLETE ({result['segments']} segments)")
    return True


def _unresolved_review_ids(run: Path) -> list[str]:
    result = []
    for path in (run / "review").glob("*.json"):
        try:
            data = _read(path)
            if data.get("publication", {}).get("human_review", {}).get("status", "PENDING") != "REJECTED":
                result.append(str(data.get("asset", {}).get("id", path.stem)))
        except (OSError, ValueError, TypeError):
            result.append(path.stem)
    return sorted(result)


def read_status(input_dir: Path) -> dict[str, Any]:
    """Read canonical persisted state only.  This function must not write."""
    run = _run_dir(input_dir)
    summary = _read(run / "progress_summary.json") if (run / "progress_summary.json").is_file() else {}
    store_path = run / "visual_event_segments_v1.json"
    store = _read(store_path) if store_path.is_file() else {}
    events = store.get("events", []) if isinstance(store.get("events"), list) else []
    batches = store.get("batches", {}) if isinstance(store.get("batches"), dict) else {}
    if not summary:
        narrative = run / "narrative-v2"
        waiting = (narrative / "chunks").is_dir() and not (narrative / "narrative_map.json").is_file()
        provider_state = _read(narrative / "narrative_provider_state.json") if (narrative / "narrative_provider_state.json").is_file() else {}
        narrative_state = _read(narrative / "narrative_run.json") if (narrative / "narrative_run.json").is_file() else {}
        waiting_provider = provider_state.get("status") in {"WAITING_PROVIDER", "PARTIAL"} or narrative_state.get("status") == "PARTIAL"
        blocked_validation = "BLOCKED_NARRATIVE_VALIDATION" in {provider_state.get("status"), narrative_state.get("status")}
        summary = {"movie_id": input_dir.name,
                   "status": "BLOCKED_NARRATIVE_VALIDATION" if blocked_validation else "WAITING_PROVIDER" if waiting_provider else "WAITING_EXTERNAL" if waiting else "NOT_STARTED",
                   "stage": "narrative-v2" if waiting else "not_started"}
    validated = [event for event in events if event.get("editorial", {}).get("status") == "VALIDATED"]
    editorial_counts = {decision: sum(event.get("editorial", {}).get("decision") == decision for event in validated)
                        for decision in ("KEEP", "REJECT", "REVIEW")}
    provisional = sum(event.get("editorial", {}).get("status") == "PROVISIONAL" for event in events)
    failure_ids = _semantic_failure_ids(run, events)
    semantic_incomplete = sum(
        event.get("editorial", {}).get("status") == "SEMANTIC_INCOMPLETE"
        and event.get("visual_event_id") in failure_ids for event in events
    )
    provider_deferred = sum(event.get("editorial", {}).get("status") == "PROVIDER_DEFERRED" for event in events)
    inconsistent_batches = _inconsistent_batches(run, batches, events)
    current_batch = next((batch_id for batch_id, batch in batches.items() if batch.get("status") != "COMPLETE"), None)
    current_event = None
    if current_batch and batches[current_batch].get("event_ids"):
        current_event = batches[current_batch]["event_ids"][0]
    contract_path = run / "canonical_semantic_contract.json"
    contract = _read(contract_path) if contract_path.is_file() else {}
    active_contract = store.get("canonical_semantic_contract", contract.get("active_contract", "V8"))
    provider = os.environ.get("SEMANTIC_PROVIDER", "openai").lower()
    model = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL) if provider == "openai" else None
    # The canonical V9 manifest is provenance, whereas environment values are
    # merely the configuration an operator would use for a future call.
    if active_contract == "V9":
        labels = contract.get("provider_models", [])
        if isinstance(labels, list) and len(labels) == 1 and isinstance(labels[0], str):
            provider, _, model = labels[0].partition("/")
    soft_warning_assets, hard_review_reasons = _finalization_qa(run / "assets")
    _, review_hard_reasons = _finalization_qa(run / "review")
    # Observation/cache status is intentionally read-only.  It does not infer
    # policy from legacy semantic decisions and never constructs a provider.
    try:
        from .semantic_observations import observation_status
        observations = observation_status(input_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        observations = {"total": len(events), "cached": 0, "missing": len(events),
                        "provider_provenance": {}, "policy_version": "broll_policy_v1",
                        "policy": {"KEEP": 0, "REJECT": 0, "REVIEW": 0},
                        "observation_requests": 0, "reused_observations": 0,
                        "usage": {"prompt_tokens": 0, "cached_tokens": 0, "response_tokens": 0,
                                  "thinking_tokens": 0, "total_tokens": 0}, "cost_usd": 0.0}
    utility_pipeline = {}
    if store.get('policy_version') == 'broll_policy_v2':
        from .visual_utility_resolution import SCHEMA
        active_contract = SCHEMA
        utility_report = _read(run/'visual_utility_resolution.json') if (run/'visual_utility_resolution.json').is_file() else {}
        states = utility_report.get('event_states', {})
        utility_pipeline = dict(Counter(states.values()))
        usage = _semantic_usage(run)
        completed = sum(v in {'VALID', 'VALID_INCONCLUSIVE', 'COMPATIBLE'} for v in states.values())
        observations = {'total':len(events),'cached':completed,'missing':len(events)-completed,
            'policy_version':'broll_policy_v2','policy':editorial_counts,
            'provider_provenance':usage['provenance'],'observation_requests':usage['requests'],
            'reused_observations':utility_report.get('reused',0),'usage':usage['usage'],
            'cost_usd':usage['estimated_cost_usd']}
    return {"movie_id": input_dir.name,
            "status": store.get("production_status", store.get("status", summary.get("status", "NOT_STARTED"))),
            "stage": summary.get("stage", "production" if events else "not_started"),
            "technical_shots": _technical_shot_count(run, summary.get("technical_shots", 0)),
            "visual_events": len(events) if store_path.is_file() else summary.get("visual_events", 0),
            "production_batches_complete": sum(x.get("status") == "COMPLETE" for x in batches.values()),
            "production_batches_total": len(batches),
            "validated": len(validated), "provisional": provisional, "semantic_incomplete": semantic_incomplete,
            "provider_deferred": provider_deferred, "inconsistent_batches": inconsistent_batches,
            "keep": editorial_counts["KEEP"], "reject": editorial_counts["REJECT"], "review": editorial_counts["REVIEW"],
            "semantic_failures": len(failure_ids),
            "assets": _complete_packages(run / "assets"), "review_packages": _complete_packages(run / "review"),
            "unresolved_review_asset_ids": _unresolved_review_ids(run),
            "soft_warning_assets": soft_warning_assets,
            "hard_review_reasons": review_hard_reasons or hard_review_reasons,
            "legacy_review_insufficient": _legacy_insufficient_packages(run / "review"),
            "current_batch": current_batch, "current_event": current_event,
            "resume_safe": bool(summary.get("resume_safe", True)),
            "active_semantic_contract": active_contract,
            "semantic_provider": provider, "semantic_model": model,
            "api_usage": _semantic_usage(run), "semantic_observations": observations,
            "visual_utility_pipeline_states": utility_pipeline,
            "source_ready": (run / "source-v1" / "source_manifest.json").is_file(),
            "narrative_ready": (run / "narrative-v2" / "narrative_map.json").is_file()}


def print_status(status: dict[str, Any], output: Callable[[str], None] = print) -> None:
    output(f"MOVIE: {status['movie_id']}")
    output(f"STATUS: {status['status']}")
    output(f"STAGE: {status['stage']}")
    output(f"SOURCE: {'READY' if status.get('source_ready') else 'PENDING'}")
    if status.get("unresolved_review_asset_ids"):
        output("HUMAN REVIEW REQUIRED: " + ", ".join(status["unresolved_review_asset_ids"]))
    output(f"NARRATIVE: {'READY' if status.get('narrative_ready') else 'PENDING'}")
    output(f"TECHNICAL SHOTS: {status['technical_shots']}")
    output(f"VISUAL EVENTS: {status['visual_events']}")
    output(f"PRODUCTION BATCHES: {status['production_batches_complete']}/{status['production_batches_total']}")
    output(f"VALIDATED: {status['validated']}")
    output(f"ACTIVE SEMANTIC CONTRACT: {status.get('active_semantic_contract', 'V8')}")
    output(f"SEMANTIC KEEP: {status['keep']}  SEMANTIC REJECT: {status['reject']}  SEMANTIC REVIEW: {status['review']}")
    output(f"PENDING/PROVISIONAL: {status['provisional']}")
    output(f"SEMANTIC INCOMPLETE: {status['semantic_incomplete']}")
    if status.get("visual_utility_pipeline_states"):
        output("VISUAL UTILITY PIPELINE: " + ", ".join(f"{k}: {v}" for k,v in sorted(status["visual_utility_pipeline_states"].items())))
    output(f"SEMANTIC FAILURES: {status['semantic_failures']}")
    output(f"ACTIVE SEMANTIC PROVIDER: {status['semantic_provider']}")
    if status["semantic_model"]:
        output(f"MODEL: {status['semantic_model']}")
    observations = status.get("semantic_observations", {})
    output("SEMANTIC OBSERVATIONS: "
           f"total: {observations.get('total', 0)}  cached: {observations.get('cached', 0)}  "
           f"missing: {observations.get('missing', 0)}")
    if observations.get("provider_provenance"):
        output("OBSERVATION PROVENANCE: " + ", ".join(
            f"{name}: {count}" for name, count in sorted(observations["provider_provenance"].items())))
    policy = observations.get("policy", {})
    output(f"POLICY: {observations.get('policy_version', 'broll_policy_v1')}  "
           f"KEEP: {policy.get('KEEP', 0)}  REJECT: {policy.get('REJECT', 0)}  REVIEW: {policy.get('REVIEW', 0)}")
    observation_usage = observations.get("usage", {})
    output("OBSERVATION API: "
           f"requests: {observations.get('observation_requests', 0)}  "
           f"reused: {observations.get('reused_observations', 0)}  "
           f"input tokens: {observation_usage.get('prompt_tokens', 0)}  "
           f"cached input: {observation_usage.get('cached_tokens', 0)}  "
           f"output tokens: {observation_usage.get('response_tokens', 0)}  "
           f"cost: ${float(observations.get('cost_usd', 0.0)):.4f}")
    usage = status["api_usage"]
    if usage["provenance"]:
        output("SEMANTIC PROVENANCE: " + ", ".join(
            f"{provider}: {count} validated" for provider, count in sorted(usage["provenance"].items())
        ))
    if usage["requests"]:
        tokens = usage["usage"]
        output(f"OPENAI API USAGE: requests: {usage['requests']}")
        output(f"INPUT TOKENS: {tokens['prompt_tokens']:,}")
        output(f"CACHED INPUT: {tokens['cached_tokens']:,}")
        output(f"OUTPUT TOKENS: {tokens['response_tokens']:,}")
        output(f"SEMANTIC API COST: ${usage['estimated_cost_usd']:.4f}")
    output(f"AUTO-PUBLISHED ASSETS: {status['assets']}")
    output(f"HUMAN REVIEW REQUIRED: {status['review_packages']}")
    if status.get("unresolved_review_asset_ids"):
        output("REVIEW ASSET IDS: " + ", ".join(status["unresolved_review_asset_ids"]))
    output(f"SOFT-WARNING ASSETS: {status['soft_warning_assets']}")
    if status['legacy_review_insufficient']:
        output(f"LEGACY REVIEW INSUFFICIENT EVIDENCE: {status['legacy_review_insufficient']}")
    if status['hard_review_reasons']:
        output("TOP HARD REVIEW REASONS: " + ", ".join(f"{reason}: {count}" for reason, count in status['hard_review_reasons'].items()))
    if status["provider_deferred"]:
        output(f"PROVIDER DEFERRED: {status['provider_deferred']}")
    if status["inconsistent_batches"]:
        output("INVALID/INCONSISTENT BATCH: " + ", ".join(status["inconsistent_batches"]))
    if status["current_batch"]:
        output(f"CURRENT: {status['current_batch']} / {status['current_event'] or 'pending'}")


def print_final_summary(input_dir: Path, output: Callable[[str], None] = print) -> None:
    status = read_status(input_dir)
    output(f"MOVIE: {status['movie_id']}")
    output(f"STATUS: {status['status']}")
    output(f"VISUAL EVENTS: {status['visual_events']}")
    output(f"VALIDATED: {status['visual_events'] - status['semantic_failures']}")
    output(f"SEMANTIC KEEP: {status['keep']}")
    output(f"SEMANTIC REJECT: {status['reject']}")
    output(f"SEMANTIC FAILURES: {status['semantic_failures']}")
    output(f"AUTO-PUBLISHED ASSETS: {status['assets']}")
    output(f"HUMAN REVIEW REQUIRED: {status['review_packages']}")
    if status.get("unresolved_review_asset_ids"):
        output("REVIEW ASSET IDS: " + ", ".join(status["unresolved_review_asset_ids"]))
    output(f"SOFT-WARNING ASSETS: {status['soft_warning_assets']}")
    if status['legacy_review_insufficient']:
        output(f"LEGACY REVIEW INSUFFICIENT EVIDENCE: {status['legacy_review_insufficient']}")
    output(f"RESUME SAFE: {'YES' if status['resume_safe'] else 'NO'}")
    usage = status["api_usage"]
    if usage["requests"]:
        tokens = usage["usage"]
        output("API USAGE")
        output(f"provider: {status['semantic_provider']}")
        output(f"model: {status['semantic_model'] or 'mixed'}")
        output(f"requests: {usage['requests']}")
        output(f"input_tokens: {tokens['prompt_tokens']}")
        output(f"cached_input_tokens: {tokens['cached_tokens']}")
        output(f"output_tokens: {tokens['response_tokens']}")
        output(f"estimated_cost_usd: {usage['estimated_cost_usd']:.6f}")


def run(input_dir: Path, output: Callable[[str], None] = print,
        supervisor_factory: Callable[..., Supervisor] = Supervisor) -> int:
    """Run or resume production after acquiring the existing supervisor lock."""
    input_dir = input_dir.resolve()
    reused = read_status(input_dir)

    def before_start() -> None:
        source_reused = ensure_source(input_dir)
        output(f"[movie-broll] movie: {input_dir.name}")
        output(f"[movie-broll] source-v1: {'REUSED' if source_reused else 'COMPLETE'}")
        if not ensure_narrative(input_dir, output):
            raise WaitingExternal()
        # Supervisor owns the title lock and has already recovered stale state.
        # Exclude this process so preflight cannot mistake that ownership for a
        # separate production owner.
        report = preflight(input_dir, exclude_owner_pid=os.getpid())
        if not report["ready"]:
            raise RuntimeError("preflight blocked: " + "; ".join(report["blockers"]))
        output("[movie-broll] preflight: READY FOR PRODUCTION")
        if reused["technical_shots"] or reused["production_batches_complete"] or reused["assets"]:
            output("[movie-broll] resume: reusing compatible persisted work")
            if reused["current_batch"]:
                output(f"[movie-broll] resume current: {reused['current_batch']} / {reused['current_event'] or 'pending'}")

    supervisor = supervisor_factory(input_dir, before_start=before_start, stream_child_output=True)
    try:
        code = supervisor.run()
    except WaitingExternal:
        return 0
    if code == 0:
        print_final_summary(input_dir, output)
    return code
