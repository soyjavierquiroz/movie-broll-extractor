"""Fast, read-only readiness checks before a title enters production."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .gemini_credentials import GeminiCredentialSource
from .broll_semantics import DEFAULT_OPENAI_MODEL
from .narrative_finalize import EXTERNAL_PROVENANCE
from .production_ownership import active_title_owner, running_state


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("root must be a JSON object")
    return value


def _readable(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            handle.read(1)
        return True
    except OSError:
        return False


def legacy_root_chunk_comparison(run: Path) -> list[dict[str, str]]:
    """Diagnose legacy root inputs only; this never moves or removes them."""
    result = []
    for legacy in sorted(run.glob("NCHUNK_*.input.json")):
        canonical = run / "chunks" / legacy.name
        status = "MISSING_CANONICAL"
        if canonical.is_file():
            if legacy.read_bytes() == canonical.read_bytes():
                status = "BYTE_IDENTICAL"
            else:
                try:
                    status = "CONTENT_IDENTICAL" if _json(legacy) == _json(canonical) else "DIFFERENT"
                except (OSError, ValueError, json.JSONDecodeError):
                    status = "DIFFERENT"
        result.append({"chunk": legacy.name.removesuffix(".input.json"), "status": status})
    return result


def _narrative_provenance(run: Path, manifest: dict[str, Any], analysis: Any) -> list[str]:
    """Validate persisted producer identity, independently of today's transport settings."""
    from .narrative_runner import PROMPT_VERSION, _checkpoint_valid, _narrative_content_inputs
    from .utils import sha256_text

    errors: list[str] = []
    if not isinstance(analysis, dict):
        return ["Narrative Map provenance mismatch: analysis must be an object"]
    external = manifest.get("provider") == EXTERNAL_PROVENANCE["provider"]
    for key in ("provider", "model", "prompt_version"):
        value = manifest.get(key)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            errors.append(f"narrative run provenance mismatch: {key} must be a nonempty string")
        if analysis.get(key) != value or not isinstance(analysis.get(key), str):
            errors.append(f"Narrative Map provenance mismatch: analysis.{key} must match narrative run")
        expected = EXTERNAL_PROVENANCE[key] if external else (PROMPT_VERSION if key == "prompt_version" else None)
        if expected is not None and value != expected:
            errors.append(f"narrative run provenance mismatch: {key} must be {expected}")
    if external or errors:
        return errors

    # API identity must be supported by every validated chunk checkpoint;
    # agreeing manifest/map labels alone are not evidence of a producer.
    prompt = Path(__file__).resolve().parents[2] / "config" / "prompts" / f"{PROMPT_VERSION}.md"
    prompt_hash = sha256_text(prompt.read_text(encoding="utf-8"))
    for input_path in sorted((run / "chunks").glob("NCHUNK_*.input.json")):
        chunk_id = input_path.name.removesuffix(".input.json")
        checkpoint = run / "maps" / f"{chunk_id}.checkpoint.json"
        try:
            producer = _json(checkpoint)
            for key in ("provider", "model", "prompt_version"):
                if producer.get(key) != manifest.get(key):
                    errors.append(f"narrative chunk provenance mismatch: {chunk_id}.{key} must match narrative run")
            expected = _narrative_content_inputs(input_path, manifest["model"], prompt_hash)
            if not _checkpoint_valid(input_path, run / "maps" / f"{chunk_id}.narrative_map.json", checkpoint, expected):
                errors.append(f"narrative chunk provenance mismatch: {chunk_id} checkpoint is invalid")
        except (OSError, ValueError, TypeError, KeyError):
            errors.append(f"narrative chunk provenance mismatch: {chunk_id} checkpoint is missing or malformed")
    return errors


def preflight(input_dir: Path, *, exclude_owner_pid: int | None = None) -> dict[str, Any]:
    movie_id = input_dir.name
    root = input_dir.resolve().parents[1] / "runs" / movie_id
    source = root / "source-v1"
    narrative = root / "narrative-v2"
    blockers: list[str] = []
    warnings: list[str] = []
    movie, srt = input_dir / "movie.mp4", input_dir / "subtitles.srt"
    source_manifest, cues = source / "source_manifest.json", source / "srt_cues.jsonl"
    run_manifest, final_map = narrative / "narrative_run.json", narrative / "narrative_map.json"
    checks = {
        "movie": movie.is_file() and _readable(movie),
        "srt": srt.is_file() and _readable(srt),
        "source_manifest": source_manifest.is_file(),
        "canonical_cues": cues.is_file() and _readable(cues),
    }
    for name, passed in checks.items():
        if not passed:
            blockers.append(f"source {name} is missing or unreadable")
    source_data: dict[str, Any] = {}
    if checks["source_manifest"]:
        try:
            source_data = _json(source_manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            checks["source_manifest"] = False
            blockers.append("source source_manifest cannot be parsed")

    narrative_data: dict[str, Any] = {}
    run_data: dict[str, Any] = {}
    if not run_manifest.is_file():
        blockers.append("narrative_run.json is missing")
    else:
        try:
            run_data = _json(run_manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            blockers.append("narrative_run.json cannot be parsed")
    if not final_map.is_file():
        blockers.append("narrative_map.json is missing")
    else:
        try:
            narrative_data = _json(final_map)
        except (OSError, ValueError, json.JSONDecodeError):
            blockers.append("narrative_map.json cannot be parsed")
    segments = narrative_data.get("segments") if narrative_data else []
    if narrative_data and (not isinstance(segments, list) or not segments):
        blockers.append("Narrative Map has no segments")
    analysis = narrative_data.get("analysis", {}) if narrative_data else {}
    if narrative_data:
        blockers.extend(_narrative_provenance(narrative, run_data, analysis))
    if not isinstance(analysis, dict):
        analysis = {}

    chunk_ids = {path.name.removesuffix(".input.json") for path in (narrative / "chunks").glob("NCHUNK_*.input.json")}
    map_ids = {path.name.removesuffix(".narrative_map.json") for path in (narrative / "maps").glob("NCHUNK_*.narrative_map.json")}
    if not chunk_ids:
        blockers.append("no canonical narrative chunks found")
    if chunk_ids != map_ids:
        blockers.append("canonical narrative chunks and imported maps do not correspond")

    # Active-picture detection needs only a readable source movie; preflight
    # intentionally does not decode frames or write its normal cache.
    active_picture_ready = checks["movie"] and isinstance(source_data.get("source", {}).get("movie"), dict)
    if not active_picture_ready:
        blockers.append("active picture prerequisites cannot be resolved")
    env_file = Path(__file__).resolve().parents[2] / ".env"
    credential_count = len(GeminiCredentialSource(env_file).discover().all_credentials())
    configured_env = os.environ
    semantic_provider = str(configured_env.get("SEMANTIC_PROVIDER", "openai")).strip().lower()
    semantic_model: str | None = None
    credentials = "FAIL"
    if semantic_provider == "openai":
        semantic_model = str(configured_env.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
        if configured_env.get("OPENAI_API_KEY"):
            credentials = "PASS"
        else:
            blockers.append("OPENAI_API_KEY is not configured")
    elif semantic_provider == "gemini":
        semantic_model = str(configured_env.get("GEMINI_MODEL", "gemini-3.6-flash"))
        if credential_count:
            credentials = "PASS"
        else:
            blockers.append("Gemini credentials are not configured")
    else:
        blockers.append("SEMANTIC_PROVIDER must be openai or gemini")
    disk_base = root if root.exists() else input_dir
    free_gb = shutil.disk_usage(disk_base).free / (1024 ** 3)
    state_path = root / "supervisor_state.json"
    state: dict[str, Any] = {}
    if state_path.is_file():
        try:
            state = _json(state_path)
        except (OSError, ValueError, json.JSONDecodeError):
            warnings.append("supervisor state cannot be parsed")
    owner = active_title_owner(input_dir, state, exclude_pid=exclude_owner_pid)
    active = owner is not None
    if active:
        blockers.append("an active supervisor appears to own this title")
    stale_runtime_state = False
    if state_path.is_file() and not active:
        stale_runtime_state = running_state(state)
    progress_path = root / "progress_summary.json"
    if progress_path.is_file() and not active:
        try:
            summary = _json(progress_path)
            stale_runtime_state = stale_runtime_state or running_state(summary)
        except (OSError, ValueError, json.JSONDecodeError):
            warnings.append("production progress summary cannot be parsed")
    if stale_runtime_state:
        warnings.append("recoverable stale runtime state is present; supervise will mark it INTERRUPTED after acquiring its lock")
    legacy = legacy_root_chunk_comparison(narrative)
    return {
        "movie_id": movie_id, "source": checks, "narrative": {"provider": analysis.get("provider"),
        "chunks": len(chunk_ids), "maps": len(map_ids), "segments": len(segments) if isinstance(segments, list) else 0,
        "status": "PASS" if not any(reason.startswith("narrative") or reason.startswith("Narrative") or "canonical narrative" in reason or "no canonical" in reason for reason in blockers) else "FAIL"},
        "runtime": {"openai_api_key": "SET" if configured_env.get("OPENAI_API_KEY") else "MISSING",
                    "gemini_credentials": credential_count, "semantic_provider": semantic_provider,
                    "semantic_model": semantic_model, "semantic_credentials": credentials,
                    "disk_free_gb": round(free_gb, 1),
                    "active_run_conflict": active, "active_owner": owner, "active_picture_prerequisites": active_picture_ready,
                    "recoverable_stale_state": stale_runtime_state},
        "legacy_root_chunks": legacy, "warnings": warnings, "blockers": blockers,
        "ready": not blockers,
    }


def print_preflight(report: dict[str, Any], output: callable = print) -> None:
    output("=== MOVIE B-ROLL PREFLIGHT ===")
    output(f"movie_id: {report['movie_id']}")
    output("\nsource:")
    for key in ("movie", "srt", "source_manifest", "canonical_cues"):
        output(f"  {key}: {'PASS' if report['source'][key] else 'FAIL'}")
    narrative = report["narrative"]
    output("\nnarrative:")
    output(f"  provider: {narrative['provider'] or 'unknown'}")
    for key in ("chunks", "maps", "segments"):
        output(f"  {key}: {narrative[key]}")
    output(f"  status: {narrative['status']}")
    runtime = report["runtime"]
    output("\nruntime:")
    output(f"  OPENAI_API_KEY: {runtime.get('openai_api_key', 'MISSING')}")
    output(f"  semantic_provider: {runtime['semantic_provider']}")
    output(f"  model: {runtime['semantic_model'] or 'unknown'}")
    output(f"  credentials: {runtime['semantic_credentials']}")
    output(f"  active_picture_prerequisites: {'PASS' if runtime['active_picture_prerequisites'] else 'FAIL'}")
    output(f"  gemini_credentials: {runtime['gemini_credentials']}")
    output(f"  disk_free_gb: {runtime['disk_free_gb']:.1f}")
    output(f"  active_run_conflict: {'YES' if runtime['active_run_conflict'] else 'NO'}")
    output(f"  recoverable_stale_state: {'YES' if runtime['recoverable_stale_state'] else 'NO'}")
    if report["legacy_root_chunks"]:
        output("\nlegacy_root_chunks:")
        for item in report["legacy_root_chunks"]:
            output(f"  {item['chunk']}: {item['status']}")
    if report["blockers"]:
        output("\nblocking_reasons:")
        for reason in report["blockers"]:
            output(f"  - {reason}")
    if report["warnings"]:
        output("\nwarnings:")
        for warning in report["warnings"]:
            output(f"  - {warning}")
    output("\nRESULT: " + ("READY FOR PRODUCTION" if report["ready"] else "BLOCKED"))
