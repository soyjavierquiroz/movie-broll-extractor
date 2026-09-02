"""Automatic, resumable Gemini narrative mapping with a constrained V2 boundary."""
from __future__ import annotations

import json, os, random, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from .inspect_source import inspect_movie
from .narrative import OVERLAP_SECONDS, TARGET_WINDOW_SECONDS, normalize_llm_v2_response, prepare_narrative_inputs, validate_narrative_map
from .narrative_provider import GeminiNarrativeProvider, NarrativeProvider, ProviderResponse
from .srt import parse_srt_file
from .utils import sha256_file, sha256_text, write_json, write_jsonl

PROMPT_VERSION = "srt_narrative_mapper_v2"
DEFAULT_MODEL = "gemini-2.5-flash"
MAX_SEMANTIC_ATTEMPTS = 2
MAX_TRANSIENT_RETRIES = 2
FREE_TIER_REQUEST_BUDGET = 18

def _utc() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
def _usage() -> dict[str, int | None]: return {key: None for key in ("prompt_tokens", "response_tokens", "thinking_tokens", "cached_tokens", "total_tokens")}
def _add_usage(total: dict[str, int | None], usage: dict[str, int | None]) -> None:
    for key in total:
        if usage.get(key) is not None: total[key] = (total[key] or 0) + usage[key]

def _ensure_source(movie: Path, srt: Path, root: Path, movie_id: str) -> Path:
    source_dir = root / "source-v1"; manifest_path = source_dir / "source_manifest.json"; cues_path = source_dir / "srt_cues.jsonl"
    movie_hash, srt_hash = sha256_file(movie), sha256_file(srt)
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8")); source = existing["source"]
        if source["movie"]["sha256"] == movie_hash and source["srt"]["sha256"] == srt_hash and cues_path.is_file(): return cues_path
    except (OSError, KeyError, TypeError, json.JSONDecodeError): pass
    metadata = inspect_movie(movie); parsed = parse_srt_file(srt)
    write_json(manifest_path, {"schema_version": "source_manifest_v1", "source": {"movie_id": movie_id, "movie": {**metadata, "sha256": movie_hash}, "srt": {"filename": srt.name, "sha256": srt_hash, "literal_transcription": False, "cue_count": len(parsed.cues)}}})
    write_jsonl(cues_path, [cue.as_dict() for cue in parsed.cues]); return cues_path

def _checkpoint_valid(input_path: Path, map_path: Path, meta_path: Path, expected: dict[str, str]) -> bool:
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return all(metadata.get(key) == value for key, value in expected.items()) and not validate_narrative_map(input_path, map_path)
    except (OSError, TypeError, json.JSONDecodeError): return False

def _status(error: Exception) -> int | None:
    value = getattr(error, "status_code", None) or getattr(error, "code", None)
    return value if isinstance(value, int) else None

def _daily_quota(error: Exception) -> bool:
    return _status(error) == 429 and "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in str(error)

def _transient(error: Exception) -> bool:
    return _status(error) in (429, 500, 502, 503, 504) or isinstance(error, (TimeoutError, ConnectionError, OSError))

def run_narrative(input_dir: Path, model: str = DEFAULT_MODEL, force: bool = False, max_chunks: int | None = None, provider: NarrativeProvider | None = None, sleep: Callable[[float], None] = time.sleep, output: Callable[[str], None] = print) -> dict[str, Any]:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if provider is None and not api_key: raise RuntimeError("GEMINI_API_KEY is not configured")
    movie_id = input_dir.name; movie = input_dir / "movie.mp4"; srt = input_dir / "subtitles.srt"
    if not movie.is_file() or not srt.is_file(): raise ValueError("input directory must contain movie.mp4 and subtitles.srt")
    root = Path("runs") / movie_id; run_dir = root / "narrative-v2"; chunks_dir = run_dir / "chunks"; maps_dir = run_dir / "maps"; responses_dir = run_dir / "responses"
    prompt_path = Path(__file__).resolve().parents[2] / "config" / "prompts" / f"{PROMPT_VERSION}.md"; prompt = prompt_path.read_text(encoding="utf-8"); prompt_hash = sha256_text(prompt)
    cues_path = _ensure_source(movie, srt, root, movie_id)
    prepare_narrative_inputs(cues_path, movie_id, chunks_dir, TARGET_WINDOW_SECONDS, OVERLAP_SECONDS, force=True)
    inputs = sorted(chunks_dir.glob("NCHUNK_*.input.json")); inputs = inputs if max_chunks is None else inputs[:max_chunks]
    active_provider = provider or GeminiNarrativeProvider(api_key, model)
    pricing_mode = os.getenv("GEMINI_PRICING_MODE", "unknown")
    budget = FREE_TIER_REQUEST_BUDGET if model == DEFAULT_MODEL and pricing_mode == "free_tier" else None
    started = _utc(); usage = _usage(); completed = reused = requests = retries = 0; failures: list[dict[str, str]] = []; quota_status = "NOT_EXHAUSTED"; stopped_status: str | None = None
    output(f"[narrative] movie: {movie_id}"); output(f"[narrative] provider: {active_provider.identifier}"); output(f"[narrative] model: {model}"); output(f"[narrative] chunks: {len(inputs)}")
    for position, input_path in enumerate(inputs, 1):
        if stopped_status: break
        chunk_id = input_path.name.removesuffix(".input.json"); map_path = maps_dir / f"{chunk_id}.narrative_map.json"; meta_path = maps_dir / f"{chunk_id}.checkpoint.json"
        expected = {"input_sha256": sha256_file(input_path), "model": model, "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_hash, "provider": active_provider.identifier}
        if not force and map_path.is_file() and _checkpoint_valid(input_path, map_path, meta_path, expected):
            reused += 1; completed += 1; output(f"[narrative] {position:02d}/{len(inputs):02d} REUSED"); continue
        chunk_input = json.loads(input_path.read_text(encoding="utf-8")); semantic_attempts = transient_retries = 0; last_error = "unknown failure"
        while True:
            if budget is not None and requests >= budget:
                stopped_status = "REQUEST_BUDGET_EXHAUSTED"; last_error = "request budget exhausted"; failures.append({"chunk_id": chunk_id, "error": last_error}); break
            try:
                output(f"[narrative] {position:02d}/{len(inputs):02d} CALL"); requests += 1
                response: ProviderResponse = active_provider.generate(prompt, chunk_input); _add_usage(usage, response.usage)
                semantic_attempts += 1
                raw_path = responses_dir / f"{chunk_id}.attempt-{semantic_attempts}.llm-v2.json"; write_json(raw_path, response.data)
                canonical = normalize_llm_v2_response(chunk_input, response.data)
                candidate = responses_dir / f"{chunk_id}.attempt-{semantic_attempts}.canonical.json"; write_json(candidate, canonical)
                errors = validate_narrative_map(input_path, candidate)
                if errors: raise ValueError("normalization validation: " + "; ".join(errors[:3]))
                write_json(map_path, canonical); write_json(meta_path, {**expected, "usage": response.usage, "validated_at": _utc()})
                completed += 1; output(f"[narrative] {position:02d}/{len(inputs):02d} VALID segments={len(canonical['segments'])}"); break
            except Exception as error:
                last_error = str(error)
                if _daily_quota(error):
                    quota_status = "DAILY_QUOTA_EXHAUSTED"; stopped_status = quota_status; failures.append({"chunk_id": chunk_id, "error": last_error}); output(f"[narrative] {position:02d}/{len(inputs):02d} DAILY_QUOTA_EXHAUSTED"); break
                semantic_failure = isinstance(error, ValueError) and last_error.startswith("validation:")
                if semantic_failure and semantic_attempts < MAX_SEMANTIC_ATTEMPTS:
                    retries += 1; output(f"[narrative] {position:02d}/{len(inputs):02d} SEMANTIC_RETRY"); continue
                if _transient(error) and transient_retries < MAX_TRANSIENT_RETRIES:
                    transient_retries += 1; retries += 1; sleep(min(20.0, 2.0 * (2 ** (transient_retries - 1))) + random.uniform(0, .25)); output(f"[narrative] {position:02d}/{len(inputs):02d} TRANSIENT_RETRY"); continue
                failures.append({"chunk_id": chunk_id, "error": last_error}); output(f"[narrative] {position:02d}/{len(inputs):02d} FAILED"); break
    status = stopped_status or ("COMPLETE" if not failures else "PARTIAL")
    manifest = {"schema_version": "narrative_run_v2", "narrative_profile": "narrative_v2", "movie_id": movie_id, "provider": active_provider.identifier, "model": model, "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_hash, "window_seconds": TARGET_WINDOW_SECONDS, "overlap_seconds": OVERLAP_SECONDS, "started_at": started, "completed_at": _utc(), "status": status, "chunk_count": len(inputs), "valid_chunks": completed, "completed_chunks": completed, "reused_chunks": reused, "failed_chunks": [item["chunk_id"] for item in failures], "request_budget": budget, "requests": requests, "retries": retries, "quota_status": quota_status, "usage": usage, "pricing_mode": pricing_mode, "errors": failures}
    manifest["api_cost_estimate_usd"] = "0.00" if pricing_mode == "free_tier" else "unknown"; write_json(run_dir / "narrative_run.json", manifest)
    output(f"[narrative] requests={requests} retries={retries}"); output(f"[narrative] status={status}"); return manifest
