"""Automatic, resumable Gemini narrative mapping with a constrained V3 boundary."""
from __future__ import annotations

import json, os, random, re, shutil, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


from .gemini_credentials import GeminiCredential, GeminiCredentialSource, gemini_secret_values
from .inspect_source import inspect_movie
from .narrative import (OVERLAP_SECONDS, PRODUCTION_NARRATIVE_PROFILE,
                        TARGET_WINDOW_SECONDS, normalize_llm_v3_response,
                        prepare_narrative_inputs, validate_narrative_map, validate_llm_v3_response)
from .narrative_provider import GeminiNarrativeProvider, NarrativeProvider, ProviderResponse
from .srt import parse_srt_file
from .utils import sha256_file, sha256_text, write_json, write_jsonl

PROMPT_VERSION = "srt_narrative_mapper_v3"
DEFAULT_MODEL = "gemini-3.6-flash"
MAX_SEMANTIC_ATTEMPTS = 2
MAX_TRANSIENT_RETRIES = 2
FREE_TIER_REQUEST_BUDGET = 18
CONTENT_FINGERPRINT_VERSION = "narrative_content_v1"
# Retry recovery is deliberately separate from the semantic checkpoint identity.
# Bump for a reviewed validator/canonicalizer or corrective-feedback fix that
# needs a fresh bounded retry opportunity. Do not bump for logging/runtime edits.
# Prompt/schema revisions already change the semantic content fingerprint.
# v2 reopens exhausted v1/unversioned validation failures once because retries
# now carry actionable semantic diagnostics (segment, cue range, duration) and
# require splitting distinct situations or justifying a continuous interaction.
# The semantic contract/cache identity is unchanged; the durable guard still
# limits each qualifying chunk to MAX_SEMANTIC_ATTEMPTS for this revision.
VALIDATION_RECOVERY_REVISION = "narrative_validation_recovery_v2"
_LEGACY_RECOVERY_REVISION = "narrative_validation_recovery_v1"

class V3ValidationError(ValueError):
    def __init__(self, stage: str, errors: list[dict[str, str]]) -> None:
        self.validation = {"status": "FAILED", "stage": stage, "errors": errors}
        super().__init__("; ".join(item["message"] for item in errors))


def _contract_errors(messages: list[str]) -> list[dict[str, str]]:
    result = []
    for message in messages:
        match = re.match(r"segment (\d+) (\w+)", message)
        path = f"segments[{int(match[1]) - 1}]" if match else "$"
        code = "LONG_SEGMENT_REASON_REQUIRED" if "exceeds 120 seconds" in message else "V3_CONTRACT_VIOLATION"
        if code == "LONG_SEGMENT_REASON_REQUIRED":
            path += ".long_segment_reason"
        result.append({"code": code, "path": path, "message": message})
    return result


def validate_v3_candidate(input_path: Path, data: Any, candidate: Path) -> dict[str, Any]:
    """The offline and production validation path; never invokes a provider."""
    import jsonschema
    from .narrative_provider import v3_schema
    schema = v3_schema()
    errors = []
    for error in jsonschema.Draft202012Validator(schema).iter_errors(data):
        # Report expected constraints, never echo arbitrary response values.
        path = "$" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.path)
        errors.append({"code": "SCHEMA_" + error.validator.upper(), "path": path,
                       "message": f"{path}: must satisfy {error.validator} constraint {json.dumps(error.validator_value, ensure_ascii=False)}"})
    if errors:
        raise V3ValidationError("structured_schema_validation", errors)
    chunk_input = json.loads(input_path.read_text(encoding="utf-8"))
    errors = _contract_errors(validate_llm_v3_response(chunk_input, data))
    if errors:
        raise V3ValidationError("canonical_semantic_validation", errors)
    canonical = normalize_llm_v3_response(chunk_input, data)
    write_json(candidate, canonical)
    errors = _contract_errors(validate_narrative_map(input_path, candidate))
    if errors:
        raise V3ValidationError("canonical_map_validation", errors)
    return canonical


def _retry_prompt(prompt: str, validation: dict[str, Any]) -> str:
    return (prompt + "\n\nCORRECTION REQUIRED: Your previous result failed validation:\n"
            + "\n".join("- " + item["message"] for item in validation["errors"])
            + "\nReturn a corrected complete V3 result. Preserve valid content unless required to fix these issues. "
              "Use only supplied cues. Split distinct situations; justify a long segment only if it is one continuous interaction/action.")


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

def _narrative_content_inputs(input_path: Path, model: str, prompt_hash: str) -> dict[str, Any]:
    """The durable identity of a chunk response; deliberately excludes runtime.

    Credential discovery, provider ordering, pool refreshes, and cooldown state
    only decide how a request is delivered.  They must never affect whether an
    already validated response is semantically reusable.
    """
    return {
        "content_fingerprint_version": CONTENT_FINGERPRINT_VERSION,
        "input_sha256": sha256_file(input_path),
        "schema_sha256": sha256_file(Path(__file__).resolve().parents[2] / "schemas/narrative_mapper_llm_v3.schema.json"),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "narrative_profile": "narrative_v3",
        "production_profile": PRODUCTION_NARRATIVE_PROFILE["profile_version"],
        "window_seconds": TARGET_WINDOW_SECONDS,
        "overlap_seconds": OVERLAP_SECONDS,
    }


def _content_fingerprint(content_inputs: dict[str, Any]) -> str:
    return sha256_text(json.dumps(content_inputs, sort_keys=True, separators=(",", ":")))


def _checkpoint_valid(input_path: Path, map_path: Path, meta_path: Path, expected: dict[str, Any]) -> bool:
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        # Old checkpoints predate content_fingerprint but have the same
        # content fields.  Keep them reusable rather than spending Gemini work
        # merely to add a diagnostic field.
        fingerprint = metadata.get("content_fingerprint")
        return (all(metadata.get(key) == value for key, value in expected.items())
                and (fingerprint is None or fingerprint == _content_fingerprint(expected))
                and not validate_narrative_map(input_path, map_path))
    except (OSError, TypeError, json.JSONDecodeError): return False


def _recover_saved_response(input_path: Path, run_dir: Path, expected: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Revalidate saved responses with proven matching semantic identity offline.

    Legacy attempts have only chunk-level execution provenance. Accept that
    provenance only when it identifies one producer and one content contract.
    New attempts carry immutable per-response provenance instead.
    """
    chunk_id = input_path.name.removesuffix(".input.json")
    fingerprint = _content_fingerprint(expected)
    directories = [run_dir / "responses"] + sorted((run_dir / "superseded").glob("**/responses"))
    for directory in directories:
        try:
            execution = json.loads((directory / f"{chunk_id}.execution.json").read_text(encoding="utf-8"))
            states = [execution] + list(execution.get("retry_history", {}).values())
            producers = {(s.get("content_fingerprint"), s.get("provider"), s.get("model")) for s in states}
        except (OSError, ValueError, TypeError, AttributeError):
            execution, producers = {}, set()
        paths = sorted(directory.glob(f"{chunk_id}.attempt-*.llm-v3.json"),
                       key=lambda p: int(p.name.split(".attempt-")[1].split(".")[0]), reverse=True)
        for raw_path in paths:
            try:
                provenance_path = raw_path.with_suffix(".provenance.json")
                if provenance_path.is_file():
                    producer = json.loads(provenance_path.read_text(encoding="utf-8"))
                    if producer.get("response_sha256") != sha256_file(raw_path):
                        continue
                elif len(producers) == 1:
                    producer = execution
                else:
                    continue
                if (producer.get("content_fingerprint") != fingerprint
                        or not producer.get("provider") or not producer.get("model")):
                    continue
                data = json.loads(raw_path.read_text(encoding="utf-8"))
                # Validation can write a candidate before rejecting it. Keep all
                # archive evidence intact, including historical canonical files.
                with tempfile.TemporaryDirectory(prefix="narrative-revalidation-") as temporary:
                    canonical = validate_v3_candidate(input_path, data, Path(temporary) / "candidate.json")
                return canonical, {"provider": producer["provider"], "model": producer["model"],
                                   "usage": producer.get("usage", {}), "validated_at": _utc(),
                                   "recovered_from": str(raw_path.relative_to(run_dir)),
                                   "response_sha256": sha256_file(raw_path)}
            except (OSError, ValueError, TypeError, KeyError):
                continue
    return None

def _status(error: Exception) -> int | None:
    value = getattr(error, "status_code", None) or getattr(error, "code", None)
    return value if isinstance(value, int) else None

def _daily_quota(error: Exception) -> bool:
    return _status(error) == 429 and "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in str(error)

def _transient(error: Exception) -> bool:
    if _status(error) in (400, 401, 403, 404):
        return False
    from .broll_semantics import classify_provider_error
    if classify_provider_error(error).get("retryable"):
        return True
    return _status(error) in (429, 500, 502, 503, 504) or isinstance(error, (TimeoutError, ConnectionError, OSError))

def _error_type(error: Exception) -> str:
    status = _status(error)
    if status == 401: return "AUTHENTICATION"
    if status == 404: return "MODEL_UNAVAILABLE"
    if status == 429: return "QUOTA_OR_RATE_LIMIT"
    if status == 503: return "HIGH_DEMAND"
    if isinstance(error, (TimeoutError, ConnectionError, OSError)): return "TIMEOUT_OR_NETWORK"
    if isinstance(error, ValueError): return "STRUCTURED_VALIDATION_FAILURE"
    return "PROVIDER_ERROR"

def _safe_error(error: Exception, api_key: str | None) -> str:
    message = str(error)
    return message.replace(api_key, "[REDACTED]") if api_key else message


class GeminiNarrativeProviderPool:
    """Small sequential failover boundary for unattended narrative mapping."""
    identifier = "gemini-pool"
    def __init__(self, credentials: list[GeminiCredential], model: str, source: GeminiCredentialSource | None = None) -> None:
        self.model=model
        self.providers=[self._provider(spec) for spec in credentials]
        self.cursor=0
        self.source = source

    def _provider(self, spec: GeminiCredential) -> tuple[GeminiCredential, GeminiNarrativeProvider]:
        provider = GeminiNarrativeProvider(spec.key, self.model)
        # The API wrapper has no logging, but retaining a safe identifier makes
        # any future provider-level diagnostics non-secret by construction.
        provider.identifier = spec.identifier
        return spec, provider

    def _refresh_credentials(self) -> None:
        if self.source is None:
            return
        try:
            credentials = list(self.source.discover().all_credentials())
        except (OSError, ValueError):
            return
        existing = {spec.key: (spec, provider) for spec, provider in self.providers}
        refreshed = []
        for spec in credentials:
            old = existing.get(spec.key)
            if old is not None and old[0].identifier == spec.identifier:
                refreshed.append(old)
            else:
                refreshed.append(self._provider(spec))
        self.providers = refreshed
        self.cursor = self.cursor % len(self.providers) if self.providers else 0

    def generate(self, prompt: str, chunk_input: dict[str, Any]) -> ProviderResponse:
        self._refresh_credentials()
        if not self.providers:
            raise RuntimeError("no Gemini credentials are currently configured")
        failures=[]
        for offset in range(len(self.providers)):
            index=(self.cursor+offset)%len(self.providers)
            try:
                response=self.providers[index][1].generate(prompt,chunk_input)
                self.cursor=(index+1)%len(self.providers)
                return response
            except Exception as error:
                failures.append(error)
                # Authentication/project access is credential-scoped. Try the
                # next configured credential before treating the chunk as a
                # provider-wide failure.
                if not _transient(error) and _status(error) not in (401, 403):
                    raise
        raise failures[-1]


def _configured_keys() -> list[str]:
    return [item.key for item in GeminiCredentialSource().discover().all_credentials()]


def _profile_matches(data: dict[str, Any]) -> bool:
    return (data.get("window_seconds") == TARGET_WINDOW_SECONDS
            and data.get("overlap_seconds") == OVERLAP_SECONDS
            and data.get("production_profile") == PRODUCTION_NARRATIVE_PROFILE["profile_version"])


def _chunk_profile_matches(value: Any) -> bool:
    """Accept numeric-equivalent historical profile labels (600 == 600.0)."""
    if not isinstance(value, str) or not value.endswith("s_overlap"):
        return False
    try:
        window, overlap = value.removesuffix("s_overlap").split("s_", 1)
        return float(window) == float(TARGET_WINDOW_SECONDS) and float(overlap) == float(OVERLAP_SECONDS)
    except ValueError:
        return False


def _automatic_manifest_compatible(run_dir: Path, manifest: dict[str, Any], srt: Path | None) -> bool:
    if (not _profile_matches(manifest)
            or manifest.get("status") != "COMPLETE"
            or manifest.get("prompt_version") != PROMPT_VERSION):
        return False
    prompt_path = Path(__file__).resolve().parents[2] / "config" / "prompts" / f"{PROMPT_VERSION}.md"
    if manifest.get("prompt_sha256") != sha256_text(prompt_path.read_text(encoding="utf-8")):
        return False
    if srt is None:
        return True
    try:
        source = json.loads((run_dir.parent / "source-v1" / "source_manifest.json").read_text(encoding="utf-8"))["source"]
        return source["srt"]["sha256"] == sha256_file(srt)
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


def narrative_artifacts_compatible(run_dir: Path, srt: Path | None = None) -> bool:
    """Whether an existing final map is explicitly compatible with production_v1.

    Old imported maps without an automatic-run manifest remain supported as the
    advanced external-mapper path. Automatic artifacts must always declare the
    current profile.
    """
    map_path, manifest_path = run_dir / "narrative_map.json", run_dir / "narrative_run.json"
    if not map_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
        # A manual prepare records neutral external provenance.  It is not an
        # automatic Gemini-run manifest and must retain the same compatibility
        # behavior as the prior manifest-less external-mapper workflow.
        if (manifest is not None and manifest.get("provider") != "external_llm"
                and not _automatic_manifest_compatible(run_dir, manifest, srt)):
            return False
        analysis = json.loads(map_path.read_text(encoding="utf-8")).get("analysis", {})
        return not analysis or _chunk_profile_matches(analysis.get("chunk_profile"))
    except (OSError, TypeError, json.JSONDecodeError):
        return False


def _archive_paths(run_dir: Path, paths: list[Path], category: str) -> None:
    paths = [path for path in paths if path.exists()]
    if not paths:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = run_dir / "superseded" / category / stamp
    for path in paths:
        target = archive / path.relative_to(run_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))


def _retire_incompatible_profile(run_dir: Path) -> bool:
    """Move, rather than delete, derived automatic artifacts for old profiles."""
    manifest_path = run_dir / "narrative_run.json"
    incompatible = False
    if manifest_path.is_file():
        try:
            incompatible = not _profile_matches(json.loads(manifest_path.read_text(encoding="utf-8")))
        except (OSError, TypeError, json.JSONDecodeError):
            incompatible = True
    else:
        for path in (run_dir / "chunks").glob("NCHUNK_*.input.json") if (run_dir / "chunks").is_dir() else ():
            try:
                chunk = json.loads(path.read_text(encoding="utf-8")).get("chunk", {})
                if (chunk.get("target_window_seconds") != TARGET_WINDOW_SECONDS
                        or chunk.get("overlap_seconds") != OVERLAP_SECONDS):
                    incompatible = True
                    break
            except (OSError, TypeError, json.JSONDecodeError):
                incompatible = True
                break
    map_path = run_dir / "narrative_map.json"
    if not incompatible and map_path.is_file():
        try:
            analysis = json.loads(map_path.read_text(encoding="utf-8")).get("analysis", {})
            incompatible = bool(analysis) and not _chunk_profile_matches(analysis.get("chunk_profile"))
        except (OSError, TypeError, json.JSONDecodeError):
            incompatible = True
    if incompatible:
        _archive_paths(run_dir, [run_dir / name for name in ("chunks", "maps", "responses", "narrative_map.json", "narrative_run.json", "reconciliation_report.json")], "incompatible-narrative-profile")
    return incompatible


def _retire_superseded_semantics(run_dir: Path, prompt_hash: str) -> bool:
    """Archive a prior automatic mapper contract before it can be overwritten.

    Only narrative-v2-owned files move. Source inspection, technical shots,
    semantic checkpoints, assets, and reviews are intentionally outside this
    path list.
    """
    manifest_path = run_dir / "narrative_run.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        superseded = (manifest.get("prompt_version") != PROMPT_VERSION
                      or manifest.get("prompt_sha256") != prompt_hash)
    except (OSError, TypeError, json.JSONDecodeError):
        superseded = True
    if superseded:
        _archive_paths(run_dir, [run_dir / name for name in (
            "chunks", "maps", "responses", "narrative_map.json",
            "narrative_run.json", "reconciliation_report.json",
        )], "superseded-narrative-semantics")
    return superseded


def _retire_stale_chunk_artifacts(run_dir: Path, input_paths: list[Path]) -> None:
    active = {path.name.removesuffix(".input.json") for path in input_paths}
    stale: list[Path] = []
    for directory, pattern in (("chunks", "NCHUNK_*.input.json"), ("maps", "NCHUNK_*"), ("responses", "NCHUNK_*")):
        base = run_dir / directory
        if not base.is_dir():
            continue
        for path in base.glob(pattern):
            chunk_id = path.name.split(".", 1)[0]
            if chunk_id not in active:
                stale.append(path)
    _archive_paths(run_dir, stale, "stale-narrative-chunks")

def run_narrative(input_dir: Path, model: str = DEFAULT_MODEL, force: bool = False, max_chunks: int | None = None, provider: NarrativeProvider | None = None, sleep: Callable[[float], None] = time.sleep, output: Callable[[str], None] = print) -> dict[str, Any]:
    env_file = Path(__file__).resolve().parents[2] / ".env"
    credential_source = GeminiCredentialSource(env_file)
    credentials = list(credential_source.discover().all_credentials())
    # Include externally supplied values in redaction even when the .env file
    # is authoritative for selecting credentials.
    api_keys = list(dict.fromkeys(
        credential_source.secret_values() + gemini_secret_values(os.environ) + (os.environ.get("OPENAI_API_KEY", ""),)
    ))
    if provider is None and not credentials: raise RuntimeError("GEMINI_API_KEY_<positive integer>, GEMINI_API_KEY_BACKUP, or GEMINI_API_KEY is not configured")
    movie_id = input_dir.name; movie = input_dir / "movie.mp4"; srt = input_dir / "subtitles.srt"
    if not movie.is_file() or not srt.is_file(): raise ValueError("input directory must contain movie.mp4 and subtitles.srt")
    root = input_dir.resolve().parents[1] / "runs" / movie_id; run_dir = root / "narrative-v2"; chunks_dir = run_dir / "chunks"; maps_dir = run_dir / "maps"; responses_dir = run_dir / "responses"
    prompt_path = Path(__file__).resolve().parents[2] / "config" / "prompts" / f"{PROMPT_VERSION}.md"; prompt = prompt_path.read_text(encoding="utf-8"); prompt_hash = sha256_text(prompt)
    cues_path = _ensure_source(movie, srt, root, movie_id)
    profile_replaced = _retire_incompatible_profile(run_dir)
    semantics_replaced = _retire_superseded_semantics(run_dir, prompt_hash)
    prepare_narrative_inputs(cues_path, movie_id, run_dir, TARGET_WINDOW_SECONDS, OVERLAP_SECONDS, force=True)
    all_inputs = sorted(chunks_dir.glob("NCHUNK_*.input.json")); _retire_stale_chunk_artifacts(run_dir, all_inputs)
    inputs = all_inputs if max_chunks is None else all_inputs[:max_chunks]
    active_provider = provider or GeminiNarrativeProviderPool(credentials, model, credential_source)
    # This pilot is intentionally configured for Gemini Free Tier; deployments
    # with a different billing configuration must override this environment value.
    pricing_mode = os.getenv("GEMINI_PRICING_MODE", "free_tier")
    budget = FREE_TIER_REQUEST_BUDGET if model == DEFAULT_MODEL and pricing_mode == "free_tier" else None
    started = _utc(); usage = _usage(); completed = reused = requests = retries = 0; failures: list[dict[str, str]] = []; quota_status = "NOT_EXHAUSTED"; stopped_status: str | None = None
    output(f"[narrative] movie: {movie_id}"); output(f"[narrative] provider: {active_provider.identifier}"); output(f"[narrative] model: {model}"); output(f"[narrative] chunks: {len(inputs)}")
    for position, input_path in enumerate(inputs, 1):
        if stopped_status: break
        chunk_id = input_path.name.removesuffix(".input.json"); map_path = maps_dir / f"{chunk_id}.narrative_map.json"; meta_path = maps_dir / f"{chunk_id}.checkpoint.json"
        expected = _narrative_content_inputs(input_path, model, prompt_hash)
        if map_path.is_file() and _checkpoint_valid(input_path, map_path, meta_path, expected):
            reused += 1; completed += 1; output(f"[narrative] {position:02d}/{len(inputs):02d} REUSED"); continue
        recovered = _recover_saved_response(input_path, run_dir, expected)
        if recovered is not None:
            canonical, provenance = recovered
            write_json(map_path, canonical)
            write_json(meta_path, {**expected, "content_fingerprint": _content_fingerprint(expected), **provenance})
            reused += 1; completed += 1
            output(f"[narrative] {position:02d}/{len(inputs):02d} RECOVERED_OFFLINE")
            continue
        execution_path = responses_dir / f"{chunk_id}.execution.json"
        identity = {"content_fingerprint": _content_fingerprint(expected), "provider": active_provider.identifier, "model": model, "validation_recovery_revision": VALIDATION_RECOVERY_REVISION}
        try:
            prior = json.loads(execution_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prior = {}
        history = prior.pop("retry_history", {})
        # Upgrade unversioned guard records as v1, never as the current revision:
        # adding this field must not reopen already-exhausted paid retries.
        if prior:
            prior.setdefault("validation_recovery_revision", _LEGACY_RECOVERY_REVISION)
            prior_identity = {key: prior.get(key) for key in identity}
            if all(value is not None for value in prior_identity.values()):
                history[sha256_text(json.dumps(prior_identity, sort_keys=True))] = prior
        identity_key = sha256_text(json.dumps(identity, sort_keys=True))
        if not all(prior.get(key) == value for key, value in identity.items()):
            prior = history.get(identity_key, {})
            if not prior and VALIDATION_RECOVERY_REVISION == _LEGACY_RECOVERY_REVISION:
                legacy_identity = {key: value for key, value in identity.items() if key != "validation_recovery_revision"}
                prior = history.get(sha256_text(json.dumps(legacy_identity, sort_keys=True)), {})
        # Delivery changes do not authorize another paid semantic budget. Carry
        # exhaustion across producer/model switches for this contract/revision.
        matching_guards = [state for state in history.values()
                           if state.get("content_fingerprint") == identity["content_fingerprint"]
                           and state.get("validation_recovery_revision", _LEGACY_RECOVERY_REVISION) == VALIDATION_RECOVERY_REVISION]
        for state in matching_guards:
            if state.get("validation_failures", 0) > prior.get("validation_failures", 0):
                prior = state
            if (state.get("validation_failures", 0) >= MAX_SEMANTIC_ATTEMPTS
                    or state.get("status") == "BLOCKED_NARRATIVE_VALIDATION"):
                prior = state
                break
            if (not prior and state.get("status") == "PARTIAL"
                    and state.get("error_type") == "STRUCTURED_VALIDATION_FAILURE"):
                prior = state

        def persist_execution(state: dict[str, Any]) -> None:
            state = {key: value for key, value in state.items() if key != "retry_history"}
            history[identity_key] = state
            write_json(execution_path, {**state, "retry_history": history})

        # A legacy execution PARTIAL alone could be an interrupted first attempt.
        # The final manifest error confirms the old runner exhausted its budget.
        try:
            previous_run = json.loads((run_dir / "narrative_run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous_run = {}
        legacy_exhausted = (prior.get("status") == "PARTIAL"
                            and prior.get("error_type") == "STRUCTURED_VALIDATION_FAILURE"
                            and any(item.get("chunk_id") == chunk_id and item.get("error_type") == "STRUCTURED_VALIDATION_FAILURE"
                                    for item in previous_run.get("errors", [])))
        validation_failures = prior.get("validation_failures", MAX_SEMANTIC_ATTEMPTS if legacy_exhausted or prior.get("status") == "BLOCKED_NARRATIVE_VALIDATION" else 0)
        if validation_failures >= MAX_SEMANTIC_ATTEMPTS:
            stopped_status = "BLOCKED_NARRATIVE_VALIDATION"
            failures.append({"chunk_id": chunk_id, "error_type": "STRUCTURED_VALIDATION_FAILURE", "error": "durable semantic retry limit exhausted"})
            persist_execution({**prior, **identity, "status": stopped_status, "validation_failures": validation_failures})
            output(f"[narrative] {position:02d}/{len(inputs):02d} {stopped_status}")
            break
        chunk_input = json.loads(input_path.read_text(encoding="utf-8")); semantic_attempts = max([prior.get("response_attempts", 0)] + [int(path.name.split(".attempt-", 1)[1].split(".", 1)[0]) for path in responses_dir.glob(f"{chunk_id}.attempt-*.llm-v3.json")]); transient_retries = 0; last_error = "unknown failure"; last_error_type = "PROVIDER_ERROR"
        validation = prior.get("validation", {})
        call_prompt = _retry_prompt(prompt, validation) if validation.get("errors") else prompt
        while True:
            if budget is not None and requests >= budget:
                stopped_status = "REQUEST_BUDGET_EXHAUSTED"; last_error = "request budget exhausted"; failures.append({"chunk_id": chunk_id, "error": last_error}); break
            try:
                persist_execution({**identity, "status": "RUNNING", "validation_failures": validation_failures, "validation": validation, "response_attempts": semantic_attempts, "failure_fingerprint": sha256_text(json.dumps(validation, sort_keys=True)) if validation else None, "attempt": semantic_attempts + transient_retries + 1})
                output(f"[narrative] {position:02d}/{len(inputs):02d} CALL"); requests += 1
                response: ProviderResponse = active_provider.generate(call_prompt, chunk_input); _add_usage(usage, response.usage)
                semantic_attempts += 1
                raw_path = responses_dir / f"{chunk_id}.attempt-{semantic_attempts}.llm-v3.json"; write_json(raw_path, response.data)
                write_json(raw_path.with_suffix(".provenance.json"), {**identity, "model": getattr(active_provider, "model", model), "usage": response.usage, "response_sha256": sha256_file(raw_path)})
                candidate = responses_dir / f"{chunk_id}.attempt-{semantic_attempts}.canonical.json"
                canonical = validate_v3_candidate(input_path, response.data, candidate)
                write_json(map_path, canonical); write_json(meta_path, {**expected, "content_fingerprint": _content_fingerprint(expected), "provider": active_provider.identifier, "model": getattr(active_provider, "model", model), "usage": response.usage, "validated_at": _utc()})
                persist_execution({**identity, "status": "COMPLETE", "usage": response.usage, "validation": {"status": "PASSED", "stage": "canonical_map_validation", "errors": []}})
                completed += 1; output(f"[narrative] {position:02d}/{len(inputs):02d} VALID segments={len(canonical['segments'])}"); break
            except Exception as error:
                if hasattr(error, "usage"):
                    _add_usage(usage, error.usage)
                semantic_failure = isinstance(error, ValueError)
                if semantic_failure:
                    validation_failures += 1
                    validation = getattr(error, "validation", {"status": "FAILED", "stage": "provider_response_decoding", "errors": [{"code": "INVALID_PROVIDER_JSON", "path": "$", "message": "Provider response could not be decoded as V3 JSON"}]})
                    # Redact diagnostics as well as the manifest and retry feedback.
                    for item in validation["errors"]:
                        for key in api_keys:
                            if key:
                                item["message"] = item["message"].replace(key, "[REDACTED]")
                                item["path"] = item["path"].replace(key, "[REDACTED]")
                    last_error = "; ".join(item["message"] for item in validation["errors"])
                    output(f"[narrative] {position:02d}/{len(inputs):02d} INVALID errors={len(validation['errors'])}")
                    for item in validation["errors"]:
                        output("[narrative]   - " + item["message"])
                else:
                    last_error = _error_type(error)
                for key in api_keys: last_error=_safe_error(Exception(last_error), key)
                last_error_type = _error_type(error)
                persist_execution({**identity, "status": ("BLOCKED_NARRATIVE_VALIDATION" if validation_failures >= MAX_SEMANTIC_ATTEMPTS else "RETRY_NARRATIVE_VALIDATION") if semantic_failure else "WAITING_PROVIDER", "error_type": last_error_type, "usage": usage, "validation_failures": validation_failures, "response_attempts": semantic_attempts, "validation": validation, "failure_fingerprint": sha256_text(json.dumps(validation, sort_keys=True)) if validation else None})
                if _daily_quota(error):
                    quota_status = "DAILY_QUOTA_EXHAUSTED"; stopped_status = quota_status; failures.append({"chunk_id": chunk_id, "error_type": last_error_type, "error": last_error}); output(f"[narrative] {position:02d}/{len(inputs):02d} DAILY_QUOTA_EXHAUSTED"); break
                if semantic_failure and validation_failures < MAX_SEMANTIC_ATTEMPTS:
                    call_prompt = _retry_prompt(prompt, validation)
                    retries += 1; output(f"[narrative] {position:02d}/{len(inputs):02d} SEMANTIC_RETRY"); continue
                if _transient(error) and transient_retries < MAX_TRANSIENT_RETRIES:
                    transient_retries += 1; retries += 1; sleep(min(20.0, 2.0 * (2 ** (transient_retries - 1))) + random.uniform(0, .25)); output(f"[narrative] {position:02d}/{len(inputs):02d} TRANSIENT_RETRY"); continue
                failures.append({"chunk_id": chunk_id, "error_type": last_error_type, "error": last_error}); stopped_status = "BLOCKED_NARRATIVE_VALIDATION" if semantic_failure else "PARTIAL"; output(f"[narrative] {position:02d}/{len(inputs):02d} FAILED"); break
    status = stopped_status or ("COMPLETE" if not failures and completed == len(all_inputs) else "PARTIAL")
    manifest = {"schema_version": "narrative_run_v2", "narrative_profile": "narrative_v3", "production_profile": PRODUCTION_NARRATIVE_PROFILE["profile_version"], "movie_id": movie_id, "provider": active_provider.identifier, "model": model, "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_hash, "window_seconds": TARGET_WINDOW_SECONDS, "overlap_seconds": OVERLAP_SECONDS, "profile_replaced": profile_replaced, "semantics_replaced": semantics_replaced, "started_at": started, "completed_at": _utc(), "status": status, "chunk_count": len(inputs), "valid_chunks": completed, "completed_chunks": completed, "reused_chunks": reused, "failed_chunks": [item["chunk_id"] for item in failures], "safety_request_budget": budget, "requests": requests, "retries": retries, "quota_status": quota_status, "usage": usage, "pricing_mode": pricing_mode, "errors": failures}
    manifest["api_cost_estimate_usd"] = "0.00" if pricing_mode == "free_tier" else "unknown"; write_json(run_dir / "narrative_run.json", manifest)
    output(f"[narrative] requests={requests} retries={retries}"); output(f"[narrative] status={status}"); return manifest


def export_external_comparison(input_dir: Path, chunk_id: str) -> Path:
    """Write one provider-neutral, text-only package for an A/B mapper review."""
    if not chunk_id.startswith("NCHUNK_"):
        raise ValueError("chunk_id must be an NCHUNK_ identifier")
    run_dir = Path("runs") / input_dir.name / "narrative-v2"
    input_path = run_dir / "chunks" / f"{chunk_id}.input.json"
    raw_paths = sorted((run_dir / "responses").glob(f"{chunk_id}.attempt-*.llm-v3.json"))
    if not input_path.is_file() or not raw_paths:
        raise ValueError(f"validated v3 input and response are required for {chunk_id}")
    prompt_path = Path(__file__).resolve().parents[2] / "config" / "prompts" / f"{PROMPT_VERSION}.md"
    package = {
        "schema_version": "narrative_external_comparison_v1",
        "purpose": "Provider-neutral A/B evaluation of narrative situation boundaries; no visual claims.",
        "prompt_version": PROMPT_VERSION,
        "prompt": prompt_path.read_text(encoding="utf-8"),
        "chunk_input": json.loads(input_path.read_text(encoding="utf-8")),
        "automatic_response": json.loads(raw_paths[-1].read_text(encoding="utf-8")),
        "evaluation_questions": [
            "Does every segment represent one coherent narrative situation?",
            "Are meaningful interaction/action transitions split without splitting camera or speaker changes?",
            "Are segments over 120 seconds justified and still coherent?",
        ],
    }
    output_path = run_dir / "external-comparison" / f"{chunk_id}.comparison.json"
    write_json(output_path, package)
    return output_path
