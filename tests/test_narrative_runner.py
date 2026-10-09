import json
from pathlib import Path

import pytest

from movie_broll import narrative_runner
from movie_broll.narrative_provider import ProviderResponse
from movie_broll.narrative_consolidate import consolidate_narrative
from movie_broll.narrative_runner import (DEFAULT_MODEL, FREE_TIER_REQUEST_BUDGET,
                                          narrative_artifacts_compatible,
                                          run_narrative)
from movie_broll.srt import Cue


def _response(data):
    return {"schema_version": "narrative_mapper_llm_v3", "chunk_summary_es": "Resumen.", "segments": [{"first_cue_id": data["cues"][0]["cue_id"], "last_cue_id": data["cues"][-1]["cue_id"], "segment_type": "conversation", "narrative_summary_es": "Resumen.", "situation_es": "Una conversación.", "participants_es": "Interlocutores no identificados.", "interaction_action_es": "Conversan.", "location_context_es": "No inferible por los subtítulos.", "narrative_tone": "neutral", "narrative_function": "conversation", "context_dependency": "medium", "continuity_previous": "unknown", "continuity_next": "outside_chunk", "continuity_rationale_es": "Un solo intercambio.", "transition_reason_start": "chunk_start", "boundary_reason_end": "chunk_end", "long_segment_reason": None, "possible_visual_opportunities": ["reaction"]}]}


class FakeProvider:
    identifier = "gemini"; model = "gemini-3.6-flash"
    def __init__(self, failures=0, invalid=False): self.calls = 0; self.failures = failures; self.invalid = invalid
    def generate(self, prompt, chunk_input):
        self.calls += 1
        if self.calls <= self.failures: raise TimeoutError("temporary")
        value = _response(chunk_input)
        if self.invalid: value["segments"][0]["first_cue_id"] = "SRT_999999"
        return ProviderResponse(value, {"prompt_tokens": 3, "response_tokens": 4, "thinking_tokens": 0, "cached_tokens": 0, "total_tokens": 7})


class HttpError(Exception):
    def __init__(self, code, message): self.status_code = code; super().__init__(message)


def _clear_narrative_gemini_credentials(monkeypatch):
    for name in list(narrative_runner.os.environ):
        if name.startswith("GEMINI_API_KEY"):
            monkeypatch.delenv(name, raising=False)
    actual_source = narrative_runner.GeminiCredentialSource
    monkeypatch.setattr(
        narrative_runner,
        "GeminiCredentialSource",
        lambda *_args, **_kwargs: actual_source(None),
    )
    assert narrative_runner._configured_keys() == []


def _srt_sensitive_source(tmp_path):
    """A lightweight source stage whose canonical cue identity follows the SRT."""
    def ensure(_movie, srt, root, movie_id):
        source_dir = root / "source-v1"; source_dir.mkdir(parents=True, exist_ok=True)
        cues_path = source_dir / "srt_cues.jsonl"
        cue = Cue("SRT_000001", 1, 0, 1, srt.read_text(encoding="utf-8"))
        cues_path.write_text(json.dumps(cue.as_dict()) + "\n", encoding="utf-8")
        (source_dir / "source_manifest.json").write_text(json.dumps({"source": {"movie_id": movie_id, "srt": {"sha256": narrative_runner.sha256_file(srt)}}}), encoding="utf-8")
        return cues_path
    return ensure


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); monkeypatch.setenv("GEMINI_PRICING_MODE", "free_tier")
    source = tmp_path / "cues.jsonl"
    cues = [Cue("SRT_000001", 1, 1, 2, "uno"), Cue("SRT_000002", 2, 1201, 1202, "dos")]
    source.write_text("".join(json.dumps(cue.as_dict()) + "\n" for cue in cues), encoding="utf-8")
    movie_dir = tmp_path / "input" / "pilot"; movie_dir.mkdir(parents=True)
    (movie_dir / "movie.mp4").touch(); (movie_dir / "subtitles.srt").write_text("x", encoding="utf-8")
    monkeypatch.setattr("movie_broll.narrative_runner._ensure_source", lambda *args: source)
    return movie_dir


def test_missing_key_fails_before_requests(prepared, monkeypatch):
    _clear_narrative_gemini_credentials(monkeypatch)

    class NoProviderConstruction:
        def __init__(self, *_):
            raise AssertionError("provider construction must not occur without credentials")

    monkeypatch.setattr("movie_broll.narrative_runner.GeminiNarrativeProvider", NoProviderConstruction)
    with pytest.raises(RuntimeError, match=r"GEMINI_API_KEY_<positive integer>, GEMINI_API_KEY_BACKUP, or GEMINI_API_KEY is not configured"):
        run_narrative(prepared)


def test_numbered_pool_key_succeeds_without_legacy_key(prepared, monkeypatch):
    _clear_narrative_gemini_credentials(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY_3", "numbered-pool-key")
    provider_keys = []

    class NumberedPoolProvider:
        def __init__(self, api_key, model):
            provider_keys.append((api_key, model))

        def generate(self, prompt, chunk_input):
            return ProviderResponse(_response(chunk_input), {"prompt_tokens": 3, "response_tokens": 4, "thinking_tokens": 0, "cached_tokens": 0, "total_tokens": 7})

    monkeypatch.setattr("movie_broll.narrative_runner.GeminiNarrativeProvider", NumberedPoolProvider)
    manifest = run_narrative(prepared, max_chunks=1, output=lambda _: None)
    assert manifest["provider"] == "gemini-pool" and manifest["requests"] == 1
    assert provider_keys == [("numbered-pool-key", DEFAULT_MODEL)]


def test_success_reuse_and_usage(prepared):
    provider = FakeProvider(); first = run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None); second = run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None)
    assert (first["requests"], first["usage"]["total_tokens"], provider.calls) == (1, 7, 1)
    assert second["reused_chunks"] == 1 and second["requests"] == 0 and first["safety_request_budget"] == FREE_TIER_REQUEST_BUDGET


def test_completed_narrative_map_reuses_after_restart_and_credential_changes(prepared, monkeypatch, tmp_path):
    monkeypatch.setattr("movie_broll.narrative_runner._ensure_source", _srt_sensitive_source(tmp_path))
    first_provider = FakeProvider(); first_provider.identifier = "gemini-pool"
    first = run_narrative(prepared, provider=first_provider, output=lambda _: None)
    assert first["requests"] == 1
    consolidate_narrative(prepared, output=lambda _: None)
    run = Path("runs") / "pilot" / "narrative-v2"
    semantic = Path("runs") / "pilot" / "semantic_checkpoints" / "sentinel.json"; semantic.parent.mkdir(parents=True); semantic.write_text("checkpoint")
    asset = Path("runs") / "pilot" / "assets" / "sentinel.mp4"; asset.parent.mkdir(parents=True); asset.write_bytes(b"asset")
    assert narrative_artifacts_compatible(run, prepared / "subtitles.srt")

    # These mutate runtime delivery only.  The distinct provider identifiers
    # also prove that provider/pool state is not a content-cache key.
    monkeypatch.setenv("GEMINI_API_KEY_7", "seven-first-value")
    monkeypatch.setenv("GEMINI_API_KEY_8", "eight")
    monkeypatch.setenv("GEMINI_API_KEY_9", "nine")
    restarted_provider = FakeProvider(); restarted_provider.identifier = "gemini-pool-refreshed"
    restarted = run_narrative(prepared, provider=restarted_provider, output=lambda _: None)
    assert restarted["requests"] == 0 and restarted["reused_chunks"] == 1 and restarted_provider.calls == 0

    monkeypatch.setenv("GEMINI_API_KEY_7", "seven-replaced-value")
    monkeypatch.delenv("GEMINI_API_KEY_8")
    refreshed_provider = FakeProvider(); refreshed_provider.identifier = "gemini-pool-reordered"
    refreshed = run_narrative(prepared, provider=refreshed_provider, output=lambda _: None)
    assert refreshed["requests"] == 0 and refreshed_provider.calls == 0
    assert (run / "narrative_map.json").is_file()
    assert semantic.read_text() == "checkpoint" and asset.read_bytes() == b"asset"
    checkpoint = (run / "maps" / "NCHUNK_0001.checkpoint.json").read_text()
    assert "seven-first-value" not in checkpoint and "seven-replaced-value" not in checkpoint


def test_profile_change_invalidates_narrative_content(prepared, monkeypatch):
    first_provider = FakeProvider(); run_narrative(prepared, provider=first_provider, max_chunks=1, output=lambda _: None)
    monkeypatch.setattr(narrative_runner, "TARGET_WINDOW_SECONDS", 601.0)
    changed_provider = FakeProvider(); result = run_narrative(prepared, provider=changed_provider, max_chunks=1, output=lambda _: None)
    assert result["profile_replaced"] is True and changed_provider.calls == 1
    assert (Path("runs") / "pilot" / "narrative-v2" / "superseded" / "incompatible-narrative-profile").is_dir()


def test_prompt_contract_change_archives_only_narrative_artifacts(prepared):
    first_provider = FakeProvider(); run_narrative(prepared, provider=first_provider, max_chunks=1, output=lambda _: None)
    run = Path("runs") / "pilot" / "narrative-v2"
    assets = Path("runs") / "pilot" / "assets" / "keep.mp4"; assets.parent.mkdir(); assets.write_bytes(b"keep")
    manifest_path = run / "narrative_run.json"; manifest = json.loads(manifest_path.read_text()); manifest["prompt_version"] = "old-contract"; manifest_path.write_text(json.dumps(manifest))
    second_provider = FakeProvider(); replacement = run_narrative(prepared, provider=second_provider, max_chunks=1, output=lambda _: None)
    assert replacement["semantics_replaced"] is True and second_provider.calls == 0
    assert replacement["reused_chunks"] == 1
    assert (run / "superseded" / "superseded-narrative-semantics").is_dir()
    assert assets.read_bytes() == b"keep"


def test_srt_change_invalidates_narrative_content(prepared, monkeypatch, tmp_path):
    monkeypatch.setattr("movie_broll.narrative_runner._ensure_source", _srt_sensitive_source(tmp_path))
    first_provider = FakeProvider(); run_narrative(prepared, provider=first_provider, output=lambda _: None)
    consolidate_narrative(prepared, output=lambda _: None)
    run = Path("runs") / "pilot" / "narrative-v2"
    assert narrative_artifacts_compatible(run, prepared / "subtitles.srt")
    (prepared / "subtitles.srt").write_text("changed SRT identity", encoding="utf-8")
    assert not narrative_artifacts_compatible(run, prepared / "subtitles.srt")
    changed_provider = FakeProvider(); result = run_narrative(prepared, provider=changed_provider, output=lambda _: None)
    assert result["profile_replaced"] is False and changed_provider.calls == 1 and result["requests"] == 1


def test_incompatible_profile_is_safely_replaced_without_reinspecting_source(prepared):
    run=Path('runs')/'pilot'/'narrative-v2'; (run/'chunks').mkdir(parents=True)
    (run/'narrative_run.json').write_text(json.dumps({'window_seconds':1200,'overlap_seconds':90,'production_profile':'production_v1'}))
    (run/'chunks'/'NCHUNK_0001.input.json').write_text('{}')
    manifest=run_narrative(prepared, provider=FakeProvider(), max_chunks=1, output=lambda _: None)
    assert manifest['profile_replaced'] is True
    assert (run/'superseded'/'incompatible-narrative-profile').is_dir()
    assert manifest['window_seconds']==600 and manifest['overlap_seconds']==60


def test_semantic_failure_retries_once(prepared):
    manifest = run_narrative(prepared, provider=FakeProvider(invalid=True), max_chunks=1, sleep=lambda _: None, output=lambda _: None)
    assert manifest["status"] == "BLOCKED_NARRATIVE_VALIDATION" and manifest["requests"] == 2 and manifest["retries"] == 1
    assert not list((Path("runs") / "pilot" / "narrative-v2" / "maps").glob("*.narrative_map.json"))


def test_503_is_bounded_and_checkpoint_invalidation(prepared):
    transient = FakeProvider(failures=1); manifest = run_narrative(prepared, provider=transient, max_chunks=1, sleep=lambda _: None, output=lambda _: None)
    assert manifest["status"] == "PARTIAL" and manifest["requests"] == 2 and manifest["retries"] == 1
    changed_model = FakeProvider(); run_narrative(prepared, provider=changed_model, model="gemini-2.5-flash", max_chunks=1, output=lambda _: None); assert changed_model.calls == 0

    class UnavailableProvider(FakeProvider):
        def generate(self, prompt, chunk_input): self.calls += 1; raise HttpError(503, "high demand")
    unavailable = UnavailableProvider(); result = run_narrative(prepared, provider=unavailable, force=True, max_chunks=2, sleep=lambda _: None, output=lambda _: None)
    assert result["status"] == "PARTIAL" and unavailable.calls == 3 and result["retries"] == 2


def test_daily_quota_stops_without_retry(prepared):
    class QuotaProvider(FakeProvider):
        def generate(self, prompt, chunk_input): self.calls += 1; raise HttpError(429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    provider = QuotaProvider(); manifest = run_narrative(prepared, provider=provider, sleep=lambda _: None, output=lambda _: None)
    assert manifest["status"] == "DAILY_QUOTA_EXHAUSTED" and manifest["quota_status"] == "DAILY_QUOTA_EXHAUSTED" and provider.calls == 1
    assert manifest["errors"][0]["error_type"] == "QUOTA_OR_RATE_LIMIT"


def test_404_model_unavailable_fails_without_retry_and_redacts_key(prepared, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-test-key")
    class UnavailableProvider(FakeProvider):
        def generate(self, prompt, chunk_input): self.calls += 1; raise HttpError(404, "model unavailable: secret-test-key")
    provider = UnavailableProvider()
    manifest = run_narrative(prepared, provider=provider, force=True, max_chunks=1, sleep=lambda _: None, output=lambda _: None)
    assert manifest["status"] == "PARTIAL" and manifest["requests"] == 1 and manifest["retries"] == 0
    assert manifest["errors"][0]["error_type"] == "MODEL_UNAVAILABLE"
    assert "secret-test-key" not in manifest["errors"][0]["error"]


def test_request_budget_stops_cleanly(prepared, monkeypatch):
    monkeypatch.setattr("movie_broll.narrative_runner.FREE_TIER_REQUEST_BUDGET", 1)
    manifest = run_narrative(prepared, provider=FakeProvider(invalid=True), max_chunks=1, sleep=lambda _: None, output=lambda _: None)
    assert manifest["status"] == "REQUEST_BUDGET_EXHAUSTED" and manifest["requests"] == 1


def test_gemini_36_is_default():
    assert DEFAULT_MODEL == "gemini-3.6-flash"
