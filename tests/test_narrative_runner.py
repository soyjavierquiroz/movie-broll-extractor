import json
from pathlib import Path

import pytest

from movie_broll.narrative_provider import ProviderResponse
from movie_broll.narrative_runner import DEFAULT_MODEL, FREE_TIER_REQUEST_BUDGET, run_narrative
from movie_broll.srt import Cue


def _response(data):
    return {"schema_version": "narrative_mapper_llm_v2", "chunk_summary_es": "Resumen.", "segments": [{"first_cue_id": data["cues"][0]["cue_id"], "last_cue_id": data["cues"][-1]["cue_id"], "segment_type": "conversation", "narrative_summary_es": "Resumen.", "narrative_tone": "neutral", "narrative_function": "conversation", "context_dependency": "medium", "continuity_previous": "unknown", "continuity_next": "outside_chunk", "possible_visual_opportunities": ["reaction"]}]}


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
    monkeypatch.delenv("GEMINI_API_KEY", raising=False); monkeypatch.setattr("movie_broll.narrative_runner.load_dotenv", lambda *_: False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not configured"): run_narrative(prepared)


def test_success_reuse_and_usage(prepared):
    provider = FakeProvider(); first = run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None); second = run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None)
    assert (first["requests"], first["usage"]["total_tokens"], provider.calls) == (1, 7, 1)
    assert second["reused_chunks"] == 1 and second["requests"] == 0 and first["safety_request_budget"] == FREE_TIER_REQUEST_BUDGET


def test_semantic_failure_retries_once(prepared):
    manifest = run_narrative(prepared, provider=FakeProvider(invalid=True), max_chunks=1, sleep=lambda _: None, output=lambda _: None)
    assert manifest["status"] == "PARTIAL" and manifest["requests"] == 2 and manifest["retries"] == 1
    assert not list((Path("runs") / "pilot" / "narrative-v2" / "maps").glob("*.narrative_map.json"))


def test_503_is_bounded_and_checkpoint_invalidation(prepared):
    transient = FakeProvider(failures=1); manifest = run_narrative(prepared, provider=transient, max_chunks=1, sleep=lambda _: None, output=lambda _: None)
    assert manifest["status"] == "COMPLETE" and manifest["requests"] == 2 and manifest["retries"] == 1
    changed_model = FakeProvider(); run_narrative(prepared, provider=changed_model, model="gemini-2.5-flash", max_chunks=1, output=lambda _: None); assert changed_model.calls == 1

    class UnavailableProvider(FakeProvider):
        def generate(self, prompt, chunk_input): self.calls += 1; raise HttpError(503, "high demand")
    unavailable = UnavailableProvider(); result = run_narrative(prepared, provider=unavailable, force=True, max_chunks=1, sleep=lambda _: None, output=lambda _: None)
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
