"""Deterministic safety checks for OpenAI semantic completion boundaries."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from movie_broll import production, production_run
from movie_broll.broll_pilot import _write_provider_output_diagnostic
from movie_broll.broll_semantics import (
    OpenAIProviderError,
    OpenAISemanticProvider,
    OpenAISemanticStructuredResult,
    OpenAIStructuredOutputError,
    build_openai_provider_from_env,
    classify_provider_error,
)
from movie_broll.processing_ledger import ProcessingLedger
from movie_broll.utils import write_json


def _parsed_response():
    value = {
        "visual": {"summary_es":"","subjects":[],"objects":[],"actions":[],"people_count_estimate":"0","setting":"","visible_interactions":[],"visible_emotions":[],"people":[],"primary_subject_position":"unclear","primary_subject_description":"","visual_focus":"","shot_focus_plan":[]},
        "relationships": [],
        "editorial": {"standalone_meaning_es":"","reusable_broll":False,"action_or_moment_complete":"false","use_cases_es":[],"negative_use_cases_es":[],"search_terms_es":[],"editorial_confidence":"low","keep_qualification":{"action_evidence_type":"unclear","action_evidence_es":"","reusable_intent_type":"unclear","reusable_use_case_es":""},"reason":"","decision":"REJECT"},
    }
    return SimpleNamespace(model="gpt-6-luna", output_parsed=OpenAISemanticStructuredResult.model_validate(value), usage=None)


def test_structured_parse_path_never_uses_output_text_and_logs_safe_timing():
    calls, lines = [], []
    response = _parsed_response()
    client = SimpleNamespace(responses=SimpleNamespace(parse=lambda **kw: calls.append(kw) or response))
    ticks = iter((10.0, 12.8))
    result = OpenAISemanticProvider("sk-not-real", client=client, reporter=lines.append, clock=lambda: next(ticks)).generate("p", {"visual_event_id":"VE_X"}, b"jpeg")
    assert result.data["editorial"]["decision"] == "REJECT"
    assert calls[0]["text_format"] is OpenAISemanticStructuredResult
    assert any("complete latency=2.8s" in line for line in lines)
    assert all("sk-not-real" not in line and "jpeg" not in line for line in lines)


def test_malformed_strict_output_is_bounded_retryable_not_editorial_failure():
    calls, sleeps = [], []
    def parse(**_kwargs):
        calls.append(1)
        raise ValueError("Extra data: line 1 column 1820 (char 1819)")
    provider = OpenAISemanticProvider("not-a-real-key", client=SimpleNamespace(responses=SimpleNamespace(parse=parse)), sleep=sleeps.append, max_retries=2)
    with pytest.raises(OpenAIProviderError) as raised:
        provider.generate("p", {"visual_event_id":"VE_BAD"}, b"jpeg")
    assert len(calls) == 2 and sleeps == [1.0]
    assert classify_provider_error(raised.value)["reason"] == "structured_output_invalid"
    assert classify_provider_error(raised.value)["retryable"] is True


def test_timeout_is_retryable_provider_deferred_and_diagnostic_redacts_secret(tmp_path):
    error = OpenAIProviderError(TimeoutError("request timed out for sk-secret"), 2)
    assert classify_provider_error(error)["reason"] == "timeout"
    candidate = {"visual_event_id":"VE_TIMEOUT", "candidate_id":"BRC_TIMEOUT"}
    artifact = _write_provider_output_diagnostic(tmp_path / "semantic_checkpoints", candidate, error, "openai", "gpt-6-luna", "fp")
    text = artifact.read_text()
    assert "sk-secret" not in text and json.loads(text)["failure_stage"] == "structured_output_parse"


def test_openai_client_has_explicit_timeouts_and_sdk_retries_disabled(monkeypatch):
    captured = {}
    class Timeout:
        def __init__(self, **kwargs): self.kwargs = kwargs
    class Client:
        def __init__(self, **kwargs): captured.update(kwargs)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Client))
    monkeypatch.setitem(sys.modules, "httpx2", SimpleNamespace(Timeout=Timeout))
    build_openai_provider_from_env(environ={
        "OPENAI_API_KEY":"not-a-real-key", "OPENAI_CONNECT_TIMEOUT_SECONDS":"11",
        "OPENAI_READ_TIMEOUT_SECONDS":"91", "OPENAI_WRITE_TIMEOUT_SECONDS":"31",
        "OPENAI_POOL_TIMEOUT_SECONDS":"12", "OPENAI_MAX_RETRIES":"2",
    })
    assert captured["max_retries"] == 0
    assert captured["timeout"].kwargs == {"connect":11.0, "read":91.0, "write":31.0, "pool":12.0}


def _event(event_id: str, candidate: str, status: str = "VALIDATED") -> dict:
    return {"visual_event_id":event_id, "candidate_id":candidate, "editorial":{"status":status, "decision":"REJECT"}, "source_shot_ids":[]}


def test_resume_integrity_requeues_only_inconsistent_completed_batch(monkeypatch, tmp_path):
    run = tmp_path / "runs" / "film"; run.mkdir(parents=True)
    good, bad = _event("VE_GOOD", "BRC_GOOD"), _event("VE_BAD", "BRC_BAD", "SEMANTIC_INCOMPLETE")
    store = {"status":"COMPLETE", "production_status":"COMPLETE", "events":[good, bad], "batches":{
        "PBATCH_GOOD":{"event_ids":["VE_GOOD"], "semantic_status":"COMPLETE", "status":"COMPLETE"},
        "PBATCH_BAD":{"event_ids":["VE_BAD"], "semantic_status":"COMPLETE", "status":"COMPLETE", "finalization_status":"COMPLETE"},
    }}
    ledger = ProcessingLedger(run, "film", {})
    monkeypatch.setattr(production, "_semantic_checkpoint", lambda _path, event, *_args: {"ok":True} if event["visual_event_id"] == "VE_GOOD" else None)
    repaired = production._repair_resume_semantic_integrity({"run":run}, store, ledger)
    assert repaired == ["PBATCH_BAD"]
    assert store["batches"]["PBATCH_GOOD"]["status"] == "COMPLETE"
    assert store["batches"]["PBATCH_BAD"]["status"] == "PENDING"
    assert store["production_status"] == "RUNNING"


def test_terminal_semantic_failure_artifact_satisfies_contract(monkeypatch, tmp_path):
    run = tmp_path / "run"; event = _event("VE_FAILURE", "BRC_FAILURE", "SEMANTIC_INCOMPLETE")
    write_json(run / "semantic_failures" / "VE_FAILURE.json", {"visual_event_id":"VE_FAILURE", "failure_stage":"semantic_validation"})
    monkeypatch.setattr(production, "_semantic_checkpoint", lambda *_: None)
    assert production._semantic_terminal_contract(run, [event]) == (True, [])


def test_openai_usage_excludes_historical_gemini_and_reports_mixed_provenance(tmp_path):
    run = tmp_path / "runs" / "film"
    write_json(run / "semantic_checkpoints" / "gemini.json", {"provider":"gemini", "model":"gemini-3.6-flash", "usage":{"prompt_tokens":999}})
    write_json(run / "semantic_checkpoints" / "openai.json", {"provider":"openai", "model":"gpt-6-luna", "usage":{"prompt_tokens":100, "cached_tokens":10, "response_tokens":20}})
    usage = production_run._semantic_usage(run)
    assert usage["requests"] == 1 and usage["usage"]["prompt_tokens"] == 100
    assert usage["provenance"] == {"gemini/gemini-3.6-flash":1, "openai/gpt-6-luna":1}
    assert usage["estimated_cost_usd"] > 0
