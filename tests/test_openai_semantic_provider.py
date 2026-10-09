from types import SimpleNamespace

import pytest

from movie_broll.broll_semantics import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_REASONING_EFFORT,
    OpenAISemanticProvider,
    OpenAIStructuredOutputError,
    OpenAISemanticStructuredResult,
    build_semantic_provider_from_env,
    classify_provider_error,
    estimate_openai_cost,
    redact_provider_error,
)


class _Responses:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _response():
    return SimpleNamespace(
        model="gpt-6-luna",
        output_parsed=OpenAISemanticStructuredResult.model_validate({
            "visual": {"summary_es":"","subjects":[],"objects":[],"actions":[],"people_count_estimate":"0","setting":"","visible_interactions":[],"visible_emotions":[],"people":[],"primary_subject_position":"unclear","primary_subject_description":"","visual_focus":"","shot_focus_plan":[]},
            "relationships": [],
            "editorial": {"standalone_meaning_es":"","reusable_broll":False,"action_or_moment_complete":"false","use_cases_es":[],"negative_use_cases_es":[],"search_terms_es":[],"editorial_confidence":"low","keep_qualification":{"action_evidence_type":"unclear","action_evidence_es":"","reusable_intent_type":"unclear","reusable_use_case_es":""},"reason":"","decision":"REJECT"},
        }),
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=200,
            total_tokens=1200,
            input_tokens_details=SimpleNamespace(cached_tokens=300),
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )


def test_openai_default_selection_and_structured_multimodal_request():
    responses = _Responses(_response())
    client = SimpleNamespace(responses=responses)
    provider = OpenAISemanticProvider("not-a-real-key", client=client)
    result = provider.generate("system", {"visual_event_id": "VE_1"}, b"jpeg")
    assert provider.model == DEFAULT_OPENAI_MODEL
    assert provider.reasoning_effort == DEFAULT_OPENAI_REASONING_EFFORT
    assert result.usage == {"prompt_tokens": 1000, "cached_tokens": 300, "response_tokens": 200,
                            "thinking_tokens": 0, "total_tokens": 1200}
    request = responses.calls[0]
    assert request["reasoning"] == {"effort": "none"}
    assert request["text_format"] is OpenAISemanticStructuredResult
    assert request["input"][0]["content"][1]["type"] == "input_image"
    assert request["input"][0]["content"][1]["detail"] == "low"


@pytest.mark.parametrize("status", [429, 500, 503])
def test_openai_transient_errors_are_retryable(status):
    error = RuntimeError(f"status {status}")
    assert classify_provider_error(error)["retryable"] is True


def test_openai_auth_error_is_non_retryable_and_cost_uses_cached_rate():
    detail = classify_provider_error(RuntimeError("status 401"))
    assert detail["reason"] == "auth_error" and detail["retryable"] is False
    assert estimate_openai_cost(
        {"prompt_tokens": 1_000_000, "cached_tokens": 400_000, "response_tokens": 1_000_000},
        {"OPENAI_PRICE_INPUT_PER_MILLION": "0.10", "OPENAI_PRICE_CACHED_INPUT_PER_MILLION": "0.01",
         "OPENAI_PRICE_OUTPUT_PER_MILLION": "0.50"},
    ) == pytest.approx(0.564)
    assert "not-a-real-secret" not in redact_provider_error(
        "request failed for not-a-real-secret", {"OPENAI_API_KEY": "not-a-real-secret"}
    )


def test_openai_is_selected_without_gemini_credentials(monkeypatch):
    import movie_broll.broll_semantics as semantics
    sentinel = SimpleNamespace(identifier="openai", model=DEFAULT_OPENAI_MODEL)
    monkeypatch.setattr(semantics, "OpenAISemanticProvider", lambda *args, **kwargs: sentinel)
    provider = build_semantic_provider_from_env(
        environ={"SEMANTIC_PROVIDER": "openai", "OPENAI_API_KEY": "not-a-real-key"},
    )
    assert provider.identifier == "openai" and provider.model == DEFAULT_OPENAI_MODEL


def test_provider_change_reuses_a_valid_canonical_checkpoint(tmp_path):
    from movie_broll.broll_pilot import (
        SEMANTIC_PROMPT_VERSION,
        SEMANTIC_SCHEMA_VERSION,
        _semantic_checkpoint,
    )
    response = {
        "visual": {"summary_es": "puerta", "subjects": [], "objects": [], "actions": [],
                   "people_count_estimate": "0", "setting": "interior", "visible_interactions": [],
                   "visible_emotions": [], "people": [], "primary_subject_position": "unclear",
                   "primary_subject_description": "", "visual_focus": "puerta",
                   "shot_focus_plan": [{"shot_id": "S1", "focus_subject": "environment",
                                        "focus_reason": "puerta", "preserve_secondary_subject": False,
                                        "interaction_requirement": "none", "focus_position": "unclear"}]},
        "relationships": [],
        "editorial": {"standalone_meaning_es": "puerta", "reusable_broll": False,
                      "action_or_moment_complete": "false", "use_cases_es": [], "negative_use_cases_es": [],
                      "search_terms_es": [], "editorial_confidence": "low",
                      "keep_qualification": {"action_evidence_type": "unclear", "action_evidence_es": "",
                                               "reusable_intent_type": "unclear", "reusable_use_case_es": ""},
                      "reason": "contexto", "decision": "REJECT"},
    }
    candidate = {"candidate_id": "BRC_1", "start_frame": 1, "end_frame_exclusive": 2,
                 "source_shot_ids": ["S1"]}
    identity = {"window_id": "FULL", "candidate_id": "BRC_1", "start_frame": 1, "end_frame_exclusive": 2}
    checkpoint = tmp_path / "BRC_1.json"
    checkpoint.write_text(__import__("json").dumps({
        **identity, "candidate_identity": identity, "candidate_fingerprint": "gemini-fingerprint",
        "provider": "gemini", "model": "gemini-3.6-flash",
        "semantic_schema_version": SEMANTIC_SCHEMA_VERSION, "semantic_prompt_version": SEMANTIC_PROMPT_VERSION,
        "response": response,
    }))
    assert _semantic_checkpoint(checkpoint, candidate, DEFAULT_OPENAI_MODEL, "FULL", "openai-fingerprint", "openai") == response
