import json
from types import SimpleNamespace

from movie_broll.narrative_provider import GEMINI_RESPONSE_SCHEMA, GeminiNarrativeProvider


class FakeInteractions:
    def __init__(self): self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=json.dumps({"schema_version": "narrative_mapper_llm_v2", "chunk_summary_es": "Resumen.", "segments": []}),
            usage=SimpleNamespace(total_input_tokens=11, total_output_tokens=7, total_thought_tokens=2, total_cached_tokens=3, total_tokens=20),
        )


def test_gemini_36_uses_interactions_structured_response_and_usage_mapping():
    provider = GeminiNarrativeProvider("not-a-real-secret")
    interactions = FakeInteractions()
    provider.client = SimpleNamespace(interactions=interactions, models=SimpleNamespace(
        generate_content=lambda **_: (_ for _ in ()).throw(AssertionError("legacy API used"))))

    response = provider.generate("mapper instructions", {"chunk_id": "NCHUNK_0001", "cues": []})

    call = interactions.calls[0]
    assert provider.model == "gemini-3.6-flash"
    assert call["model"] == "gemini-3.6-flash"
    assert call["system_instruction"] == "mapper instructions"
    assert json.loads(call["input"])["chunk_id"] == "NCHUNK_0001"
    assert call["generation_config"] == {"thinking_level": "minimal"}
    assert "thinking_budget" not in call["generation_config"]
    assert not {"temperature", "top_p", "top_k"} & set(call["generation_config"])
    assert call["response_format"] == {"type": "text", "mime_type": "application/json", "schema": GEMINI_RESPONSE_SCHEMA}
    assert call.get("tools") is None
    assert response.usage == {"prompt_tokens": 11, "response_tokens": 7, "thinking_tokens": 2, "cached_tokens": 3, "total_tokens": 20}


def test_structured_schema_preserves_all_enums():
    segment = GEMINI_RESPONSE_SCHEMA["properties"]["segments"]["items"]
    for key in ("segment_type", "narrative_tone", "narrative_function", "context_dependency", "continuity_previous", "continuity_next"):
        assert segment["properties"][key]["enum"]
    assert segment["properties"]["possible_visual_opportunities"]["items"]["enum"]
