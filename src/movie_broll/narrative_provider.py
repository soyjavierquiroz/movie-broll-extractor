"""Small provider boundary for SRT-only narrative mapping."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ProviderResponse:
    data: dict[str, Any]
    usage: dict[str, int | None]


class NarrativeProvider(Protocol):
    identifier: str
    model: str

    def generate(self, prompt: str, chunk_input: dict[str, Any]) -> ProviderResponse: ...


# Gemini's response schema is deliberately a compatible subset.  The local
# Phase 2A validator remains the authoritative full contract.
_ASSERTION = {"type": "object", "properties": {"value": {}, "source": {"type": "string"}, "confidence": {"type": "number"}}, "required": ["value", "source", "confidence"]}
_SEGMENT = {"type": "object", "properties": {
    "segment_id": {"type": "string"}, "start_seconds": {"type": "number"}, "end_seconds": {"type": "number"},
    "cue_ids": {"type": "array", "items": {"type": "string"}}, "segment_type": _ASSERTION,
    "narrative_summary": _ASSERTION, "dialogue_density": _ASSERTION, "narrative_tone": _ASSERTION,
    "narrative_function": _ASSERTION, "continuity": {"type": "object", "properties": {"previous": {"type": "string"}, "next": {"type": "string"}}, "required": ["previous", "next"]},
    "possible_visual_opportunities": {"type": "array", "items": _ASSERTION}, "context_dependency": _ASSERTION,
    "boundary": {"type": "object", "properties": {"start_confidence": {"type": "number"}, "end_confidence": {"type": "number"}}, "required": ["start_confidence", "end_confidence"]},
}, "required": ["segment_id", "start_seconds", "end_seconds", "cue_ids", "segment_type", "narrative_summary", "dialogue_density", "narrative_tone", "narrative_function", "continuity", "possible_visual_opportunities", "context_dependency", "boundary"]}
GEMINI_RESPONSE_SCHEMA = {"type": "object", "properties": {
    "schema_version": {"type": "string"}, "movie_id": {"type": "string"},
    "chunk": {"type": "object", "properties": {"chunk_id": {"type": "string"}, "start_seconds": {"type": "number"}, "end_seconds": {"type": "number"}}, "required": ["chunk_id", "start_seconds", "end_seconds"]},
    "source": {"type": "object", "properties": {"type": {"type": "string"}, "literal_transcription": {"type": "boolean"}}, "required": ["type", "literal_transcription"]},
    "chunk_summary": _ASSERTION, "segments": {"type": "array", "items": _SEGMENT},
}, "required": ["schema_version", "movie_id", "chunk", "source", "chunk_summary", "segments"]}


class GeminiNarrativeProvider:
    identifier = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, chunk_input: dict[str, Any]) -> ProviderResponse:
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model,
            contents=json.dumps(chunk_input, ensure_ascii=False, separators=(",", ":")),
            config=types.GenerateContentConfig(system_instruction=prompt, temperature=0.1,
                response_mime_type="application/json", response_json_schema=GEMINI_RESPONSE_SCHEMA,
                thinking_config=types.ThinkingConfig(thinking_budget=0)),
        )
        data = response.parsed if isinstance(response.parsed, dict) else json.loads(response.text)
        metadata = response.usage_metadata
        def attr(*names: str) -> int | None:
            for name in names:
                value = getattr(metadata, name, None) if metadata else None
                if isinstance(value, int): return value
            return None
        usage = {"prompt_tokens": attr("prompt_token_count"), "response_tokens": attr("candidates_token_count"), "thinking_tokens": attr("thoughts_token_count"), "cached_tokens": attr("cached_content_token_count")}
        usage["total_tokens"] = attr("total_token_count") if metadata else None
        return ProviderResponse(data=data, usage=usage)
