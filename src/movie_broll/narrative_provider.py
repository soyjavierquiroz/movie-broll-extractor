"""Small provider boundary for semantic-only SRT narrative mapping."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from .narrative import ENUMS, LLM_V2_SCHEMA_VERSION


@dataclass(frozen=True)
class ProviderResponse:
    data: dict[str, Any]
    usage: dict[str, int | None]


class NarrativeProvider(Protocol):
    identifier: str
    model: str
    def generate(self, prompt: str, chunk_input: dict[str, Any]) -> ProviderResponse: ...


def _enum(values: set[str]) -> dict[str, Any]:
    return {"type": "string", "enum": sorted(values)}


LLM_V2_SEGMENT_SCHEMA = {"type": "object", "properties": {
    "first_cue_id": {"type": "string", "pattern": "^SRT_[0-9]{6}$"},
    "last_cue_id": {"type": "string", "pattern": "^SRT_[0-9]{6}$"},
    "segment_type": _enum(ENUMS["segment_type"]),
    "narrative_summary_es": {"type": "string"},
    "narrative_tone": _enum(ENUMS["narrative_tone"]),
    "narrative_function": _enum(ENUMS["narrative_function"]),
    "context_dependency": _enum(ENUMS["context_dependency"]),
    "continuity_previous": _enum(ENUMS["continuity"]),
    "continuity_next": _enum(ENUMS["continuity"]),
    "possible_visual_opportunities": {"type": "array", "items": _enum(ENUMS["possible_visual_opportunities"])},
}, "required": ["first_cue_id", "last_cue_id", "segment_type", "narrative_summary_es", "narrative_tone", "narrative_function", "context_dependency", "continuity_previous", "continuity_next", "possible_visual_opportunities"]}

# This is the actual JSON Schema passed to Gemini, intentionally not the
# downstream canonical map schema. Every classifier is a real string enum.
GEMINI_RESPONSE_SCHEMA = {"type": "object", "properties": {
    "schema_version": {"type": "string", "enum": [LLM_V2_SCHEMA_VERSION]},
    "chunk_summary_es": {"type": "string"},
    "segments": {"type": "array", "items": LLM_V2_SEGMENT_SCHEMA},
}, "required": ["schema_version", "chunk_summary_es", "segments"]}


class GeminiNarrativeProvider:
    identifier = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-3.6-flash") -> None:
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, chunk_input: dict[str, Any]) -> ProviderResponse:
        """Map one text-only chunk through Gemini's Interactions API.

        Keep this request deliberately small: Gemini supplies semantic boundaries
        only; canonical timestamps and cue expansion remain local.
        """
        response = self.client.interactions.create(
            model=self.model,
            input=json.dumps(chunk_input, ensure_ascii=False, separators=(",", ":")),
            system_instruction=prompt,
            generation_config={"thinking_level": "MINIMAL"},
            response_format={"type": "text", "mime_type": "application/json", "schema": GEMINI_RESPONSE_SCHEMA},
        )
        data = json.loads(response.output_text)
        metadata = getattr(response, "usage", None)
        def attr(*names: str) -> int | None:
            for name in names:
                value = getattr(metadata, name, None) if metadata else None
                if isinstance(value, int): return value
            return None
        usage = {
            "prompt_tokens": attr("total_input_tokens"),
            "response_tokens": attr("total_output_tokens"),
            "thinking_tokens": attr("total_thought_tokens"),
            "cached_tokens": attr("total_cached_tokens"),
            "total_tokens": attr("total_tokens"),
        }
        return ProviderResponse(data=data, usage=usage)
