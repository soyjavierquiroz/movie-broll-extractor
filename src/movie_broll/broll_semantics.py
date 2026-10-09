"""Constrained multimodal semantic boundary for B-roll pilot candidates."""
from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from .gemini_credentials import GeminiCredential, GeminiCredentialSource

EMOTIONS = ["smiling", "crying", "tense_appearance", "surprised_appearance", "neutral", "unclear"]
POSITIONS = ["left", "center", "right", "multiple", "unclear"]
PRESENTATIONS = ["woman", "man", "unclear"]
AGE_GROUPS = ["young_adult", "adult", "middle_aged", "older_adult", "unclear"]
FRAME_ROLES = ["primary", "secondary", "background", "unclear"]
RELATIONSHIP_SOURCES = ["visual", "srt", "narrative", "combined"]
DECISIONS = ["KEEP", "REVIEW", "REJECT"]
FOCUS_SUBJECTS = ["woman", "man", "multiple_people", "action_region", "environment", "unclear"]
INTERACTION_REQUIREMENTS = ["none", "sequence", "simultaneous", "unclear"]
TARGET_BINDING_CONFIDENCE = ["high", "medium", "low", "unclear"]
DIRECTIVE_REQUIRED_FIELDS = ("shot_id", "focus_subject", "focus_reason", "preserve_secondary_subject", "interaction_requirement", "focus_position")
FOCUS_POSITION_DOMAINS = {
    "woman": ("left", "center", "right"),
    "man": ("left", "center", "right"),
    "multiple_people": ("multiple",),
}
KEEP_ACTION_EVIDENCE_TYPES = ["distinct_visible_action_or_reaction", "generic_presence_or_movement", "unclear"]
KEEP_REUSABLE_INTENT_TYPES = ["specific_visual_need", "generic_ambience_or_context", "unclear"]
DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_OPENAI_REASONING_EFFORT = "none"
DEFAULT_OPENAI_IMAGE_DETAIL = "low"
DEFAULT_OPENAI_PRICE_INPUT_PER_MILLION = 0.10
DEFAULT_OPENAI_PRICE_CACHED_INPUT_PER_MILLION = 0.01
DEFAULT_OPENAI_PRICE_OUTPUT_PER_MILLION = 0.50
DEFAULT_OPENAI_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_OPENAI_READ_TIMEOUT_SECONDS = 90.0
DEFAULT_OPENAI_WRITE_TIMEOUT_SECONDS = 30.0
DEFAULT_OPENAI_POOL_TIMEOUT_SECONDS = 10.0
DEFAULT_OPENAI_MAX_RETRIES = 2

SEMANTIC_SCHEMA: dict[str, Any] = {"type": "object", "properties": {
    "visual": {"type": "object", "properties": {
        "summary_es": {"type": "string"}, "subjects": {"type": "array", "items": {"type": "string"}},
        "objects": {"type": "array", "items": {"type": "string"}}, "actions": {"type": "array", "items": {"type": "string"}},
        "people_count_estimate": {"type": "string"}, "setting": {"type": "string"},
        "visible_interactions": {"type": "array", "items": {"type": "string"}},
        "visible_emotions": {"type": "array", "items": {"type": "string", "enum": EMOTIONS}},
        "people": {"type": "array", "items": {"type": "object", "properties": {
            "presentation": {"type": "string", "enum": PRESENTATIONS},
            "apparent_age_group": {"type": "string", "enum": AGE_GROUPS},
            "frame_role": {"type": "string", "enum": FRAME_ROLES},
            "position": {"type": "string", "enum": POSITIONS},
        }, "required": ["presentation", "apparent_age_group", "frame_role", "position"]}},
        "primary_subject_position": {"type": "string", "enum": POSITIONS}, "primary_subject_description": {"type": "string"}, "visual_focus": {"type": "string"},
        "shot_focus_plan": {"type": "array", "items": {"type": "object", "properties": {
            "shot_id": {"type": "string"}, "focus_subject": {"type": "string", "enum": FOCUS_SUBJECTS}, "focus_reason": {"type": "string"}, "preserve_secondary_subject": {"type": "boolean"}, "interaction_requirement": {"type": "string", "enum": INTERACTION_REQUIREMENTS}, "focus_position": {"type": "string", "enum": POSITIONS, "description": "woman/man: left|center|right; multiple_people: multiple; non-person focus: any listed position or unclear"},
            "target_person_ids": {"type": "array", "items": {"type": "string"}},
            "target_binding_confidence": {"type": "string", "enum": TARGET_BINDING_CONFIDENCE}
        }, "required": ["shot_id", "focus_subject", "focus_reason", "preserve_secondary_subject", "interaction_requirement", "focus_position", "target_person_ids", "target_binding_confidence"]}},
    }, "required": ["summary_es", "subjects", "objects", "actions", "people_count_estimate", "setting", "visible_interactions", "visible_emotions", "people", "primary_subject_position", "primary_subject_description", "visual_focus", "shot_focus_plan"]},
    "relationships": {"type": "array", "items": {"type": "object", "properties": {
        "type": {"type": "string"}, "source": {"type": "string", "enum": RELATIONSHIP_SOURCES},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }, "required": ["type", "source", "confidence"]}},
    "editorial": {"type": "object", "properties": {
        "standalone_meaning_es": {"type": "string"}, "reusable_broll": {"type": "boolean"},
        "action_or_moment_complete": {"type": "string", "enum": ["true", "false", "unclear"]},
        "use_cases_es": {"type": "array", "items": {"type": "string"}}, "negative_use_cases_es": {"type": "array", "items": {"type": "string"}},
        "search_terms_es": {"type": "array", "items": {"type": "string"}}, "editorial_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "keep_qualification": {"type": "object", "properties": {
            "action_evidence_type": {"type": "string", "enum": KEEP_ACTION_EVIDENCE_TYPES},
            "action_evidence_es": {"type": "string"},
            "reusable_intent_type": {"type": "string", "enum": KEEP_REUSABLE_INTENT_TYPES},
            "reusable_use_case_es": {"type": "string"},
        }, "required": ["action_evidence_type", "action_evidence_es", "reusable_intent_type", "reusable_use_case_es"]},
        "reason": {"type": "string"}, "decision": {"type": "string", "enum": DECISIONS},
    }, "required": ["standalone_meaning_es", "reusable_broll", "action_or_moment_complete", "use_cases_es", "negative_use_cases_es", "search_terms_es", "editorial_confidence", "keep_qualification", "reason", "decision"]},
}, "required": ["visual", "relationships", "editorial"]}


# This is deliberately a typed mirror of SEMANTIC_SCHEMA.  OpenAI 3.26.0's
# Responses ``parse`` API derives strict JSON Schema from this model and gives
# us ``response.output_parsed`` without concatenating output parts ourselves.
class _StrictResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _SemanticPerson(_StrictResponseModel):
    presentation: Literal["woman", "man", "unclear"]
    apparent_age_group: Literal["young_adult", "adult", "middle_aged", "older_adult", "unclear"]
    frame_role: Literal["primary", "secondary", "background", "unclear"]
    position: Literal["left", "center", "right", "multiple", "unclear"]


class _ShotFocusDirective(_StrictResponseModel):
    shot_id: str
    focus_subject: Literal["woman", "man", "multiple_people", "action_region", "environment", "unclear"]
    focus_reason: str
    preserve_secondary_subject: bool
    interaction_requirement: Literal["none", "sequence", "simultaneous", "unclear"]
    focus_position: Literal["left", "center", "right", "multiple", "unclear"]
    target_person_ids: list[str]
    target_binding_confidence: Literal["high", "medium", "low", "unclear"]


class _SemanticVisual(_StrictResponseModel):
    summary_es: str
    subjects: list[str]
    objects: list[str]
    actions: list[str]
    people_count_estimate: str
    setting: str
    visible_interactions: list[str]
    visible_emotions: list[Literal["smiling", "crying", "tense_appearance", "surprised_appearance", "neutral", "unclear"]]
    people: list[_SemanticPerson]
    primary_subject_position: Literal["left", "center", "right", "multiple", "unclear"]
    primary_subject_description: str
    visual_focus: str
    shot_focus_plan: list[_ShotFocusDirective]


class _SemanticRelationship(_StrictResponseModel):
    type: str
    source: Literal["visual", "srt", "narrative", "combined"]
    confidence: float


class _KeepQualification(_StrictResponseModel):
    action_evidence_type: Literal["distinct_visible_action_or_reaction", "generic_presence_or_movement", "unclear"]
    action_evidence_es: str
    reusable_intent_type: Literal["specific_visual_need", "generic_ambience_or_context", "unclear"]
    reusable_use_case_es: str


class _SemanticEditorial(_StrictResponseModel):
    standalone_meaning_es: str
    reusable_broll: bool
    action_or_moment_complete: Literal["true", "false", "unclear"]
    use_cases_es: list[str]
    negative_use_cases_es: list[str]
    search_terms_es: list[str]
    editorial_confidence: Literal["high", "medium", "low"]
    keep_qualification: _KeepQualification
    reason: str
    decision: Literal["KEEP", "REVIEW", "REJECT"]


class OpenAISemanticStructuredResult(_StrictResponseModel):
    visual: _SemanticVisual
    relationships: list[_SemanticRelationship]
    editorial: _SemanticEditorial

PROMPT = """You validate a movie B-roll candidate. The images are the only authority for visual facts. Spanish output. Describe every visually relevant person in visual.people using only the supplied approximate enums; do not identify people or exact ages. Return visual.shot_focus_plan, exactly one compact directive for every supplied technical shot ID. Use only woman, man, multiple_people, action_region, environment, or unclear. For focus_position, use left, center, or right for woman/man; use multiple only for multiple_people; for action_region, environment, or unclear use the visible position or unclear. The representative images are labelled SHOT ID/order; do not copy an event-level 'woman and man' description into every shot. Set interaction_requirement to sequence for dialogue, reaction, argument, flirting and shot/reverse-shot: each shot may focus only its relevant person. Set simultaneous only when both people/action must be visible in the same shot (hug, kiss, handshake, handoff, contact, joint object action). Prefer a clear visible interlocutor face over a large over-the-shoulder back/head silhouette. Relationships are separate evidence: visual source is allowed only for directly visible interaction labels such as talking_face_to_face, embracing, or arguing. Never infer couple, romantic_partner, married_couple, mother_daughter, father_daughter, siblings, or any family relation from images alone. Those specific relations require supporting SRT/narrative evidence and source narrative, srt, or combined. If evidence is insufficient, return []. SRT/narrative are synchronized context, not literal proof of what is visible. Be conservative about emotions and use only the provided enum.

The core editorial question is: could an editor plausibly search for this visible action, state, interaction, reaction, movement, environment, object, or composition and reuse it without knowing this movie's story? KEEP when the answer is yes and the visual is clear enough to reuse; REJECT when it is no. Reusable and searchable does NOT mean distinctive, dramatic, unusual, narratively important, or memorable. Ordinary, subtle, brief-but-visually-complete activities are valid B-roll: walking or walking away, entering/leaving, sitting/standing/resting in bed, writing/reading/working, using or holding a phone, driving or sitting at the wheel, waiting, looking through a window, opening a drawer/searching belongings, carrying food, eating/drinking/cooking, smelling or examining an object, shopping, listening/reacting, looking worried, smiling/crying/hesitating, comforting, touching an arm, hugging, arguing, serious discussion, a clear two-person interaction, an establishing environment, object detail, or a transitional movement. Do not reject a visible action merely because it is routine, generic, subtle, common, part of a conversation, or lacks narrative importance.

Self-contained means understandable enough to reuse as a visual asset; it does not require a beginning/middle/end or a mini-story. A clear visual state may be reusable even when no narrative action arc applies. Mark action_or_moment_complete true for a clear reusable action, state, interaction, reaction, movement, object detail, or composition; mark false only when an incomplete action has no reusable visual state. Keep it as metadata, not as a demand for dramatic narrative completion.

Conversation is not automatically valuable: REJECT generic alternating dialogue or talking-head coverage when it has no useful gesture, visible reaction, strong interaction, useful composition/state, or activity and depends on dialogue/context. KEEP can be appropriate when the visual itself shows listening/reaction, support, tension, argument, meaningful silence, physical interaction, clear emotion, useful two-person composition, or a recognizable activity. Judge the visual asset, never the dialogue importance.

Hard-reject credits, logos, title cards, production-specific text overlays, black/fade/transitions, corrupt/frozen/severely blurred content, unusable composition, visually meaningless presence, context-dependent fragments, redundant weaker duplicate coverage, and incomplete action with no reusable visual state. Use cases must be concrete visible actions/moments; negative use cases must prevent unsupported claims. For editorial.keep_qualification, copy action_evidence_es exactly from visual.actions or visual.visible_interactions and reusable_use_case_es exactly from use_cases_es. Use distinct_visible_action_or_reaction for a clear visible action or reaction, generic_presence_or_movement for an ordinary but visibly clear searchable activity/state/movement, and unclear only when there is no grounded visual evidence. Use specific_visual_need for a concrete search target and generic_ambience_or_context for a reusable environment/composition; neither generic enum automatically means REJECT. Do not use narrative or SRT to manufacture an action. Decide KEEP only if the candidate is visually clear, grounded, searchable, reusable B-roll. There is no quota and no gender preference."""

@dataclass(frozen=True)
class SemanticResponse:
    data: dict[str, Any]
    usage: dict[str, int | None]
    provider: str | None = None
    model: str | None = None
    attempts: int = 1
    provider_trace: tuple[dict[str, Any], ...] = ()

class SemanticProvider(Protocol):
    identifier: str
    model: str
    def generate(self, prompt: str, context: dict[str, Any], jpeg: bytes) -> SemanticResponse: ...

class GeminiBrollSemanticProvider:
    identifier = "gemini"
    def __init__(self, api_key: str, model: str = "gemini-3.6-flash", identifier: str = "gemini", response_schema: dict[str, Any] | None = None) -> None:
        from google import genai
        self.client = genai.Client(api_key=api_key); self.model = model; self.identifier = identifier; self.response_schema = response_schema or SEMANTIC_SCHEMA

    def generate(self, prompt: str, context: dict[str, Any], jpeg: bytes) -> SemanticResponse:
        # google-genai 2.22.0 Interactions accepts image content as base64 data.
        images = jpeg if isinstance(jpeg, list) else [jpeg]
        content = [{"type": "text", "text": json.dumps(context, ensure_ascii=False)}]
        for index, item in enumerate(images):
            if isinstance(jpeg, list):
                content.append({"type": "text", "text": "Event image " + str(index + 1) + ": " + context["events"][index]["event_id"]})
            content.append({"type": "image", "data": base64.b64encode(item).decode("ascii"), "mime_type": "image/jpeg"})
        response = self.client.interactions.create(model=self.model, input=[{"type": "user_input", "content": content}], system_instruction=prompt, generation_config={"thinking_level": "minimal"}, response_format={"type": "text", "mime_type": "application/json", "schema": self.response_schema})
        data = json.loads(response.output_text); usage = getattr(response, "usage", None)
        def value(*names: str) -> int | None:
            for name in names:
                item = getattr(usage, name, None) if usage else None
                if isinstance(item, int): return item
            return None
        return SemanticResponse(data, {"prompt_tokens": value("total_input_tokens"), "response_tokens": value("total_output_tokens"), "thinking_tokens": value("total_thought_tokens"), "cached_tokens": value("total_cached_tokens"), "total_tokens": value("total_tokens")})


def openai_structured_output_schema() -> dict[str, Any]:
    """Return the canonical schema in Responses strict-mode form.

    Responses requires closed objects and all properties declared required.  A
    canonical optional field is represented as nullable rather than omitted, so
    the response shape and local validator remain provider-neutral.
    """
    def convert(value: Any) -> Any:
        if isinstance(value, list):
            return [convert(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: convert(item) for key, item in value.items()}
        if result.get("type") == "object":
            properties = result.get("properties", {})
            original_required = set(result.get("required", []))
            converted: dict[str, Any] = {}
            for key, item in properties.items():
                # Strict Responses schemas have no omitted object fields.
                converted[key] = item if key in original_required else {
                    "anyOf": [item, {"type": "null"}],
                }
            result["properties"] = converted
            result["required"] = list(properties)
            result["additionalProperties"] = False
        return result
    return convert(SEMANTIC_SCHEMA)


def _env_float(environ: dict[str, str], name: str, default: float) -> float:
    try:
        return float(environ.get(name, default))
    except (TypeError, ValueError):
        return default


def estimate_openai_cost(usage: dict[str, Any], environ: dict[str, str] | None = None) -> float:
    """Estimate one recorded OpenAI response using centrally configurable rates."""
    env = os.environ if environ is None else environ
    input_tokens = int(usage.get("prompt_tokens") or 0)
    cached_tokens = int(usage.get("cached_tokens") or 0)
    output_tokens = int(usage.get("response_tokens") or 0)
    # Cached input is a subset of input.  Never double charge it at full price.
    uncached_tokens = max(0, input_tokens - cached_tokens)
    return (
        uncached_tokens * _env_float(env, "OPENAI_PRICE_INPUT_PER_MILLION", DEFAULT_OPENAI_PRICE_INPUT_PER_MILLION)
        + cached_tokens * _env_float(env, "OPENAI_PRICE_CACHED_INPUT_PER_MILLION", DEFAULT_OPENAI_PRICE_CACHED_INPUT_PER_MILLION)
        + output_tokens * _env_float(env, "OPENAI_PRICE_OUTPUT_PER_MILLION", DEFAULT_OPENAI_PRICE_OUTPUT_PER_MILLION)
    ) / 1_000_000


def redact_provider_error(value: Any, environ: dict[str, str] | None = None) -> str:
    """Make persisted provider diagnostics safe even for unusual SDK errors."""
    env = os.environ if environ is None else environ
    text = str(value)
    for secret_name in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
        secret = env.get(secret_name)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    # Covers numbered Gemini keys and normal OpenAI sk-* keys without trying to
    # serialize any request/header object into operational artifacts.
    for key, secret in env.items():
        if (key.startswith("GEMINI_API_KEY_") or key == "GEMINI_API_KEY_BACKUP") and secret:
            text = text.replace(secret, "[REDACTED]")
    return re.sub(r"\bsk-[A-Za-z0-9_-]+\b", "[REDACTED]", text)


class OpenAIProviderError(RuntimeError):
    """Safe provider error retaining retry metadata, never request credentials."""
    def __init__(self, error: Exception, attempts: int) -> None:
        self.error, self.attempts = error, attempts
        super().__init__(str(error))


class OpenAIStructuredOutputError(ValueError):
    """A response was received but its strict structured result was unusable."""

    def __init__(self, error: Exception, diagnostic: dict[str, Any] | None = None) -> None:
        self.error = error
        self.diagnostic = diagnostic or {}
        super().__init__(str(error))


def _safe_output_shape(response: Any) -> dict[str, Any]:
    """Return response-only diagnostics; never serialize input images or headers."""
    output = getattr(response, "output", None)
    item_types: list[str] = []
    text_parts: list[dict[str, Any]] = []
    if isinstance(output, list):
        for item in output:
            item_type = getattr(item, "type", None)
            item_types.append(str(item_type) if item_type is not None else type(item).__name__)
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for part in content:
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        text_parts.append({
                            "type": str(getattr(part, "type", type(part).__name__)),
                            "length": len(text), "prefix": text[:240], "suffix": text[-240:],
                        })
    return {
        "response_id": getattr(response, "id", None),
        "response_status": getattr(response, "status", None),
        "output_item_types": item_types,
        "output_text_parts": text_parts,
        "output_text_length": sum(part["length"] for part in text_parts),
        "usage": OpenAISemanticProvider._usage(response),
    }


class OpenAISemanticProvider:
    """Minimal Responses API adapter for the provider-neutral semantic request."""
    identifier = "openai"

    def __init__(
        self, api_key: str, model: str = DEFAULT_OPENAI_MODEL, *,
        reasoning_effort: str = DEFAULT_OPENAI_REASONING_EFFORT,
        image_detail: str = DEFAULT_OPENAI_IMAGE_DETAIL,
        response_model: type[BaseModel] = OpenAISemanticStructuredResult,
        client: Any = None, sleep: Callable[[float], None] = time.sleep,
        max_retries: int = DEFAULT_OPENAI_MAX_RETRIES,
        connect_timeout: float = DEFAULT_OPENAI_CONNECT_TIMEOUT_SECONDS,
        read_timeout: float = DEFAULT_OPENAI_READ_TIMEOUT_SECONDS,
        write_timeout: float = DEFAULT_OPENAI_WRITE_TIMEOUT_SECONDS,
        pool_timeout: float = DEFAULT_OPENAI_POOL_TIMEOUT_SECONDS,
        reporter: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        if image_detail not in {"low", "high", "auto"}:
            raise ValueError("OPENAI_IMAGE_DETAIL must be low, high, or auto")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.image_detail = image_detail
        self.response_model = response_model
        self.sleep = sleep
        self.max_retries = max(1, int(max_retries))
        self.reporter = reporter or (lambda _line: None)
        self.clock = clock
        self.timeout_config = {
            "connect": float(connect_timeout), "read": float(read_timeout),
            "write": float(write_timeout), "pool": float(pool_timeout),
        }
        if client is None:
            from openai import OpenAI
            # OpenAI SDK retries are deliberately disabled.  This adapter owns
            # the bounded, observable retry/backoff policy below.
            import httpx2
            client = OpenAI(
                api_key=api_key, max_retries=0,
                timeout=httpx2.Timeout(
                    connect=self.timeout_config["connect"], read=self.timeout_config["read"],
                    write=self.timeout_config["write"], pool=self.timeout_config["pool"],
                ),
            )
        self.client = client

    @staticmethod
    def _usage(response: Any) -> dict[str, int | None]:
        usage = getattr(response, "usage", None)
        def item(parent: Any, *names: str) -> int | None:
            for name in names:
                value = getattr(parent, name, None) if parent is not None else None
                if isinstance(value, int):
                    return value
            return None
        details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        return {
            "prompt_tokens": item(usage, "input_tokens", "prompt_tokens"),
            "cached_tokens": item(details, "cached_tokens") or 0,
            "response_tokens": item(usage, "output_tokens", "completion_tokens"),
            "thinking_tokens": item(output_details, "reasoning_tokens") or 0,
            "total_tokens": item(usage, "total_tokens"),
        }

    def _delay(self, error: Exception, retry_index: int) -> float:
        retry_after = _provider_retry_after(error)
        # A bounded 1/2/4 second backoff avoids hot retries while still making
        # transient provider failures self-healing in unattended production.
        return min(30.0, retry_after if retry_after is not None else 2.0 ** (retry_index - 1))

    def generate(self, prompt: str, context: dict[str, Any], jpeg: bytes) -> SemanticResponse:
        images = jpeg if isinstance(jpeg, list) else [jpeg]
        content = [{"type": "input_text", "text": json.dumps(context, ensure_ascii=False, separators=(",", ":"))}]
        for index, item in enumerate(images):
            if isinstance(jpeg, list):
                content.append({"type": "input_text", "text": "Event image " + str(index + 1) + ": " + context["events"][index]["event_id"]})
            content.append({"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(item).decode("ascii"), "detail": self.image_detail})
        request = {
            "model": self.model,
            "instructions": prompt,
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": self.reasoning_effort},
        }
        for attempt in range(1, self.max_retries + 1):
            started = self.clock()
            event_id = context.get("visual_event_id", "unknown")
            self.reporter(f"[openai] event={event_id} attempt={attempt} start")
            try:
                # Canonical OpenAI 3.26.0 strict Responses path.  It produces
                # one parsed model from a response output item, rather than
                # JSON-decoding the convenience output_text concatenation.
                response = self.client.responses.parse(**request, text_format=self.response_model)
                parsed = getattr(response, "output_parsed", None)
                if not isinstance(parsed, self.response_model):
                    raise OpenAIStructuredOutputError(
                        ValueError("OpenAI Responses result has no parsed strict output"),
                        _safe_output_shape(response),
                    )
                usage = self._usage(response)
                latency = self.clock() - started
                self.reporter(
                    f"[openai] event={event_id} attempt={attempt} complete latency={latency:.1f}s "
                    f"input={usage.get('prompt_tokens') or 0} output={usage.get('response_tokens') or 0} "
                    f"cost={estimate_openai_cost(usage):.6f}"
                )
                return SemanticResponse(
                    parsed.model_dump(mode="json"), usage, provider=self.identifier,
                    model=getattr(response, "model", None) or self.model, attempts=attempt,
                    provider_trace=({"provider": self.identifier, "model": self.model, "attempt": attempt, "status": "COMPLETE"},),
                )
            except Exception as error:
                if not isinstance(error, OpenAIStructuredOutputError) and _looks_like_structured_parse_error(error):
                    # Some SDK exceptions retain the raw response; when they
                    # do, preserve only its output shape/text edges.  Parsing
                    # failures without a retained response still get a safe
                    # diagnostic artifact with the parser error.
                    raw_response = getattr(error, "response", None)
                    error = OpenAIStructuredOutputError(
                        error, _safe_output_shape(raw_response) if raw_response is not None else None
                    )
                detail = classify_provider_error(error)
                latency = self.clock() - started
                action = "RETRY" if detail.get("retryable") and attempt < self.max_retries else "DEFER"
                self.reporter(
                    f"[openai] event={event_id} attempt={attempt} {detail.get('reason')} "
                    f"latency={latency:.1f}s action={action}"
                )
                if not detail.get("retryable") or attempt >= self.max_retries:
                    if detail.get("retryable"):
                        self.reporter(f"[openai] event={event_id} deferred reason={detail.get('reason')}")
                    raise OpenAIProviderError(error, attempt) from error
                self.sleep(self._delay(error, attempt))
        raise AssertionError("unreachable")


def build_openai_provider_from_env(
    model: str | None = None, *, environ: dict[str, str] | None = None,
    client: Any = None, sleep: Callable[[float], None] = time.sleep,
    reporter: Callable[[str], None] | None = None,
    response_model: type[BaseModel] = OpenAISemanticStructuredResult,
) -> OpenAISemanticProvider | None:
    env = os.environ if environ is None else environ
    key = env.get("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAISemanticProvider(
        key, model or env.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        reasoning_effort=env.get("OPENAI_REASONING_EFFORT", DEFAULT_OPENAI_REASONING_EFFORT),
        image_detail=env.get("OPENAI_IMAGE_DETAIL", DEFAULT_OPENAI_IMAGE_DETAIL),
        client=client, sleep=sleep, max_retries=int(env.get("OPENAI_MAX_RETRIES", str(DEFAULT_OPENAI_MAX_RETRIES))),
        connect_timeout=_env_float(env, "OPENAI_CONNECT_TIMEOUT_SECONDS", DEFAULT_OPENAI_CONNECT_TIMEOUT_SECONDS),
        read_timeout=_env_float(env, "OPENAI_READ_TIMEOUT_SECONDS", DEFAULT_OPENAI_READ_TIMEOUT_SECONDS),
        write_timeout=_env_float(env, "OPENAI_WRITE_TIMEOUT_SECONDS", DEFAULT_OPENAI_WRITE_TIMEOUT_SECONDS),
        pool_timeout=_env_float(env, "OPENAI_POOL_TIMEOUT_SECONDS", DEFAULT_OPENAI_POOL_TIMEOUT_SECONDS),
        reporter=reporter, response_model=response_model,
    )


def build_semantic_provider_from_env(
    model: str | None = None, *, reporter: Any = None, environ: dict[str, str] | None = None,
    env_file: Path | None = None, response_model: type[BaseModel] | None = None,
    response_schema: dict[str, Any] | None = None,
) -> SemanticProvider | None:
    """Select exactly the configured semantic provider; there is no fallback."""
    env = os.environ if environ is None else environ
    provider = env.get("SEMANTIC_PROVIDER", "openai").strip().lower()
    if provider == "openai":
        return build_openai_provider_from_env(model, environ=env, reporter=reporter,
                                              response_model=response_model or OpenAISemanticStructuredResult)
    if provider == "gemini":
        return build_gemini_provider_from_env(model or "gemini-3.6-flash", reporter=reporter, environ=env, env_file=env_file,
                                               response_schema=response_schema)
    raise ValueError("SEMANTIC_PROVIDER must be openai or gemini")


class GeminiProviderPoolError(RuntimeError):
    """All eligible Gemini providers failed for one semantic request."""

    def __init__(self, failures: list[dict[str, Any]], model: str) -> None:
        self.failures = [dict(x) for x in failures]
        self.model = model
        self.attempts = sum(
            1
            for x in self.failures
            if x.get("attempted", True)
        )
        self.providers_attempted = [
            x.get("provider")
            for x in self.failures
            if x.get("provider") and x.get("attempted", True)
        ]

        quota_only = bool(self.failures) and all(
            x.get("reason") == "quota_exceeded"
            for x in self.failures
        )
        auth_only = bool(self.failures) and all(
            x.get("reason") == "auth_error"
            for x in self.failures
        )

        self.quota_exhausted = quota_only

        if quota_only:
            self.reason = "quota_exceeded"
        elif auth_only:
            self.reason = "auth_error"
        else:
            self.reason = "provider_unavailable"

        # Auth is not retryable against the same credential, but the
        # production job itself is resume-safe after credentials are fixed.
        self.retryable = (
            auth_only
            or any(bool(x.get("retryable")) for x in self.failures)
        )

        retry_values = [
            float(x["retry_after_seconds"])
            for x in self.failures
            if x.get("retry_after_seconds") is not None
        ]
        self.retry_after_seconds = min(retry_values) if retry_values else None
        self.http_status = 429 if quota_only else next(
            (
                x.get("http_status")
                for x in reversed(self.failures)
                if x.get("http_status") is not None
            ),
            None,
        )

        super().__init__(
            "Gemini provider pool exhausted: "
            + "; ".join(
                f"{x.get('provider')}={x.get('reason')}"
                for x in self.failures
            )
        )


def _provider_http_status(error: Exception) -> int | None:
    if isinstance(error, OpenAIProviderError):
        error = error.error
    if isinstance(error, OpenAIStructuredOutputError):
        error = error.error
    for attr in ("status_code", "code"):
        value = getattr(error, attr, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value

    text = str(error)

    match = re.search(
        r"(?:error code|status(?: code)?)\s*[:=]?\s*(\d{3})",
        text,
        re.I,
    )
    if match:
        return int(match.group(1))

    for code in (429, 401, 403, 500, 502, 503, 504):
        if re.search(rf"(?<!\d){code}(?!\d)", text):
            return code

    return None


def _provider_retry_after(error: Exception) -> float | None:
    if isinstance(error, OpenAIProviderError):
        error = error.error
    if isinstance(error, OpenAIStructuredOutputError):
        error = error.error
    for attr in ("retry_after_seconds", "retry_after"):
        value = getattr(error, attr, None)
        if isinstance(value, (int, float)) and value >= 0:
            return float(value)

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            if value is not None:
                return max(0.0, float(value))
        except (TypeError, ValueError):
            pass

    match = re.search(
        r"retry(?:\s+after|\s+in)?\s*:?\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*s",
        str(error),
        re.I,
    )
    return float(match.group(1)) if match else None


def classify_provider_error(error: Exception) -> dict[str, Any]:
    attempts = error.attempts if isinstance(error, OpenAIProviderError) else None
    source_error = error.error if isinstance(error, OpenAIProviderError) else error
    if isinstance(source_error, OpenAIStructuredOutputError):
        return {
            "http_status": _provider_http_status(source_error),
            "reason": "structured_output_invalid",
            "retryable": True,
            "retry_after_seconds": None,
            "quota_exhausted": False,
            **({"attempts": attempts} if attempts is not None else {}),
        }
    if isinstance(error, GeminiProviderPoolError):
        failures = error.failures
        return {
            "provider": (
                failures[-1].get("provider")
                if failures
                else "gemini"
            ),
            "model": error.model,
            "http_status": error.http_status,
            "reason": error.reason,
            "retryable": error.retryable,
            "retry_after_seconds": error.retry_after_seconds,
            "quota_exhausted": error.quota_exhausted,
            "providers_attempted": error.providers_attempted,
            "attempts": error.attempts,
        }

    text = str(source_error).lower()
    status = _provider_http_status(source_error)
    retry_after = _provider_retry_after(source_error)

    quota = (
        status == 429 or "resource_exhausted" in text
    ) and any(
        token in text
        for token in ("quota", "exhaust", "free tier", "daily")
    )

    body = getattr(source_error, "body", None)
    body_error = body.get("error", body) if isinstance(body, dict) else {}
    code = getattr(source_error, "code", None) or (body_error.get("code") if isinstance(body_error, dict) else None)
    input_rejection = code in {"invalid_prompt", "content_policy_violation"} or (
        status == 400 and any(token in text for token in ("invalid_prompt", "content_policy_violation"))
    )
    if input_rejection:
        reason = "provider_input_rejection"
        retryable = False
    elif quota:
        reason = "quota_exceeded"
        retryable = True
    elif status == 429:
        reason = "rate_limited"
        retryable = True
    elif isinstance(source_error, TimeoutError) or "timeout" in text or "timed out" in text:
        reason = "timeout"
        retryable = True
    elif (
        "connection" in text
        or "temporar" in text
        or "unavailable" in text
        or (status is not None and 500 <= status <= 599)
    ):
        reason = "provider_unavailable"
        retryable = True
    elif status in {401, 403}:
        reason = "auth_error"
        retryable = False
    else:
        reason = "provider_error"
        retryable = False

    return {
        "http_status": status,
        "reason": reason,
        "error_code": code or ("invalid_prompt" if input_rejection and "invalid_prompt" in text else None),
        "retryable": retryable,
        "retry_after_seconds": retry_after,
        "quota_exhausted": quota,
        **({"attempts": attempts} if attempts is not None else {}),
    }


def _looks_like_structured_parse_error(error: Exception) -> bool:
    """Recognize SDK/Pydantic strict-output parsing errors without text payloads."""
    module = type(error).__module__.lower()
    name = type(error).__name__.lower()
    text = str(error).lower()
    return (
        "pydantic" in module or "validationerror" in name
        or "extra data" in text or "invalid json" in text
        or "json invalid" in text
    )


class GeminiProviderPool:
    """
    Round-robin primaries with independent cooldown and optional backup.

    identifier intentionally remains "gemini" so adding the pool does not
    alter existing semantic fingerprint identity.
    """

    identifier = "gemini"

    def __init__(
        self,
        primaries: list[SemanticProvider],
        backup: SemanticProvider | None = None,
        *,
        reporter: Any = None,
        clock: Any = None,
        default_cooldown_seconds: float = 30.0,
        credential_source: GeminiCredentialSource | None = None,
        provider_factory: Callable[[GeminiCredential], SemanticProvider] | None = None,
    ) -> None:
        if not primaries and backup is None:
            raise ValueError("GeminiProviderPool requires at least one provider")

        self.primaries = [self._member(provider) for provider in primaries]
        self.backup = (
            self._member(backup)
            if backup is not None
            else None
        )

        first = primaries[0] if primaries else backup
        self.model = first.model
        self.reporter = reporter
        self.clock = clock or time.monotonic
        self.default_cooldown_seconds = float(default_cooldown_seconds)
        self.cursor = 0
        self.credential_source = credential_source
        self.provider_factory = provider_factory

    @staticmethod
    def _member(provider: SemanticProvider, credential: str | None = None) -> dict[str, Any]:
        return {
            "provider": provider,
            "credential": credential,
            "cooldown_until": 0.0,
            "last_failure": None,
            "disabled": False,
        }

    def _refresh_credentials(self) -> None:
        """Apply a changed .env only before a new provider request starts."""
        if self.credential_source is None or self.provider_factory is None:
            return
        try:
            configured = self.credential_source.discover()
        except (OSError, ValueError):
            # A transient partial write must not discard a working pool.
            return
        previous = {
            member["credential"]: member
            for member in self.primaries + ([self.backup] if self.backup else [])
            if member.get("credential") is not None
        }

        def member_for(spec: GeminiCredential) -> dict[str, Any]:
            old = previous.get(spec.key)
            if old is not None and old["provider"].identifier == spec.identifier:
                return old
            member = self._member(self.provider_factory(spec), spec.key)
            if old is not None:
                for name in ("cooldown_until", "last_failure", "disabled"):
                    member[name] = old[name]
            return member

        primaries = [member_for(spec) for spec in configured.primaries]
        backup = member_for(configured.backup) if configured.backup else None
        changed = (
            [(x["provider"].identifier, x.get("credential")) for x in primaries]
            != [(x["provider"].identifier, x.get("credential")) for x in self.primaries]
            or (backup and (backup["provider"].identifier, backup.get("credential")))
            != (self.backup and (self.backup["provider"].identifier, self.backup.get("credential")))
        )
        self.primaries, self.backup = primaries, backup
        if self.primaries:
            self.cursor %= len(self.primaries)
        else:
            self.cursor = 0
        if changed:
            self._log(
                "[gemini-pool] credentials_refreshed "
                f"primaries={len(self.primaries)} backup={bool(self.backup)}"
            )

    def _log(self, message: str) -> None:
        if self.reporter is not None:
            self.reporter(message)

    def _try_member(
        self,
        member: dict[str, Any],
        prompt: str,
        context: dict[str, Any],
        jpeg: bytes,
        attempt: int,
    ) -> tuple[SemanticResponse | None, dict[str, Any]]:
        provider = member["provider"]

        try:
            response = provider.generate(prompt, context, jpeg)

            trace = {
                "provider": provider.identifier,
                "model": provider.model,
                "attempt": attempt,
                "status": "COMPLETE",
            }

            self._log(
                "[gemini-pool] "
                f"event={context.get('visual_event_id') or context.get('candidate_id', '?')} "
                f"provider={provider.identifier} "
                f"attempt={attempt} status=COMPLETE"
            )

            return response, trace

        except Exception as error:
            detail = classify_provider_error(error)
            detail.update(
                provider=provider.identifier,
                model=provider.model,
                attempt=attempt,
                attempted=True,
            )

            if detail["reason"] == "auth_error":
                member["disabled"] = True
                member["last_failure"] = dict(detail)

                self._log(
                    "[gemini-pool] "
                    f"event={context.get('visual_event_id') or context.get('candidate_id', '?')} "
                    f"provider={provider.identifier} "
                    f"attempt={attempt} "
                    f"status={detail.get('http_status')} "
                    "reason=auth_error "
                    "action=DISABLE"
                )

                return None, detail

            if not detail["retryable"]:
                raise

            cooldown = detail["retry_after_seconds"]
            if cooldown is None:
                cooldown = self.default_cooldown_seconds

            member["cooldown_until"] = self.clock() + max(0.0, float(cooldown))
            member["last_failure"] = dict(detail)

            self._log(
                "[gemini-pool] "
                f"event={context.get('visual_event_id') or context.get('candidate_id', '?')} "
                f"provider={provider.identifier} "
                f"attempt={attempt} "
                f"status={detail.get('http_status') or detail['reason']} "
                f"reason={detail['reason']} "
                f"retry_after={detail.get('retry_after_seconds')} "
                "action=COOLDOWN"
            )

            return None, detail

    def generate(
        self,
        prompt: str,
        context: dict[str, Any],
        jpeg: bytes,
    ) -> SemanticResponse:
        self._refresh_credentials()
        trace: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []

        count = len(self.primaries)
        now = self.clock()

        if count:
            order = [
                (self.cursor + offset) % count
                for offset in range(count)
            ]

            for index in order:
                member = self.primaries[index]

                if member.get("disabled"):
                    previous = member.get("last_failure")
                    if previous:
                        current = dict(previous)
                        current["attempted"] = False
                        failures.append(current)
                    continue

                if member["cooldown_until"] > now:
                    previous = member.get("last_failure")
                    if previous:
                        current = dict(previous)
                        current["attempted"] = False
                        current["retry_after_seconds"] = max(
                            0.0,
                            member["cooldown_until"] - now,
                        )
                        failures.append(current)
                    continue

                # Next request starts after this provider.
                self.cursor = (index + 1) % count

                response, detail = self._try_member(
                    member,
                    prompt,
                    context,
                    jpeg,
                    len(trace) + 1,
                )

                trace.append(detail)

                if response is not None:
                    return SemanticResponse(
                        response.data,
                        response.usage,
                        provider=response.provider or member["provider"].identifier,
                        model=response.model or member["provider"].model,
                        attempts=len(trace),
                        provider_trace=tuple(trace),
                    )

                failures.append(detail)

        if self.backup is not None:
            member = self.backup
            now = self.clock()

            if member.get("disabled"):
                previous = member.get("last_failure")
                if previous:
                    current = dict(previous)
                    current["attempted"] = False
                    failures.append(current)

            elif member["cooldown_until"] <= now:
                response, detail = self._try_member(
                    member,
                    prompt,
                    context,
                    jpeg,
                    len(trace) + 1,
                )

                trace.append(detail)

                if response is not None:
                    return SemanticResponse(
                        response.data,
                        response.usage,
                        provider=response.provider or member["provider"].identifier,
                        model=response.model or member["provider"].model,
                        attempts=len(trace),
                        provider_trace=tuple(trace),
                    )

                failures.append(detail)
            else:
                previous = member.get("last_failure")
                if previous:
                    current = dict(previous)
                    current["attempted"] = False
                    current["retry_after_seconds"] = max(
                        0.0,
                        member["cooldown_until"] - now,
                    )
                    failures.append(current)

        raise GeminiProviderPoolError(failures, self.model)


def build_gemini_provider_from_env(
    model: str = "gemini-3.6-flash",
    *,
    reporter: Any = None,
    environ: dict[str, str] | None = None,
    env_file: Path | None = None, response_schema: dict[str, Any] | None = None,
) -> SemanticProvider | None:
    source = GeminiCredentialSource(env_file, os.environ if environ is None else environ)
    configured = source.discover()
    if not configured.primaries and configured.backup is None:
        return None

    # Exact backward compatibility with the original one-key semantic path.
    if configured.legacy_only:
        if response_schema is None:
            return GeminiBrollSemanticProvider(configured.primaries[0].key, model)
        return GeminiBrollSemanticProvider(configured.primaries[0].key, model, response_schema=response_schema)

    def create(spec: GeminiCredential) -> SemanticProvider:
        if response_schema is None:
            return GeminiBrollSemanticProvider(spec.key, model, identifier=spec.identifier)
        return GeminiBrollSemanticProvider(spec.key, model, identifier=spec.identifier, response_schema=response_schema)

    pool = GeminiProviderPool(
        [create(spec) for spec in configured.primaries],
        create(configured.backup) if configured.backup else None,
        reporter=reporter,
        credential_source=source,
        provider_factory=create,
    )
    for member, spec in zip(pool.primaries, configured.primaries):
        member["credential"] = spec.key
    if pool.backup and configured.backup:
        pool.backup["credential"] = configured.backup.key
    return pool

def directive_validation_errors(directive: Any) -> list[dict[str, Any]]:
    """Return canonical, field-level syntax and cross-field errors for one directive."""
    if not isinstance(directive, dict):
        return [{
            'field': '$directive', 'value': directive,
            'reason': 'invalid_type', 'expected': 'object',
        }]

    errors: list[dict[str, Any]] = []

    def missing(field: str) -> None:
        errors.append({'field': field, 'reason': 'missing_required_field', 'expected': 'present'})

    for field in DIRECTIVE_REQUIRED_FIELDS:
        if field not in directive:
            missing(field)

    def typed_enum(field: str, values: list[str] | tuple[str, ...]) -> None:
        if field not in directive:
            return
        value = directive[field]
        if not isinstance(value, str):
            errors.append({'field': field, 'value': value, 'reason': 'invalid_type', 'expected': 'string'})
        elif value not in values:
            errors.append({'field': field, 'value': value, 'reason': 'unsupported_enum', 'allowed': list(values)})

    if 'shot_id' in directive and not isinstance(directive['shot_id'], str):
        errors.append({'field': 'shot_id', 'value': directive['shot_id'], 'reason': 'invalid_type', 'expected': 'string'})
    typed_enum('focus_subject', FOCUS_SUBJECTS)
    typed_enum('interaction_requirement', INTERACTION_REQUIREMENTS)
    typed_enum('focus_position', POSITIONS)
    if 'focus_reason' in directive and not isinstance(directive['focus_reason'], str):
        errors.append({'field': 'focus_reason', 'value': directive['focus_reason'], 'reason': 'invalid_type', 'expected': 'string'})
    if ('preserve_secondary_subject' in directive
            and not isinstance(directive['preserve_secondary_subject'], bool)):
        errors.append({'field': 'preserve_secondary_subject', 'value': directive['preserve_secondary_subject'], 'reason': 'invalid_type', 'expected': 'boolean'})

    subject = directive.get('focus_subject')
    position = directive.get('focus_position')
    allowed_positions = FOCUS_POSITION_DOMAINS.get(subject, tuple(POSITIONS))
    if (isinstance(position, str)
            and position in POSITIONS
            and position not in allowed_positions):
        errors.append({
            'field': 'focus_position', 'value': position,
            'reason': 'incompatible_focus_position',
            'allowed': list(allowed_positions),
            'depends_on': {'focus_subject': subject},
        })

    if 'target_person_ids' in directive:
        ids = directive['target_person_ids']
        if not isinstance(ids, list):
            errors.append({'field': 'target_person_ids', 'value': ids, 'reason': 'invalid_type', 'expected': 'array[string]'})
        elif any(not isinstance(value, str) for value in ids):
            errors.append({'field': 'target_person_ids', 'value': ids, 'reason': 'invalid_item_type', 'expected': 'array[string]'})
        elif len(ids) != len(set(ids)):
            errors.append({'field': 'target_person_ids', 'value': ids, 'reason': 'duplicate_person_id', 'expected': 'unique array[string]'})
    if 'target_binding_confidence' in directive:
        typed_enum('target_binding_confidence', TARGET_BINDING_CONFIDENCE)
    return errors


def validate_response(data: dict[str, Any]) -> list[str]:
    try: visual, editorial = data["visual"], data["editorial"]
    except (KeyError, TypeError): return ["missing visual or editorial"]
    required_visual = SEMANTIC_SCHEMA["properties"]["visual"]["required"]
    required_editorial = SEMANTIC_SCHEMA["properties"]["editorial"]["required"]
    errors = [f"visual missing {x}" for x in required_visual if x not in visual] + [f"editorial missing {x}" for x in required_editorial if x not in editorial]
    if visual.get("primary_subject_position") not in POSITIONS: errors.append("invalid subject position")
    plan=visual.get('shot_focus_plan', [])
    if plan and (not isinstance(plan,list) or any(directive_validation_errors(x) for x in plan)): errors.append('invalid shot focus plan')
    if isinstance(plan,list):
        for directive in plan:
            if not isinstance(directive,dict): continue
    for person in visual.get("people", []):
        if not isinstance(person, dict) or person.get("presentation") not in PRESENTATIONS: errors.append("invalid person presentation")
        elif person.get("apparent_age_group") not in AGE_GROUPS: errors.append("invalid person age group")
        elif person.get("frame_role") not in FRAME_ROLES: errors.append("invalid person frame role")
        elif person.get("position") not in POSITIONS: errors.append("invalid person position")
    for relationship in data.get("relationships", []):
        if not isinstance(relationship, dict) or relationship.get("source") not in RELATIONSHIP_SOURCES: errors.append("invalid relationship provenance"); continue
        if not isinstance(relationship.get("confidence"), (int, float)) or not 0 <= relationship["confidence"] <= 1: errors.append("invalid relationship confidence")
        if relationship.get("source") == "visual" and relationship.get("type") in {"romantic_partner", "married_couple", "mother_daughter", "father_daughter", "mother_son", "father_son", "siblings"}: errors.append("visual relationship overreach")
    if any(x not in EMOTIONS for x in visual.get("visible_emotions", [])): errors.append("invalid visible emotion")
    if editorial.get("decision") not in DECISIONS: errors.append("invalid decision")
    if editorial.get("action_or_moment_complete") not in {"true", "false", "unclear"}: errors.append("invalid completeness")
    forbidden = {"mother", "daughter", "husband", "wife", "couple", "therapist", "trauma", "jealousy", "betrayal"}
    visual_text = " ".join(str(v).lower() for key in ("summary_es", "subjects", "objects", "actions", "visible_interactions") for v in (visual.get(key, []) if isinstance(visual.get(key), list) else [visual.get(key, "")]))
    if any(term in visual_text for term in forbidden): errors.append("visual relationship or narrative hallucination")
    # A semantic KEEP is never accepted without grounded visual usefulness.
    # V9 deliberately permits ordinary/generic-but-searchable activity and
    # composition; "generic" is descriptive evidence, not a rejection rule.
    # The qualification strings must still be copied from structured visual
    # action and use-case fields so a KEEP cannot be manufactured from prose.
    if editorial.get("decision") == "KEEP":
        if not editorial.get("reusable_broll") or editorial.get("action_or_moment_complete") == "false": errors.append("KEEP lacks semantic usefulness")
        qualification=editorial.get("keep_qualification")
        if not isinstance(qualification, dict):
            errors.append("KEEP lacks editorial qualification")
        else:
            action_type=qualification.get("action_evidence_type")
            intent_type=qualification.get("reusable_intent_type")
            if action_type not in KEEP_ACTION_EVIDENCE_TYPES: errors.append("invalid KEEP action evidence type")
            if intent_type not in KEEP_REUSABLE_INTENT_TYPES: errors.append("invalid KEEP reusable intent type")
            if action_type == "unclear": errors.append("KEEP lacks grounded visible action or state")
            if intent_type == "unclear": errors.append("KEEP lacks searchable reusable use case")

            def normalized(value: Any) -> str:
                return value.strip().casefold() if isinstance(value, str) else ""

            action_evidence=normalized(qualification.get("action_evidence_es"))
            visual_actions={normalized(value) for value in [*visual.get("actions", []), *visual.get("visible_interactions", [])] if normalized(value)}
            if not action_evidence or action_evidence not in visual_actions: errors.append("KEEP action evidence is not grounded in visible structured action")
            use_case_evidence=normalized(qualification.get("reusable_use_case_es"))
            use_cases={normalized(value) for value in editorial.get("use_cases_es", []) if normalized(value)}
            if not use_case_evidence or use_case_evidence not in use_cases: errors.append("KEEP reusable use case is not grounded in structured use cases")
    return errors
