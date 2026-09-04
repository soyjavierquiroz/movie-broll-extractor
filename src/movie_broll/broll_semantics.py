"""Constrained multimodal semantic boundary for B-roll pilot candidates."""
from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

EMOTIONS = ["smiling", "crying", "tense_appearance", "surprised_appearance", "neutral", "unclear"]
POSITIONS = ["left", "center", "right", "multiple", "unclear"]
PRESENTATIONS = ["woman", "man", "unclear"]
AGE_GROUPS = ["young_adult", "adult", "middle_aged", "older_adult", "unclear"]
FRAME_ROLES = ["primary", "secondary", "background", "unclear"]
RELATIONSHIP_SOURCES = ["visual", "srt", "narrative", "combined"]
DECISIONS = ["KEEP", "REVIEW", "REJECT"]
FOCUS_SUBJECTS = ["woman", "man", "multiple_people", "action_region", "environment", "unclear"]
INTERACTION_REQUIREMENTS = ["none", "sequence", "simultaneous", "unclear"]

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
            "shot_id": {"type": "string"}, "focus_subject": {"type": "string", "enum": FOCUS_SUBJECTS}, "focus_reason": {"type": "string"}, "preserve_secondary_subject": {"type": "boolean"}, "interaction_requirement": {"type": "string", "enum": INTERACTION_REQUIREMENTS}, "focus_position": {"type": "string", "enum": POSITIONS}
        }, "required": ["shot_id", "focus_subject", "focus_reason", "preserve_secondary_subject", "interaction_requirement", "focus_position"]}},
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
        "reason": {"type": "string"}, "decision": {"type": "string", "enum": DECISIONS},
    }, "required": ["standalone_meaning_es", "reusable_broll", "action_or_moment_complete", "use_cases_es", "negative_use_cases_es", "search_terms_es", "editorial_confidence", "reason", "decision"]},
}, "required": ["visual", "relationships", "editorial"]}

PROMPT = """You validate a movie B-roll candidate. The images are the only authority for visual facts. Spanish output. Describe every visually relevant person in visual.people using only the supplied approximate enums; do not identify people or exact ages. Return visual.shot_focus_plan, exactly one compact directive for every supplied technical shot ID. Use only woman, man, multiple_people, action_region, environment, or unclear. The representative images are labelled SHOT ID/order; do not copy an event-level 'woman and man' description into every shot. Set interaction_requirement to sequence for dialogue, reaction, argument, flirting and shot/reverse-shot: each shot may focus only its relevant person. Set simultaneous only when both people/action must be visible in the same shot (hug, kiss, handshake, handoff, contact, joint object action). Prefer a clear visible interlocutor face over a large over-the-shoulder back/head silhouette. Relationships are separate evidence: visual source is allowed only for directly visible interaction labels such as talking_face_to_face, embracing, or arguing. Never infer couple, romantic_partner, married_couple, mother_daughter, father_daughter, siblings, or any family relation from images alone. Those specific relations require supporting SRT/narrative evidence and source narrative, srt, or combined. If evidence is insufficient, return []. SRT/narrative are synchronized context, not literal proof of what is visible. Be conservative about emotions and use only the provided enum. A shot/reverse-shot sequence can be KEEP when it is a coherent complete mini-event with standalone reusable meaning; do not reject it merely because it has multiple shots or people. Reject incomplete dialogue-coverage fragments. Use cases must be concrete visible actions/moments; negative use cases must prevent unsupported claims. Decide KEEP only if the candidate is visually clear, standalone reusable B-roll and has a complete action/moment; there is no quota and no gender preference."""

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
    def __init__(self, api_key: str, model: str = "gemini-3.6-flash", identifier: str = "gemini") -> None:
        from google import genai
        self.client = genai.Client(api_key=api_key); self.model = model; self.identifier = identifier

    def generate(self, prompt: str, context: dict[str, Any], jpeg: bytes) -> SemanticResponse:
        # google-genai 2.22.0 Interactions accepts image content as base64 data.
        content = [{"type": "text", "text": json.dumps(context, ensure_ascii=False)}, {"type": "image", "data": base64.b64encode(jpeg).decode("ascii"), "mime_type": "image/jpeg"}]
        response = self.client.interactions.create(model=self.model, input=[{"type": "user_input", "content": content}], system_instruction=prompt, generation_config={"thinking_level": "minimal"}, response_format={"type": "text", "mime_type": "application/json", "schema": SEMANTIC_SCHEMA})
        data = json.loads(response.output_text); usage = getattr(response, "usage", None)
        def value(*names: str) -> int | None:
            for name in names:
                item = getattr(usage, name, None) if usage else None
                if isinstance(item, int): return item
            return None
        return SemanticResponse(data, {"prompt_tokens": value("total_input_tokens"), "response_tokens": value("total_output_tokens"), "thinking_tokens": value("total_thought_tokens"), "cached_tokens": value("total_cached_tokens"), "total_tokens": value("total_tokens")})


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
    for attr in ("retry_after_seconds", "retry_after"):
        value = getattr(error, attr, None)
        if isinstance(value, (int, float)) and value >= 0:
            return float(value)

    match = re.search(
        r"retry(?:\s+after|\s+in)?\s*:?\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*s",
        str(error),
        re.I,
    )
    return float(match.group(1)) if match else None


def classify_provider_error(error: Exception) -> dict[str, Any]:
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

    text = str(error).lower()
    status = _provider_http_status(error)
    retry_after = _provider_retry_after(error)

    quota = (
        status == 429 or "resource_exhausted" in text
    ) and any(
        token in text
        for token in ("quota", "exhaust", "free tier", "daily")
    )

    if quota:
        reason = "quota_exceeded"
        retryable = True
    elif status == 429:
        reason = "rate_limited"
        retryable = True
    elif "timeout" in text or "timed out" in text:
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
        "retryable": retryable,
        "retry_after_seconds": retry_after,
        "quota_exhausted": quota,
    }


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
    ) -> None:
        if not primaries and backup is None:
            raise ValueError("GeminiProviderPool requires at least one provider")

        self.primaries = [
            {
                "provider": provider,
                "cooldown_until": 0.0,
                "last_failure": None,
                "disabled": False,
            }
            for provider in primaries
        ]
        self.backup = (
            {
                "provider": backup,
                "cooldown_until": 0.0,
                "last_failure": None,
                "disabled": False,
            }
            if backup is not None
            else None
        )

        first = primaries[0] if primaries else backup
        self.model = first.model
        self.reporter = reporter
        self.clock = clock or time.monotonic
        self.default_cooldown_seconds = float(default_cooldown_seconds)
        self.cursor = 0

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
) -> SemanticProvider | None:
    env = os.environ if environ is None else environ

    primary_specs = [
        (f"gemini-primary-{index}", env.get(f"GEMINI_API_KEY_{index}"))
        for index in (1, 2, 3)
    ]
    primary_specs = [
        (name, key)
        for name, key in primary_specs
        if key
    ]

    backup_key = env.get("GEMINI_API_KEY_BACKUP")
    legacy_key = env.get("GEMINI_API_KEY")

    # Exact backward compatibility with the old configuration.
    if not primary_specs and not backup_key:
        return (
            GeminiBrollSemanticProvider(legacy_key, model)
            if legacy_key
            else None
        )

    # If a backup is configured but numbered primaries are not,
    # the legacy key can remain the primary.
    if not primary_specs and legacy_key:
        primary_specs = [
            ("gemini-primary-1", legacy_key)
        ]

    primaries = [
        GeminiBrollSemanticProvider(
            key,
            model,
            identifier=name,
        )
        for name, key in primary_specs
    ]

    backup = (
        GeminiBrollSemanticProvider(
            backup_key,
            model,
            identifier="gemini-backup",
        )
        if backup_key
        else None
    )

    return GeminiProviderPool(
        primaries,
        backup,
        reporter=reporter,
    )

def validate_response(data: dict[str, Any]) -> list[str]:
    try: visual, editorial = data["visual"], data["editorial"]
    except (KeyError, TypeError): return ["missing visual or editorial"]
    required_visual = SEMANTIC_SCHEMA["properties"]["visual"]["required"]
    required_editorial = SEMANTIC_SCHEMA["properties"]["editorial"]["required"]
    errors = [f"visual missing {x}" for x in required_visual if x not in visual] + [f"editorial missing {x}" for x in required_editorial if x not in editorial]
    if visual.get("primary_subject_position") not in POSITIONS: errors.append("invalid subject position")
    plan=visual.get('shot_focus_plan', [])
    if plan and (not isinstance(plan,list) or any(not isinstance(x,dict) or x.get('focus_subject') not in FOCUS_SUBJECTS or x.get('interaction_requirement') not in INTERACTION_REQUIREMENTS or x.get('focus_position') not in POSITIONS or not all(k in x for k in ('shot_id','focus_reason','preserve_secondary_subject')) for x in plan)): errors.append('invalid shot focus plan')
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
    # A semantic KEEP is never accepted without the two explicit usefulness gates.
    if editorial.get("decision") == "KEEP" and (not editorial.get("reusable_broll") or editorial.get("action_or_moment_complete") != "true"): errors.append("KEEP lacks semantic usefulness")
    return errors
