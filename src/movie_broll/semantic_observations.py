"""Immutable semantic observations and deterministic local B-roll policy.

This module is deliberately independent of the legacy V8/V9 contracts.  A
semantic observation says what was seen; ``broll_policy_v1`` is the only place
that turns that evidence into KEEP/REJECT/REVIEW.  The split lets operators
replay policy or finalization changes without paying for another observation.
"""
from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .broll_semantics import estimate_openai_cost
from .processing_ledger import fingerprint
from .utils import sha256_file, write_json


OBSERVATION_SCHEMA_VERSION = "semantic_observation_v1"
OBSERVATION_PROMPT_VERSION = "semantic_observation_prompt_v1"
POLICY_VERSION = "broll_policy_v1"
OBSERVATION_ROOT = ("semantic_observations", "v1")

VISUAL_UTILITY_KINDS = frozenset({
    "concrete_action", "useful_state", "clear_reaction", "physical_interaction",
    "movement", "object_activity", "environment", "strong_nonverbal_interaction",
    "generic_dialogue_only", "generic_presence_only", "unclear",
})
CONVERSATION_SIGNALS = frozenset({
    "none", "generic_dialogue_only", "clear_reaction", "strong_body_language",
    "physical_interaction", "concrete_activity", "useful_environment_or_composition",
    "unclear",
})
USEFUL_UTILITY_KINDS = frozenset({
    "concrete_action", "useful_state", "clear_reaction", "physical_interaction",
    "movement", "object_activity", "environment", "strong_nonverbal_interaction",
})
USEFUL_CONVERSATION_SIGNALS = frozenset({
    "clear_reaction", "strong_body_language", "physical_interaction",
    "concrete_activity", "useful_environment_or_composition",
})
HARD_REJECT_FLAGS = (
    "title_card", "credits", "logo", "dominant_text", "black_or_empty",
    "corrupt_or_unusable",
)

# This is intentionally observation-only.  No decision/reusable_broll/policy
# field exists in the response shape.
SEMANTIC_OBSERVATION_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "properties": {
        "event_id": {"type": "string"},
        "represented_shot_ids": {"type": "array", "items": {"type": "string"}},
        "people_count": {"type": "string", "enum": ["0", "1", "2_plus", "unclear"]},
        "visible_person_ids": {"type": "array", "items": {"type": "string"}},
        "action_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "object_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "visible_states": {"type": "array", "items": {"type": "string"}},
        "movement": {"type": "string", "enum": ["none", "present", "unclear"]},
        "physical_interactions": {"type": "array", "items": {"type": "string"}},
        "visible_reactions": {"type": "array", "items": {"type": "string"}},
        "conversation_present": {"type": "string", "enum": ["true", "false", "unclear"]},
        "conversation_visual_signal": {"type": "string", "enum": sorted(CONVERSATION_SIGNALS)},
        "visual_utility_kind": {"type": "string", "enum": sorted(VISUAL_UTILITY_KINDS)},
        "action_or_moment_complete": {"type": "string", "enum": ["true", "false", "unclear"]},
        "context_dependency": {"type": "string", "enum": ["low", "medium", "high", "unclear"]},
        "technical_observations": {"type": "object", "additionalProperties": False,
            "properties": {flag: {"type": "boolean"} for flag in HARD_REJECT_FLAGS},
            "required": list(HARD_REJECT_FLAGS)},
        "shot_focus_plan": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["event_id", "represented_shot_ids", "people_count", "visible_person_ids",
                 "action_evidence_ids", "object_evidence_ids", "visible_states", "movement",
                 "physical_interactions", "visible_reactions", "conversation_present",
                 "conversation_visual_signal", "visual_utility_kind", "action_or_moment_complete",
                 "context_dependency", "technical_observations", "shot_focus_plan"],
}

OBSERVATION_PROMPT = """Describe only what is visibly present in this Visual Event.
Return semantic_observation_v1.  Do not return KEEP, REJECT, REVIEW,
reusable_broll, editorial recommendations, rankings, or policy advice.  Bind
actions, objects, people, and focus targets only by the supplied canonical
evidence IDs.  Human-readable labels are metadata, never IDs.  When evidence
is absent, use the specified unclear/empty value rather than inventing it."""


class _StrictObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _TechnicalObservations(_StrictObservation):
    title_card: bool; credits: bool; logo: bool; dominant_text: bool
    black_or_empty: bool; corrupt_or_unusable: bool


class _FocusDirective(_StrictObservation):
    shot_id: str
    focus_subject: Literal["woman", "man", "multiple_people", "action_region", "environment", "unclear"]
    focus_reason: str
    preserve_secondary_subject: bool
    interaction_requirement: Literal["none", "sequence", "simultaneous", "unclear"]
    focus_position: Literal["left", "center", "right", "multiple", "unclear"]
    target_person_ids: list[str]
    target_binding_confidence: Literal["high", "medium", "low", "unclear"]


class OpenAISemanticObservationResult(_StrictObservation):
    """Strict provider model for the observation-only contract."""
    event_id: str; represented_shot_ids: list[str]
    people_count: Literal["0", "1", "2_plus", "unclear"]
    visible_person_ids: list[str]; action_evidence_ids: list[str]; object_evidence_ids: list[str]
    visible_states: list[str]; movement: Literal["none", "present", "unclear"]
    physical_interactions: list[str]; visible_reactions: list[str]
    conversation_present: Literal["true", "false", "unclear"]
    conversation_visual_signal: Literal["none", "generic_dialogue_only", "clear_reaction", "strong_body_language", "physical_interaction", "concrete_activity", "useful_environment_or_composition", "unclear"]
    visual_utility_kind: Literal["concrete_action", "useful_state", "clear_reaction", "physical_interaction", "movement", "object_activity", "environment", "strong_nonverbal_interaction", "generic_dialogue_only", "generic_presence_only", "unclear"]
    action_or_moment_complete: Literal["true", "false", "unclear"]
    context_dependency: Literal["low", "medium", "high", "unclear"]
    technical_observations: _TechnicalObservations
    shot_focus_plan: list[_FocusDirective]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def run_dir(input_dir: Path) -> Path:
    return input_dir.resolve().parents[1] / "runs" / input_dir.name


def observation_dir(run: Path) -> Path:
    return run.joinpath(*OBSERVATION_ROOT)


def observation_path(run: Path, event_id: str) -> Path:
    return observation_dir(run) / "events" / f"{event_id}.json"


def _active_picture_identity(active_picture: dict[str, Any] | None) -> dict[str, Any]:
    source = active_picture or {}
    return {key: source.get(key) for key in ("x", "y", "width", "height", "source_width",
                                               "source_height", "structural_bars", "detection_profile")}


def canonical_evidence_catalog(event: dict[str, Any], contact_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Expose deterministic IDs; labels are display metadata, never bindings."""
    event_id = str(event["visual_event_id"])
    people: list[dict[str, Any]] = []
    for shot in (contact_evidence or {}).get("technical_shots", []):
        if not isinstance(shot, dict):
            continue
        shot_id = shot.get("shot_id")
        for person in shot.get("candidates", []):
            if isinstance(person, dict) and isinstance(person.get("person_id"), str):
                people.append({"id": f"{shot_id}:{person['person_id']}", "shot_id": shot_id,
                               "canonical_person_id": person["person_id"]})
    # Deterministic scene analysis may attach action/object records.  Do not
    # turn provider prose into a canonical fact for a new observation.
    def entries(names: tuple[str, ...], prefix: str) -> list[dict[str, Any]]:
        raw: list[Any] = []
        for name in names:
            value = event.get(name, [])
            raw.extend(value if isinstance(value, list) else [])
        result = []
        for ordinal, value in enumerate(raw, 1):
            if isinstance(value, dict):
                supplied = value.get("id") or value.get("evidence_id")
                label = value.get("canonical_label") or value.get("label")
            else:
                supplied, label = None, value
            if isinstance(supplied, str) and supplied:
                identifier = supplied
            else:
                # A deterministic upstream label is metadata only; the ID is
                # stable and avoids prose-to-prose equality validation.
                identifier = f"{event_id}:{prefix}_{ordinal:02d}"
            result.append({"id": identifier, "canonical_label": label if isinstance(label, str) else None})
        return result
    return {"people": people, "actions": entries(("visual_actions", "action_evidence"), "ACTION"),
            "objects": entries(("object_evidence", "visual_objects"), "OBJECT")}


def observation_fingerprint(*, movie_sha256: str, event: dict[str, Any], contact_sheet_sha256: str | None,
                            active_picture: dict[str, Any] | None, narrative_context_fingerprint: str | None,
                            provider_compatibility: dict[str, Any] | None = None,
                            evidence_profile: str | None = None, input_evidence: dict[str, Any] | None = None) -> str:
    """Identity of pixels/evidence/schema only; deliberately excludes policy."""
    represented = [{key: shot.get(key) for key in ("shot_id", "start_frame", "end_frame_exclusive",
                                                    "start_seconds", "end_seconds")}
                   for shot in event.get("technical_shots", []) if isinstance(shot, dict)]
    if not represented:
        represented = [{"shot_id": shot_id} for shot_id in event.get("source_shot_ids", [])]
    identity = {"source_movie_sha256": movie_sha256, "visual_event_id": event.get("visual_event_id"),
                        "range": [event.get("start_frame"), event.get("end_frame_exclusive")],
                        "represented_technical_shots": represented,
                        "contact_sheet_sha256": contact_sheet_sha256,
                        "active_picture": _active_picture_identity(active_picture),
                        "narrative_context_fingerprint": narrative_context_fingerprint,
                        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
                        "observation_prompt_version": OBSERVATION_PROMPT_VERSION,
                        "provider_output_compatibility": provider_compatibility or {}}
    if evidence_profile is not None:
        identity.update(evidence_profile=evidence_profile, input_evidence=input_evidence or {})
    return fingerprint(identity)


def effective_observation(run: Path, event: dict[str, Any]) -> dict[str, Any]:
    """Read a valid enrichment preferentially; never replace legacy provenance."""
    from .temporal_semantics import latest_record, policy_projection
    temporal = latest_record(run, event)
    if temporal is not None:
        return policy_projection(temporal)
    return _read(observation_path(run, event["visual_event_id"]))


def validate_observation(observation: dict[str, Any], event: dict[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    if observation.get("schema_version") != OBSERVATION_SCHEMA_VERSION: errors.append("schema_version")
    body = observation.get("observation")
    if not isinstance(body, dict): return errors + ["observation"]
    for key in SEMANTIC_OBSERVATION_SCHEMA["required"]:
        if key not in body: errors.append(f"missing:{key}")
    if body.get("visual_utility_kind") not in VISUAL_UTILITY_KINDS: errors.append("visual_utility_kind")
    if body.get("conversation_visual_signal") not in CONVERSATION_SIGNALS: errors.append("conversation_visual_signal")
    technical = body.get("technical_observations")
    if not isinstance(technical, dict) or any(not isinstance(technical.get(flag), bool) for flag in HARD_REJECT_FLAGS): errors.append("technical_observations")
    if event is not None and body.get("event_id") != event.get("visual_event_id"): errors.append("event_id")
    return errors


def persist_observation(run: Path, event: dict[str, Any], body: dict[str, Any], *, observation_fingerprint_value: str,
                        evidence_catalog: dict[str, Any], provider_provenance: dict[str, Any]) -> dict[str, Any]:
    """Atomically persist one immutable observation after local validation."""
    record = {"schema_version": OBSERVATION_SCHEMA_VERSION, "event_id": event["visual_event_id"],
              "observation_fingerprint": observation_fingerprint_value, "observation": body,
              "evidence_catalog": evidence_catalog, "provider_provenance": provider_provenance,
              "created_at": _utc()}
    errors = validate_observation(record, event)
    if errors:
        raise ValueError("invalid semantic observation: " + ", ".join(errors))
    destination = observation_path(run, event["visual_event_id"])
    if destination.is_file():
        existing = _read(destination)
        if existing.get("observation_fingerprint") != observation_fingerprint_value:
            raise RuntimeError("immutable observation fingerprint conflict; use explicit observe --refresh")
        return existing
    write_json(destination, record)
    return record


def observe_missing_events(input_dir: Path, events: list[dict[str, Any]], *, movie_sha256: str,
                           provider: Any, make_contact_sheet: Callable[[dict[str, Any], dict[str, Any]], bytes],
                           make_context: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
                           active_picture: dict[str, Any] | None = None,
                           provider_compatibility: dict[str, Any] | None = None) -> dict[str, Any]:
    """Observe only cache misses.  Provider calls are isolated to this function.

    ``provider.generate`` receives the observation-only prompt and canonical
    evidence catalog.  It is intentionally not invoked while replaying policy.
    """
    run = run_dir(input_dir); requests = reused = 0; usage = {key: 0 for key in ("prompt_tokens", "cached_tokens", "response_tokens", "thinking_tokens", "total_tokens")}
    cost = 0.0
    for event in events:
        # Contact evidence may be populated by the contact-sheet producer.
        contact_evidence: dict[str, Any] = {}
        sheet = make_contact_sheet(event, contact_evidence)
        catalog = canonical_evidence_catalog(event, contact_evidence)
        context = make_context(event, catalog)
        narrative_fp = fingerprint(context.get("narrative", {})) if isinstance(context, dict) else None
        context = {**context, "visual_event_id": event["visual_event_id"],
                   "expected_event_id": event["visual_event_id"],
                   "identity_instruction": "Return event_id exactly equal to expected_event_id; never use candidate_id or invent an ID.",
                   "canonical_evidence": catalog, "observation_schema": OBSERVATION_SCHEMA_VERSION}
        sheet_hash = __import__("hashlib").sha256(sheet).hexdigest()
        obs_fp = observation_fingerprint(movie_sha256=movie_sha256, event=event, contact_sheet_sha256=sheet_hash,
                                         active_picture=active_picture, narrative_context_fingerprint=narrative_fp,
                                         provider_compatibility=provider_compatibility)
        path = observation_path(run, event["visual_event_id"])
        if path.is_file():
            try:
                existing = _read(path)
                if existing.get("observation_fingerprint") == obs_fp and not validate_observation(existing, event):
                    reused += 1; continue
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        # Persist the parsed provider payload before validation, including failures.
        # A valid saved response also closes the crash window before canonical persistence.
        attempts_dir = observation_dir(run) / "responses" / event["visual_event_id"] / obs_fp
        saved = []
        for attempt_path in sorted(attempts_dir.glob("*.json")):
            attempt_record = _read(attempt_path)
            if not validate_observation({"schema_version": OBSERVATION_SCHEMA_VERSION,
                                         "observation": attempt_record["response"]}, event):
                saved.append(attempt_record)
        if saved:
            record = saved[-1]
            persist_observation(run, event, record["response"], observation_fingerprint_value=obs_fp,
                                evidence_catalog=catalog, provider_provenance=record.get("cumulative_provider_provenance", record["provider_provenance"]))
            reused += 1
            continue
        event_requests = 0; event_cost = 0.0; event_usage = {key: 0 for key in usage}
        for validation_attempt in range(1, 3):
            response = provider.generate(OBSERVATION_PROMPT, context, sheet)
            body = response.data
            provenance = {"provider": response.provider or getattr(provider, "identifier", None),
                          "model": response.model or getattr(provider, "model", None),
                          "usage": response.usage or {}, "request_count": max(1, int(response.attempts or 1)),
                          "cost_usd": estimate_openai_cost(response.usage or {}) if (response.provider or getattr(provider, "identifier", None)) == "openai" else 0.0}
            requests += provenance["request_count"]; cost += provenance["cost_usd"]
            event_requests += provenance["request_count"]; event_cost += provenance["cost_usd"]
            for key in event_usage: event_usage[key] += int((response.usage or {}).get(key, 0) or 0)
            for key in usage: usage[key] += int((response.usage or {}).get(key, 0) or 0)
            errors = validate_observation({"schema_version": OBSERVATION_SCHEMA_VERSION,
                                           "observation": body}, event)
            diagnostic = {"errors": errors, "expected_event_id": event["visual_event_id"],
                          "returned_event_id": body.get("event_id"),
                          "event_id_present": "event_id" in body}
            attempt_number = len(list(attempts_dir.glob("*.json"))) + 1
            write_json(attempts_dir / f"{attempt_number:04d}.json",
                       {"response": body, "provider_provenance": provenance,
                        "cumulative_provider_provenance": {**provenance, "request_count": event_requests,
                                                           "usage": dict(event_usage), "cost_usd": event_cost},
                        "provider_trace": list(response.provider_trace or ()),
                        "observation_fingerprint": obs_fp, "diagnostic": diagnostic, "created_at": _utc()})
            if errors:
                if validation_attempt == 2:
                    raise ValueError("invalid semantic observation: " + json.dumps(diagnostic, sort_keys=True))
                context = {**context, "validation_feedback": diagnostic,
                           "retry_instruction": "Correct the listed fields using the supplied authoritative IDs and visual evidence."}
                continue
            # Account for both paid validation attempts in the durable success.
            provenance = {**provenance, "request_count": event_requests, "usage": event_usage, "cost_usd": event_cost}
            persist_observation(run, event, body, observation_fingerprint_value=obs_fp,
                                evidence_catalog=catalog, provider_provenance=provenance)
            break
    return {"requests": requests, "reused": reused, "usage": usage, "cost_usd": cost}


def policy_evaluate(observation: dict[str, Any], *, policy_version: str = POLICY_VERSION,
                    event: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pure local policy.  It intentionally never examines descriptive prose."""
    if policy_version == "broll_policy_v2":
        from .broll_policy_v2 import evaluate
        return evaluate(observation, event)
    if policy_version != POLICY_VERSION:
        raise ValueError(f"unsupported policy: {policy_version}")
    body = observation.get("observation", observation)
    reasons: list[str] = []
    if not isinstance(body, dict) or validate_observation({"schema_version": OBSERVATION_SCHEMA_VERSION,
                                                             "observation": body}):
        return {"policy_version": policy_version, "policy_decision": "REVIEW",
                "policy_reasons": ["observation_incomplete_or_invalid"], "policy_cost_usd": 0.0}
    flags = body["technical_observations"]
    hard = [flag for flag in HARD_REJECT_FLAGS if flags.get(flag)]
    if hard:
        return {"policy_version": policy_version, "policy_decision": "REJECT",
                "policy_reasons": [f"hard_exclusion:{flag}" for flag in hard], "policy_cost_usd": 0.0}
    utility = body["visual_utility_kind"]
    signal = body["conversation_visual_signal"]
    action_ids = body.get("action_evidence_ids", [])
    complete = body.get("action_or_moment_complete")
    context = body.get("context_dependency")
    if utility in {"generic_dialogue_only", "generic_presence_only"}:
        if signal not in USEFUL_CONVERSATION_SIGNALS:
            return {"policy_version": policy_version, "policy_decision": "REJECT",
                    "policy_reasons": ["generic_dialogue_or_presence_without_independent_visual_signal"], "policy_cost_usd": 0.0}
    if utility == "unclear" or complete == "unclear" or context == "unclear":
        return {"policy_version": policy_version, "policy_decision": "REVIEW",
                "policy_reasons": ["insufficient_observation"], "policy_cost_usd": 0.0}
    if context == "high":
        return {"policy_version": policy_version, "policy_decision": "REJECT",
                "policy_reasons": ["high_context_dependency"], "policy_cost_usd": 0.0}
    if complete == "false":
        return {"policy_version": policy_version, "policy_decision": "REJECT",
                "policy_reasons": ["incomplete_moment_without_reusable_state"], "policy_cost_usd": 0.0}
    if utility in USEFUL_UTILITY_KINDS or signal in USEFUL_CONVERSATION_SIGNALS:
        if utility in {"concrete_action", "object_activity"} and not action_ids:
            return {"policy_version": policy_version, "policy_decision": "REVIEW",
                    "policy_reasons": ["action_utility_without_canonical_action_evidence"], "policy_cost_usd": 0.0}
        reasons.append(f"useful_visual_utility:{utility}")
        if signal in USEFUL_CONVERSATION_SIGNALS: reasons.append(f"independent_conversation_signal:{signal}")
        return {"policy_version": policy_version, "policy_decision": "KEEP", "policy_reasons": reasons,
                "policy_cost_usd": 0.0}
    return {"policy_version": policy_version, "policy_decision": "REVIEW",
            "policy_reasons": ["unclassified_visual_utility"], "policy_cost_usd": 0.0}


def apply_policy_result(event: dict[str, Any], observation: dict[str, Any], result: dict[str, Any]) -> None:
    """Materialize local policy for downstream finalization, never in the cache."""
    body = observation["observation"]
    labels = {row.get("id"): row.get("canonical_label") for row in observation.get("evidence_catalog", {}).get("actions", [])
              if isinstance(row, dict)}
    event["visual"] = {**event.get("visual", {}),
                       "actions": [labels[x] for x in body.get("action_evidence_ids", []) if labels.get(x)],
                       "visible_interactions": list(body.get("physical_interactions", [])),
                       "visible_emotions": list(body.get("visible_reactions", [])),
                       "shot_focus_plan": copy.deepcopy(body.get("shot_focus_plan", []))}
    # Observation-only schemas do not supply the legacy event.people array.
    # Keep uncertain/plural counts honest; shot-local IDs are not global people.
    count = body.get("people_count", "unclear")
    event["semantic_people"] = {
        "contains_people": True if count in {"1", "2_plus"} or body.get("visible_person_ids") else False if count == "0" else None,
        "people_count": int(count) if count in {"0", "1"} else None,
        "people_count_evidence": count,
        "composition": "single_subject" if count == "1" else "multiple_people" if count == "2_plus" else "not_applicable" if count == "0" else "unclear",
        "visible_person_ids": copy.deepcopy(body.get("visible_person_ids", [])),
    }
    event["editorial"] = {"decision": result["policy_decision"], "status": "VALIDATED",
                          "policy_version": result["policy_version"],
                          "policy_reasons": list(result["policy_reasons"]),
                          "observation_event_id": observation["event_id"]}
    if "temporal_completeness" in observation:
        event["editorial"]["temporal_completeness"] = copy.deepcopy(observation["temporal_completeness"])


def apply_cached_policy(input_dir: Path, events: list[dict[str, Any]], *, policy_version: str = POLICY_VERSION) -> dict[str, Any]:
    """Apply persisted observations to mutable production state locally."""
    if policy_version == "broll_policy_v2":
        raise ValueError("policy v2 requires explicit asset-window migration; use read-only policy evaluate")
    run = run_dir(input_dir); counts: Counter[str] = Counter(); missing: list[str] = []
    for event in events:
        try: observation = effective_observation(run, event)
        except (OSError, ValueError): missing.append(event["visual_event_id"]); continue
        if validate_observation(observation, event): missing.append(event["visual_event_id"]); continue
        result = policy_evaluate(observation, policy_version=policy_version)
        apply_policy_result(event, observation, result); counts[result["policy_decision"]] += 1
    return {"policy_version": policy_version, "counts": dict(counts), "missing_event_ids": missing,
            "provider_requests": 0, "api_cost_usd": 0.0}


def _legacy_to_observation(event: dict[str, Any], response: dict[str, Any], *, provenance: dict[str, Any]) -> dict[str, Any]:
    """Best-effort read-only migration; legacy decision fields are ignored."""
    visual = response.get("visual", {}) if isinstance(response, dict) else {}
    editorial = response.get("editorial", {}) if isinstance(response, dict) else {}
    event_id = str(event["visual_event_id"])
    actions = [value for value in [*visual.get("actions", []), *visual.get("visible_interactions", [])] if isinstance(value, str)]
    action_catalog = [{"id": f"{event_id}:LEGACY_ACTION_{i:02d}", "canonical_label": value}
                      for i, value in enumerate(actions, 1)]
    people = visual.get("people", []) if isinstance(visual.get("people"), list) else []
    kind = editorial.get("visual_utility_kind")
    if kind not in VISUAL_UTILITY_KINDS:
        # Legacy data did not distinguish policy evidence.  Preserve only
        # safely observable structure and make uncertainty explicit.
        kind = "unclear"
    signal = editorial.get("conversation_visual_signal")
    if signal not in CONVERSATION_SIGNALS: signal = "unclear"
    text = " ".join(actions + [str(visual.get("setting", ""))]).casefold()
    flags = {flag: False for flag in HARD_REJECT_FLAGS}
    flags["title_card"] = "title" in text or "título" in text
    flags["credits"] = "credit" in text
    flags["logo"] = "logo" in text
    flags["black_or_empty"] = "negra" in text or "black" in text
    body = {"event_id": event_id, "represented_shot_ids": list(event.get("source_shot_ids", [])),
            "people_count": "0" if not people else "1" if len(people) == 1 else "2_plus",
            "visible_person_ids": [], "action_evidence_ids": [x["id"] for x in action_catalog],
            "object_evidence_ids": [], "visible_states": list(visual.get("visible_emotions", [])),
            "movement": "present" if kind == "movement" else "unclear",
            "physical_interactions": list(visual.get("visible_interactions", [])),
            "visible_reactions": list(visual.get("visible_emotions", [])),
            "conversation_present": "true" if signal != "none" else "false",
            "conversation_visual_signal": signal, "visual_utility_kind": kind,
            "action_or_moment_complete": editorial.get("action_or_moment_complete", "unclear"),
            "context_dependency": "unclear", "technical_observations": flags,
            "shot_focus_plan": copy.deepcopy(visual.get("shot_focus_plan", []))}
    return {"schema_version": OBSERVATION_SCHEMA_VERSION, "event_id": event_id,
            "observation_fingerprint": provenance["observation_fingerprint"], "observation": body,
            "evidence_catalog": {"actions": action_catalog, "objects": [], "people": []},
            "provider_provenance": provenance["provider_provenance"], "migration": {"source": "persisted_legacy_semantic", "lossy": True},
            "created_at": _utc()}


def migrate_existing_observations(input_dir: Path, *, policy_version: str = POLICY_VERSION) -> dict[str, Any]:
    """Migrate checkpoint/V9/V9.1 evidence only.  This function never calls a provider or edits canonical events."""
    run = run_dir(input_dir); store_path = run / "visual_event_segments_v1.json"
    if not store_path.is_file(): raise FileNotFoundError(f"canonical Visual Events required: {store_path}")
    store = _read(store_path); events = store.get("events", [])
    movie = input_dir / "movie.mp4"; movie_hash = sha256_file(movie) if movie.is_file() else "unknown"
    created = reused = unavailable = 0; decisions: Counter[str] = Counter()
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("visual_event_id"), str): continue
        event_id = event["visual_event_id"]
        destination = observation_path(run, event_id)
        if destination.is_file():
            try:
                existing = _read(destination)
                if not validate_observation(existing, event):
                    decisions[policy_evaluate(existing, policy_version=policy_version)["policy_decision"]] += 1; reused += 1; continue
            except (OSError, ValueError, json.JSONDecodeError): pass
        response = None; metadata: dict[str, Any] = {}
        checkpoint = run / "semantic_checkpoints" / f"{event.get('candidate_id')}.json"
        if checkpoint.is_file():
            try:
                checkpoint_value = _read(checkpoint); response = checkpoint_value.get("response"); metadata = checkpoint_value
            except (OSError, ValueError, json.JSONDecodeError): pass
        if response is None:
            for directory in (run / "semantic_reclassifications" / "semantic-v9" / "events",
                              run / "semantic_reclassifications" / "semantic-v9.1" / "events"):
                path = directory / f"{event_id}.json"
                if path.is_file():
                    try:
                        candidate = _read(path)
                        if isinstance(candidate.get("response"), dict): response, metadata = candidate["response"], candidate; break
                    except (OSError, ValueError, json.JSONDecodeError): pass
        if not isinstance(response, dict): unavailable += 1; continue
        obs_fp = observation_fingerprint(movie_sha256=movie_hash, event=event,
                                         contact_sheet_sha256=None, active_picture=None,
                                         narrative_context_fingerprint=None)
        provenance = {"observation_fingerprint": obs_fp, "provider_provenance": {
            "provider": metadata.get("provider"), "model": metadata.get("model"),
            # This is historical provenance, not an observation-v1 request.
            # Migration is deliberately free and must never make status claim
            # that replay just contacted a provider.
            "legacy_source_usage": metadata.get("usage", metadata.get("token_usage", {})),
            "usage": {}, "cost_usd": 0.0, "request_count": 0,
        }}
        record = _legacy_to_observation(event, response, provenance=provenance)
        write_json(destination, record); created += 1
        decisions[policy_evaluate(record, policy_version=policy_version)["policy_decision"]] += 1
    manifest = {"schema_version": "semantic_observation_manifest_v1", "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
                "movie_id": input_dir.name, "migration": "existing_persisted_evidence_only", "provider_requests": 0,
                "api_cost_usd": 0.0, "created": created, "reused": reused, "unavailable": unavailable,
                "policy": {"policy_version": policy_version, **dict(decisions)}, "updated_at": _utc()}
    write_json(observation_dir(run) / "manifest.json", manifest)
    return manifest


def evaluate_cached_observations(input_dir: Path, *, policy_version: str = POLICY_VERSION) -> dict[str, Any]:
    """Evaluate every cached observation locally and persist only a policy sidecar."""
    if policy_version == "broll_policy_v2":
        from .broll_policy_v2 import offline_projection
        return offline_projection(input_dir)
    run = run_dir(input_dir); store = _read(run / "visual_event_segments_v1.json")
    results: dict[str, Any] = {}; counts: Counter[str] = Counter(); missing = 0
    for event in store.get("events", []):
        if not isinstance(event, dict) or not isinstance(event.get("visual_event_id"), str): continue
        event_id = event["visual_event_id"]
        try: observation = effective_observation(run, event)
        except (OSError, ValueError, json.JSONDecodeError): missing += 1; continue
        decision = policy_evaluate(observation, policy_version=policy_version)
        results[event_id] = decision; counts[decision["policy_decision"]] += 1
    report = {"schema_version": "broll_policy_evaluation_v1", "movie_id": input_dir.name,
              "policy_version": policy_version, "provider_requests": 0, "api_cost_usd": 0.0,
              "total_events": len(store.get("events", [])), "cached": sum(counts.values()), "missing": missing,
              "counts": {key: counts[key] for key in ("KEEP", "REJECT", "REVIEW")}, "decisions": results,
              "evaluated_at": _utc()}
    write_json(observation_dir(run) / "policy" / f"{policy_version}.json", report)
    return report


def observation_status(input_dir: Path, *, policy_version: str = POLICY_VERSION) -> dict[str, Any]:
    run = run_dir(input_dir); store_path = run / "visual_event_segments_v1.json"
    events = _read(store_path).get("events", []) if store_path.is_file() else []
    records = []
    provenance: Counter[str] = Counter(); usage = {key: 0 for key in ("prompt_tokens", "cached_tokens", "response_tokens", "thinking_tokens", "total_tokens")}; cost = 0.0; requests = 0
    for event in events:
        if isinstance(event, dict):
            try: records.append(effective_observation(run, event))
            except (OSError, ValueError, json.JSONDecodeError): pass
    for record in records:
        p = record.get("provider_provenance", {}); label = "/".join(str(p.get(x)) for x in ("provider", "model") if p.get(x))
        if label: provenance[label] += 1
        if record.get("migration", {}).get("source") == "persisted_legacy_semantic":
            continue
        requests += int(p.get("request_count", 0) or 0); cost += float(p.get("cost_usd", 0) or 0)
        for key in usage:
            usage[key] += int((p.get("usage") or {}).get(key, 0) or 0)
    policy_path = observation_dir(run) / "policy" / f"{policy_version}.json"
    policy = _read(policy_path) if policy_path.is_file() else {"counts": {"KEEP": 0, "REJECT": 0, "REVIEW": 0}}
    return {"total": len(events), "cached": len(records), "missing": len(events) - len(records),
            "provider_provenance": dict(provenance), "policy_version": policy_version,
            "policy": policy.get("counts", {}), "observation_requests": requests,
            "reused_observations": len(records), "usage": usage, "cost_usd": cost}
