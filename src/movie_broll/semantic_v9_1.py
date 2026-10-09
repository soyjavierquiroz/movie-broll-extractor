"""The independently versioned V9.1 semantic contract.

V9 remains an historical calibration contract in :mod:`broll_semantics`.  This
module deliberately owns the V9.1 schema, prompt, and local eligibility rule
so a future calibration cannot silently reinterpret a V9 result.
"""
from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .broll_semantics import (
    AGE_GROUPS, DECISIONS, EMOTIONS, FOCUS_SUBJECTS, FRAME_ROLES,
    INTERACTION_REQUIREMENTS, KEEP_ACTION_EVIDENCE_TYPES,
    KEEP_REUSABLE_INTENT_TYPES, POSITIONS, PRESENTATIONS,
    RELATIONSHIP_SOURCES, TARGET_BINDING_CONFIDENCE, directive_validation_errors,
)

SEMANTIC_CONTRACT_V9_1 = "semantic_contract_v9_1"
SEMANTIC_PROMPT_V9_1 = "semantic_prompt_v9_1"
SEMANTIC_SCHEMA_VERSION_V9_1 = "broll_semantics_v9_1"
VISUAL_UTILITY_KINDS = [
    "concrete_action", "useful_state", "clear_reaction", "physical_interaction",
    "movement", "object_activity", "environment", "strong_nonverbal_interaction",
    "generic_dialogue_only", "generic_presence_only", "unclear",
]
CONVERSATION_VISUAL_SIGNALS = [
    "none", "clear_reaction", "strong_body_language", "physical_interaction",
    "concrete_activity", "useful_environment_or_composition", "generic_dialogue_only",
]
INELIGIBLE_KEEP_UTILITY_KINDS = frozenset({
    "generic_dialogue_only", "generic_presence_only", "unclear",
})


SEMANTIC_SCHEMA_V9_1: dict[str, Any] = {
    "type": "object", "properties": {
        "visual": {"type": "object", "properties": {
            "summary_es": {"type": "string"}, "subjects": {"type": "array", "items": {"type": "string"}},
            "objects": {"type": "array", "items": {"type": "string"}}, "actions": {"type": "array", "items": {"type": "string"}},
            "people_count_estimate": {"type": "string"}, "setting": {"type": "string"},
            "visible_interactions": {"type": "array", "items": {"type": "string"}},
            "visible_emotions": {"type": "array", "items": {"type": "string", "enum": EMOTIONS}},
            "people": {"type": "array", "items": {"type": "object", "properties": {
                "presentation": {"type": "string", "enum": PRESENTATIONS}, "apparent_age_group": {"type": "string", "enum": AGE_GROUPS},
                "frame_role": {"type": "string", "enum": FRAME_ROLES}, "position": {"type": "string", "enum": POSITIONS},
            }, "required": ["presentation", "apparent_age_group", "frame_role", "position"]}},
            "primary_subject_position": {"type": "string", "enum": POSITIONS}, "primary_subject_description": {"type": "string"}, "visual_focus": {"type": "string"},
            "shot_focus_plan": {"type": "array", "items": {"type": "object", "properties": {
                "shot_id": {"type": "string"}, "focus_subject": {"type": "string", "enum": FOCUS_SUBJECTS}, "focus_reason": {"type": "string"},
                "preserve_secondary_subject": {"type": "boolean"}, "interaction_requirement": {"type": "string", "enum": INTERACTION_REQUIREMENTS},
                "focus_position": {"type": "string", "enum": POSITIONS}, "target_person_ids": {"type": "array", "items": {"type": "string"}},
                "target_binding_confidence": {"type": "string", "enum": TARGET_BINDING_CONFIDENCE},
            }, "required": ["shot_id", "focus_subject", "focus_reason", "preserve_secondary_subject", "interaction_requirement", "focus_position", "target_person_ids", "target_binding_confidence"]}},
        }, "required": ["summary_es", "subjects", "objects", "actions", "people_count_estimate", "setting", "visible_interactions", "visible_emotions", "people", "primary_subject_position", "primary_subject_description", "visual_focus", "shot_focus_plan"]},
        "relationships": {"type": "array", "items": {"type": "object", "properties": {
            "type": {"type": "string"}, "source": {"type": "string", "enum": RELATIONSHIP_SOURCES}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        }, "required": ["type", "source", "confidence"]}},
        "editorial": {"type": "object", "properties": {
            "standalone_meaning_es": {"type": "string"}, "reusable_broll": {"type": "boolean"},
            "action_or_moment_complete": {"type": "string", "enum": ["true", "false", "unclear"]},
            "use_cases_es": {"type": "array", "items": {"type": "string"}}, "negative_use_cases_es": {"type": "array", "items": {"type": "string"}},
            "search_terms_es": {"type": "array", "items": {"type": "string"}}, "editorial_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "keep_qualification": {"type": "object", "properties": {
                "action_evidence_type": {"type": "string", "enum": KEEP_ACTION_EVIDENCE_TYPES}, "action_evidence_es": {"type": "string"},
                "reusable_intent_type": {"type": "string", "enum": KEEP_REUSABLE_INTENT_TYPES}, "reusable_use_case_es": {"type": "string"},
            }, "required": ["action_evidence_type", "action_evidence_es", "reusable_intent_type", "reusable_use_case_es"]},
            "visual_utility_kind": {"type": "string", "enum": VISUAL_UTILITY_KINDS},
            "conversation_visual_signal": {"type": "string", "enum": CONVERSATION_VISUAL_SIGNALS},
            "reason": {"type": "string"}, "decision": {"type": "string", "enum": DECISIONS},
        }, "required": ["standalone_meaning_es", "reusable_broll", "action_or_moment_complete", "use_cases_es", "negative_use_cases_es", "search_terms_es", "editorial_confidence", "keep_qualification", "visual_utility_kind", "conversation_visual_signal", "reason", "decision"]},
    }, "required": ["visual", "relationships", "editorial"],
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Person(_Strict):
    presentation: Literal["woman", "man", "unclear"]
    apparent_age_group: Literal["young_adult", "adult", "middle_aged", "older_adult", "unclear"]
    frame_role: Literal["primary", "secondary", "background", "unclear"]
    position: Literal["left", "center", "right", "multiple", "unclear"]


class _Directive(_Strict):
    shot_id: str; focus_subject: Literal["woman", "man", "multiple_people", "action_region", "environment", "unclear"]
    focus_reason: str; preserve_secondary_subject: bool
    interaction_requirement: Literal["none", "sequence", "simultaneous", "unclear"]
    focus_position: Literal["left", "center", "right", "multiple", "unclear"]
    target_person_ids: list[str]; target_binding_confidence: Literal["high", "medium", "low", "unclear"]


class _Visual(_Strict):
    summary_es: str; subjects: list[str]; objects: list[str]; actions: list[str]; people_count_estimate: str; setting: str
    visible_interactions: list[str]; visible_emotions: list[Literal["smiling", "crying", "tense_appearance", "surprised_appearance", "neutral", "unclear"]]
    people: list[_Person]; primary_subject_position: Literal["left", "center", "right", "multiple", "unclear"]
    primary_subject_description: str; visual_focus: str; shot_focus_plan: list[_Directive]


class _Relationship(_Strict):
    type: str; source: Literal["visual", "srt", "narrative", "combined"]; confidence: float


class _Qualification(_Strict):
    action_evidence_type: Literal["distinct_visible_action_or_reaction", "generic_presence_or_movement", "unclear"]
    action_evidence_es: str; reusable_intent_type: Literal["specific_visual_need", "generic_ambience_or_context", "unclear"]
    reusable_use_case_es: str


class _Editorial(_Strict):
    standalone_meaning_es: str; reusable_broll: bool; action_or_moment_complete: Literal["true", "false", "unclear"]
    use_cases_es: list[str]; negative_use_cases_es: list[str]; search_terms_es: list[str]
    editorial_confidence: Literal["high", "medium", "low"]; keep_qualification: _Qualification
    visual_utility_kind: Literal["concrete_action", "useful_state", "clear_reaction", "physical_interaction", "movement", "object_activity", "environment", "strong_nonverbal_interaction", "generic_dialogue_only", "generic_presence_only", "unclear"]
    conversation_visual_signal: Literal["none", "clear_reaction", "strong_body_language", "physical_interaction", "concrete_activity", "useful_environment_or_composition", "generic_dialogue_only"]
    reason: str; decision: Literal["KEEP", "REVIEW", "REJECT"]


class OpenAISemanticV9_1StructuredResult(_Strict):
    visual: _Visual; relationships: list[_Relationship]; editorial: _Editorial


PROMPT_V9_1 = """You validate a movie B-roll candidate under semantic_contract_v9_1. Images are the only authority for visual facts; output Spanish. Return every required structured field and one compact visual.shot_focus_plan directive for every supplied technical shot ID. Do not identify people or infer family/romantic relations from imagery. SRT/narrative are synchronized context, never proof of visual facts.

The editorial question is: would an editor intentionally search for THIS visual behavior, state, interaction, reaction, activity, object, environment, composition, or movement? Ordinary reusable visual targets are valid KEEP candidates; they need not be distinctive, dramatic, special, memorable, or narratively important. Valid ordinary targets include walking/walking away, writing, reading, phone use, resting in bed, eating, driving or sitting at a steering wheel, waiting, looking through a window, searching a drawer, carrying food, examining an object, cooking, working, hugging, touch/support, visible reaction, and useful transport/environment/object imagery.

Set editorial.visual_utility_kind to the single best grounded kind. Set editorial.conversation_visual_signal to none when conversation is not material; otherwise state the independent visual signal. Generic dialogue alone, talking-head coverage, alternating shot/reverse-shot conversation, generic seated conversation, talking off-camera, generic presence, sitting, standing, or merely looking at someone are not searchable visual utility and normally REJECT. For those cases use generic_dialogue_only or generic_presence_only. Conversation may KEEP only where independent visual value is clear: reaction/listening, strong body language/emotion, physical interaction/support, meaningful silence, concrete activity, meaningful movement, or useful environment/composition independent of speech. Do not treat 'conversation is searchable' as a reason to KEEP.

Hard-reject black/fade, title cards, credits, logos, production-specific text/UI, corrupt/frozen/blurred content, and context-dependent fragments. For KEEP set reusable_broll true and ground keep_qualification evidence exactly in visual actions/interactions and use cases. A model KEEP with visual_utility_kind generic_dialogue_only, generic_presence_only, or unclear is locally ineligible and will not become effective KEEP."""


def validate_response_v9_1(data: dict[str, Any]) -> list[str]:
    """V9.1-local validation; V9's validator is deliberately untouched."""
    try:
        visual, editorial = data["visual"], data["editorial"]
    except (KeyError, TypeError):
        return ["missing visual or editorial"]
    required_visual = SEMANTIC_SCHEMA_V9_1["properties"]["visual"]["required"]
    required_editorial = SEMANTIC_SCHEMA_V9_1["properties"]["editorial"]["required"]
    errors = [f"visual missing {x}" for x in required_visual if x not in visual]
    errors += [f"editorial missing {x}" for x in required_editorial if x not in editorial]
    if visual.get("primary_subject_position") not in POSITIONS: errors.append("invalid subject position")
    plan = visual.get("shot_focus_plan", [])
    if plan and (not isinstance(plan, list) or any(directive_validation_errors(x) for x in plan)): errors.append("invalid shot focus plan")
    for person in visual.get("people", []):
        if not isinstance(person, dict) or person.get("presentation") not in PRESENTATIONS: errors.append("invalid person presentation")
        elif person.get("apparent_age_group") not in AGE_GROUPS: errors.append("invalid person age group")
        elif person.get("frame_role") not in FRAME_ROLES: errors.append("invalid person frame role")
        elif person.get("position") not in POSITIONS: errors.append("invalid person position")
    for relation in data.get("relationships", []):
        if not isinstance(relation, dict) or relation.get("source") not in RELATIONSHIP_SOURCES: errors.append("invalid relationship provenance"); continue
        if not isinstance(relation.get("confidence"), (int, float)) or not 0 <= relation["confidence"] <= 1: errors.append("invalid relationship confidence")
    if any(x not in EMOTIONS for x in visual.get("visible_emotions", [])): errors.append("invalid visible emotion")
    if editorial.get("decision") not in DECISIONS: errors.append("invalid decision")
    if editorial.get("visual_utility_kind") not in VISUAL_UTILITY_KINDS: errors.append("invalid visual utility kind")
    if editorial.get("conversation_visual_signal") not in CONVERSATION_VISUAL_SIGNALS: errors.append("invalid conversation visual signal")
    if editorial.get("decision") == "KEEP":
        if not editorial.get("reusable_broll") or editorial.get("action_or_moment_complete") == "false": errors.append("KEEP lacks semantic usefulness")
        if editorial.get("visual_utility_kind") in INELIGIBLE_KEEP_UTILITY_KINDS: errors.append("KEEP has ineligible visual utility kind")
        qualification = editorial.get("keep_qualification")
        if not isinstance(qualification, dict): errors.append("KEEP lacks editorial qualification")
        else:
            if qualification.get("action_evidence_type") not in KEEP_ACTION_EVIDENCE_TYPES: errors.append("invalid KEEP action evidence type")
            if qualification.get("reusable_intent_type") not in KEEP_REUSABLE_INTENT_TYPES: errors.append("invalid KEEP reusable intent type")
            if qualification.get("action_evidence_type") == "unclear": errors.append("KEEP lacks grounded visible action or state")
            if qualification.get("reusable_intent_type") == "unclear": errors.append("KEEP lacks searchable reusable use case")
            norm = lambda x: x.strip().casefold() if isinstance(x, str) else ""
            actions = {norm(x) for x in [*visual.get("actions", []), *visual.get("visible_interactions", [])] if norm(x)}
            if not norm(qualification.get("action_evidence_es")) or norm(qualification.get("action_evidence_es")) not in actions: errors.append("KEEP action evidence is not grounded in visible structured action")
            use_cases = {norm(x) for x in editorial.get("use_cases_es", []) if norm(x)}
            if not norm(qualification.get("reusable_use_case_es")) or norm(qualification.get("reusable_use_case_es")) not in use_cases: errors.append("KEEP reusable use case is not grounded in structured use cases")
    return sorted(set(errors))


def effective_decision(data: dict[str, Any] | None, local_validation: dict[str, Any] | None) -> str | None:
    """The only countable semantic decision; raw model decision remains auditable."""
    decision = data.get("editorial", {}).get("decision") if isinstance(data, dict) else None
    if decision not in DECISIONS:
        return "REVIEW" if data is not None else None
    return decision if isinstance(local_validation, dict) and local_validation.get("valid") is True else "REVIEW"
