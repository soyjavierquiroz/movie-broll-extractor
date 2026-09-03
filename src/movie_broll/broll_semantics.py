"""Constrained multimodal semantic boundary for B-roll pilot candidates."""
from __future__ import annotations

import base64
import json
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
        }, "required": ["shot_id", "focus_subject", "focus_reason", "preserve_secondary_subject", "interaction_requirement"]}},
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

class SemanticProvider(Protocol):
    identifier: str
    model: str
    def generate(self, prompt: str, context: dict[str, Any], jpeg: bytes) -> SemanticResponse: ...

class GeminiBrollSemanticProvider:
    identifier = "gemini"
    def __init__(self, api_key: str, model: str = "gemini-3.6-flash") -> None:
        from google import genai
        self.client = genai.Client(api_key=api_key); self.model = model

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

def validate_response(data: dict[str, Any]) -> list[str]:
    try: visual, editorial = data["visual"], data["editorial"]
    except (KeyError, TypeError): return ["missing visual or editorial"]
    required_visual = SEMANTIC_SCHEMA["properties"]["visual"]["required"]
    required_editorial = SEMANTIC_SCHEMA["properties"]["editorial"]["required"]
    errors = [f"visual missing {x}" for x in required_visual if x not in visual] + [f"editorial missing {x}" for x in required_editorial if x not in editorial]
    if visual.get("primary_subject_position") not in POSITIONS: errors.append("invalid subject position")
    plan=visual.get('shot_focus_plan', [])
    if plan and (not isinstance(plan,list) or any(not isinstance(x,dict) or x.get('focus_subject') not in FOCUS_SUBJECTS or x.get('interaction_requirement') not in INTERACTION_REQUIREMENTS or not all(k in x for k in ('shot_id','focus_reason','preserve_secondary_subject')) for x in plan)): errors.append('invalid shot focus plan')
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
