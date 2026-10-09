"""Immutable temporal observations and a local, targeted enrichment workflow.

The existing policy is unchanged. This module projects a separately established
reusable moment/state into its legacy completeness field; raw action completeness
and provider payloads remain independently inspectable.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError

from . import semantic_observations as legacy
from .broll_semantics import KEEP_ACTION_EVIDENCE_TYPES, classify_provider_error
from .processing_ledger import fingerprint
from .temporal_evidence import PROFILE, MAX_FRAMES, sample_plan
from .utils import write_json

SCHEMA = "semantic_observation_v2"
PROMPT_VERSION = "semantic_observation_prompt_v2"
MAX_RESPONSE_ATTEMPTS = 2
# Bump only after explicit review authorizes another bounded validation budget.
# v2 is authorized because deterministic validation feedback now identifies the
# exact reusable_state / visual_utility_kind contradiction. Legacy archives stay v1.
TEMPORAL_VALIDATION_RECOVERY_REVISION = "temporal_validation_recovery_v2"
_LEGACY_RECOVERY_REVISION = "temporal_validation_recovery_v1"

RETRY_INSTRUCTION = (
    "Correct the listed errors using authoritative IDs and visual samples. "
    "Do not change moment_status merely to satisfy validation. Return useful_state "
    "only if temporal evidence actually supports a reusable sustained visual state; "
    "otherwise choose the truthful moment/utility classification. Incomplete, unclear, "
    "generic dialogue and otherwise unusable conclusions are allowed."
)


def response_diagnostic(body, event, evidence):
    errors = validate_response(body, event, evidence)
    details = []
    for code in errors:
        if code == "reusable_state_requires_useful_state_utility":
            path = "visual_utility_kind"
            message = ("moment_status=reusable_state requires visual_utility_kind=useful_state; "
                       f"returned visual_utility_kind={body.get('visual_utility_kind')}. "
                       + RETRY_INSTRUCTION)
        else:
            path = {"event_id": "event_id", "represented_shot_ids": "represented_shot_ids",
                    "reusable_state_requires_visible_state": "visible_states",
                    "insufficient_temporal_support": "temporal_support_sample_ids"}.get(code, "$")
            message = code
        details.append({"code": code, "path": path, "message": message})
    try:
        TemporalObservationResult.model_validate(body)
    except ValidationError as error:
        details = [{"code": row["type"], "path": ".".join(map(str, row["loc"])),
                    "message": row["msg"]} for row in error.errors(include_input=False, include_url=False)]
    return {"errors": errors, "validation_errors": details,
            "expected_event_id": event["visual_event_id"],
            "returned_event_id": body.get("event_id"), "event_id_present": "event_id" in body}


def validation_attempt_directory(base, provider, compatibility):
    """Adopt legacy archives without rewriting them; each execution key is bounded.

    Unknown provider objects conservatively adopt the existing identity. Real
    providers expose identifier/model; explicit compatibility overrides are useful
    for provider pools and offline callers.
    """
    legacy_saved = [legacy._read(p) for p in sorted(base.glob("*.json"))]
    old = (legacy_saved[-1].get("execution_identity") or
           legacy_saved[-1]["cumulative_provider_provenance"]) if legacy_saved else {}
    provider_id = compatibility.get("provider", getattr(provider, "identifier", old.get("provider", "unknown")))
    model = compatibility.get("model", getattr(provider, "model", old.get("model", "unknown")))
    identity = {"observation_fingerprint": base.name, "provider": provider_id, "model": model,
                "validation_recovery_revision": TEMPORAL_VALIDATION_RECOVERY_REVISION}
    if (TEMPORAL_VALIDATION_RECOVERY_REVISION == _LEGACY_RECOVERY_REVISION
            and (not old or (provider_id == old.get("provider") and model == old.get("model")))):
        return base, identity
    return base / "executions" / fingerprint(identity), identity


class ObservedAction(legacy._StrictObservation):
    canonical_label: str
    evidence_type: Literal["distinct_visible_action_or_reaction", "generic_presence_or_movement", "unclear"]
    sample_ids: list[str]


class TemporalObservationResult(legacy.OpenAISemanticObservationResult):
    moment_status: Literal["complete_action", "incomplete_action", "reusable_state", "unclear"] = Field(
        description="complete_action requires raw completeness true and beginning/development/end; "
        "incomplete_action requires false; reusable_state requires raw completeness false/unclear, "
        "visual_utility_kind useful_state, nonblank visible_states and multiple temporal timestamps; "
        "unclear permits raw completeness false/unclear: action incompleteness may be known "
        "while the overall moment remains unresolved; it cannot claim complete action.")
    temporal_support_sample_ids: list[str]
    visual_actions: list[str]
    observed_actions: list[ObservedAction]


RESPONSE_SCHEMA = TemporalObservationResult.model_json_schema()
PROMPT = legacy.OBSERVATION_PROMPT.replace("semantic_observation_v1", SCHEMA).replace(
    "actions, objects, people, and focus targets", "objects, people, and focus targets") + """
Temporal input profile: temporal_evidence_v2. Read the labelled samples in time
order within their technical shots. Images alone establish visible actions and
completeness; subtitles/narrative are context and never visual action evidence.
Return moment_status=complete_action only when the samples visibly establish
beginning, development and completion; event end is NOT proof of completion.
Return incomplete_action when a visible action is truncated and no independently
reusable sustained state is established. Return reusable_state for a clearly
visible, independently reusable sustained state supported across time, even if
its real-world beginning/end are outside the event; use visual_utility_kind=
useful_state. Do NOT claim the action is complete for reusable_state: keep
action_or_moment_complete=false or unclear. Return unclear when temporal evidence
cannot establish any of these. complete_action requires action_or_moment_complete
true; incomplete_action requires false; unclear permits false or unclear (known
action incompleteness does not resolve overall moment utility). Truthful unclear
completeness, utility and context are valid observations, evaluated by policy,
never reasons to retry. Cite supplied
sample IDs in temporal_support_sample_ids. Complete actions and reusable states
require multiple distinct timestamps; complete actions must cite event begin,
middle/development and end. Sparse samples do not prove unseen transitions.
Describe visually observed actions in visual_actions (the existing visual.actions
contract). For observed_actions copy canonical_label exactly from visual_actions
or physical_interactions, cite supporting sample IDs, and use only the existing
KEEP_ACTION_EVIDENCE_TYPES supplied in context. Only distinct_visible_action_or_reaction
with temporal visual support becomes canonical action evidence locally. Empty
catalogs are valid. Return action_evidence_ids=[]; the local validator assigns IDs
after grounding checks. People IDs are shot-local canonical reference identities;
their labels appear only on middle reference frames, never new identities at cuts.
"""


def event_identity(event: dict[str, Any]) -> str:
    return fingerprint({"event_id": event["visual_event_id"],
                        "range": [event["start_frame"], event["end_frame_exclusive"]],
                        "source_shot_ids": event["source_shot_ids"],
                        "technical_shots": event.get("technical_shots", [])})


def record_path(run: Path, event_id: str, observation_fp: str) -> Path:
    # IDs originate in production, but reject path components at this boundary.
    if any(Path(value).name != value or value in {"", ".", ".."} for value in (event_id, observation_fp)):
        raise ValueError("invalid observation identity path")
    return run / "semantic_observations" / "v2" / "events" / event_id / f"{observation_fp}.json"


def validate_response(body: dict[str, Any], event: dict[str, Any], evidence: dict[str, Any]) -> list[str]:
    errors = []
    try:
        TemporalObservationResult.model_validate(body)
    except ValueError as error:
        return [str(error)]
    if body["event_id"] != event["visual_event_id"]:
        errors.append("event_id")
    if (len(body["represented_shot_ids"]) != len(event["source_shot_ids"])
            or set(body["represented_shot_ids"]) != set(event["source_shot_ids"])):
        errors.append("represented_shot_ids")
    samples = {row["sample_id"]: row for row in evidence.get("samples", [])}
    if evidence.get("evidence_profile") != PROFILE or not samples or len(samples) > MAX_FRAMES:
        errors.append("temporal_evidence_profile_or_samples")
    try:
        if evidence.get("samples") != sample_plan(event, float(evidence["fps"])):
            errors.append("noncanonical_temporal_sample_plan")
    except (KeyError, ValueError, TypeError):
        errors.append("noncanonical_temporal_sample_plan")

    def supported(ids):
        return (len(ids) == len(set(ids)) and all(s in samples for s in ids)
                and len({samples[s]["frame"] for s in ids if s in samples}) >= 2)

    support = body["temporal_support_sample_ids"]
    if any(s not in samples for s in support):
        errors.append("unknown_temporal_sample_id")
    status, complete = body["moment_status"], body["action_or_moment_complete"]
    if status != "unclear" and not supported(support):
        errors.append("insufficient_temporal_support")
    if status == "complete_action":
        cited = [samples[s] for s in support if s in samples]
        roles = {role for row in cited for role in row["roles"]}
        frames = {row["frame"] for row in cited}
        if (complete != "true" or len(frames) < 3 or not {"begin", "end"} <= roles
                or not {"middle", "development"} & roles
                or min(frames, default=-1) != event["start_frame"]
                or max(frames, default=-1) != event["end_frame_exclusive"] - 1):
            errors.append("complete_action_requires_begin_development_end")
    elif status == "incomplete_action" and complete != "false":
        errors.append("incomplete_action_completeness")
    elif status == "reusable_state":
        if complete == "true":
            errors.append("reusable_state_raw_action_completeness_must_be_false_or_unclear")
        if body["visual_utility_kind"] != "useful_state":
            errors.append("reusable_state_requires_useful_state_utility")
        if not any(state.strip() for state in body["visible_states"]):
            errors.append("reusable_state_requires_visible_state")
    elif status == "unclear" and complete == "true":
        errors.append("unclear_moment_cannot_claim_complete_action")
    if body["action_evidence_ids"]:
        errors.append("canonical_action_ids_are_assigned_locally")
    labels = set(body["visual_actions"] + body["physical_interactions"])
    for action in body["observed_actions"]:
        if not action["canonical_label"].strip() or action["canonical_label"] not in labels:
            errors.append("action_not_grounded_in_visible_structured_action")
        if action["evidence_type"] not in KEEP_ACTION_EVIDENCE_TYPES:
            errors.append("action_evidence_type")
        if action["evidence_type"] != "unclear" and not supported(action["sample_ids"]):
            errors.append("action_without_temporal_visual_support")
        if any(s not in samples for s in action["sample_ids"]):
            errors.append("unknown_action_sample_id")
    catalog = legacy.canonical_evidence_catalog(event, evidence)
    people = {row["id"] for row in catalog["people"]}
    if any(pid not in people for pid in body["visible_person_ids"]):
        errors.append("unknown_visible_person_id")
    for directive in body["shot_focus_plan"]:
        if directive["shot_id"] not in event["source_shot_ids"] or any(
            pid not in people or not pid.startswith(directive["shot_id"] + ":")
            for pid in directive["target_person_ids"]
        ):
            errors.append("focus_target_not_in_shot_evidence")
    if len(body["shot_focus_plan"]) != len(event["source_shot_ids"]) or {
        row["shot_id"] for row in body["shot_focus_plan"]
    } != set(event["source_shot_ids"]):
        errors.append("one_focus_directive_per_technical_shot")
    return errors


def grounded_catalog(event: dict[str, Any], body: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    errors = validate_response(body, event, evidence)
    if errors:
        raise ValueError("invalid temporal observation: " + "; ".join(errors))
    catalog = legacy.canonical_evidence_catalog(event, evidence)
    # Ignore unverified upstream action labels. Only this observation's visual
    # claims, exact structured labels and canonical temporal citations qualify.
    catalog["actions"] = []
    for action in body["observed_actions"]:
        if action["evidence_type"] == "distinct_visible_action_or_reaction":
            catalog["actions"].append({**action, "id": f"{event['visual_event_id']}:ACTION_{fingerprint(action)[:16]}",
                                       "source": "temporal_visual_observation"})
    return catalog


def policy_projection(record: dict[str, Any]) -> dict[str, Any]:
    """Compatibility adapter, not a new policy or a raw completeness rewrite."""
    body = copy.deepcopy(record["observation"])
    body["action_evidence_ids"] = [row["id"] for row in record["evidence_catalog"]["actions"]]
    if body["moment_status"] == "reusable_state":
        # The legacy field is action OR moment completeness. A temporally
        # established reusable state satisfies the moment branch, not action.
        body["action_or_moment_complete"] = "true"
    return {**record, "schema_version": legacy.OBSERVATION_SCHEMA_VERSION, "observation": body,
            "temporal_completeness": {
                "moment_status": record["observation"]["moment_status"],
                "raw_action_or_moment_complete": record["observation"]["action_or_moment_complete"],
                "effective_action_or_moment_complete": body["action_or_moment_complete"],
            }}


def valid_record(record: dict[str, Any], event: dict[str, Any], movie_sha256: str | None = None) -> bool:
    try:
        return (record["schema_version"] == SCHEMA and record["evidence_profile"] == PROFILE
                and record["prompt_version"] == PROMPT_VERSION
                and record["event_id"] == event["visual_event_id"]
                and isinstance(record.get("created_at"), str)
                and record["event_input_identity"] == event_identity(event)
                and (movie_sha256 is None or record["source_movie_sha256"] == movie_sha256)
                and not validate_response(record["observation"], event, record["input_evidence"])
                and record["evidence_catalog"] == grounded_catalog(event, record["observation"], record["input_evidence"]))
    except (KeyError, TypeError, ValueError):
        return False


def latest_record(run: Path, event: dict[str, Any]) -> dict[str, Any] | None:
    source = run / "source_fingerprint.json"
    movie_sha = legacy._read(source).get("movie_sha256") if source.is_file() else None
    directory = record_path(run, event["visual_event_id"], "lookup").parent
    records = []
    for path in directory.glob("*.json"):
        try:
            record = legacy._read(path)
            if valid_record(record, event, movie_sha):
                records.append(record)
        except (OSError, ValueError):
            continue
    return max(records, key=lambda record: (record["created_at"], record["observation_fingerprint"]), default=None)


def persist(run: Path, event: dict[str, Any], body: dict[str, Any], *, observation_fp: str,
            evidence: dict[str, Any], provenance: dict[str, Any], movie_sha256: str) -> dict[str, Any]:
    catalog = grounded_catalog(event, body, evidence)
    destination = record_path(run, event["visual_event_id"], observation_fp)
    if destination.is_file():
        existing = legacy._read(destination)
        if not valid_record(existing, event, movie_sha256):
            raise RuntimeError("immutable temporal observation is invalid")
        return existing
    record = {"schema_version": SCHEMA, "evidence_profile": PROFILE, "prompt_version": PROMPT_VERSION,
              "event_id": event["visual_event_id"], "event_input_identity": event_identity(event),
              "source_movie_sha256": movie_sha256, "observation_fingerprint": observation_fp,
              "observation": body, "evidence_catalog": catalog, "input_evidence": evidence,
              "provider_provenance": provenance, "created_at": legacy._utc()}
    write_json(destination, record)
    return record


def recover_saved_response_offline(run: Path, event: dict[str, Any], attempt_path: Path) -> dict[str, Any]:
    """Recover an archived response using its exact evidence, without a provider or rendering.

    Historical diagnostics, payloads and cumulative usage remain immutable. The
    current source/event identity and archive fingerprints must still match.
    """
    attempt = legacy._read(attempt_path)
    source_sha = legacy._read(run / "source_fingerprint.json")["movie_sha256"]
    observation_fp = attempt["observation_fingerprint"]
    base = run / "semantic_observations/v2/responses" / event["visual_event_id"] / observation_fp
    if (not attempt_path.resolve().is_relative_to(base.resolve())
            or attempt["source_movie_sha256"] != source_sha
            or attempt["event_input_identity"] != event_identity(event)
            or attempt["expected_event_id"] != event["visual_event_id"]
            or attempt["evidence_profile"] != PROFILE
            or attempt["prompt_version"] != PROMPT_VERSION
            or attempt["evidence_fingerprint"] != fingerprint(attempt["input_evidence"])):
        raise ValueError("saved temporal response identity/evidence mismatch")
    errors = validate_response(attempt["response"], event, attempt["input_evidence"])
    if errors:
        raise ValueError("invalid saved temporal response: " + "; ".join(errors))
    saved = [legacy._read(path) for path in sorted(attempt_path.parent.glob("*.json"))]
    return persist(run, event, attempt["response"], observation_fp=observation_fp,
                   evidence=attempt["input_evidence"], movie_sha256=source_sha,
                   provenance=saved[-1]["cumulative_provider_provenance"])


def historical_rejection_identity(run, event, movie_sha256, provider, compatibility=None):
    """Conservative offline quarantine when the old runner saved no request.

    Snapshot all persisted input dependencies rather than inventing a request
    fingerprint. Provider/model, source, event, profile, prompt or explicit
    compatibility changes identify a separate future input. Validator revisions
    cannot release the quarantine.
    """
    compatibility = compatibility or {}
    dependencies = {}
    for name in ("narrative-v2/narrative_map.json", "technical_shots.json",
                 "active_picture.json", "movie_metadata.json"):
        path = run / name
        dependencies[name] = (__import__("hashlib").sha256(path.read_bytes()).hexdigest()
                              if path.is_file() else None)
    subtitles = run.parent.parent / "input" / run.name / "subtitles.srt"
    dependencies["subtitles.srt"] = (__import__("hashlib").sha256(subtitles.read_bytes()).hexdigest()
                                     if subtitles.is_file() else None)
    return {"event_input_identity": event_identity(event), "source_movie_sha256": movie_sha256,
            "provider": compatibility.get("provider", getattr(provider, "identifier", "unknown")),
            "model": compatibility.get("model", getattr(provider, "model", "unknown")),
            "evidence_profile": PROFILE, "prompt_fingerprint": fingerprint(PROMPT),
            "provider_compatibility": compatibility, "persisted_input_dependencies": dependencies}


def historical_rejection_path(run, event, identity):
    return (run / "semantic_observations/v2/historical_provider_blocks" / event["visual_event_id"]
            / f"{fingerprint(identity)}.json")


def quarantine_reported_rejection_offline(run, event, *, movie_sha256, provider_id, model):
    """Add a new diagnostic only; never alter archived responses or observations."""
    from types import SimpleNamespace
    identity = historical_rejection_identity(run, event, movie_sha256,
                                             SimpleNamespace(identifier=provider_id, model=model))
    path = historical_rejection_path(run, event, identity)
    if not path.exists():
        write_json(path, {"event_id": event["visual_event_id"], "status": "PROVIDER_BLOCKED",
                          "input_snapshot_fingerprint": fingerprint(identity), "input_identity": identity,
                          "exact_request_fingerprint": None,
                          "identity_basis": "operator_report_and_first_unresolved_enrichment_order",
                          "diagnostic": {"http_status": 400, "error_code": "invalid_prompt",
                                         "reason": "provider_input_rejection", "retryable": False},
                          "semantic_attempts": 0, "provider_input_rejections": 1,
                          "editorial_decision": "REVIEW", "renderable": False, "created_at": legacy._utc()})
    return path


def terminal_path(run, event, identity, status):
    # Provider rejection survives validator revision changes. Input/profile and
    # configured provider/model changes have separately identifiable cache keys.
    key = dict(identity)
    if status == "PROVIDER_BLOCKED":
        key.pop("validation_recovery_revision", None)
    return (run / "semantic_observations/v2/terminal" / event["visual_event_id"]
            / status / f"{fingerprint(key)}.json")


def persist_terminal(run, event, identity, status, *, evidence, semantic_attempts, detail=None):
    path = terminal_path(run, event, identity, status)
    if not path.exists():
        write_json(path, {"status": status, "event_id": event["visual_event_id"],
                          "event_input_identity": event_identity(event),
                          "observation_fingerprint": identity["observation_fingerprint"],
                          "execution_identity": identity, "evidence_profile": PROFILE,
                          "evidence_fingerprint": fingerprint(evidence),
                          "semantic_attempts": semantic_attempts,
                          "provider_input_rejections": int(status == "PROVIDER_BLOCKED"),
                          "diagnostic": detail or {}, "editorial_decision": "REVIEW",
                          "renderable": False, "created_at": legacy._utc()})


def batch_summary(run, events, blocked):
    records = [latest_record(run, event) for event in events]
    completed = sum(record is not None for record in records)
    review = sum(record is not None and legacy.policy_evaluate(policy_projection(record))["policy_decision"] == "REVIEW"
                 for record in records)
    provider_blocked = sum(status == "PROVIDER_BLOCKED" for status in blocked.values())
    validation_blocked = sum(status == "BLOCKED_VALIDATION" for status in blocked.values())
    remaining = len(events) - completed - len(blocked)
    return {"completed": completed, "review": review, "provider_blocked": provider_blocked,
            "validation_blocked": validation_blocked, "remaining": remaining,
            "status": "PARTIAL_PROVIDER" if remaining else (
                "COMPLETE_WITH_REVIEW" if review or blocked else "COMPLETE")}


def observe_events(input_dir: Path, events: list[dict[str, Any]], *, movie_sha256: str, provider: Any,
                   make_contact_sheet: Any, make_context: Any, active_picture=None,
                   provider_compatibility=None, preserve_legacy=True) -> dict[str, Any]:
    """No rendering/finalization. Completed records and saved responses are free.

    Each fingerprint/provider/model/recovery revision has at most two semantic
    response attempts across interruptions. Transport retries are provider-owned. Raw
    responses are durable before any validation/canonical persistence. The batch
    records terminal event failures and continues; earlier successes remain
    reusable. Saved diagnostics are historical; recovery always revalidates.
    """
    run = legacy.run_dir(input_dir)
    requests = reused = 0
    blocked = {}
    transport_stopped = False
    usage = {key: 0 for key in ("prompt_tokens", "cached_tokens", "response_tokens", "thinking_tokens", "total_tokens")}
    cost = 0.0
    for event in events:
        old_path = legacy.observation_path(run, event["visual_event_id"])
        if preserve_legacy and latest_record(run, event) is None and old_path.is_file():
            old = legacy._read(old_path)
            if not legacy.validate_observation(old, event):
                reused += 1
                continue
        quarantine_identity = historical_rejection_identity(run, event, movie_sha256, provider, provider_compatibility)
        if historical_rejection_path(run, event, quarantine_identity).is_file():
            blocked[event["visual_event_id"]] = "PROVIDER_BLOCKED"
            continue
        evidence: dict[str, Any] = {}
        sheet = make_contact_sheet(event, evidence)
        if evidence.get("evidence_profile") != PROFILE:
            raise ValueError("temporal input producer did not supply temporal_evidence_v2")
        context = make_context(event, legacy.canonical_evidence_catalog(event, evidence))
        compatibility = {"schema": SCHEMA, "prompt": PROMPT_VERSION, **(provider_compatibility or {})}
        obs_fp = legacy.observation_fingerprint(
            movie_sha256=movie_sha256, event=event,
            contact_sheet_sha256=__import__("hashlib").sha256(sheet).hexdigest(), active_picture=active_picture,
            narrative_context_fingerprint=fingerprint(context.get("narrative", {})),
            provider_compatibility=compatibility, evidence_profile=PROFILE, input_evidence=evidence)
        destination = record_path(run, event["visual_event_id"], obs_fp)
        if destination.is_file() and valid_record(legacy._read(destination), event, movie_sha256):
            reused += 1
            continue
        context = {**context, "visual_event_id": event["visual_event_id"], "expected_event_id": event["visual_event_id"],
                   "identity_instruction": "Return event_id exactly equal to expected_event_id.",
                   "canonical_evidence": legacy.canonical_evidence_catalog(event, evidence),
                   "temporal_evidence": evidence, "observation_schema": SCHEMA,
                   "action_evidence_types": KEEP_ACTION_EVIDENCE_TYPES}
        attempts_base = run / "semantic_observations/v2/responses" / event["visual_event_id"] / obs_fp
        attempts_dir, execution_identity = validation_attempt_directory(
            attempts_base, provider, provider_compatibility or {})
        rejection_identity = {**execution_identity, "request_input_fingerprint": fingerprint({
            "prompt": PROMPT, "context": context,
            "contact_sheet_sha256": __import__("hashlib").sha256(sheet).hexdigest()})}
        provider_block = terminal_path(run, event, rejection_identity, "PROVIDER_BLOCKED")
        if provider_block.is_file():
            blocked[event["visual_event_id"]] = "PROVIDER_BLOCKED"
            continue
        saved = [legacy._read(path) for path in sorted(attempts_dir.glob("*.json"))]
        for attempt in saved:
            # Recompute feedback in memory; never overwrite historical diagnostics.
            attempt["diagnostic"] = response_diagnostic(attempt["response"], event, evidence)
        recovered = False
        for attempt in saved:
            if not validate_response(attempt["response"], event, evidence):
                persist(run, event, attempt["response"], observation_fp=obs_fp, evidence=evidence,
                        provenance=saved[-1]["cumulative_provider_provenance"], movie_sha256=movie_sha256)
                reused += 1
                recovered = True
                break
        if recovered:
            continue
        event_usage = dict(saved[-1]["cumulative_provider_provenance"]["usage"]) if saved else dict.fromkeys(usage, 0)
        event_requests = int(saved[-1]["cumulative_provider_provenance"]["request_count"]) if saved else 0
        event_cost = float(saved[-1]["cumulative_provider_provenance"]["cost_usd"]) if saved else 0.0
        if saved:
            context["validation_feedback"] = saved[-1]["diagnostic"]
            context["retry_instruction"] = RETRY_INSTRUCTION
        if len(saved) >= MAX_RESPONSE_ATTEMPTS:
            persist_terminal(run, event, execution_identity, "BLOCKED_VALIDATION", evidence=evidence,
                             semantic_attempts=len(saved), detail=saved[-1]["diagnostic"])
            blocked[event["visual_event_id"]] = "BLOCKED_VALIDATION"
            continue
        for number in range(len(saved) + 1, MAX_RESPONSE_ATTEMPTS + 1):
            try:
                response = provider.generate(PROMPT, context, sheet)
            except Exception as error:
                detail = classify_provider_error(error)
                # Persist only allowlisted metadata; exception text, requests,
                # headers and prompt content are intentionally excluded.
                transport_count = max(1, int(detail.get("attempts") or 1))
                detail = {key: detail.get(key) for key in
                          ("reason", "http_status", "error_code", "retryable", "attempts")}
                detail["transport_request_count"] = transport_count
                if detail["reason"] == "provider_input_rejection":
                    requests += transport_count
                    persist_terminal(run, event, rejection_identity, "PROVIDER_BLOCKED", evidence=evidence,
                                     semantic_attempts=number - 1, detail=detail)
                    blocked[event["visual_event_id"]] = "PROVIDER_BLOCKED"
                    break
                # Transport policy remains provider-owned and bounded per call.
                # Journal failed calls separately from parsed semantic responses.
                transport_dir = attempts_dir / "transport_failures"
                number_failed = len(list(transport_dir.glob("*.json"))) + 1
                write_json(transport_dir / f"{number_failed:04d}.json",
                           {"execution_identity": execution_identity, "event_id": event["visual_event_id"],
                            "diagnostic": detail, "semantic_attempts": number - 1, "created_at": legacy._utc()})
                if detail["retryable"]:
                    requests += transport_count
                    transport_stopped = True
                    break
                raise
            body = response.data
            current_usage = response.usage or {}
            current_requests = max(1, int(response.attempts or 1))
            current_cost = legacy.estimate_openai_cost(current_usage) if response.provider == "openai" else 0.0
            requests += current_requests; cost += current_cost
            event_requests += current_requests; event_cost += current_cost
            for key in usage:
                usage[key] += int(current_usage.get(key, 0) or 0)
                event_usage[key] += int(current_usage.get(key, 0) or 0)
            provenance = {"provider": response.provider, "model": response.model,
                          "request_count": event_requests, "usage": dict(event_usage), "cost_usd": event_cost}
            # Raw first: even an invalid/missing identity must remain recoverable.
            attempt_path = attempts_dir / f"{number:04d}.json"
            attempt_record = {"response": body, "input_evidence": copy.deepcopy(evidence),
                              "input_context": copy.deepcopy(context),
                              "evidence_profile": PROFILE, "evidence_fingerprint": fingerprint(evidence),
                              "prompt_version": PROMPT_VERSION, "execution_identity": execution_identity,
                              "validation_recovery_revision": TEMPORAL_VALIDATION_RECOVERY_REVISION,
                              "attempt_number": number, "transport_request_count": current_requests,
                              "expected_event_id": event["visual_event_id"],
                              "returned_event_id": body.get("event_id"),
                              "event_input_identity": event_identity(event),
                              "source_movie_sha256": movie_sha256, "observation_fingerprint": obs_fp, "cumulative_provider_provenance": provenance,
                              "provider_trace": list(response.provider_trace or ()), "created_at": legacy._utc()}
            write_json(attempt_path, attempt_record)
            diagnostic = response_diagnostic(body, event, evidence)
            errors = diagnostic["errors"]
            write_json(attempt_path, {**attempt_record, "diagnostic": diagnostic})
            if errors:
                if number == MAX_RESPONSE_ATTEMPTS:
                    persist_terminal(run, event, execution_identity, "BLOCKED_VALIDATION", evidence=evidence,
                                     semantic_attempts=number, detail=diagnostic)
                    blocked[event["visual_event_id"]] = "BLOCKED_VALIDATION"
                    break
                context = {**context, "validation_feedback": diagnostic,
                           "retry_instruction": RETRY_INSTRUCTION}
                continue
            persist(run, event, body, observation_fp=obs_fp, evidence=evidence,
                    provenance=provenance, movie_sha256=movie_sha256)
            break
        if transport_stopped:
            break
    return {"requests": requests, "reused": reused, "usage": usage, "cost_usd": cost,
            "blocked_events": blocked, **batch_summary(run, events, blocked)}


def enrichment_plan(input_dir: Path) -> dict[str, Any]:
    """Read-only pre-call gate using authoritative final decisions and old facts.

    Known hard exclusions, high context and unsupported utility can be resolved
    locally without upgrading uncertain completeness. No prose is reinterpreted.
    """
    run = legacy.run_dir(input_dir)
    events = legacy._read(run / "visual_event_segments_v1.json").get("events", [])
    paid = []; resolved = []; completed = []; skipped = []
    for event in events:
        editorial = event.get("editorial", {})
        if (editorial.get("decision") != "REVIEW" or editorial.get("status") != "VALIDATED"
                or editorial.get("policy_version") != legacy.POLICY_VERSION
                or editorial.get("policy_reasons") != ["insufficient_observation"]):
            skipped.append(event["visual_event_id"]); continue
        temporal = latest_record(run, event)
        if temporal is not None:
            completed.append({"event_id": event["visual_event_id"],
                              **legacy.policy_evaluate(policy_projection(temporal))}); continue
        path = legacy.observation_path(run, event["visual_event_id"])
        if not path.is_file():
            skipped.append(event["visual_event_id"]); continue
        record = legacy._read(path)
        if legacy.validate_observation(record, event):
            skipped.append(event["visual_event_id"]); continue
        current = legacy.policy_evaluate(record)
        body = record["observation"]
        if current["policy_decision"] != "REVIEW":
            resolved.append({"event_id": event["visual_event_id"], **current}); continue
        if body["context_dependency"] == "high":
            # The unchanged policy rejects high context once its earlier
            # uncertainty branch is resolved. Record a necessary-condition
            # exclusion; never alter the observation to force a policy result.
            resolved.append({"event_id": event["visual_event_id"], "resolution": "ineligible_high_context_dependency"})
            continue
        paid.append(event["visual_event_id"])
    return {"schema_version": "temporal_enrichment_plan_v2", "evidence_profile": PROFILE,
            "eligible_event_ids": paid, "eligible": len(paid), "locally_resolved": resolved,
            "completed_temporal": completed, "skipped_event_ids": skipped,
            "maximum_generate_attempts": MAX_RESPONSE_ATTEMPTS * len(paid), "provider_requests": 0}


def enrich_reviews(input_dir: Path, *, movie_sha256: str, provider: Any, make_contact_sheet: Any,
                   make_context: Any, active_picture=None, max_events: int | None = None) -> dict[str, Any]:
    """Explicit targeted operation; never invoked by a normal production resume."""
    plan = enrichment_plan(input_dir)
    events = legacy._read(legacy.run_dir(input_dir) / "visual_event_segments_v1.json")["events"]
    selected = [event for event in events if event["visual_event_id"] in set(plan["eligible_event_ids"])]
    if max_events is not None:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        selected = selected[:max_events]
    return observe_events(input_dir, selected, movie_sha256=movie_sha256, provider=provider,
                          make_contact_sheet=make_contact_sheet, make_context=make_context,
                          active_picture=active_picture, preserve_legacy=False)


def run_enrichment(input_dir: Path, *, execute: bool = False, max_events: int | None = None,
                   provider: Any = None) -> dict[str, Any]:
    """Explicit recovery entry point; dry by default and never resumes production.

    Execution writes v2 observations, response archives and failure diagnostics. Policy
    results are returned, not written into event/narrative/shot/asset manifests.
    """
    if max_events is not None and max_events < 1:
        raise ValueError("max_events must be positive")
    plan = enrichment_plan(input_dir)
    if not execute or not plan["eligible"]:
        return plan
    run = legacy.run_dir(input_dir)
    events = legacy._read(run / "visual_event_segments_v1.json")["events"]
    selected_ids = plan["eligible_event_ids"][:max_events]
    selected = [item for item in events if item["visual_event_id"] in set(selected_ids)]
    from .broll_pilot import candidate_contact_sheet, semantic_request_context
    from .broll_semantics import build_semantic_provider_from_env
    from .srt import parse_srt_file
    from .utils import sha256_file
    from .active_picture import full_frame
    shots = legacy._read(run / "technical_shots.json")
    fps = float(shots["fps"])
    for item in selected:
        sample_plan(item, fps)  # Local cap/range validation before constructing a provider.
    movie_sha = sha256_file(input_dir / "movie.mp4")
    if movie_sha != legacy._read(run / "source_fingerprint.json")["movie_sha256"]:
        raise ValueError("source changed; temporal enrichment cannot use old events")
    active_path = run / "active_picture.json"
    if active_path.is_file():
        active = legacy._read(active_path)["active_picture"]
    else:
        video = legacy._read(run / "movie_metadata.json")["video"]
        active = full_frame(int(video["width"]), int(video["height"]))
    cues = parse_srt_file(input_dir / "subtitles.srt").cues
    segments = legacy._read(run / "narrative-v2/narrative_map.json")["segments"]
    if provider is None:
        provider = build_semantic_provider_from_env(
            env_file=input_dir.resolve().parents[1] / ".env", response_model=TemporalObservationResult,
            response_schema=RESPONSE_SCHEMA)
        if provider is None:
            raise RuntimeError("temporal enrichment requires a configured semantic provider")

    def sheet(item, evidence):
        return candidate_contact_sheet(input_dir / "movie.mp4", item, fps, evidence, active,
                                       evidence_profile=PROFILE)

    observed = observe_events(input_dir, selected, movie_sha256=movie_sha, provider=provider,
                              make_contact_sheet=sheet,
                              make_context=lambda item, catalog: semantic_request_context(item, cues, segments, {}, "FULL"),
                              active_picture=active, preserve_legacy=False)
    decisions = []
    for item in selected:
        record = latest_record(run, item)
        decisions.append({"event_id": item["visual_event_id"], **(
            legacy.policy_evaluate(policy_projection(record)) if record else
            {"policy_decision": "REVIEW", "status": observed["blocked_events"].get(item["visual_event_id"], "UNRESOLVED"),
             "renderable": False})})
    all_events = [item for item in events if item["visual_event_id"] in
                  set(plan["eligible_event_ids"]) | {row["event_id"] for row in plan["completed_temporal"]}]
    summary = batch_summary(run, all_events, observed["blocked_events"])
    return {"plan": plan, "observations": observed, "policy_results": decisions, **summary}
