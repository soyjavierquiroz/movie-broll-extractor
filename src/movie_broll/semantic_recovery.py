"""Offline recovery of observation-v1 records from historical semantic sidecars.

This is intentionally a reporting/reconciliation boundary.  It never imports a
provider, calls a renderer, or changes the canonical V8 event store.  Recovered
observations live beside (rather than replacing) the earlier lossy migration so
that the evidence trail remains inspectable.
"""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .semantic_observations import (
    CONVERSATION_SIGNALS, HARD_REJECT_FLAGS, OBSERVATION_SCHEMA_VERSION,
    VISUAL_UTILITY_KINDS, canonical_evidence_catalog, observation_path,
    policy_evaluate, run_dir, validate_observation,
)
from .utils import write_json

RECOVERY_SCHEMA_VERSION = "e02_semantic_recovery_v1"
RECOVERY_ROOT = ("semantic_observations", "v1", "recovered")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def recovered_path(run: Path, event_id: str) -> Path:
    return run.joinpath(*RECOVERY_ROOT, "events", f"{event_id}.json")


def _v9_record(run: Path, event_id: str) -> dict[str, Any] | None:
    path = run / "semantic_reclassifications" / "semantic-v9" / "events" / f"{event_id}.json"
    try:
        record = _read(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    response = record.get("response")
    return record if record.get("state") == "VALID" and isinstance(response, dict) else None


def _benchmark_records(run: Path) -> dict[str, dict[str, Any]]:
    """Return only completed V9.1 benchmark rows, never dry-run placeholders."""
    candidates: list[Path] = []
    root = run / "benchmarks" / "semantic-v9.1"
    if root.is_dir():
        for directory in sorted(root.glob("*/events")):
            if "dry_run" not in directory.parent.name and any(directory.glob("*.json")):
                candidates.append(directory)
    if not candidates:
        return {}
    # A completed benchmark has effective decisions; this also makes a newer
    # incomplete/dry run directory unable to shadow the paid result.
    for directory in candidates:
        rows: dict[str, dict[str, Any]] = {}
        for path in sorted(directory.glob("*.json")):
            try:
                row = _read(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(row.get("event_id"), str) and row.get("effective_decision") in {"KEEP", "REJECT", "REVIEW"}:
                rows[row["event_id"]] = row
        if rows:
            return rows
    return {}


def _norm(value: Any) -> str:
    return value.casefold().strip() if isinstance(value, str) else ""


def _strings(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str) and item.strip()] if isinstance(value, list) else []


_DIALOGUE_WORDS = ("convers", "habla", "hablando", "escuch", "interlocutor", "talking", "discusi")
_INDEPENDENT_WORDS = ("teléfono", "telefono", "comiendo", "escrib", "leer", "examin", "oler", "cajón", "cajon", "volante", "conduce", "camina", "alej", "lleva", "sostiene", "abraza", "apoya", "tose", "escupe", "maquill", "bus", "autobús", "autobus")
_MOVEMENT_WORDS = ("camina", "alej", "marcha", "desplaza", "entra", "sale", "lleva")
_STATE_WORDS = ("recost", "cama", "dorm", "descansa", "repos", "ojos cerrados")
_PHYSICAL_WORDS = ("abraza", "apoya", "examen médico", "examen medico", "cuidado", "acompañamiento", "inclina")


def _contains(values: list[str], terms: tuple[str, ...]) -> bool:
    normalized = " | ".join(_norm(value) for value in values)
    return any(term in normalized for term in terms)


def _is_generic_dialogue(actions: list[str], interactions: list[str], relationships: list[dict[str, Any]]) -> bool:
    labels = [*actions, *interactions]
    labels.extend(row.get("type", "") for row in relationships if isinstance(row, dict))
    if not labels or not _contains(labels, _DIALOGUE_WORDS + ("cara a cara",)):
        return False
    return not _contains(actions + interactions, _INDEPENDENT_WORDS)


def _evidence_catalog(event: dict[str, Any], existing: dict[str, Any] | None, actions: list[str], objects: list[str]) -> tuple[dict[str, Any], list[str], list[str]]:
    """Use canonical IDs first; namespaced V9 IDs only fill absent bindings."""
    catalog = canonical_evidence_catalog(event)
    for key, source in (("actions", (existing or {}).get("evidence_catalog", {}).get("actions", [])),
                        ("objects", (existing or {}).get("evidence_catalog", {}).get("objects", []))):
        if not isinstance(source, list):
            continue
        for row in source:
            if isinstance(row, dict) and isinstance(row.get("id"), str):
                catalog[key].append(copy.deepcopy(row))

    def bind(values: list[str], key: str, prefix: str) -> list[str]:
        by_label = {_norm(row.get("canonical_label")): row.get("id") for row in catalog[key]
                    if isinstance(row, dict) and isinstance(row.get("id"), str)}
        result: list[str] = []
        for ordinal, label in enumerate(values, 1):
            identifier = by_label.get(_norm(label)) or f"{event['visual_event_id']}:V9_{prefix}_{ordinal:02d}"
            if identifier not in result:
                result.append(identifier)
            if not any(row.get("id") == identifier for row in catalog[key] if isinstance(row, dict)):
                catalog[key].append({"id": identifier, "canonical_label": label})
        return result
    return catalog, bind(actions, "actions", "ACTION"), bind(objects, "objects", "OBJECT")


def recover_observation_from_historical_semantics(
    event: dict[str, Any], *, v9: dict[str, Any] | None, existing: dict[str, Any] | None,
    v91: dict[str, Any] | None,
) -> dict[str, Any]:
    """Construct one observation solely from persisted structured fields.

    V9 supplies action, qualification, completeness, object, interaction, and
    focus-plan facts.  V9.1 only fills its explicitly structured utility and
    conversation-signal fields for its exact benchmark events.
    """
    event_id = str(event["visual_event_id"])
    old = (existing or {}).get("observation", {}) if isinstance(existing, dict) else {}
    response = (v9 or {}).get("response", {})
    visual = response.get("visual", {}) if isinstance(response, dict) else {}
    editorial = response.get("editorial", {}) if isinstance(response, dict) else {}
    qualification = editorial.get("keep_qualification", {}) if isinstance(editorial, dict) else {}
    actions = _strings(visual.get("actions"))
    interactions = _strings(visual.get("visible_interactions"))
    objects = _strings(visual.get("objects"))
    emotions = _strings(visual.get("visible_emotions"))
    relationships = response.get("relationships", []) if isinstance(response.get("relationships"), list) else []
    catalog, action_ids, object_ids = _evidence_catalog(event, existing, [*actions, *interactions], objects)

    benchmark_kind = (v91 or {}).get("visual_utility_kind")
    benchmark_signal = (v91 or {}).get("conversation_visual_signal")
    generic_dialogue = _is_generic_dialogue(actions, interactions, relationships)
    if benchmark_kind in VISUAL_UTILITY_KINDS:
        utility = benchmark_kind
    elif generic_dialogue:
        utility = "generic_dialogue_only"
    elif _contains(actions, _MOVEMENT_WORDS):
        utility = "movement"
    elif _contains(actions, _STATE_WORDS):
        utility = "useful_state"
    elif _contains(interactions + actions, _PHYSICAL_WORDS):
        utility = "physical_interaction"
    elif editorial.get("reusable_broll") is True and qualification.get("action_evidence_type") == "distinct_visible_action_or_reaction":
        utility = "concrete_action"
    elif qualification.get("action_evidence_type") == "generic_presence_or_movement":
        utility = "generic_presence_only"
    else:
        utility = old.get("visual_utility_kind") if old.get("visual_utility_kind") in VISUAL_UTILITY_KINDS else "unclear"
    if benchmark_signal in CONVERSATION_SIGNALS:
        signal = benchmark_signal
    elif generic_dialogue:
        signal = "generic_dialogue_only"
    elif utility == "physical_interaction":
        signal = "physical_interaction"
    elif utility in {"concrete_action", "movement", "object_activity"}:
        signal = "concrete_activity"
    else:
        signal = "none" if not generic_dialogue else "generic_dialogue_only"
    complete = editorial.get("action_or_moment_complete")
    if complete not in {"true", "false", "unclear"}:
        complete = old.get("action_or_moment_complete", "unclear")
    context = "low" if editorial.get("reusable_broll") is True and qualification.get("reusable_intent_type") == "specific_visual_need" else old.get("context_dependency", "unclear")
    if context not in {"low", "medium", "high", "unclear"}:
        context = "unclear"
    flags = old.get("technical_observations", {}) if isinstance(old.get("technical_observations"), dict) else {}
    flags = {flag: bool(flags.get(flag, False)) for flag in HARD_REJECT_FLAGS}
    people = visual.get("people", []) if isinstance(visual.get("people"), list) else []
    people_count = "0" if not people else "1" if len(people) == 1 else "2_plus"
    body = {
        "event_id": event_id, "represented_shot_ids": list(event.get("source_shot_ids", [])),
        "people_count": people_count, "visible_person_ids": list(old.get("visible_person_ids", [])),
        "action_evidence_ids": action_ids, "object_evidence_ids": object_ids,
        "visible_states": emotions, "movement": "present" if utility == "movement" else "none",
        "physical_interactions": interactions, "visible_reactions": [x for x in emotions if x != "neutral"],
        "conversation_present": "true" if generic_dialogue or any("convers" in _norm(x) or "talking" in _norm(x) for x in interactions) else "false",
        "conversation_visual_signal": signal, "visual_utility_kind": utility,
        "action_or_moment_complete": complete, "context_dependency": context,
        "technical_observations": flags, "shot_focus_plan": copy.deepcopy(visual.get("shot_focus_plan", old.get("shot_focus_plan", []))),
    }
    record = {"schema_version": OBSERVATION_SCHEMA_VERSION, "event_id": event_id,
              "observation_fingerprint": (existing or {}).get("observation_fingerprint", f"historical-v9:{event_id}"),
              "observation": body, "evidence_catalog": catalog,
              "provider_provenance": {"provider": "historical_persisted_semantics", "model": "V9/V9.1", "request_count": 0, "usage": {}, "cost_usd": 0.0},
              "recovery": {"source": "semantic-v9", "v9_structured": bool(v9), "v9_1_benchmark": bool(v91), "existing_observation": bool(existing)}}
    errors = validate_observation(record, event)
    if errors:
        raise ValueError(f"recovery produced invalid observation for {event_id}: {', '.join(errors)}")
    return record


def _existing_packages(run: Path) -> dict[str, list[dict[str, Any]]]:
    """Find existing 5-file packages by their immutable source event identity."""
    found: dict[str, list[dict[str, Any]]] = {}
    for location in ("assets", "review"):
        directory = run / location
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
            try:
                data = _read(path)
                event_id = data.get("source_timeline", {}).get("visual_event_id")
                asset = data.get("asset", {})
                base = f"{asset.get('id')}-{asset.get('slug')}"
                names = (f"{base}.mp4", f"v{base}.mp4", f"{base}.jpg", f"v{base}.jpg", f"{base}.json")
                if not isinstance(event_id, str) or path.name != names[-1] or not all((directory / name).is_file() for name in names):
                    continue
                found.setdefault(event_id, []).append({"location": location, "asset_id": asset.get("id"), "base": base})
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    return found


def recover_e02_report(input_dir: Path) -> dict[str, Any]:
    """Write recovered observations and the deterministic no-provider closure report."""
    run = run_dir(input_dir)
    store = _read(run / "visual_event_segments_v1.json")
    events = sorted((event for event in store.get("events", []) if isinstance(event, dict)), key=lambda row: (row.get("timeline_ordinal", 0), row.get("visual_event_id", "")))
    v91 = _benchmark_records(run)
    packages = _existing_packages(run)
    decisions: Counter[str] = Counter(); matrix: Counter[str] = Counter(); keeps: list[dict[str, Any]] = []
    recovered = unresolved = 0
    for event in events:
        event_id = event.get("visual_event_id")
        if not isinstance(event_id, str):
            continue
        try:
            existing = _read(observation_path(run, event_id)) if observation_path(run, event_id).is_file() else None
        except (OSError, ValueError, json.JSONDecodeError):
            existing = None
        record = recover_observation_from_historical_semantics(event, v9=_v9_record(run, event_id), existing=existing, v91=v91.get(event_id))
        write_json(recovered_path(run, event_id), record)
        result = policy_evaluate(record)
        decision = result["policy_decision"]
        decisions[decision] += 1; recovered += 1
        if decision == "REVIEW": unresolved += 1
        v8 = event.get("editorial", {}).get("decision", "REJECT")
        if v8 == "KEEP": matrix[f"V8 KEEP -> final {decision}"] += 1
        elif v8 == "REJECT": matrix[f"V8 REJECT -> final {decision}"] += 1
        if decision == "KEEP":
            keeps.append({"ordinal": event.get("timeline_ordinal"), "event_id": event_id,
                          "visual_utility_kind": record["observation"]["visual_utility_kind"],
                          "policy_reasons": result["policy_reasons"], "existing_package": "yes" if packages.get(event_id) else "no"})
    keeps.sort(key=lambda row: (row["ordinal"], row["event_id"]))
    report = {"schema_version": RECOVERY_SCHEMA_VERSION, "movie_id": input_dir.name,
              "total_events": len(events), "existing_rendered_v8_packages": sum(len(value) for value in packages.values()),
              "recovered_observations": recovered, "unresolved_observations": unresolved,
              "zero_api_usage": {"provider_requests": 0, "api_cost_usd": 0.0},
              "final_local_policy": {key: decisions[key] for key in ("KEEP", "REJECT", "REVIEW")},
              "v8_to_final_matrix": {key: matrix[key] for key in ("V8 KEEP -> final KEEP", "V8 KEEP -> final REJECT", "V8 KEEP -> final REVIEW", "V8 REJECT -> final KEEP", "V8 REJECT -> final REJECT", "V8 REJECT -> final REVIEW")},
              "final_keep": keeps,
              "finalization_plan": {"command": f"movie-broll finalize-recovered {input_dir}", "status": "NOT_EXECUTED_PENDING_REPORT_ACCEPTANCE", "behavior": ["reuse accepted existing valid 5-file packages", "render only accepted events without a package", "archive packages no longer accepted", "consolidate one Atlas-ready directory", "preserve chronological event IDs", "make no provider calls"]},
              "recovery_notes": {"v9_fields_recovered": ["visual.actions", "visual.objects", "visual.visible_interactions", "visual.visible_emotions", "visual.people", "visual.shot_focus_plan", "editorial.action_or_moment_complete", "editorial.reusable_broll", "editorial.keep_qualification.action_evidence_type", "editorial.keep_qualification.reusable_intent_type"], "v9_1_exact_fields_recovered": ["visual_utility_kind", "conversation_visual_signal"]}}
    write_json(run / "e02_recovery_report.json", report)
    return report
