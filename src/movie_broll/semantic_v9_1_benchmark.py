"""Read-only V9.1 semantic calibration benchmark (30 fixed E02 controls)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .broll_pilot import candidate_contact_sheet
from .broll_semantics import SemanticProvider
from .semantic_v9_1 import (
    OpenAISemanticV9_1StructuredResult, PROMPT_V9_1, SEMANTIC_CONTRACT_V9_1,
    effective_decision, validate_response_v9_1,
)
from .semantic_v9_benchmark import run_semantic_v9_benchmark

BENCHMARK_NAME_V9_1 = "semantic-v9.1"
BENCHMARK_SCHEMA_V9_1 = "semantic_v9_1_read_only_benchmark_v1"

# Fixed identities make calibration comparable.  Categories are controls, not
# forced outcomes: the provider must still judge the actual contact sheet.
CONTROLS: tuple[tuple[str, int, str], ...] = (
    ("ordinary_action", 2, "medical_examination"), ("ordinary_action", 21, "writing"),
    ("ordinary_action", 31, "object_activity_makeup"), ("ordinary_action", 41, "carrying_food"),
    ("ordinary_action", 51, "phone_use"), ("ordinary_action", 62, "steering_wheel"),
    ("ordinary_action", 67, "looking_at_bus"), ("ordinary_action", 72, "walking"),
    ("ordinary_action", 121, "resting_state"), ("ordinary_action", 122, "drawer_search"),
    *( ("generic_dialogue_or_presence", ordinal, "generic_dialogue_control") for ordinal in (13, 54, 58, 81, 87, 91, 101, 106, 127, 141) ),
    ("borderline_reaction_interaction", 32, "expressive_conversation"),
    ("borderline_reaction_interaction", 35, "physical_support"),
    ("borderline_reaction_interaction", 37, "listening_reaction"),
    ("borderline_reaction_interaction", 113, "off_camera_conversation"),
    ("borderline_reaction_interaction", 145, "close_dialogue"),
    ("hard_negative", 1, "black_or_text"), ("hard_negative", 23, "title_card"),
    ("hard_negative", 155, "credits"), ("hard_negative", 166, "logo"), ("hard_negative", 170, "credits_and_logo"),
)


def _run(input_dir: Path) -> Path:
    return input_dir.resolve().parents[1] / "runs" / input_dir.name


def _read(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def select_v9_1_benchmark_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ordinal = {event.get("timeline_ordinal"): event for event in events}
    selected: list[dict[str, Any]] = []
    missing: list[int] = []
    for category, ordinal, label in CONTROLS:
        event = by_ordinal.get(ordinal)
        if not isinstance(event, dict):
            missing.append(ordinal); continue
        selected.append({**event, "_benchmark_category": category, "_benchmark_label": label})
    if missing:
        raise ValueError("V9.1 benchmark controls absent from canonical event store: " + ", ".join(map(str, missing)))
    if len(selected) != 30 or len({event["visual_event_id"] for event in selected}) != 30:
        raise ValueError("semantic V9.1 benchmark must contain 30 unique events")
    return selected


def _v9_decisions(input_dir: Path) -> dict[str, str | None]:
    directory = _run(input_dir) / "semantic_reclassifications" / "semantic-v9" / "events"
    decisions: dict[str, str | None] = {}
    if not directory.is_dir(): return decisions
    for path in directory.glob("*.json"):
        try:
            record = _read(path)
            decisions[str(record.get("event_id"))] = record.get("effective_decision") or record.get("response", {}).get("editorial", {}).get("decision")
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return decisions


def run_semantic_v9_1_benchmark(
    input_dir: Path, *, provider: SemanticProvider | None = None, run_id: str | None = None,
    contact_sheet: Callable[..., bytes] = candidate_contact_sheet, dry_run: bool = False,
) -> dict[str, Any]:
    run = _run(input_dir)
    store = _read(run / "visual_event_segments_v1.json")
    events = store.get("events")
    if not isinstance(events, list): raise ValueError("canonical event store has no events list")
    decisions = _v9_decisions(input_dir)
    selected = [{**event, "_v9_decision": decisions.get(str(event["visual_event_id"]))}
                for event in select_v9_1_benchmark_events(events)]
    report = run_semantic_v9_benchmark(
        input_dir, provider=provider, run_id=run_id, contact_sheet=contact_sheet, dry_run=dry_run,
        selected_events=selected, benchmark_name=BENCHMARK_NAME_V9_1, prompt=PROMPT_V9_1,
        validator=validate_response_v9_1, response_model=OpenAISemanticV9_1StructuredResult,
        contract_version="V9.1", benchmark_schema=BENCHMARK_SCHEMA_V9_1,
        contract_identity=SEMANTIC_CONTRACT_V9_1, effective_decider=effective_decision,
    )
    return report
