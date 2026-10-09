"""Atlas-oriented producer-package publication readiness and human review."""
from __future__ import annotations

import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .processing_ledger import ProcessingLedger, fingerprint
from .utils import write_json

PUBLICATION_SCHEMA_VERSION = "atlas_publication_readiness_v1"
HUMAN_REVIEW_SCHEMA_VERSION = "vertical_human_review_v1"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _editorially_eligible(data: dict[str, Any]) -> bool:
    editorial = data.get("editorial", {})
    media = data.get("media", {})
    vertical = data.get("visual", {}).get("final_vertical", {})
    return (
        data.get("schema_version") == "asset_metadata_v1"
        and editorial.get("status") == "VALIDATED"
        and editorial.get("decision") == "KEEP"
        and not any(key in editorial for key in ("rejection_reason", "rejected", "rejection_flag"))
        and all(
            media.get(kind, {}).get("technical_validated") is True
            and media.get(kind, {}).get("semantic_validated") is True
            for kind in ("horizontal", "vertical")
        )
        and vertical.get("validation_status") in {"PASS", "REVIEW"}
    )


def human_review_status(data: dict[str, Any]) -> str:
    review = data.get("publication", {}).get("human_review", {})
    return str(review.get("status", "PENDING"))


def is_publish_ready(data: dict[str, Any]) -> bool:
    """Canonical producer readiness predicate; never consult legacy Hub fields."""
    if not _editorially_eligible(data):
        return False
    vertical = data["visual"]["final_vertical"]["validation_status"]
    if human_review_status(data) == "REJECTED":
        return False
    if vertical == "PASS":
        return True
    return vertical == "REVIEW" and human_review_status(data) == "APPROVED"


def is_atlas_ready(data: dict[str, Any]) -> bool:
    """Atlas ingest eligibility is intentionally identical to producer readiness."""
    return is_publish_ready(data)


def apply_publication_projection(
    data: dict[str, Any],
    human_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach canonical readiness and the deprecated compatibility projection."""
    result = copy.deepcopy(data)
    vertical_status = result.get("visual", {}).get("final_vertical", {}).get("validation_status")
    existing = result.get("publication", {}).get("human_review", {})
    review = dict(human_review or existing)
    if vertical_status == "PASS":
        review = {"status": "NOT_REQUIRED"}
    elif vertical_status == "REVIEW" and not review:
        review = {"status": "PENDING"}
    result["publication"] = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "vertical_validation_status": vertical_status,
        "human_review": review,
    }
    ready = is_publish_ready(result)
    result["publication"].update(publish_ready=ready, atlas_ready=ready)
    # This name is retained only for consumers that have not moved to Atlas.
    result["asset_hub_ready"] = ready
    return result


def _owned_run(run: Path) -> Path:
    resolved = run.resolve()
    if resolved.name in {"", "."} or resolved.parent.name != "runs":
        raise ValueError("review decision requires one concrete runs/<movie-id> directory")
    if not (resolved / "asset_registry.json").is_file():
        raise ValueError(f"run has no asset registry: {resolved}")
    return resolved


def _package(run: Path, asset_id: str) -> tuple[str, dict[str, Any], Path, str, dict[str, Any]]:
    registry = json.loads((run / "asset_registry.json").read_text(encoding="utf-8"))
    matches = [(event_id, entry) for event_id, entry in registry.get("events", {}).items()
               if entry.get("asset_id") == asset_id]
    if len(matches) != 1:
        raise ValueError(f"asset ID must resolve to exactly one registry event: {asset_id}")
    event_id, entry = matches[0]
    base = f"{asset_id}-{entry['slug']}"
    review, assets = run / "review", run / "assets"
    review_meta, assets_meta = review / f"{base}.json", assets / f"{base}.json"
    locations = [location for location, metadata in ((review, review_meta), (assets, assets_meta)) if metadata.is_file()]
    if len(locations) != 1:
        raise ValueError(f"asset must have exactly one package location: {asset_id}")
    location = locations[0]
    return event_id, entry, location, base, json.loads((location / f"{base}.json").read_text(encoding="utf-8"))


def _eligible_review_package(run: Path, asset_id: str) -> tuple[str, dict[str, Any], Path, str, dict[str, Any]]:
    event_id, entry, review, base, data = _package(run, asset_id)
    if review.name != "review":
        raise ValueError(f"asset is already in the published producer location: {asset_id}")
    # Import here so finalization remains independent of the decision interface.
    from .finalization import _asset_metadata_contract_valid, _complete_package
    if not _complete_package(review, base) or not _asset_metadata_contract_valid(data, review):
        raise ValueError(f"review package is not a current valid 5-file package: {asset_id}")
    if data.get("asset", {}).get("id") != asset_id:
        raise ValueError(f"review package producer identity mismatch: {asset_id}")
    if data.get("source_timeline", {}).get("visual_event_id") != event_id:
        raise ValueError(f"review package source event mismatch: {asset_id}")
    if data.get("visual", {}).get("final_vertical", {}).get("validation_status") != "REVIEW":
        raise ValueError(f"human review is allowed only for vertical REVIEW packages: {asset_id}")
    if not _editorially_eligible(data):
        raise ValueError(f"review package failed a hard editorial/technical eligibility gate: {asset_id}")
    return event_id, entry, review, base, data


def decide_review_vertical(run: Path, asset_id: str, decision: str, *, override_qa: bool = False, reason: str | None = None, reviewer: str | None = None) -> dict[str, Any]:
    """Persist one explicit APPROVE/REJECT decision for one pending review asset."""
    if decision not in {"APPROVE", "REJECT"}:
        raise ValueError("human review decision must be APPROVE or REJECT")
    run = _owned_run(run)
    review_dir = run / "editorial_reviews"
    record_path = review_dir / f"{asset_id}.json"
    status = "APPROVED" if decision == "APPROVE" else "REJECTED"
    event_id, entry, location, base, current = _package(run, asset_id)
    if record_path.exists() and location.name == "assets":
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("asset_id") != asset_id or existing.get("visual_event_id") != event_id:
            raise ValueError(f"existing human review record identity mismatch: {asset_id}")
        if existing.get("decision") != decision or existing.get("status") != status:
            raise ValueError(f"conflicting immutable human review decision exists: {asset_id}")
        if decision != "APPROVE" or not is_publish_ready(current):
            raise ValueError(f"published package does not match immutable approval: {asset_id}")
        from .closure import validate_package
        validate_package(location, base, current)
        return {"asset_id": asset_id, "decision": decision, "status": status,
                "publish_ready": True, "idempotent": True, "record": str(record_path)}
    event_id, entry, review, base, data = _eligible_review_package(run, asset_id)
    if decision == "APPROVE":
        hard = data["visual"]["final_vertical"].get("hard_failures", [])
        if hard and not override_qa:
            raise ValueError("automatic hard QA requires explicit --override-qa and --reason")
        if override_qa and not (reason and reason.strip()):
            raise ValueError("--override-qa requires a nonempty --reason")
        from .closure import validate_package
        validate_package(review, base, data)
    existing_record = None
    if record_path.exists():
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("asset_id") != asset_id or existing.get("visual_event_id") != event_id:
            raise ValueError(f"existing human review record identity mismatch: {asset_id}")
        if existing.get("decision") != decision or existing.get("status") != status:
            raise ValueError(f"conflicting immutable human review decision exists: {asset_id}")
        projected = apply_publication_projection(data, existing)
        if decision == "REJECT":
            if data != projected:
                write_json(review / f"{base}.json", projected)
            return {"asset_id": asset_id, "decision": decision, "status": status,
                    "publish_ready": False, "idempotent": True,
                    "record": str(record_path)}
        record = existing
        existing_record = existing
    else:
        record = {
            "schema_version": HUMAN_REVIEW_SCHEMA_VERSION,
            "asset_id": asset_id,
            "visual_event_id": event_id,
            "decision": decision,
            "status": status,
            "decided_at": _utc(),
            "reviewer": reviewer,
            "reason": reason,
            "automatic_qa_overridden": bool(override_qa),
            "decision_fingerprint": fingerprint({
                "asset_id": asset_id,
                "visual_event_id": event_id,
                "decision": decision,
                "vertical_validation": data["visual"]["final_vertical"],
            }),
        }
        # The decision record is durable first; a rerun deterministically completes
        # the metadata projection if an interruption occurs before package promotion.
        write_json(record_path, record)
    projected = apply_publication_projection(data, record)

    if decision == "REJECT":
        write_json(review / f"{base}.json", projected)
        location = review
    else:
        from .finalization import _asset_metadata_contract_valid, _complete_package, _promote_complete_package, _retire_package
        assets = run / "assets"
        final = [assets / f"{prefix}{base}{suffix}" for prefix, suffix in
                 (("", ".mp4"), ("v", ".mp4"), ("", ".jpg"), ("v", ".jpg"), ("", ".json"))]
        if any(path.exists() for path in final):
            raise ValueError(f"published destination is not clean for {asset_id}")
        stage = run / ".work" / "editorial_approval" / asset_id / record["decision_fingerprint"]
        stage.mkdir(parents=True, exist_ok=True)
        staged = [stage / path.name for path in final]
        # Preserve the exact validated media members; only metadata gains the
        # explicit decision/projection. This is promotion, never a rerender.
        for source, target in zip(
            [review / f"{base}.mp4", review / f"v{base}.mp4", review / f"{base}.jpg", review / f"v{base}.jpg"],
            staged[:4],
        ):
            shutil.copy2(source, target)
        write_json(staged[4], projected)
        if not _complete_package(stage, base) or not _asset_metadata_contract_valid(projected, stage):
            raise RuntimeError(f"staged approval package failed validation: {asset_id}")
        _promote_complete_package(staged, final)
        _retire_package(review, base)
        location = assets

    ledger = ProcessingLedger(run, run.name, {})
    ledger.stage(event_id, "human_review", "COMPLETE", decision=decision, human_review_status=status,
                 record=str(record_path.relative_to(run)), publish_ready=is_publish_ready(projected),
                 atlas_ready=is_atlas_ready(projected), asset_hub_ready=projected["asset_hub_ready"])
    ledger.stage(event_id, "finalization", "COMPLETE",
                 decision="HUMAN_APPROVED" if decision == "APPROVE" else "HUMAN_REJECTED",
                 publish_ready=is_publish_ready(projected), atlas_ready=is_atlas_ready(projected),
                 asset_hub_ready=projected["asset_hub_ready"], human_review_record=str(record_path.relative_to(run)))
    return {"asset_id": asset_id, "decision": decision, "status": status,
            "publish_ready": is_publish_ready(projected), "idempotent": existing_record is not None,
            "record": str(record_path), "location": str(location)}
