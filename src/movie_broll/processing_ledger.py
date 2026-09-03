"""Small, durable operational ledger used by resumable B-roll stages."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import sha256_text, write_json

STATUSES = {"PENDING", "RUNNING", "COMPLETE", "FAILED_RETRYABLE", "FAILED_FINAL", "STALE"}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def fingerprint(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


class ProcessingLedger:
    """JSON is sufficient here: records are compact and every update is atomic."""
    def __init__(self, run_dir: Path, movie_id: str, inputs: dict[str, str]) -> None:
        self.run_dir, self.movie_id = run_dir, movie_id
        self.path, self.log_path, self.summary_path = run_dir / "processing_ledger.json", run_dir / "progress.jsonl", run_dir / "progress_summary.json"
        self.data = self._load(inputs)
        # A process which died while a request was outstanding must be safe to rerun.
        for event in self.data["events"].values():
            for name,stage in event.get("stages", {}).items():
                # A short-lived 3E.2 bug persisted an editorial disposition as a
                # generic lifecycle status. Preserve all completed work while
                # repairing only the vertical/finalization contexts it created.
                if name in {"vertical_validation", "finalization"} and stage.get("status") == "REVIEW_VERTICAL":
                    stage.update(status="COMPLETE", decision="REVIEW_VERTICAL", migrated_from_status="REVIEW_VERTICAL", updated_at=utc())
                if stage.get("status") == "RUNNING":
                    stage.update(status="FAILED_RETRYABLE", error="interrupted before durable completion", updated_at=utc())
        self.save()

    def _load(self, inputs: dict[str, str]) -> dict[str, Any]:
        try:
            existing = json.loads(self.path.read_text())
            if existing.get("movie_id") == self.movie_id:
                existing.setdefault("events", {})
                if inputs:
                    existing["inputs"] = inputs
                return existing
        except (OSError, json.JSONDecodeError):
            pass
        return {"schema_version": "processing_ledger_v1", "movie_id": self.movie_id,
                "created_at": utc(), "updated_at": utc(), "inputs": inputs, "events": {}}

    def save(self) -> None:
        self.data["updated_at"] = utc()
        write_json(self.path, self.data)

    def log(self, event: str, **fields: Any) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"ts": utc(), "event": event, **fields}
        # O_APPEND + fsync makes completed work auditable even if a later JSON write dies.
        fd = os.open(self.log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
            os.fsync(fd)
        finally:
            os.close(fd)

    def register(self, item: dict[str, Any], candidate_fingerprint: str) -> dict[str, Any]:
        eid = item["visual_event_id"]
        record = self.data["events"].get(eid)
        if record and record.get("candidate_fingerprint") != candidate_fingerprint:
            for stage in record.get("stages", {}).values():
                if stage.get("status") == "COMPLETE": stage["status"] = "STALE"
        if not record:
            record = {"visual_event_id": eid, "source_start_frame": item["start_frame"],
                      "source_end_frame_exclusive": item["end_frame_exclusive"], "stages": {}}
            self.data["events"][eid] = record
        record.update(candidate_fingerprint=candidate_fingerprint, source_shot_ids=item["source_shot_ids"])
        record["stages"].setdefault("grouping", {"status": "COMPLETE", "updated_at": utc()})
        record["stages"].setdefault("semantic", {"status": "PENDING"})
        record["stages"].setdefault("export", {"status": "PENDING"})
        record["stages"].setdefault("validation", {"status": "PENDING"})
        for name in ("horizontal_export", "horizontal_validation", "vertical_reframe",
                     "vertical_validation", "horizontal_thumbnail", "vertical_thumbnail",
                     "metadata", "cleanup", "finalization"):
            record["stages"].setdefault(name, {"status": "PENDING"})
        self.save()
        return record

    def stage(self, eid: str, stage: str, status: str, **fields: Any) -> None:
        if status not in STATUSES: raise ValueError(f"invalid ledger status {status}")
        item = self.data["events"][eid]["stages"].setdefault(stage, {})
        item.update(status=status, updated_at=utc(), **fields)
        self.save()
        self.log(f"{stage.upper()}_{status}", visual_event_id=eid, **fields)

    def summary(self, **values: Any) -> dict[str, Any]:
        result = {"movie_id": self.movie_id, "resume_safe": True, **values}
        write_json(self.summary_path, result)
        return result
