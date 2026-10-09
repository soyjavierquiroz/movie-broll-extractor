import hashlib
import json
from pathlib import Path

import pytest

from movie_broll.cli import main
from movie_broll.processing_ledger import ProcessingLedger
from movie_broll.finalization import reconcile_review_packages
from movie_broll.publication import (
    apply_publication_projection,
    decide_review_vertical,
    is_atlas_ready,
    is_publish_ready,
)
from movie_broll.utils import write_json


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _review_package(tmp_path, asset_id="m001", event_id="VE_000001"):
    run = tmp_path / "runs" / "film"
    review, assets = run / "review", run / "assets"
    review.mkdir(parents=True, exist_ok=True)
    assets.mkdir(exist_ok=True)
    slug, base = "reviewed-event", f"{asset_id}-reviewed-event"
    event = {
        "visual_event_id": event_id,
        "start_frame": 0,
        "end_frame_exclusive": 24,
        "source_shot_ids": ["S1"],
    }
    registry_path = run / "asset_registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {
        "schema_version": "asset_registry_v1", "movie_id": "film", "events": {},
    }
    registry["events"][event_id] = {"asset_id": asset_id, "slug": slug}
    write_json(registry_path, registry)
    ledger = ProcessingLedger(run, "film", {})
    ledger.register(event, "test")
    ledger.stage(event_id, "vertical_validation", "COMPLETE", decision="REVIEW_VERTICAL")
    ledger.stage(event_id, "finalization", "COMPLETE", decision="REVIEW_VERTICAL")
    members = [f"{base}.mp4", f"v{base}.mp4", f"{base}.jpg", f"v{base}.jpg"]
    import subprocess
    for name in members:
        vertical = name.startswith('v')
        size = '32x48' if vertical else '48x32'
        command = ['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', f'color=c=blue:s={size}:d=0.2']
        command += ['-frames:v', '1'] if name.endswith('.jpg') else ['-c:v', 'libx264', '-pix_fmt', 'yuv420p']
        subprocess.run(command + [str(review / name)], check=True, capture_output=True)
    def rendition(name, thumbnail, orientation):
        media = review / name
        image = review / thumbnail
        return {
            "file": name, "sha256": _digest(media), "size_bytes": media.stat().st_size,
            "technical_validated": True, "semantic_validated": True,
            "orientation": orientation, "duration_seconds": 1.0,
            "thumbnail": {"file": thumbnail, "sha256": _digest(image), "size_bytes": image.stat().st_size},
        }
    data = {
        "schema_version": "asset_metadata_v1",
        "asset": {"id": asset_id, "slug": slug, "source_movie_id": "film"},
        "source_timeline": {"visual_event_id": event_id, "start_seconds": 0.0, "end_seconds": 1.0},
        "editorial": {"status": "VALIDATED", "decision": "KEEP"},
        "visual": {"final_vertical": {"validation_status": "REVIEW"}},
        "media": {
            "horizontal": rendition(f"{base}.mp4", f"{base}.jpg", "landscape"),
            "vertical": rendition(f"v{base}.mp4", f"v{base}.jpg", "portrait"),
        },
    }
    data = apply_publication_projection(data)
    write_json(review / f"{base}.json", data)
    return run, base, data


def _metadata(run, location, base):
    return json.loads((run / location / f"{base}.json").read_text())


def test_review_without_human_decision_is_not_publishable(tmp_path):
    _, _, data = _review_package(tmp_path)
    assert not is_publish_ready(data)
    assert not is_atlas_ready(data)
    assert data["asset_hub_ready"] is False


def test_review_approval_is_publishable_and_preserves_review_fact(tmp_path):
    run, base, _ = _review_package(tmp_path)
    result = decide_review_vertical(run, "m001", "APPROVE")
    data = _metadata(run, "assets", base)
    assert result["publish_ready"] and not result["idempotent"]
    assert data["visual"]["final_vertical"]["validation_status"] == "REVIEW"
    assert data["publication"]["human_review"]["status"] == "APPROVED"
    assert data["publication"]["publish_ready"] and data["publication"]["atlas_ready"]
    assert data["asset_hub_ready"] is True
    assert not (run / "review" / f"{base}.json").exists()
    assert (run / "editorial_reviews" / "m001.json").is_file()


def test_review_rejection_remains_not_publishable(tmp_path):
    run, base, _ = _review_package(tmp_path)
    result = decide_review_vertical(run, "m001", "REJECT")
    data = _metadata(run, "review", base)
    assert not result["publish_ready"]
    assert data["visual"]["final_vertical"]["validation_status"] == "REVIEW"
    assert data["publication"]["human_review"]["status"] == "REJECTED"
    assert not data["publication"]["atlas_ready"] and data["asset_hub_ready"] is False


def test_automatic_pass_is_publishable_without_human_override(tmp_path):
    _, _, data = _review_package(tmp_path)
    data["visual"]["final_vertical"]["validation_status"] = "PASS"
    data = apply_publication_projection(data)
    assert data["publication"]["human_review"]["status"] == "NOT_REQUIRED"
    assert is_publish_ready(data) and is_atlas_ready(data)


def test_hard_technical_failure_cannot_be_human_approved(tmp_path):
    run, base, data = _review_package(tmp_path)
    data["media"]["vertical"]["technical_validated"] = False
    write_json(run / "review" / f"{base}.json", data)
    with pytest.raises(ValueError, match="current valid 5-file package"):
        decide_review_vertical(run, "m001", "APPROVE")
    assert not (run / "editorial_reviews" / "m001.json").exists()


def test_approval_is_idempotent(tmp_path):
    run, _, _ = _review_package(tmp_path)
    first = decide_review_vertical(run, "m001", "APPROVE")
    record = run / "editorial_reviews" / "m001.json"
    before = _digest(record)
    second = decide_review_vertical(run, "m001", "APPROVE")
    assert not first["idempotent"] and second["idempotent"]
    assert _digest(record) == before


def test_wrong_or_nonexistent_asset_fails_closed(tmp_path):
    run, _, _ = _review_package(tmp_path)
    with pytest.raises(ValueError, match="exactly one registry"):
        decide_review_vertical(run, "m999", "APPROVE")


def test_approval_of_one_asset_does_not_modify_another(tmp_path):
    run, _, _ = _review_package(tmp_path, "m001", "VE_000001")
    _, second_base, _ = _review_package(tmp_path, "m002", "VE_000002")
    other = run / "review" / f"{second_base}.json"
    before = _digest(other)
    decide_review_vertical(run, "m001", "APPROVE")
    assert other.is_file() and _digest(other) == before


def test_legacy_hub_projection_never_drives_readiness(tmp_path):
    _, _, data = _review_package(tmp_path)
    data["asset_hub_ready"] = True
    assert not is_publish_ready(data) and not is_atlas_ready(data)


def test_cli_approval_is_explicit_and_bounded(tmp_path):
    run, base, _ = _review_package(tmp_path)
    assert main(["review-vertical", "approve", "--run", str(run), "--asset-id", "m001"]) == 0
    assert _metadata(run, "assets", base)["publication"]["atlas_ready"] is True


def test_reconciliation_promotes_soft_only_package_atomically_and_is_idempotent(tmp_path, monkeypatch):
    run, base, data = _review_package(tmp_path)
    final = data["visual"]["final_vertical"]
    final.update(hard_failures=[], soft_warnings=["minor_centering_preference"], review_required=False)
    write_json(run / "review" / f"{base}.json", data)
    monkeypatch.setattr("movie_broll.finalization.render_vertical", lambda *_: pytest.fail("reconciliation must not rerender"))

    first = reconcile_review_packages(run)

    assert first["promoted"] == 1
    audit=json.loads((run / "review_reconciliation.json").read_text())
    assert audit["packages"] == [{"package": base, "decision": "PROMOTED", "reason": "no_hard_failures", "qa_source": "persisted_qa", "soft_warnings": ["minor_centering_preference"]}]
    assert not (run / "review" / f"{base}.json").exists()
    promoted = _metadata(run, "assets", base)
    assert promoted["asset"]["id"] == "m001"
    assert promoted["source_timeline"]["visual_event_id"] == "VE_000001"
    assert promoted["visual"]["final_vertical"]["validation_status"] == "PASS"
    assert promoted["visual"]["final_vertical"]["soft_warnings"] == ["minor_centering_preference"]
    assert promoted["publication"]["publish_ready"] is True
    assert len(list((run / "assets").glob(f"{base}*"))) == 3
    assert (run / "assets" / f"v{base}.mp4").is_file()
    assert len(list((run / "assets").iterdir())) == 5
    assert reconcile_review_packages(run)["promoted"] == 0


def test_reconciliation_keeps_hard_review_and_incomplete_packages(tmp_path):
    run, base, data = _review_package(tmp_path)
    data["visual"]["final_vertical"].update(hard_failures=["primary_subject_materially_clipped"], soft_warnings=[])
    write_json(run / "review" / f"{base}.json", data)
    result = reconcile_review_packages(run)
    assert result["hard_review"] == 1
    assert (run / "review" / f"{base}.json").is_file()
    assert not (run / "assets" / f"{base}.json").exists()


def test_reconciliation_locally_revalidates_legacy_package_before_promoting(tmp_path, monkeypatch):
    run, base, _ = _review_package(tmp_path)
    calls=[]
    monkeypatch.setattr(
        "movie_broll.finalization._local_legacy_vertical_qa",
        lambda run_path, video, data: calls.append((run_path, video.name)) or {
            "hard_failures": [], "soft_warnings": ["focused_person_persistently_at_crop_edge"],
            "locally_revalidated": True,
        },
    )
    result=reconcile_review_packages(run)
    assert result["promoted"] == 1 and calls == [(run, f"v{base}.mp4")]
    data=_metadata(run,"assets",base)
    assert data["visual"]["final_vertical"]["soft_warnings"] == ["focused_person_persistently_at_crop_edge"]


def test_reconciliation_persists_explicit_insufficient_legacy_evidence(tmp_path, monkeypatch):
    run, base, _ = _review_package(tmp_path)
    monkeypatch.setattr(
        "movie_broll.finalization._local_legacy_vertical_qa",
        lambda *_: {"hard_failures": [], "soft_warnings": [], "insufficient_reason": "legacy_evidence_insufficient"},
    )
    result=reconcile_review_packages(run)
    assert result["insufficient_evidence"] == 1
    data=_metadata(run,"review",base)
    final=data["visual"]["final_vertical"]
    assert final["review_required"] and final["review_reason"] == "legacy_evidence_insufficient"
