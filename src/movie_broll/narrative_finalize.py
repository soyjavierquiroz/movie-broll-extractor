"""Read-only-safe orchestration for prepared external narrative exchanges."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .narrative import import_external_v3_map, validate_narrative_map
from .narrative_consolidate import consolidate_narrative


EXTERNAL_PROVENANCE = {
    "provider": "external_llm",
    "model": "external_unspecified",
    "prompt_version": "srt_narrative_mapper_v3",
}
_INPUT = re.compile(r"^(NCHUNK_[0-9]+)\.input\.json$")
_RESPONSE = re.compile(r"^(NCHUNK_[0-9]+)\.external-v3\.json$")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _collect(directory: Path, pattern: re.Pattern[str], label: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"required {label} directory missing: {directory}")
    paths = sorted(directory.glob("NCHUNK_*.json"))
    result: dict[str, Path] = {}
    for path in paths:
        match = pattern.fullmatch(path.name)
        if not match:
            raise ValueError(f"invalid {label} filename: {path.name}")
        chunk_id = match.group(1)
        if chunk_id in result:
            raise ValueError(f"duplicate {label} chunk ID: {chunk_id}")
        result[chunk_id] = path
    if not result:
        raise ValueError(f"no {label} files found: {directory}")
    return result


def _validate_layout(input_dir: Path) -> tuple[Path, dict[str, Path], dict[str, Path], dict[str, Any]]:
    movie_id = input_dir.name
    root = input_dir.resolve().parents[1] / "runs" / movie_id
    run = root / "narrative-v2"
    cues = root / "source-v1" / "srt_cues.jsonl"
    if not cues.is_file():
        raise FileNotFoundError(f"canonical SRT cues missing: {cues}")
    manifest_path = run / "narrative_run.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"narrative run manifest missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("movie_id") != movie_id:
        raise ValueError("narrative_run movie_id does not match input directory")
    for key, expected in EXTERNAL_PROVENANCE.items():
        if manifest.get(key) != expected:
            raise ValueError(f"external narrative provenance mismatch: {key} must be {expected}")
    inputs = _collect(run / "chunks", _INPUT, "chunk input")
    responses = _collect(run / "external-v3-inbox", _RESPONSE, "external response")
    if set(inputs) != set(responses):
        missing = sorted(set(inputs) - set(responses))
        orphaned = sorted(set(responses) - set(inputs))
        details = []
        if missing:
            details.append("missing external responses: " + ", ".join(missing))
        if orphaned:
            details.append("orphan external responses: " + ", ".join(orphaned))
        raise ValueError("; ".join(details))
    for chunk_id, path in inputs.items():
        if _read_json(path).get("chunk", {}).get("chunk_id") != chunk_id:
            raise ValueError(f"chunk input ID does not match filename: {path.name}")
    return run, inputs, responses, manifest


def finalize_external(input_dir: Path, output: callable = print) -> dict[str, Any]:
    """Import every prepared external response, validate it, then consolidate.

    The complete correspondence check deliberately happens before the first
    import. Existing imports are reused only when their preserved raw response
    is byte-identical to the current inbox response.
    """
    run, inputs, responses, manifest = _validate_layout(input_dir)
    maps = run / "maps"
    imported = 0
    for chunk_id in sorted(inputs):
        source, destination = responses[chunk_id], maps / f"{chunk_id}.narrative_map.json"
        archive = run / "responses" / f"{chunk_id}.external-v3.import-source.json"
        if destination.is_file():
            if not archive.is_file() or archive.read_bytes() != source.read_bytes():
                raise ValueError(f"existing canonical map cannot be matched to authoritative external response: {chunk_id}")
        else:
            import_external_v3_map(inputs[chunk_id], source, destination)
            imported += 1
        errors = validate_narrative_map(inputs[chunk_id], destination)
        if errors:
            raise ValueError(f"invalid canonical map {chunk_id}: {errors[0]}")

    report = consolidate_narrative(input_dir, output=lambda _line: None)
    if report.get("status") != "PASS":
        raise RuntimeError("narrative consolidation did not pass")
    final_path = run / "narrative_map.json"
    if not final_path.is_file():
        raise FileNotFoundError(f"final narrative map missing: {final_path}")
    final = _read_json(final_path)
    analysis = final.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("final narrative map analysis is missing")
    for key, expected in EXTERNAL_PROVENANCE.items():
        if analysis.get(key) != expected:
            raise ValueError(f"final external provenance mismatch: analysis.{key} must be {expected}")
    result = {
        "movie_id": input_dir.name,
        "chunks": len(inputs),
        "external_responses": len(responses),
        "imported_maps": len(inputs),
        "maps_created": imported,
        "segments": len(final.get("segments", [])),
        "seams": len(report.get("chunk_boundaries", [])),
        **EXTERNAL_PROVENANCE,
        "status": "PASS",
    }
    output("=== NARRATIVE FINALIZE ===")
    for key in ("movie_id", "chunks", "external_responses", "imported_maps", "segments", "seams", "provider", "prompt_version", "status"):
        output(f"{key}: {result[key]}")
    return result
