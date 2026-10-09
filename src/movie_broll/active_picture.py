"""Conservative, cached detection of the usable picture inside a video container."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .processing_ledger import fingerprint
from .utils import sha256_file, write_json

VERSION = "active_picture_v1"


def full_frame(width: int, height: int, *, sampled_frames: int = 0, confidence: float = 1.0) -> dict[str, Any]:
    return {"source_width": width, "source_height": height, "x": 0, "y": 0,
            "width": width, "height": height, "detection_profile": VERSION,
            "confidence": confidence, "sampled_frames": sampled_frames, "structural_bars": False}


def load_cached_active_picture(path: Path) -> dict[str, Any]:
    """Return canonical geometry from an active-picture cache without writing."""
    document = json.loads(path.read_text(encoding="utf-8"))
    active = document.get("active_picture") if isinstance(document, dict) else None
    if not isinstance(active, dict):
        raise ValueError(f"active-picture cache has no geometry object: {path}")
    missing = [key for key in ("x", "y", "width", "height") if key not in active]
    if missing:
        raise ValueError(
            f"active-picture geometry is missing {', '.join(missing)}: {path}"
        )
    return dict(active)


def crop_frame(frame: np.ndarray, active_picture: dict[str, Any] | None = None) -> np.ndarray:
    """Return content pixels in the active-picture coordinate system.

    Source-frame numbering and timestamps intentionally stay outside this helper.
    Every consumer that measures or detects content pixels should call it directly
    after decoding, before any colour conversion or detector invocation.
    """
    if frame is None or frame.size == 0:
        raise ValueError("cannot crop an empty decoded frame")
    if active_picture is None:
        return frame
    source_height, source_width = frame.shape[:2]
    x, y, width, height = (int(active_picture[key]) for key in ("x", "y", "width", "height"))
    declared_width = active_picture.get("source_width")
    declared_height = active_picture.get("source_height")
    if ((declared_width is not None and int(declared_width) != source_width)
            or (declared_height is not None and int(declared_height) != source_height)
            or x < 0 or y < 0 or width <= 0 or height <= 0
            or x + width > source_width or y + height > source_height):
        raise ValueError(
            "active_picture geometry is outside decoded source frame: "
            f"active={active_picture!r}, frame={source_width}x{source_height}"
        )
    return frame[y:y + height, x:x + width]


def _edge_runs(gray: np.ndarray, limit: int) -> tuple[int, int, int, int]:
    """Return dark rows/columns touching each edge, tolerating compression noise."""
    row_dark = np.mean(gray <= limit, axis=1) >= .985
    col_dark = np.mean(gray <= limit, axis=0) >= .985
    def begin(values: np.ndarray) -> int:
        return int(np.argmax(~values)) if np.any(~values) else len(values)
    return begin(row_dark), begin(row_dark[::-1]), begin(col_dark), begin(col_dark[::-1])


def detect_frames(frames: Iterable[np.ndarray], *, dark_luma_max: int = 18,
                  edge_coverage_required: float = .78, minimum_bar_pixels: int = 24,
                  minimum_active_fraction: float = .55) -> dict[str, Any]:
    rows: list[tuple[int, int, int, int]] = []
    width = height = 0
    for frame in frames:
        if frame is None or frame.size == 0:
            continue
        height, width = frame.shape[:2]
        rows.append(_edge_runs(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dark_luma_max))
    if not rows:
        raise ValueError("active-picture detector received no decodable frames")
    values = np.asarray(rows)
    # A percentile prevents a one-off dark scene from establishing a bar while
    # retaining a stable bar even if a few samples include fade-ins/out.
    required = max(1, int(np.ceil(len(rows) * edge_coverage_required)))
    stable = [int(np.partition(values[:, i], len(rows) - required)[len(rows) - required]) for i in range(4)]
    top, bottom, left, right = [v if v >= minimum_bar_pixels else 0 for v in stable]
    active_w, active_h = width - left - right, height - top - bottom
    if active_w <= 0 or active_h <= 0 or active_w / width < minimum_active_fraction or active_h / height < minimum_active_fraction:
        return full_frame(width, height, sampled_frames=len(rows), confidence=0.0)
    structural = any((top, bottom, left, right))
    evidence = [float(np.mean(values[:, i] >= stable[i])) if stable[i] else 1.0 for i in range(4)]
    return {"source_width": width, "source_height": height, "x": left, "y": top,
            "width": active_w, "height": active_h, "detection_profile": VERSION,
            "confidence": round(min(evidence), 3), "sampled_frames": len(rows),
            "structural_bars": structural, "bars": {"top": top, "bottom": bottom, "left": left, "right": right}}


def detect_video(movie: Path, *, sample_count: int = 18, **thresholds: Any) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(movie))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or count < 1:
            raise RuntimeError(f"cannot open video for active-picture detection: {movie}")
        # Avoid the very first/last frame where fades and logos are common.
        indexes = np.linspace(max(0, int(count * .03)), max(0, int(count * .97) - 1), max(3, sample_count), dtype=int)
        frames = []
        for index in sorted(set(indexes.tolist())):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        return detect_frames(frames, **thresholds)
    finally:
        cap.release()


def load_or_detect(movie: Path, run: Path, profile: dict[str, Any]) -> dict[str, Any]:
    path = run / "active_picture.json"
    source_sha = sha256_file(movie)
    settings = {k: v for k, v in profile.items() if k != "enabled"}
    key = fingerprint({"version": VERSION, "source_movie_sha256": source_sha, "settings": settings})
    try:
        cached = json.loads(path.read_text())
        if cached.get("fingerprint") == key:
            return load_cached_active_picture(path)
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        pass
    result = detect_video(movie, **settings) if profile.get("enabled", True) else full_frame(0, 0, confidence=0.0)
    write_json(path, {"schema_version": "active_picture_v1", "fingerprint": key,
                      "source_movie_sha256": source_sha, "active_picture": result})
    return result


def crop_dimensions(active: dict[str, Any]) -> tuple[int, int]:
    return min(int(active["width"]), round(int(active["height"]) * 3 / 4)), int(active["height"])


def source_x(active: dict[str, Any], active_x: float) -> float:
    return float(active["x"]) + active_x
