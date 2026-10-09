"""Cheap sampled-frame gate for text that dominates reusable B-roll."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from .active_picture import crop_frame

VERSION = "intrusive_text_v1"


def _regions(frame: np.ndarray) -> list[dict[str, float]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Both white subtitles and dark title lettering become edges. Connecting
    # nearby glyphs makes this a layout detector, not an OCR/transcription job.
    edges = cv2.Canny(gray, 60, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, frame.shape[1] // 70), max(3, frame.shape[0] // 180)))
    joined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = frame.shape[0] * frame.shape[1]
    result=[]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h / total
        # Text blocks are broad but not a whole scene edge. Requiring a modest
        # density filters thin horizon/architecture contours.
        density = cv2.contourArea(contour) / max(1, w * h)
        if w >= frame.shape[1] * .12 and h >= frame.shape[0] * .018 and area >= .012 and density >= .008:
            result.append({"x": x / frame.shape[1], "y": y / frame.shape[0],
                           "width": w / frame.shape[1], "height": h / frame.shape[0],
                           "area_fraction": area, "density": density})
    return result


def evaluate_frames(frames: Iterable[np.ndarray], *, minimum_persistent_samples: int = 3,
                    large_region_fraction: float = .045, subtitle_band_fraction: float = .025,
                    **_: Any) -> dict[str, Any]:
    observations=[]
    for index, frame in enumerate(frames):
        regions=_regions(frame)
        invasive=[]
        for region in regions:
            lower_third = region["y"] + region["height"] >= .68
            if region["area_fraction"] >= large_region_fraction or (lower_third and region["area_fraction"] >= subtitle_band_fraction):
                invasive.append(region)
        observations.append({"sample": index, "regions": regions, "invasive_regions": invasive})
    hits=[x for x in observations if x["invasive_regions"]]
    persistent=len(hits) >= minimum_persistent_samples
    reason = "INTRUSIVE_BURNED_IN_TEXT" if persistent else None
    return {"version": VERSION, "decision": "REJECT" if persistent else "PASS", "reason": reason,
            "sample_count": len(observations), "intrusive_samples": len(hits),
            "minimum_persistent_samples": minimum_persistent_samples, "samples": observations}


def inspect_event(movie: Path, event: dict[str, Any], *, sample_count: int = 5,
                  active_picture: dict[str, Any] | None = None, **settings: Any) -> dict[str, Any]:
    cap=cv2.VideoCapture(str(movie))
    try:
        start, end=float(event["start_seconds"]), float(event["end_seconds"])
        if end <= start:
            raise ValueError("text gate event must have positive duration")
        frames=[]
        for timestamp in np.linspace(start + .05, max(start + .05, end - .05), max(1, sample_count)):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000)
            ok, frame=cap.read()
            if ok:
                frames.append(crop_frame(frame,active_picture))
        return evaluate_frames(frames, **settings)
    finally:
        cap.release()
