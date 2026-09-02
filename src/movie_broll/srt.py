"""Small, deterministic SubRip parsing and timeline validation."""
from __future__ import annotations
from dataclasses import dataclass
import re
from pathlib import Path

_TIMING = re.compile(r"^\s*(?P<start>\d{1,3}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(?P<end>\d{1,3}:\d{2}:\d{2}[,.]\d{1,3})(?:\s+.*)?\s*$")
_STAMP = re.compile(r"^(?P<hours>\d{1,3}):(?P<minutes>\d{2}):(?P<seconds>\d{2})[,.](?P<millis>\d{1,3})$")

@dataclass(frozen=True)
class Cue:
    cue_id: str; source_index: int; start_seconds: float; end_seconds: float; text: str
    def as_dict(self) -> dict[str, object]:
        return {"cue_id": self.cue_id, "source_index": self.source_index, "start_seconds": self.start_seconds, "end_seconds": self.end_seconds, "duration_seconds": round(self.end_seconds-self.start_seconds, 3), "text": self.text}
@dataclass(frozen=True)
class ParseResult:
    cues: list[Cue]; malformed: list[str]

def timestamp_to_seconds(value: str) -> float:
    match = _STAMP.fullmatch(value.strip())
    if not match: raise ValueError(f"invalid SRT timestamp: {value!r}")
    h, m, s, ms = (int(match.group(key)) for key in ("hours", "minutes", "seconds", "millis"))
    if m >= 60 or s >= 60: raise ValueError(f"invalid SRT timestamp: {value!r}")
    return h * 3600 + m * 60 + s + ms / (10 ** len(match.group("millis")))

def parse_srt_text(text: str) -> ParseResult:
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")); cues=[]; malformed=[]
    for block_number, block in enumerate(blocks, 1):
        lines=[line.strip() for line in block.split("\n") if line.strip()]
        if not lines: continue
        timing_at=1 if re.fullmatch(r"\d+", lines[0]) else 0
        if len(lines)<=timing_at: malformed.append(f"block {block_number}: missing timing line"); continue
        timing=_TIMING.match(lines[timing_at])
        if not timing: malformed.append(f"block {block_number}: malformed timing line {lines[timing_at]!r}"); continue
        try: start,end=timestamp_to_seconds(timing.group("start")),timestamp_to_seconds(timing.group("end"))
        except ValueError as error: malformed.append(f"block {block_number}: {error}"); continue
        cue_text=" ".join(lines[timing_at+1:])
        if not cue_text: malformed.append(f"block {block_number}: missing cue text"); continue
        source_index=int(lines[0]) if timing_at else block_number
        cues.append(Cue(f"SRT_{len(cues)+1:06d}", source_index, start, end, cue_text))
    return ParseResult(cues, malformed)

def parse_srt_file(path: Path) -> ParseResult: return parse_srt_text(path.read_text(encoding="utf-8-sig", errors="replace"))
def cue_statistics(cues: list[Cue]) -> dict[str, float|int|None]:
    if not cues: return {"cue_count":0,"first_cue_start":None,"last_cue_end":None,"total_subtitle_covered_seconds":0.0,"average_cue_duration":0.0,"average_gap_seconds":0.0,"maximum_gap_seconds":0.0}
    durations=[cue.end_seconds-cue.start_seconds for cue in cues]; gaps=[max(0.0,c.start_seconds-p.end_seconds) for p,c in zip(cues,cues[1:])]
    return {"cue_count":len(cues),"first_cue_start":cues[0].start_seconds,"last_cue_end":cues[-1].end_seconds,"total_subtitle_covered_seconds":round(sum(durations),3),"average_cue_duration":round(sum(durations)/len(durations),3),"average_gap_seconds":round(sum(gaps)/len(gaps),3) if gaps else 0.0,"maximum_gap_seconds":round(max(gaps),3) if gaps else 0.0}
def validate_timeline(cues: list[Cue], movie_duration: float, tolerance: float=2.0) -> dict[str, object]:
    warnings=[]; errors=[]; prior=None
    for cue in cues:
        if cue.start_seconds>cue.end_seconds: errors.append(f"{cue.cue_id}: start is after end")
        if cue.start_seconds < -tolerance: errors.append(f"{cue.cue_id}: start is before zero")
        if prior is not None and cue.start_seconds < prior: errors.append(f"{cue.cue_id}: timestamp regression")
        prior=cue.start_seconds
    if cues and cues[-1].end_seconds>movie_duration+tolerance: warnings.append("last SRT cue ends after movie duration tolerance")
    return {"status":"ERROR" if errors else "WARNING" if warnings else "OK","warnings":warnings,"errors":errors}
