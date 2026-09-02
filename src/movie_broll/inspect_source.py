"""ffprobe normalization for source inspection."""
from __future__ import annotations
import json, shutil, subprocess
from pathlib import Path
from typing import Any
def _num(value: Any) -> float|None:
    try: return float(value) if value not in (None,"N/A","") else None
    except (TypeError,ValueError): return None
def _rate(value: Any) -> float|None:
    try:
        a,b=str(value).split("/"); return float(a)/float(b) if float(b) else None
    except (ValueError,ZeroDivisionError): return None
def normalize_ffprobe(payload: dict[str,Any], movie: Path) -> dict[str,Any]:
    streams=payload.get("streams",[]); videos=[s for s in streams if s.get("codec_type")=="video"]
    if not videos: raise ValueError("ffprobe found no video stream")
    v=videos[0]
    video={"stream_index":v.get("index"),"codec":v.get("codec_name"),"width":v.get("width"),"height":v.get("height"),"pixel_format":v.get("pix_fmt"),"fps":_rate(v.get("avg_frame_rate")),"raw_avg_frame_rate":v.get("avg_frame_rate"),"raw_r_frame_rate":v.get("r_frame_rate"),"time_base":v.get("time_base"),"start_time":_num(v.get("start_time")),"duration":_num(v.get("duration")),"bitrate":_num(v.get("bit_rate")),"nb_frames":v.get("nb_frames") if v.get("nb_frames") not in (None,"N/A") else None}
    audios=[{"stream_index":s.get("index"),"codec":s.get("codec_name"),"sample_rate":_num(s.get("sample_rate")),"channels":s.get("channels"),"channel_layout":s.get("channel_layout"),"start_time":_num(s.get("start_time")),"duration":_num(s.get("duration")),"bitrate":_num(s.get("bit_rate")),"language":s.get("tags",{}).get("language")} for s in streams if s.get("codec_type")=="audio"]
    subs=[{"stream_index":s.get("index"),"codec":s.get("codec_name"),"language":s.get("tags",{}).get("language"),"title":s.get("tags",{}).get("title")} for s in streams if s.get("codec_type")=="subtitle"]
    fmt=payload.get("format",{})
    return {"filename":movie.name,"absolute_path":str(movie.resolve()),"file_size_bytes":movie.stat().st_size,"duration_seconds":_num(fmt.get("duration")),"overall_bitrate":_num(fmt.get("bit_rate")),"container":{"format_name":fmt.get("format_name"),"format_long_name":fmt.get("format_long_name")},"video":video,"audio_tracks":audios,"subtitle_streams":{"count":len(subs),"streams":subs}}
def inspect_movie(movie: Path, ffprobe_binary: str="ffprobe") -> dict[str,Any]:
    binary=shutil.which(ffprobe_binary)
    if not binary: raise RuntimeError("ffprobe is unavailable; ensure it is on PATH")
    result=subprocess.run([binary,"-v","error","-show_format","-show_streams","-of","json",str(movie)],capture_output=True,text=True)
    if result.returncode: raise RuntimeError(f"ffprobe failed: {result.stderr.strip() or 'unknown error'}")
    try: return normalize_ffprobe(json.loads(result.stdout),movie)
    except json.JSONDecodeError as error: raise RuntimeError(f"ffprobe returned invalid JSON: {error}") from error
