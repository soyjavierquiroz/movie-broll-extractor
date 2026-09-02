"""Phase 1 CLI."""
from __future__ import annotations
import argparse,sys,uuid
from datetime import datetime,timezone
from pathlib import Path
from . import __version__
from .inspect_source import inspect_movie
from .srt import parse_srt_file,cue_statistics,validate_timeline
from .utils import sha256_file,write_json,write_jsonl
from .narrative import OVERLAP_SECONDS, TARGET_WINDOW_SECONDS, prepare_narrative_inputs, validate_narrative_map
from .narrative_runner import run_narrative
def utc(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def main(argv=None):
 p=argparse.ArgumentParser(prog="movie-broll",description="Manifest-first movie source inspection."); sub=p.add_subparsers(dest="command",required=True); i=sub.add_parser("inspect",help="inspect a movie and synchronized external SRT"); i.add_argument("--movie",required=True);i.add_argument("--srt",required=True);i.add_argument("--run-dir",required=True)
 n=sub.add_parser("narrative",help="prepare and validate deterministic narrative mapper exchanges"); narrative_sub=n.add_subparsers(dest="narrative_command",required=True)
 prepare=narrative_sub.add_parser("prepare",help="create deterministic external-LLM input chunks")
 prepare.add_argument("--srt-cues",required=True); prepare.add_argument("--movie-id",required=True); prepare.add_argument("--output-dir",required=True); prepare.add_argument("--window-seconds",type=float,default=TARGET_WINDOW_SECONDS); prepare.add_argument("--overlap-seconds",type=float,default=OVERLAP_SECONDS); prepare.add_argument("--force",action="store_true")
 validate=narrative_sub.add_parser("validate",help="strictly validate one external narrative map")
 validate.add_argument("--input",required=True); validate.add_argument("--map",required=True)
 run=narrative_sub.add_parser("run",help="automatically map SRT narrative chunks with Gemini")
 run.add_argument("input_dir", help="input/<movie-id> directory containing movie.mp4 and subtitles.srt")
 run.add_argument("--model", default="gemini-2.5-flash"); run.add_argument("--force", action="store_true")
 run.add_argument("--max-chunks", type=int, help="limit chunks for development smoke tests")
 a=p.parse_args(argv)
 if a.command == "narrative":
  if a.narrative_command == "prepare":
   try:
    paths=prepare_narrative_inputs(Path(a.srt_cues),a.movie_id,Path(a.output_dir),a.window_seconds,a.overlap_seconds,a.force)
    print(f"[narrative] inputs written: {len(paths)}")
    return 0
   except (OSError,ValueError,FileExistsError) as error: print(f"error: {error}",file=sys.stderr); return 2
  if a.narrative_command == "run":
   if a.max_chunks is not None and a.max_chunks < 1: print("error: --max-chunks must be positive",file=sys.stderr); return 2
   try:
    manifest=run_narrative(Path(a.input_dir),model=a.model,force=a.force,max_chunks=a.max_chunks)
    return 0 if manifest["status"] == "COMPLETE" else 1
   except RuntimeError as error: print(f"ERROR: {error}",file=sys.stderr); return 2
   except (OSError,ValueError,FileNotFoundError) as error: print(f"error: {error}",file=sys.stderr); return 2
  errors=validate_narrative_map(Path(a.input),Path(a.map))
  if errors:
   for error in errors: print(f"ERROR: {error}",file=sys.stderr)
   return 1
  print("VALID")
  return 0
 movie,srt,run=Path(a.movie),Path(a.srt),Path(a.run_dir)
 for label,path in (("movie",movie),("SRT",srt)):
  if not path.is_file(): print(f"error: {label} file does not exist: {path}",file=sys.stderr);return 2
 if run.exists() and any(run.iterdir()): print(f"error: run directory must be new or empty: {run}",file=sys.stderr);return 2
 run.mkdir(parents=True,exist_ok=True); started=utc(); run_id="inspect-"+uuid.uuid4().hex[:12]
 try:
  md=inspect_movie(movie); parsed=parse_srt_file(srt); stats=cue_statistics(parsed.cues); tv=validate_timeline(parsed.cues,md["duration_seconds"] or 0)
  if parsed.malformed: tv["warnings"].append(f"{len(parsed.malformed)} malformed SRT cue block(s)"); tv["status"]="WARNING" if tv["status"]=="OK" else tv["status"]
  manifest={"schema_version":"source_manifest_v1","source":{"movie_id":movie.parent.name,"movie":{**md,"sha256":sha256_file(movie)},"srt":{"filename":srt.name,"absolute_path":str(srt.resolve()),"sha256":sha256_file(srt),"literal_transcription":False,"timing_assumption":"synchronized_external_srt","cue_count":stats["cue_count"],"first_cue_start_seconds":stats["first_cue_start"],"last_cue_end_seconds":stats["last_cue_end"],"statistics":stats}},"validation":{"movie_readable":True,"srt_readable":True,"srt_timeline_status":tv["status"],"warnings":tv["warnings"]+parsed.malformed,"errors":tv["errors"]}}
  write_json(run/"source_manifest.json",manifest);write_jsonl(run/"srt_cues.jsonl",[x.as_dict() for x in parsed.cues]);write_json(run/"run_manifest.json",{"schema_version":"run_manifest_v1","run_id":run_id,"command":"inspect","started_at":started,"completed_at":utc(),"status":"completed","producer":"movie_broll_extractor","producer_version":__version__,"outputs":{"source_manifest":"source_manifest.json","srt_cues":"srt_cues.jsonl"},"errors":[]})
  seconds=int(md["duration_seconds"] or 0); video=md["video"]
  print("[inspect] movie: readable");print(f"[inspect] duration: {seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}");print(f"[inspect] video: {video['width']}x{video['height']} @ {video['fps'] or 'unknown'}");print(f"[inspect] audio tracks: {len(md['audio_tracks'])}");print(f"[inspect] srt cues: {len(parsed.cues)}");print(f"[inspect] srt timeline: {tv['status']}");print("[inspect] source_manifest.json: written");print("[inspect] srt_cues.jsonl: written");print("[inspect] status: COMPLETE");return 0
 except Exception as e:
  write_json(run/"run_manifest.json",{"schema_version":"run_manifest_v1","run_id":run_id,"command":"inspect","started_at":started,"completed_at":utc(),"status":"failed","producer":"movie_broll_extractor","producer_version":__version__,"outputs":{},"errors":[str(e)]});print(f"error: {e}",file=sys.stderr);return 1
