"""Phase 1 CLI."""
from __future__ import annotations
import argparse,sys,uuid,subprocess
from datetime import datetime,timezone
from pathlib import Path
from . import __version__
from .inspect_source import inspect_movie
from .srt import parse_srt_file,cue_statistics,validate_timeline
from .utils import sha256_file,write_json,write_jsonl
from .narrative import OVERLAP_SECONDS, TARGET_WINDOW_SECONDS, prepare_narrative_inputs, validate_narrative_map
from .narrative_runner import run_narrative
from .narrative_consolidate import consolidate_narrative
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
 run.add_argument("--model", default="gemini-3.6-flash"); run.add_argument("--force", action="store_true")
 run.add_argument("--max-chunks", type=int, help="limit chunks for development smoke tests")
 consolidate=narrative_sub.add_parser("consolidate",help="deterministically reconcile validated narrative-v2 chunk maps")
 consolidate.add_argument("input_dir", help="input/<movie-id> directory associated with the current narrative-v2 run")
 v=sub.add_parser("visual",help="technical visual analysis"); visual_sub=v.add_subparsers(dest="visual_command",required=True)
 smoke=visual_sub.add_parser("smoke",help="run representative technical shot-detection smoke windows")
 smoke.add_argument("input_dir",help="input/<movie-id> directory containing movie.mp4")
 smoke.add_argument("--threshold",type=float,help="debug detector threshold override")
 smoke.add_argument("--window-seconds",type=float,default=60.0,help="debug smoke window duration")
 audit=visual_sub.add_parser("threshold-audit",help="audit existing smoke windows only")
 audit.add_argument("input_dir",help="input/<movie-id> directory containing movie.mp4")
 pilot=sub.add_parser("pilot",help="run bounded evaluation pilots")
 pilot_sub=pilot.add_subparsers(dest="pilot_command",required=True)
 broll=pilot_sub.add_parser("broll",help="create B-roll candidates from a persisted visual smoke window")
 broll.add_argument("input_dir",help="input/<movie-id> directory")
 broll.add_argument("--window",default="SW_02",metavar="WINDOW",help="persisted visual smoke window ID (default: SW_02)")
 finalize=pilot_sub.add_parser("finalize",help="finalize semantic KEEP assets into the flat production library")
 finalize.add_argument("input_dir",help="input/<movie-id> directory")
 finalize.add_argument("--window",default="SW_02",metavar="WINDOW",help="existing pilot window ID")
 select_next=pilot_sub.add_parser("select-next",help="select the next diverse narrative pilot window")
 select_next.add_argument("input_dir",help="input/<movie-id> directory")
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
  if a.narrative_command == "consolidate":
   try:
    report=consolidate_narrative(Path(a.input_dir))
    return 0 if report["status"] == "PASS" else 1
   except (OSError,ValueError,FileNotFoundError) as error: print(f"error: {error}",file=sys.stderr); return 2
  errors=validate_narrative_map(Path(a.input),Path(a.map))
  if errors:
   for error in errors: print(f"ERROR: {error}",file=sys.stderr)
   return 1
  print("VALID")
  return 0
 if a.command == "visual":
  if a.visual_command == "threshold-audit":
   try:
    from .visual import run_threshold_audit
    report=run_threshold_audit(Path(a.input_dir)); print(f"[visual threshold-audit] SW_03: {report['sw_03_classification']}"); return 0
   except (OSError,ValueError,FileNotFoundError,RuntimeError) as error: print(f"error: {error}",file=sys.stderr); return 2
  if a.window_seconds <= 0: print("error: --window-seconds must be positive",file=sys.stderr); return 2
  try:
   from .visual import run_visual_smoke
   manifest=run_visual_smoke(Path(a.input_dir),a.threshold,a.window_seconds)
   print(f"[visual smoke] status: {manifest['status']}; shots: {manifest['shot_count']}"); return 0 if manifest['status']=="COMPLETE" else 1
  except (OSError,ValueError,FileNotFoundError,RuntimeError) as error: print(f"error: {error}",file=sys.stderr); return 2
 if a.command == "pilot":
  try:
   if a.pilot_command == "finalize":
    from .finalization import finalize_pilot
    report=finalize_pilot(Path(a.input_dir),a.window)
    print(f"[pilot-finalize] assets: {report['completed']}; review: {report['review']}; reused: {report['reused']}")
    print(f"[pilot-finalize] directory: {report['assets']}")
    print("[pilot-finalize] status: COMPLETE")
    return 0
   if a.pilot_command == "select-next":
    from .pilot_selector import select_next as choose_pilot_window
    window=choose_pilot_window(Path(a.input_dir))
    print(f"[pilot-selector] selected: {window['window_id']}")
    print(f"[pilot-selector] start: {window['start_seconds']:.3f}")
    print(f"[pilot-selector] end: {window['end_seconds']:.3f}")
    print(f"[pilot-selector] narrative segment: {', '.join(window['narrative_segment_ids'])}")
    print(f"[pilot-selector] reason: {', '.join(window['selection_reason'])}")
    print("[pilot-selector] status: COMPLETE")
    return 0
   from .broll_pilot import run_broll_pilot
   report=run_broll_pilot(Path(a.input_dir),window_id=a.window)
   from .pilot_selector import mark_attempted
   mark_attempted(Path(a.input_dir),a.window,str(report.get('status','COMPLETE')))
   output=report['output']; print(f"[broll-pilot] window: {report['window']}"); print(f"[broll-pilot] shots: {report['shots']}"); print(f"[broll-pilot] visual events: {report.get('visual_events',report['candidates'])}"); print(f"[broll-pilot] candidates: {report['candidates']}")
   for key in ('KEEP','REVIEW','REJECT','exported'): print(f"[broll-pilot] {key}: {report[key]}")
   print(f"[broll-pilot] average KEEP duration: {report['average_keep_duration']:.1f}s")
   print(f"[broll-pilot] semantic complete: {report.get('semantic_complete',0)}; reused: {report.get('semantic_reused',0)}; pending: {report.get('semantic_pending',0)}")
   print(f"[broll-pilot] review reel: {output/'review_reel.mp4'}"); print(f"[broll-pilot] status: {report.get('status','COMPLETE')}"); return 0 if report.get('status','COMPLETE') == 'COMPLETE' else 1
  except (OSError,ValueError,FileNotFoundError,RuntimeError,subprocess.CalledProcessError) as error: print(f"error: {error}",file=sys.stderr); return 2
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
