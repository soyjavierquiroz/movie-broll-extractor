"""Phase 1 CLI."""
from __future__ import annotations
import argparse,json,sys,uuid,subprocess
from datetime import datetime,timezone
from pathlib import Path
from . import __version__
from .inspect_source import inspect_movie
from .srt import parse_srt_file,cue_statistics,validate_timeline
from .utils import sha256_file,write_json,write_jsonl
from .narrative import OVERLAP_SECONDS, TARGET_WINDOW_SECONDS, import_external_v3_map, prepare_narrative_inputs, validate_narrative_map
from .narrative_runner import export_external_comparison, run_narrative
from .narrative_consolidate import consolidate_narrative
def utc(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")


def _external_prepare_manifest(movie_id, window_seconds, overlap_seconds, chunk_count):
 return {
  "schema_version":"narrative_run_v2", "narrative_profile":"narrative_v3",
  "movie_id":movie_id, "provider":"external_llm", "model":"external_unspecified",
  "prompt_version":"srt_narrative_mapper_v3", "window_seconds":window_seconds,
  "overlap_seconds":overlap_seconds, "status":"PREPARED", "chunk_count":chunk_count,
  "prepared_at":utc(),
 }


def _write_process_outcome(input_dir, error, exit_code=2):
 classification=getattr(error,"outcome_classification",None)
 if classification is None:
  classification="TRANSIENT" if isinstance(error,(TimeoutError,ConnectionError)) else "DETERMINISTIC"
 try:
  run=input_dir.resolve().parents[1]/"runs"/input_dir.name
  write_json(run/"process_outcome.json",{"schema_version":"process_outcome_v1","classification":classification,"reason":str(error),"exit_code":exit_code,"completed_at":utc()})
  # The child is the first process that knows the precise failure.  Never leave
  # a durable RUNNING summary behind after it has terminated deterministically.
  summary_path=run/"progress_summary.json"
  if summary_path.is_file():
   try: summary=json.loads(summary_path.read_text(encoding="utf-8"))
   except json.JSONDecodeError: summary={}
   summary.update(status="FAILED",run_state="FAILED",updated_at=utc(),
                  failure_classification=classification,failure_reason=str(error))
   write_json(summary_path,summary)
 except OSError:
  pass

def main(argv=None):
 from .environment import load_environment
 load_environment()
 p=argparse.ArgumentParser(prog="movie-broll",description="Manifest-first movie source inspection."); sub=p.add_subparsers(dest="command",required=True); i=sub.add_parser("inspect",help="inspect a movie and synchronized external SRT"); i.add_argument("--movie",required=True);i.add_argument("--srt",required=True);i.add_argument("--run-dir",required=True)
 n=sub.add_parser("narrative",help="prepare and validate deterministic narrative mapper exchanges"); narrative_sub=n.add_subparsers(dest="narrative_command",required=True)
 prepare=narrative_sub.add_parser("prepare",help="create deterministic external-LLM input chunks")
 prepare.add_argument("--srt-cues",required=True); prepare.add_argument("--movie-id",required=True); prepare.add_argument("--output-dir",required=True); prepare.add_argument("--window-seconds",type=float,default=TARGET_WINDOW_SECONDS); prepare.add_argument("--overlap-seconds",type=float,default=OVERLAP_SECONDS); prepare.add_argument("--force",action="store_true")
 validate=narrative_sub.add_parser("validate",help="strictly validate one external narrative map")
 validate.add_argument("--input",required=True); validate.add_argument("--map",required=True)
 external_import=narrative_sub.add_parser("import-external",help="archive an external-v3 semantic map and write its deterministic canonical import")
 external_import.add_argument("--input",required=True,help="authoritative NCHUNK_####.input.json")
 external_import.add_argument("--map",required=True,help="external-v3 map; replaced with the canonical import after archival")
 external_import.add_argument("--output",help="canonical destination (default: --map)")
 run=narrative_sub.add_parser("run",help="automatically map SRT narrative chunks with Gemini")
 run.add_argument("input_dir", help="input/<movie-id> directory containing movie.mp4 and subtitles.srt")
 run.add_argument("--model", default="gemini-3.6-flash"); run.add_argument("--force", action="store_true")
 run.add_argument("--max-chunks", type=int, help="limit chunks for development smoke tests")
 consolidate=narrative_sub.add_parser("consolidate",help="deterministically reconcile validated narrative-v2 chunk maps")
 consolidate.add_argument("input_dir", help="input/<movie-id> directory associated with the current narrative-v2 run")
 export=narrative_sub.add_parser("export-comparison",help="export one v3 mapper chunk for provider-neutral boundary comparison")
 export.add_argument("input_dir", help="input/<movie-id> directory associated with the current narrative-v2 run")
 export.add_argument("--chunk",required=True,help="one NCHUNK_#### identifier")
 finalize_external=narrative_sub.add_parser("finalize-external",help="import, validate, and consolidate a prepared external-v3 narrative run")
 finalize_external.add_argument("input_dir",help="input/<movie-id> directory associated with the prepared external narrative run")
 preflight_cmd=sub.add_parser("preflight",help="report whether a title is ready to enter production without launching it")
 preflight_cmd.add_argument("input_dir",help="input/<movie-id> directory")
 run_cmd=sub.add_parser("run",help="run or resume complete production, including the external narrative handoff")
 run_cmd.add_argument("input_dir",help="input/<movie-id> directory")
 status_cmd=sub.add_parser("status",help="read persisted production progress without changing state")
 status_cmd.add_argument("input_dir",help="input/<movie-id> directory")
 policy_cmd=sub.add_parser("policy",help="evaluate cached semantic observations locally; never calls a provider")
 policy_sub=policy_cmd.add_subparsers(dest="policy_command",required=True)
 policy_evaluate=policy_sub.add_parser("evaluate",help="replay a local B-roll policy from cached observations")
 policy_evaluate.add_argument("input_dir",help="input/<movie-id> directory")
 policy_evaluate.add_argument("--policy",default="broll_policy_v1",help="local policy version (default: broll_policy_v1)")
 observe_cmd=sub.add_parser("observe",help="manage semantic observation cache")
 observe_sub=observe_cmd.add_subparsers(dest="observe_command",required=True)
 observe_migrate=observe_sub.add_parser("migrate",help="migrate persisted semantic evidence without provider calls")
 observe_migrate.add_argument("input_dir",help="input/<movie-id> directory")
 observe_plan=observe_sub.add_parser("plan-temporal-enrichment",help="read-only targeted temporal-v2 recovery plan; no provider calls")
 observe_plan.add_argument("input_dir",help="input/<movie-id> directory")
 utility_plan=observe_sub.add_parser("plan-visual-utility",help="read-only Policy V2 batch request plan; never calls providers")
 utility_plan.add_argument("input_dir")
 utility_plan.add_argument("--provider",default="openai")
 utility_plan.add_argument("--model",default="unspecified")
 utility_plan.add_argument("--transport-attempts",type=int,default=1)
 observe_enrich=observe_sub.add_parser("enrich-temporal",help="targeted temporal enrichment, dry-run unless --execute; never renders")
 observe_enrich.add_argument("input_dir",help="input/<movie-id> directory")
 observe_enrich.add_argument("--execute",action="store_true",help="explicitly enable bounded provider requests for eligible unresolved reviews")
 observe_enrich.add_argument("--max-events",type=int,help="bound the number of unresolved events handled")
 recovery_cmd=sub.add_parser("recover",help="recover observations from persisted semantic sidecars; never calls a provider or renders")
 recovery_cmd.add_argument("input_dir",help="input/<movie-id> directory")
 finalize_recovered=sub.add_parser("finalize-recovered",help="reserved gated recovery finalization; requires a separately accepted recovery report")
 finalize_recovered.add_argument("input_dir",help="input/<movie-id> directory")
 v=sub.add_parser("visual",help="technical visual analysis"); visual_sub=v.add_subparsers(dest="visual_command",required=True)
 smoke=visual_sub.add_parser("smoke",help="run representative technical shot-detection smoke windows")
 smoke.add_argument("input_dir",help="input/<movie-id> directory containing movie.mp4")
 smoke.add_argument("--threshold",type=float,help="debug detector threshold override")
 smoke.add_argument("--window-seconds",type=float,default=60.0,help="debug smoke window duration")
 audit=visual_sub.add_parser("threshold-audit",help="audit existing smoke windows only")
 audit.add_argument("input_dir",help="input/<movie-id> directory containing movie.mp4")
 event_audit=visual_sub.add_parser("audit-events",help="report duration and shot-grouping health for a persisted Visual Event manifest")
 event_audit.add_argument("input_dir",help="input/<movie-id> directory")
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
 benchmark=sub.add_parser("benchmark",help="run an explicitly requested read-only calibration")
 benchmark_sub=benchmark.add_subparsers(dest="benchmark_command",required=True)
 semantic_v9=benchmark_sub.add_parser("semantic-v9",help="run the paid, read-only OpenAI V9 semantic benchmark")
 semantic_v9.add_argument("input_dir",help="input/<movie-id> directory with a completed canonical event store")
 semantic_v9.add_argument("--run-id",help="optional unique output directory name under runs/<movie>/benchmarks/semantic-v9")
 semantic_v9.add_argument("--dry-run",action="store_true",help="prepare every semantic request through the provider boundary without calling a provider")
 semantic_v91=benchmark_sub.add_parser("semantic-v9.1",help="run the read-only OpenAI V9.1 semantic benchmark")
 semantic_v91.add_argument("input_dir",help="input/<movie-id> directory with a completed canonical event store")
 semantic_v91.add_argument("--run-id",help="optional unique output directory name under runs/<movie>/benchmarks/semantic-v9.1")
 semantic_v91.add_argument("--dry-run",action="store_true",help="prepare every semantic request through the provider boundary without calling a provider")
 reclassify=sub.add_parser("reclassify",help="run an explicit versioned semantic migration; never invoked by normal run")
 reclassify_sub=reclassify.add_subparsers(dest="reclassify_command",required=True)
 reclassify_v9=reclassify_sub.add_parser("semantic-v9",help="create or resume the isolated semantic V9 decision workspace")
 reclassify_v9.add_argument("input_dir",help="input/<movie-id> directory with canonical Visual Events")
 reclassify_v9.add_argument("--dry-run",action="store_true",help="verify request preparation without provider calls; does not create promotable results")
 reclassify_v9.add_argument("--max-events",type=int,help="bounded development/interruption test; remaining events resume later")
 reclassify_v91=reclassify_sub.add_parser("semantic-v9.1",help="create or resume the isolated semantic V9.1 decision workspace")
 reclassify_v91.add_argument("input_dir",help="input/<movie-id> directory with canonical Visual Events")
 reclassify_v91.add_argument("--dry-run",action="store_true",help="verify request preparation without provider calls; does not create promotable results")
 reclassify_v91.add_argument("--max-events",type=int,help="bounded development/interruption test; remaining events resume later")
 promote=sub.add_parser("promote",help="explicitly promote one complete versioned semantic migration")
 promote_sub=promote.add_subparsers(dest="promote_command",required=True)
 promote_v9=promote_sub.add_parser("semantic-v9",help="atomically activate a complete semantic V9 decision set")
 promote_v9.add_argument("input_dir",help="input/<movie-id> directory with a complete V9 workspace")
 promote_v9.add_argument("--no-finalize",action="store_true",help="promote decisions but leave newly promoted KEEPs pending finalization")
 promote_v91=promote_sub.add_parser("semantic-v9.1",help="atomically activate a complete semantic V9.1 decision set")
 promote_v91.add_argument("input_dir",help="input/<movie-id> directory with a complete V9.1 workspace")
 promote_v91.add_argument("--no-finalize",action="store_true",help="promote decisions but leave newly promoted KEEPs pending finalization")
 process_cmd=sub.add_parser("process",help="process one complete movie production job")
 process_cmd.add_argument("input_dir",help="input/<movie-id> directory containing canonical movie.mp4 and subtitles.srt")
 supervise_cmd=sub.add_parser("supervise",help="supervise resumable complete movie production")
 supervise_cmd.add_argument("input_dir",help="input/<movie-id> directory containing canonical movie.mp4 and subtitles.srt")
 supervise_cmd.add_argument("--stale-timeout-seconds",type=float,default=1800,help="progress.jsonl stale timeout (default: 1800)")
 supervise_cmd.add_argument("--grace-period-seconds",type=float,default=30,help="child termination grace period")
 reset_cmd=sub.add_parser("reset",help="remove derived production state while preserving the canonical narrative map")
 reset_cmd.add_argument("input_dir",help="canonical input/<movie-id> directory")
 reset_mode=reset_cmd.add_mutually_exclusive_group(required=True)
 reset_mode.add_argument("--dry-run",action="store_true",help="show exactly what would be removed without changing files")
 reset_mode.add_argument("--execute",action="store_true",help="execute the reset after all safety checks pass")
 for command in ('audit-verticals','repair-verticals'):
  vertical_cmd=sub.add_parser(command,help='audit existing verticals or selectively replace audited REPAIR renditions')
  vertical_cmd.add_argument('location',help='input/<movie>, runs/<movie>, or assets directory')
  vertical_cmd.add_argument('--asset-id',help='restrict to one existing producer asset ID')
  if command=='audit-verticals': vertical_cmd.add_argument('--output',help='report JSON outside assets')
  else: vertical_cmd.add_argument('--audit',required=True,help='vertical_repair_audit_v1 JSON')
 review_cmd=sub.add_parser('review-vertical',help='record an explicit human decision for one or more review packages')
 review_sub=review_cmd.add_subparsers(dest='review_vertical_command',required=True)
 for decision in ('approve','reject'):
  action=review_sub.add_parser(decision,help=f'{decision} explicitly named REVIEW_VERTICAL producer assets')
  action.add_argument('--override-qa',action='store_true',help='explicitly override visual QA only; requires --reason')
  action.add_argument('--reason',help='human decision rationale')
  action.add_argument('--reviewer',help='human reviewer identity')
  action.add_argument('--run',required=True,help='movie run ID under runs/, or an explicit runs/<movie-id> path')
  action.add_argument('--asset-id',action='append',required=True,help='one exact producer asset ID; repeat for an explicit bounded batch')
 close_cmd=sub.add_parser("close",help="validate and close a producer handoff without providers")
 close_cmd.add_argument("input_dir")
 a=p.parse_args(argv)
 if a.command == "close":
  from .closure import close
  try:
   return 0 if close(Path(a.input_dir))["verdict"] == "HANDOFF_COMPLETE" else 2
  except (OSError,ValueError,KeyError,TypeError) as error:
   print(f"VERDICT: BLOCKED\nBLOCKER: {error}\nSEMANTIC CALLS DURING CLOSURE: 0"); return 2
 if a.command == 'policy':
  try:
   from .semantic_observations import evaluate_cached_observations
   report=evaluate_cached_observations(Path(a.input_dir),policy_version=a.policy)
   if a.policy == 'broll_policy_v2':
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0
   counts=report['counts']
   print(f"POLICY: {report['policy_version']}")
   print(f"KEEP: {counts['KEEP']}  REJECT: {counts['REJECT']}  REVIEW: {counts['REVIEW']}")
   print(f"CACHED: {report['cached']}  MISSING: {report['missing']}")
   print("PROVIDER REQUESTS: 0  API COST: $0")
   return 0
  except (OSError,ValueError,FileNotFoundError,RuntimeError,json.JSONDecodeError) as error:
   print(f'error: {error}',file=sys.stderr); return 2
 if a.command == 'observe':
  try:
   if a.observe_command == 'plan-visual-utility':
    from .visual_utility_resolution import offline_plan
    print(json.dumps(offline_plan(Path(a.input_dir),provider=a.provider,model=a.model,transport_attempts=a.transport_attempts),ensure_ascii=False,indent=2))
    return 0
   if a.observe_command == 'plan-temporal-enrichment':
    from .temporal_semantics import enrichment_plan
    print(json.dumps(enrichment_plan(Path(a.input_dir)),ensure_ascii=False,indent=2))
    return 0
   if a.observe_command == 'enrich-temporal':
    from .temporal_semantics import run_enrichment
    print(json.dumps(run_enrichment(Path(a.input_dir),execute=a.execute,max_events=a.max_events),ensure_ascii=False,indent=2))
    return 0
   from .semantic_observations import migrate_existing_observations
   report=migrate_existing_observations(Path(a.input_dir))
   print(f"OBSERVATION MIGRATION: created={report['created']} reused={report['reused']} unavailable={report['unavailable']}")
   print("PROVIDER REQUESTS: 0  API COST: $0")
   return 0
  except (OSError,ValueError,FileNotFoundError,RuntimeError,json.JSONDecodeError) as error:
   print(f'error: {error}',file=sys.stderr); return 2
 if a.command == 'recover':
  try:
   from .semantic_recovery import recover_e02_report
   report=recover_e02_report(Path(a.input_dir)); counts=report['final_local_policy']
   print(f"RECOVERY: recovered={report['recovered_observations']} unresolved={report['unresolved_observations']}")
   print(f"KEEP: {counts['KEEP']}  REJECT: {counts['REJECT']}  REVIEW: {counts['REVIEW']}")
   print(f"EXISTING V8 PACKAGES: {report['existing_rendered_v8_packages']}")
   print("PROVIDER REQUESTS: 0  API COST: $0")
   print(f"REPORT: {Path(a.input_dir).resolve().parents[1] / 'runs' / Path(a.input_dir).name / 'e02_recovery_report.json'}")
   return 0
  except (OSError,ValueError,FileNotFoundError,RuntimeError,json.JSONDecodeError) as error:
   print(f'error: {error}',file=sys.stderr); return 2
 if a.command == 'finalize-recovered':
  print("error: recovery finalization is gated pending explicit acceptance of runs/<movie>/e02_recovery_report.json; no assets were changed",file=sys.stderr)
  return 2
 if a.command == 'reclassify':
  try:
   if a.max_events is not None and a.max_events < 1: raise ValueError('--max-events must be positive')
   from .semantic_reclassification import run_reclassification, run_reclassification_v9_1
   report=(run_reclassification_v9_1 if a.reclassify_command == 'semantic-v9.1' else run_reclassification)(Path(a.input_dir),dry_run=a.dry_run,max_events=a.max_events)
   totals=report['totals']
   tag=a.reclassify_command
   print(f"[{tag}] workspace: {report['workspace']}")
   print(f"[{tag}] processed: {report['processed']}; reused: {report['reused']}; complete: {report['complete']}")
   print(f"[{tag}] KEEP: {totals['KEEP']}; REJECT: {totals['REJECT']}; REVIEW: {totals['REVIEW']}; failures: {totals['failures']}")
   print(f"[{tag}] requests: {totals['requests']}; input tokens: {totals['prompt_tokens']}; cached tokens: {totals['cached_tokens']}; output tokens: {totals['response_tokens']}; cost: ${totals['cost_usd']:.6f}")
   return 0 if report['complete'] else 1
  except (OSError,ValueError,FileNotFoundError,RuntimeError,json.JSONDecodeError) as error:
   print(f'error: {error}',file=sys.stderr); return 2
 if a.command == 'promote':
  try:
   from .semantic_reclassification import promote_reclassification, promote_reclassification_v9_1
   result=(promote_reclassification_v9_1 if a.promote_command == 'semantic-v9.1' else promote_reclassification)(Path(a.input_dir),finalize=not a.no_finalize)
   print(f"[{a.promote_command}] promotion: {result['status']}; archived stale packages: {len(result['archived_stale_event_ids'])}; newly promoted KEEP: {len(result['newly_keep_event_ids'])}")
   return 0
  except (OSError,ValueError,FileNotFoundError,RuntimeError,json.JSONDecodeError) as error:
   print(f'error: {error}',file=sys.stderr); return 2
 if a.command == 'review-vertical':
  try:
   from .publication import decide_review_vertical
   raw=Path(a.run)
   run=raw if raw.is_absolute() or len(raw.parts)>1 else Path.cwd()/'runs'/raw
   if len(set(a.asset_id)) != len(a.asset_id): raise ValueError('duplicate --asset-id is not allowed')
   decision='APPROVE' if a.review_vertical_command=='approve' else 'REJECT'
   results=[]
   for asset_id in a.asset_id:
    result=decide_review_vertical(run,asset_id,decision,override_qa=a.override_qa,reason=a.reason,reviewer=a.reviewer); results.append(result)
    print(f"[review-vertical] {asset_id}: {result['status']}; publish_ready={result['publish_ready']}; idempotent={result['idempotent']}")
   print(f"[review-vertical] decision: {decision}; assets: {len(results)}; status: COMPLETE")
   return 0
  except (OSError,ValueError,KeyError,RuntimeError,json.JSONDecodeError) as error:
   print(f'error: {error}',file=sys.stderr); return 2
 if a.command == 'benchmark':
  try:
   from .semantic_v9_benchmark import BenchmarkSystemicFailure, run_semantic_v9_benchmark
   from .semantic_v9_1_benchmark import run_semantic_v9_1_benchmark
   tag=a.benchmark_command
   report=(run_semantic_v9_1_benchmark if tag == 'semantic-v9.1' else run_semantic_v9_benchmark)(Path(a.input_dir),run_id=a.run_id,dry_run=a.dry_run)
   print(f"[{tag}-benchmark] selected events: {report['selected_events']}; provider requests: {report['provider_requests']}; semantic results: {report['semantic_results']}; locally valid: {report['locally_valid_results']}; failed events: {report['failed_events']}")
   if report['semantic_results']:
    print(f"[{tag}-benchmark] effective KEEP: {report['counts']['KEEP']}; REVIEW: {report['counts']['REVIEW']}; REJECT: {report['counts']['REJECT']}")
   print(f"[{tag}-benchmark] failures by stage: {report['failures_by_stage']}; cost: ${report['cost_usd']:.6f}")
   print(f"[{tag}-benchmark] read-only verified: {report['read_only_verified']}")
   print(f"[{tag}-benchmark] output: {report['output']}")
   return 0
  except BenchmarkSystemicFailure as error:
   report=error.report
   print("BENCHMARK FAILED",file=sys.stderr)
   print(f"provider requests: {report['provider_requests']}",file=sys.stderr)
   print(f"successful semantic results: {report['semantic_results']}",file=sys.stderr)
   print(f"dominant failure: {report.get('dominant_failure')}",file=sys.stderr)
   print(f"output: {report['output']}",file=sys.stderr)
   return 2
  except (OSError,ValueError,FileNotFoundError,RuntimeError,json.JSONDecodeError) as error:
   print(f'error: {error}',file=sys.stderr); return 2
 if a.command in {'audit-verticals','repair-verticals'}:
  try:
   from .vertical_repair import audit_verticals, repair_verticals
   if a.command=='audit-verticals':
    report=audit_verticals(a.location,a.output,a.asset_id,reporter=lambda line: print(line,flush=True))
    return 1 if report['counts']['ERROR'] else 0
   report=repair_verticals(a.location,a.audit,a.asset_id,reporter=lambda line: print(line,flush=True))
   return 1 if any(r['status']=='REVIEW_VERTICAL' for r in report['results']) else 0
  except (OSError,ValueError,KeyError,RuntimeError,subprocess.CalledProcessError) as error:
   print(f'error: {error}',file=sys.stderr); return 2

 if a.command == "narrative":
  if a.narrative_command == "prepare":
   try:
    paths=prepare_narrative_inputs(Path(a.srt_cues),a.movie_id,Path(a.output_dir),a.window_seconds,a.overlap_seconds,a.force)
    write_json(Path(a.output_dir)/"narrative_run.json",_external_prepare_manifest(a.movie_id,a.window_seconds,a.overlap_seconds,len(paths)))
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
  if a.narrative_command == "finalize-external":
   try:
    from .narrative_finalize import finalize_external
    finalize_external(Path(a.input_dir))
    return 0
   except (OSError,ValueError,FileNotFoundError,RuntimeError,json.JSONDecodeError) as error:
    print(f"error: {error}",file=sys.stderr); return 2
  if a.narrative_command == "export-comparison":
   try:
    path=export_external_comparison(Path(a.input_dir),a.chunk); print(f"[narrative] comparison export: {path}"); return 0
   except (OSError,ValueError,FileNotFoundError) as error: print(f"error: {error}",file=sys.stderr); return 2
  if a.narrative_command == "import-external":
   try:
    path=import_external_v3_map(Path(a.input),Path(a.map),Path(a.output) if a.output else None)
    print(f"[narrative] external v3 imported and validated: {path}")
    return 0
   except (OSError,ValueError,FileNotFoundError,RuntimeError) as error: print(f"error: {error}",file=sys.stderr); return 2
  errors=validate_narrative_map(Path(a.input),Path(a.map))
  if errors:
   for error in errors: print(f"ERROR: {error}",file=sys.stderr)
   return 1
  print("VALID")
  return 0
 if a.command == "visual":
  if a.visual_command == "audit-events":
   try:
    from .visual_event_audit import audit_events, print_audit
    print_audit(audit_events(Path(a.input_dir)))
    return 0
   except (OSError,ValueError,FileNotFoundError,RuntimeError,json.JSONDecodeError) as error:
    print(f"error: {error}",file=sys.stderr); return 2
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
 if a.command == "preflight":
  try:
   from .production_preflight import preflight, print_preflight
   report=preflight(Path(a.input_dir)); print_preflight(report)
   return 0 if report["ready"] else 1
  except (OSError,ValueError,FileNotFoundError,RuntimeError,json.JSONDecodeError) as error:
   print(f"error: {error}",file=sys.stderr); return 2
 if a.command == "status":
  try:
   from .production_run import print_status, read_status
   print_status(read_status(Path(a.input_dir)))
   return 0
  except (OSError,ValueError,FileNotFoundError,RuntimeError,json.JSONDecodeError) as error:
   print(f"error: {error}",file=sys.stderr); return 2
 if a.command == "run":
  try:
   from .production_run import run as run_production
   return run_production(Path(a.input_dir),output=lambda line: print(line,flush=True))
  except (OSError,ValueError,FileNotFoundError,RuntimeError,json.JSONDecodeError) as error:
   print(f"error: {error}",file=sys.stderr); return 2
 if a.command == "pilot":
  try:
   if a.pilot_command == "finalize":
    from .finalization import finalize_pilot
    report=finalize_pilot(Path(a.input_dir),a.window)
    print(f"[pilot-finalize] assets: {report['completed']}; review: {report['review']}; reused: {report['reused']}")
    print(f"[pilot-finalize] reuse: semantic={report.get('semantic_reused',0)} horizontal={report.get('horizontal_reused',0)} vertical={report.get('reused',0)} review={report.get('review_reused',0)}")
    print(f"[pilot-finalize] directory: {report['assets']}")
    print(f"[pilot-finalize] status: {report['status']}")
    return 0 if report['status'] == 'COMPLETE' else 1
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
   print(f"[broll-pilot] semantic complete: {report.get('semantic_complete',0)}; reused: {report.get('semantic_reused',0)}; pending: {report.get('semantic_pending',0)}; retryable: {report.get('semantic_failed_retryable',0)}; failed: {report.get('semantic_failed_final',0)}")
   print(f"[broll-pilot] review reel: {output/'review_reel.mp4'}"); print(f"[broll-pilot] status: {report.get('status','COMPLETE')}"); return 0 if report.get('status','COMPLETE') == 'COMPLETE' else 1
  except (OSError,ValueError,FileNotFoundError,RuntimeError,subprocess.CalledProcessError) as error: print(f"error: {error}",file=sys.stderr); return 2
 if a.command == "reset":
  try:
   from .reset import reset_run
   report=reset_run(Path(a.input_dir),execute=bool(a.execute))
   print(f"[reset] movie: {report['movie_id']}")
   print(f"[reset] mode: {report['mode']}")
   print(f"[reset] run: {report['run']}")
   preserved=report['preserve']
   print(f"[reset] PRESERVE {preserved['path']} ({preserved['size_bytes']} bytes)")
   for item in report['delete']:
    print(f"[reset] DELETE {item['path']} ({item['size_bytes']} bytes)")
   print(f"[reset] delete total: {len(report['delete'])} paths; {report['delete_bytes']} bytes")
   print(f"[reset] status: {report['status']}")
   return 0
  except (OSError,ValueError,FileNotFoundError,RuntimeError) as error:
   print(f"error: {error}",file=sys.stderr)
   return 2
 if a.command == "process":
  try:
   from .production import process
   report=process(Path(a.input_dir),reporter=lambda line: print(line,flush=True))
   summary=report['summary']; source=summary['source']['movie_sha256'][:12]
   print(f"[process] movie: {summary['movie_id']}; source: {source}")
   print(f"[process] shots: {summary['technical_shots']}; visual events: {summary['visual_events']}")
   print(f"[process] semantic: {summary['semantic']['complete']} complete; {summary['semantic']['retryable']} retryable")
   print(f"[process] editorial: {summary['editorial']}")
   print(f"[process] status: {report['status']}")
   if report['status']=='COMPLETE': return 0
   from .production import ProductionFailure
   _write_process_outcome(Path(a.input_dir),ProductionFailure(f"production stopped with status {report['status']}"),exit_code=1)
   return 1
  except (OSError,ValueError,FileNotFoundError,RuntimeError,subprocess.CalledProcessError) as error:
   _write_process_outcome(Path(a.input_dir),error)
   print(f"error: {error}",file=sys.stderr); return 2
 if a.command == "supervise":
  if a.stale_timeout_seconds <= 0 or a.grace_period_seconds < 0: print("error: timeouts must be positive",file=sys.stderr); return 2
  try:
   from .supervisor import supervise
   return supervise(Path(a.input_dir),stale_timeout_seconds=a.stale_timeout_seconds,grace_period_seconds=a.grace_period_seconds)
  except (OSError,ValueError,FileNotFoundError,RuntimeError) as error: print(f"error: {error}",file=sys.stderr); return 2
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


if __name__ == "__main__":
 raise SystemExit(main())
