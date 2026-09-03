# Movie B-Roll Extractor

Manifest-first tooling for building a small, curated collection of reusable movie B-roll. **SHOT != ASSET**: later phases may group multiple shots into one coherent visual asset.

Phase 2B automates the SRT narrative mapper with Gemini. It does not cut media, score candidates, or export assets.

## Setup and usage

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/movie-broll inspect --movie input/romper-el-circulo/movie.mp4 --srt input/romper-el-circulo/subtitles.srt --run-dir runs/romper-el-circulo/source-inspect
```

`input/` is for local copyrighted sources; `runs/`, `output/`, and `cache/` are generated/local. They are ignored by Git. The command produces `source_manifest.json`, `srt_cues.jsonl`, and `run_manifest.json`.

## Automatic narrative mapper

Put `movie.mp4` and `subtitles.srt` in `input/<movie-id>/`, configure
`GEMINI_API_KEY` once in the project `.env`, then run:

```bash
movie-broll narrative run input/<movie-id>
```

The command automatically inspects source material as needed, creates deterministic
1200-second chunks with 90-second overlap, sends subtitle text/timestamps only to
Gemini 3.6 Flash through the Interactions API, validates every response locally,
and resumes validated checkpoints under `runs/<movie-id>/narrative-v2/`. Use
`--max-chunks 1` for a smoke test or
`--force` for explicit regeneration.

Then consolidate the validated chunk maps without making any external request:

```bash
movie-broll narrative consolidate input/<movie-id>
```

The resulting flow is `SRT → Gemini narrative chunks → deterministic overlap
consolidation → global Narrative Map`. The global map and a reconciliation report
are written under `runs/<movie-id>/narrative-v2/`.

## Legacy narrative mapper interchange

`SRT → canonical cues → deterministic chunks → external LLM → validation`.
The extractor owns canonical timeline identity (`SRT_######`), chunk boundaries, and validation; the external LLM interprets narrative only. Prepare manually managed LLM inputs without overwriting existing exchanges:

```bash
movie-broll narrative prepare --srt-cues runs/romper-el-circulo/source-inspect-v1/srt_cues.jsonl --movie-id romper-el-circulo --output-dir input/romper-el-circulo --window-seconds 600 --overlap-seconds 60
movie-broll narrative validate --input input/romper-el-circulo/NCHUNK_0001.input.json --map input/romper-el-circulo/NCHUNK_0001.narrative_map.json
```

Chunks advance by 540 seconds (a 600-second window with 60 seconds overlap). A cue belongs to a chunk when its half-open interval intersects the half-open window; cues are never split. Use `--force` only to replace generated `.input.json` files.

Future phases flow from source inspection to shots, scene/context blocks, visual events, candidates, editorial decisions, and final MP4/JPG plus `asset_metadata_v1` JSON. The schemas directory establishes those contracts now. `asset_metadata_v1` represents independently exported assets in any supported orientation.

Keep this project simple: good enough is enough, route of least resistance, manifest first, quality over quantity. Do not modify `/opt/cortadora` or `/opt/apps/kurukin-asset-hub`; they are separate systems.
## Local vertical reframing

Vertical reframing is per technical source shot. The semantic event request returns a
bounded `shot_focus_plan`; it is not one provider request per shot. CPU geometry uses
local face detection and a project-owned YOLOv5n ONNX model at
`cache/models/movie-broll/yolov5n.onnx`. The first preflight downloads official
YOLOv5 v7.0 `yolov5n.pt` weights and exports ONNX locally; install the project
`detector` extra (`torch`, `onnx`) for that one-time export. No model or cache is read from another
application. Missing required person geometry is sent to `REVIEW_VERTICAL`, never
silently passed. `REFRAME_ALGORITHM_VERSION` is stored in the vertical fingerprint,
metadata, validation, and thumbnail-dependent package reuse path, so a reframe change
reprocesses only vertical outputs while retaining completed semantic and horizontal work.
