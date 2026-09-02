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
600-second chunks with 60-second overlap, sends subtitle text/timestamps only to
Gemini, validates every response locally, and resumes validated checkpoints under
`runs/<movie-id>/narrative-v1/`. Use `--max-chunks 1` for a smoke test or
`--force` for explicit regeneration.

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
