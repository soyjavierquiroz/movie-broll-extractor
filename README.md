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

For an external v3 semantic map, import it instead of manually adding canonical
IDs, cue arrays, timestamps, density, or envelope metadata. The command archives
the exact external JSON under `narrative-v2/responses/` and writes a strict
canonical map at `--map` (or `--output`):

```bash
movie-broll narrative import-external \
  --input runs/<movie-id>/narrative-v2/chunks/NCHUNK_0001.input.json \
  --map runs/<movie-id>/narrative-v2/maps/NCHUNK_0001.narrative_map.json
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

### Phase F: face-safe verticals and selective repair

Single-person primary shots now require associated face evidence in addition to
person composition. Semantic `target_person_ids` remain authoritative: the
tracker binds the person, and YuNet supplies only face coordinates. Face
association rejects ambiguous overlapping people. `face_priority` centers the
associated face with lateral margin, using the existing bounded tracking and
source renderer. An initial repairable face failure gets one `face_priority`
retry; a source limit or inconclusive face audit stops automatic retries.

The CPU model is OpenCV Zoo `face_detection_yunet_2026may.onnx` (229,738 bytes),
pinned at revision `47534e27c9851bb1128ccc0102f1145e27f23f98`, SHA-256
`ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`.
The [official model directory](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet)
contains its MIT license and OpenCV 5 compatibility notes. Production detector
preflight provisions it once into `cache/models/movie-broll`, verifies the
checksum, and runs a smoke inference. There is no inference service or new
Python dependency. Existing corrupt files are reported rather than repeatedly
downloaded. Offline/missing models produce unavailable evidence. To explicitly
provision before an otherwise read-only legacy audit:

```bash
.venv/bin/python -c 'from movie_broll.face_safe import preflight; print(preflight())'
```

Audit existing assets without changing their packages or production ledger:

```bash
movie-broll audit-verticals runs/romper-el-circulo/assets
movie-broll audit-verticals runs/romper-el-circulo/assets --asset-id rc214 \
  --output runs/romper-el-circulo/rc214_vertical_audit_before.json
```

The default output is `runs/<movie>/vertical_repair_audit.json`. Each asset has
`PASS`, `REPAIR`, `SOURCE_LIMIT`, `AMBIGUOUS`, or `ERROR`, reasons, checksums and
sample evidence. Audit never provisions a model or changes an asset. It compares
five matching source/render frames per technical shot, measures the crop from
encoded pixels using image registration, and tests the face belonging to the
tracked target against that measured crop. Registration failure is inconclusive.
YuNet eye landmarks pass a geometric consistency gate; their states describe
retention of source landmark evidence, not an independent visibility/occlusion
classifier. No per-eye confidence is invented. Proxies alone cannot establish
face-safe PASS. Frames are bounded by the sampling constant; no detector runs at
movie frame rate and no frames accumulate across assets.

Selective replacement consumes only `REPAIR` rows, checks that their metadata
and vertical hashes are current, and performs a fresh audit before rendering:

```bash
movie-broll repair-verticals input/romper-el-circulo \
  --audit runs/romper-el-circulo/rc214_vertical_audit_before.json --asset-id rc214
```

Omitting `--asset-id` processes all `REPAIR` rows in the supplied audit; inspect
that report before doing so. Each repair encodes directly from the canonical
source and validates its MP4 and regenerated JPG before replacing anything.
Horizontal files, metadata for the horizontal rendition, and `asset.id` are
preserved. Only the vertical MP4, vertical JPG and necessary vertical metadata
are promoted. Backups, staged failures, plans and validation reports live under
`runs/<movie>/vertical_repairs/<asset-id>/<transaction>/`, outside routine work
cleanup. A failed validation leaves the old package intact and records
`REVIEW_VERTICAL`. A failed promotion rolls back from a checksummed backup.
Three flat-file replacements cannot be globally atomic to concurrent readers;
metadata is last, and a durable journal allows an interrupted replacement to be
rolled back on the next repair invocation. The supervisor lock excludes another
supervisor/repair; do not run a standalone production process concurrently.

Previously approved packages retain their legacy reuse eligibility when their
old fingerprint and package hashes still match. Phase F does not invalidate the
whole library; use the separate audit and selective repair commands.
# Movie B-Roll Extractor

For normal production, see [the operator runbook](docs/PRODUCTION_RUNBOOK.md). The canonical entry point is:

```bash
movie-broll run input/<movie-id>
```

The first run prepares the intentional External LLM Narrative Mapper handoff and reports `WAITING_EXTERNAL`; after the responses are placed in the canonical inbox, run the same command again. Use `movie-broll status input/<movie-id>` for a read-only progress summary.
