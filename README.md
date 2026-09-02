# Movie B-Roll Extractor

Manifest-first tooling for building a small, curated collection of reusable movie B-roll. **SHOT != ASSET**: later phases may group multiple shots into one coherent visual asset.

Phase 1 only performs deterministic source inspection and normalized external-SRT parsing. It does not cut media, call models, score candidates, or export assets.

## Setup and usage

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/movie-broll inspect --movie input/romper-el-circulo/movie.mp4 --srt input/romper-el-circulo/subtitles.srt --run-dir runs/romper-el-circulo/source-inspect
```

`input/` is for local copyrighted sources; `runs/`, `output/`, and `cache/` are generated/local. They are ignored by Git. The command produces `source_manifest.json`, `srt_cues.jsonl`, and `run_manifest.json`.

Future phases flow from source inspection to shots, scene/context blocks, visual events, candidates, editorial decisions, and final MP4/JPG plus `asset_metadata_v1` JSON. The schemas directory establishes those contracts now. `asset_metadata_v1` represents independently exported assets in any supported orientation.

Keep this project simple: good enough is enough, route of least resistance, manifest first, quality over quantity. Do not modify `/opt/cortadora` or `/opt/apps/kurukin-asset-hub`; they are separate systems.
