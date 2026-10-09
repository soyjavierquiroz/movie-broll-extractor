# Production runbook

## Normal workflow

1. Choose a lowercase, hyphenated movie ID, for example `mi-otra-yo-s03e02`.
2. Create `input/<movie-id>/`.
3. Put the video at `input/<movie-id>/movie.mp4` and its synchronized external subtitles at `input/<movie-id>/subtitles.srt`.
4. Start or resume the production run with one command:

```bash
movie-broll run input/mi-otra-yo-s03e02
```

On its first invocation the command writes canonical source-v1 data and prepares the external Narrative Mapper chunk files, then exits with `WAITING_EXTERNAL`. Send every chunk to the approved External LLM and place its External V3 responses in the printed inbox. Run the exact same command again: it verifies one-to-one coverage, imports, validates, consolidates narrative-v2, and continues technical analysis, semantics, exports, and packaging. It never uses Gemini for narrative mapping.

Re-running `movie-broll run` resumes durable checkpoints; do not delete the run directory. `movie-broll status input/<movie-id>` is a read-only concise progress view. `supervise` and `process` remain available as lower-level operational commands.

## Monitoring and recovery

Production writes `runs/<movie-id>/process.log`, `progress.jsonl`, and `progress_summary.json`. To watch the current log:

```bash
tail -f runs/<movie-id>/process.log
```

An SSH disconnect does not stop a normally running supervisor process. If it is stopped, run the same `movie-broll run ...` command again. It acquires the per-movie lock, automatically normalizes a lockless stale `RUNNING` marker to the existing interrupted/resumable state, and preserves completed checkpoints, assets, and packages. A real owner causes a safe refusal rather than duplicate work.

Transient Gemini failures such as HTTP 503 are retried by the provider boundary and remain resumable. If a failure becomes permanent, inspect `process.log` and `progress_summary.json`, correct credentials/input/provider availability, then run the same command again. Never expose or paste API keys into logs or issue reports.

## Output

`runs/<movie-id>/assets/` contains local producer packages that are ready for Atlas ingest. `runs/<movie-id>/review/` contains non-blocking vertical-review packages; production continues past them and canonical `publication.atlas_ready` is false. `asset_hub_ready` is retained only as a deprecated compatibility projection and must not drive publication logic.

Every package is exactly five files:

- `<base>.mp4`
- `v<base>.mp4`
- `<base>.jpg`
- `v<base>.jpg`
- `<base>.json`

A completed movie has `status: COMPLETE` and matching `segments_complete` / `segments_total` in `progress_summary.json`. Process the next episode by creating its own `input/<movie-id>/` directory and using the same one command.

## Explicit vertical-review decisions

Automatic vertical `PASS` packages are Atlas-ready. A `REVIEW_VERTICAL` package is not ready until an explicit human decision is recorded. Approve or reject only named producer IDs; there is intentionally no broad approval command:

```bash
movie-broll review-vertical approve --run <movie-id> --asset-id <producer-id>
movie-broll review-vertical reject --run <movie-id> --asset-id <producer-id>
```

Approval records are durable under `runs/<movie-id>/editorial_reviews/`. Approval preserves `visual.final_vertical.validation_status: REVIEW`, records `publication.human_review.status: APPROVED`, and projects `publication.publish_ready` / `publication.atlas_ready` to true. It promotes the existing validated five-file package without rerendering; this command does not ingest into Atlas.
