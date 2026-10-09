# Automatic batched visual utility resolution

The generic resolver is implemented and integrated into new `movie-broll run input/<episode>` jobs through the existing supervisor/process lifecycle. New jobs default to Policy V2. Historical manifests retain their recorded policy; installing this code does not implicitly migrate them. No E03 provider work, migration, rendering or production was executed.

## Batch design

The default is **eight independent events per request**, bounded by **128 temporal samples**, **120,000 event-context UTF-8 bytes**, and **24,000 estimated output tokens**. Smaller groups are selected deterministically when limits require them. A final partial batch is valid. Each event has a separate, labeled contact-sheet image; sheets are sent as separate images rather than compressed into one giant montage. OpenAI batch sheets use high image detail. Both existing provider adapters accept multiple images while retaining their single-image interface.

Each member supplies its event ID, source frame range, canonical technical-shot IDs/boundaries, temporal sample identities, person/evidence catalog and relevant saved visual observation. No subtitles enter this request. Provider judgments describe visible moments and suggested semantic spans; the local planner selects final render windows.

`visual_utility_observation_v1` extends the existing grounded moment contract. It contains reuse true/false/unclear, visual moment type, sustained-state assessment, confidence, visual/narrative context dependency, temporal citations, local action/state ranges and an assessment for every supported utility category. These cover conversation, therapy/session, discussion/argument, meditation/stillness, reaction, sustained emotion, physical/nonverbal interaction, walking, concrete actions, object interaction, environment/composition, establishing visuals and useful states. Speaking is not a rejection gate; sustained states do not require completed actions. Grounded local actions and coherent states retain the existing deterministic 5–20-second planner and explicit whole-action/short-span exceptions. Multiple windows require distinct reusable moments; no target asset count exists.

The transport envelope is `visual_utility_batch_v1`: `events` is a list of individually JSON-encoded event observations. Only that envelope is parsed by the SDK. Individual strings are validated locally, preventing an invalid event body from failing SDK parsing for valid siblings. The provider receives the complete per-event schema separately.

## E03 offline plan

[Exact batch membership and preparation identities](reports/e03-visual-utility-batch-plan.json):

| Metric | Result |
| --- | ---: |
| Unresolved logical observations | **136** |
| Initial provider requests | **17** |
| Batch distribution | **17 × 8 events** |
| Maximum additional per-event semantic assessments | **136** |
| Conservative provider-request ceiling, including retries and full isolation trees | **510** |
| HTTP ceiling with the production OpenAI one-attempt transport | **510** |
| Contact-sheet images | **136** |
| Temporal samples represented | **1,385** |
| Existing base observations | **175** |
| Existing completed Temporal V2 observations | **71** |
| Saved base observations reused in unresolved request contexts | **136** |
| Events already resolved by compatibility | **38** (17 KEEP + 21 REJECT) |
| Completed observations in the new batch cache | **0** |
| Quarantined events excluded | **1** |
| Real provider calls | **0** |

**17 is the normal initial request count. 510 is a conservative safety ceiling, not a forecast or a plan to issue 510 calls.** It is `2 × Σ(2 × batch_size − 1)`: two semantic passes, each bounded by a full binary isolation tree. Without malformed outputs or content rejection, only the 17 initial requests occur. Without rejection isolation, each original batch needs at most one retry request containing its failed subset. Successful siblings are excluded.

The ceiling is durably reserved before dispatch and cannot reset through interruption or regrouping. Each individual observation has at most two received semantic attempts. Provider adapter transport bounds also multiply the HTTP ceiling and are reserved separately; OpenAI uses one transport attempt per batch invocation, while configured Gemini pool members are counted explicitly. Changing a pool roster beyond the planned bound stops locally; content is never rewritten to bypass policy.

The offline report records **569,293 bytes of known planned event context**, a byte-based text-token estimate of **810,064** across all initial requests, and **324,670 estimated output tokens**. These are aggregate planning estimates, not measured tokenizer usage. Exact image tokens, final detected-person catalogs, encoded image sizes and total cost are unknown without runtime preparation/model-specific accounting. The offline event fingerprints identify source/sample preparation recipes; actual durable request fingerprints additionally seal the prepared image hash and full event evidence. No images were regenerated for this report. Provider/model values are explicit planning parameters; the report leaves the model unspecified rather than guessing a configuration.

## Fail-soft behavior and cache

Per-event states are `VALID`, `VALID_INCONCLUSIVE`, `VALIDATION_BLOCKED`, `PROVIDER_BLOCKED` and `MISSING_FROM_BATCH_RESPONSE`; unrequested/deferred work is `PENDING_OBSERVATION`.

- Valid and valid-inconclusive members are saved independently and reused, including after batch membership/order changes. Inconclusive judgments are genuine editorial ambiguity and do not automatically trigger another paid observation.
- Malformed, duplicate, unknown-citation or missing members retain independent diagnostics. A retry contains only the failed subset, with validation feedback. A successful sibling is never included again.
- Content rejection archives the unchanged request and deterministically bisects it. A rejected singleton is quarantined. Clean siblings continue. Transport/auth/quota errors do not become content quarantines or cause content bisection.
- A received raw batch is persisted atomically before member validation. Resume recovers its members offline, closing the response-to-event-persistence crash window. An interrupted request without an archived response remains uncertain and can consume another bounded request; remote completion without durable receipt cannot be guaranteed free.
- Per-event semantic identity seals schema, prompt, evidence profile, source movie, event/shot/sample identity, original observation fingerprint, active-picture recipe, full prepared evidence and image hash. It excludes batch membership and provider/model selection. Successful results therefore survive regrouping and model changes while stale source/evidence is rejected.
- Batch request identity includes provider/model, schema/prompt/evidence versions and ordered event semantic fingerprints. Request archives own usage/cost once; each sibling links to its request. Resume does not rely solely on this batch identity.

Original observations, Temporal V2 records, source media, event/shot identities, sample plans, narratives, provider response archives and source fingerprints remain reusable. Historical quarantines remain in force. The new observer fills missing base/temporal visual fields in the same batched observation on fresh jobs, avoiding an unnecessary independent per-event provider prepass.

## Run integration and REVIEW contract

The normal new-job lifecycle is Narrative → technical shots/Visual Events → reuse saved base/temporal evidence → Policy V2 compatibility → automatic batched resolution of remaining events → final Policy V2 decisions → deterministic window handoff → existing finalization/vertical reframe v4/QA.

No manual `observe enrich-temporal` command is required for future normal jobs. That command remains diagnostic. Historical jobs retain their existing lifecycle until explicitly migrated; this task does not switch E03's manifest or activate its new stage.

Successful siblings proceed to window finalization even when other events are blocked. Window identities and source-shot provenance use the previously implemented stable producer handoff. The finalizer consumes selected windows with independently reusable ledger/package identities. Fully cached reruns need no provider construction. Status reports expose visual-utility pipeline states and count batch usage once rather than multiplying it by the number of siblings.

After this stage, **REVIEW means a valid semantic result with unresolved visual/editorial context or a genuine technical/render ambiguity**. Missing schema fields, invalid evidence formats, a missing provider response, provider quarantine and request budgets are pipeline states: they leave production PARTIAL and have no validated editorial REVIEW decision. Incomplete real-world action or dialogue presence alone is not a review/reject criterion. A fully assessed weak/unusable event can REJECT; a useful supported sustained state can KEEP with raw completion false/unclear.

## Safety and changed files

Safe to execute the **E03 resolution stage** after separate authorization: **YES**, with durable bounds, fail-soft validation and existing quarantine preserved. This is readiness of the resolver, not a promise about provider decisions or approval to render/migrate E03. **E03 resolution was not executed.**

Files changed by this task (pre-existing workspace changes preserved):

- `src/movie_broll/visual_utility_resolution.py`
- `src/movie_broll/visual_utility_production.py`
- `src/movie_broll/broll_semantics.py`
- `src/movie_broll/broll_policy_v2.py`
- `src/movie_broll/production.py`
- `src/movie_broll/production_run.py`
- `src/movie_broll/cli.py`
- `tests/test_visual_utility_resolution.py`
- `tests/test_production.py`
- `docs/visual-utility-batches.md`
- `docs/reports/e03-visual-utility-batch-plan.json`

The read-only operator plan is available as `movie-broll observe plan-visual-utility input/<episode>`. It is optional inspection, not a required enrichment step, and never constructs or calls a provider.

Validation: focused resolver/production/status/provider-adapter/policy tests **122 passed**; full pytest **637 passed, 1 skipped**; `git diff --check` **passed**. Tests use fake providers and injected clients; resolver tests forbid socket connections. Coverage includes every requested batch/resume/integration case, request-budget persistence, independent citations, weak-dialogue assessment and request-level usage accounting. The read-only CLI output exactly matches the saved E03 plan. E03 artifacts unchanged **YES**: 551 non-media SHA-256 entries and 3 media size/mtime entries were identical before/after the audit. No E01/E02, Atlas, approval, production, rendering, commit/push or reset/clean/stash/restore action was performed. **Real provider calls: 0.**
