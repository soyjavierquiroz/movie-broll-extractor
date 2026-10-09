# B-roll editorial policy v2

The original replay counts and handoff limitation below are historical. The [compatibility audit](policy-v2-evidence-compatibility.md) supersedes them: KEEP 17 / REVIEW 136 / REJECT 21 / BLOCKED 1, with 17 windows and an explicit stable multi-window handoff. E03 remains unmigrated.

ROOT CAUSE: V1 equated reuse with independent action/moment completion. Its generic-dialogue gate ignored useful conversation composition; incomplete-action and high-context gates excluded sustained states and story-dependent but visually understandable footage. Temporal V2 restricted reusable states to `useful_state` and required complete-action citations at the Visual Event's absolute endpoints. The latter hid local actions. Event merging also did not establish asset continuity.

POLICY V2: `broll_policy_v2` is an explicit opt-in, with v1/defaults unchanged. The pure decision API is `broll_policy_v2.evaluate(record, event, selected=...)`; `semantic_observations.policy_evaluate` dispatches explicitly when supplied v2 and an event. Decision order:

1. Any true hard exclusion (title card, credits, logo, dominant text, black/empty, corrupt/unusable) REJECTS.
2. Missing flags/event, identity mismatch, invalid observation, missing temporal grounding, or unresolved legacy utility produces REVIEW and a targeted observation candidate.
3. Each valid moment requires useful visual strength, usable moment, complete asset window, a resolved moment kind, and low/medium **visual** context dependency. High/unclear visual dependency goes to REVIEW. Raw action completeness is independent.
4. Any accepted window makes the event KEEP; other unresolved moments remain targeted candidates. The result preserves raw completeness and contains every accepted asset window.
5. Fully assessed weak/unusable footage, or all usable windows suppressed as duplicates, REJECTS. An incompletely assessed empty catalog remains REVIEW.

Supported moment utility classes: `conversation_scene`, `therapy_or_session`, `discussion_or_argument`, `meditation_or_stillness`, `reaction`, `sustained_emotional_state`, `physical_interaction`, `nonverbal_interaction`, `walking_or_movement`, `concrete_action`, `object_interaction`, `useful_state`, `environment_or_composition`, `establishing_visual`. Existing utility names are retained in the event summary and mapped by structural aliases when replaying. No episode-specific branches exist.

DIALOGUE: Dialogue presence has no rejection gate. Useful conversations, sessions, family arguments, listening exchanges and group interaction can KEEP when their visual moments meet the contract. Explicitly assessed weak flat talking heads can still REJECT. Old generic-dialogue labels alone cannot establish weakness under the new contract; they become REVIEW rather than automatic KEEP or REJECT. Subtitles never establish visual utility.

COMPLETENESS: Each moment independently records `action_completeness`, `moment_usability`, and `asset_window_complete`. Walking without a destination, stillness, meditation, an embrace or expressive seated interaction can be usable with raw action completeness false/unclear. Complete local actions require supplied beginning, development and completion samples in temporal order, at the local moment bounds. They need not include the Visual Event endpoints. Neither the policy nor planner rewrites completeness.

ASSET WINDOWS: Bounds are integer frames, end exclusive; seconds derive from the evidence fps. Every moment needs at least two distinct cited timestamps, actual bound support, represented-shot support, and explicit supported continuity across cuts. Complete actions require a third development timestamp and ordered local stages.

- A coherent 5–20 second moment stays whole.
- A longer supported sustained state/composition gets one central window of at most 20 seconds (`floor(20 * fps)` frames). The complete observed span must be supported first; no inference of unseen transitions is allowed.
- A complete action stays whole even above 20 seconds. Short moments stay whole below 5 seconds. Both carry `duration_exception=true`; there is no padding, truncation, or quota.
- Other moments longer than 20 seconds go to REVIEW for a supported boundary. They are never arbitrarily divided.
- One window per genuinely distinct moment; multiple moments can yield multiple assets in one event. Multi-shot windows retain all intersecting technical-shot IDs.
- A shared `reuse_key` or overlap of at least 80% of the shorter window suppresses the later window. `selected` supports scarcity across events. New observations must share a reuse key for near-duplicate visual needs/compositions. Old caches support exact structured-state identity and overlap, so broad cross-event visual similarity cannot be certified from them.

OBSERVATION: The additive, separately identified contract is `semantic_observation_v2_visual_reuse`, prompt `semantic_observation_prompt_v2_visual_reuse`. `VisualReuseObservation`, `RESPONSE_SCHEMA`, `PROMPT`, and `validate_visual_reuse` are ready for an explicitly configured future observer. The new `visual_moments` catalog explicitly assesses conversation/session visuals, stillness, emotions, reactions, physical contact, objects, movement, environments and local actions. It separates narrative context from visual context. Historical schema/prompt/validator identities remain unchanged. A revised cache lives under `semantic_observations/v2_visual_reuse/events`, with explicit persistence and validated latest-record loading; no record has been created there for E03.

The revised validator preserves historical identity, canonical sample-plan, action-label, person-binding, and shot-focus checks. It replaces the event-endpoint requirement with grounded local action validation. Existing `temporal_evidence_v2` samples remain usable; sparse samples cannot establish an unseen completion. Targeted observation may reuse existing temporal images when adequate; missing local stages require targeted visual evidence rather than narrative inference. No images or observations were regenerated in this task.

CACHE: All 175 events have a cached observation: 71 valid temporal V2 records and 104 older records. All existing observations, samples, archived responses, provider provenance, narrative and technical-shot caches remain reusable and unchanged. Policy version is excluded from old observation identity. Replay uses raw temporal records rather than the v1 compatibility adapter that projects state completeness to true. Older records support hard exclusions but cannot establish temporal asset windows. Existing provider quarantine remains blocked if saved evidence cannot resolve the event locally; a policy change does not release it.

E03 OFFLINE PROJECTION: KEEP **6**, REVIEW **147**, REJECT **21**, PROVIDER_BLOCKED **1**, total **175**. Six supported candidate windows are planned; one is a preserved 4.04-second state with a duration exception. These are current evidence-resolved decisions, **not a forecast of decisions after new observations**. Determining final episode counts without the missing visual assessments would require inventing evidence. No target asset count was used.

TARGETED REOBSERVATION: **147 events / 147 expected initial semantic calls**. A two-response validation allowance would budget **294 semantic attempts**, excluding transport retries; this is a planning allowance, not an executed retry workflow. The blocked event is excluded from that call plan. Candidate reasons:

| Missing evidence | Events |
| --- | ---: |
| Older observation lacks temporal visual moments | 83 |
| Local action/state usability unresolved | 31 |
| Old generic class did not assess visual reuse | 19 |
| Visual continuity across asset cuts unestablished | 11 |
| Narrative vs visual context unresolved | 3 |

The [complete read-only projection](reports/broll-policy-v2-e03-offline-projection.json) includes every event decision, proposed window, quarantine and candidate reason. Re-run without writing any run artifacts:

```bash
.venv/bin/movie-broll policy evaluate input/mi-otra-yo-s03e03 --policy broll_policy_v2
```

PRODUCTION HANDOFF: The legacy mutable `apply_cached_policy` path explicitly refuses v2, because its downstream event-to-asset behavior cannot silently discard or replace this window plan. Default production stays on v1. A later explicitly authorized migration must consume the v2 planned windows and resolve missing observations; this task does not provide or execute a migration command.

REAL PROVIDER CALLS: **0**. E03 ARTIFACTS: **unchanged YES**, verified by a complete before/after SHA-256 inventory of the episode's run and input files. No production, rendering, approval, commit/push, reset/clean/stash/restore, E01/E02, or Atlas work was performed.

SAFE TO MIGRATE E03 TO POLICY V2: **NO** for a full production migration now: 147 visual assessments are unresolved, one event remains provider-blocked, and production must explicitly consume the multi-window plan. Read-only v2 replay is available now. Migration was not executed.

FILES CHANGED BY THIS TASK (pre-existing workspace changes preserved):

- `src/movie_broll/broll_policy_v2.py`
- `src/movie_broll/semantic_observations.py`
- `src/movie_broll/cli.py`
- `tests/test_broll_policy_v2.py`
- `docs/broll-policy-v2.md`
- `docs/reports/broll-policy-v2-e03-offline-projection.json`

TESTS: Focused policy/temporal/observation suite: **112 passed**. Full suite: **603 passed, 1 skipped**. `git diff --check`: **passed**. The read-only CLI replay matches the saved report counts. New tests use synthetic temporal evidence and forbid socket connections; no providers are called. Coverage includes all utility classes, weak dialogue, visual vs narrative context, false/unclear raw action completeness, local completed actions, duration exceptions, representative state windows, multiple shots, distinct moments, duplicate/scarcity protection, hard exclusions, immutable versioned cache, read-only projection/quarantine, no episode-specific policy code and the production handoff guard.
