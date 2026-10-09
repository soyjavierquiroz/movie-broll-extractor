# Temporal semantic evidence v2

The observation bottleneck was a single midpoint image per technical shot,
combined with a completeness field that cannot reliably be established from a
still image. Canonical action catalogs read only optional upstream action fields;
production populated neither field. Policy correctly held unsupported utility.

## Input contract

Normal production uses `temporal_evidence_v2`; legacy semantic/pilot callers retain
their explicit midpoint contract. Narrative, technical-shot and Visual Event
identity functions are unchanged.

`sample_plan(event, fps)` uses source frame numbers, clipped to each represented
technical shot and the existing event range. It selects:

1. Each technical shot's midpoint, `(start + end_exclusive - 1) // 2`.
2. The event's inclusive first frame and last frame, `end_exclusive - 1`.
3. Shot beginnings in chronological order, then shot endings, until the cap.
4. If the existing structural hint is action/interaction/movement/activity or
   motion is at least 12, quarter and three-quarter samples from longest shots
   first, with chronological tie breaking, while budget remains.

The hard cap is **16 distinct frames**, with duplicates coalescing roles. The
output is sorted by source frame, and every sample has a deterministic sample ID,
technical-shot ID, frame, timestamp in seconds, and role(s). If mandatory midpoint
coverage and event endpoints exceed the cap, fail locally rather than silently
omit a shot. No narrative/subtitle content chooses timestamps or invents actions.

Temporal tiles preserve aspect ratio within 480×240 pixels, plus a 38-pixel
caption. A maximum four-column grid has at most 1920×1112 pixels. Human identities
are labelled only on each shot's middle reference frame; they are not renumbered
as people move between temporal samples. The supplied catalog retains those
reference identities. Sample count, profile and geometry are fingerprinted.

## Observation and unchanged policy

The strict v2 response adds `moment_status`, `temporal_support_sample_ids`,
`visual_actions`, and `observed_actions` to the observation-only contract:

- `complete_action`: explicitly visible beginning/development/completion;
  raw `action_or_moment_complete=true`. Requires at least three distinct cited
  frames including event endpoints and middle/development. Ending an event never
  establishes completion by itself.
- `incomplete_action`: visibly truncated action without an established reusable
  state; raw completeness is false.
- `reusable_state`: temporally established independent sustained visual state;
  raw action completeness remains false/unclear, utility is `useful_state`, and
  visible state plus at least two distinct cited timestamps are required.
- `unclear`: evidence cannot establish the preceding cases; raw completeness
  remains unclear. No automatic promotion.

A compatibility projection supplies the unchanged `broll_policy_v1` with moment
completeness for an established reusable state. This projection does not rewrite
raw action completeness or bypass context, exclusion, utility or QA rules.
Structured sample citations are validation constraints, not a substitute for
the model's visual assessment; sparse input can still require REVIEW.

Canonical actions use the existing `KEEP_ACTION_EVIDENCE_TYPES` contract, not a
new action taxonomy. Only `distinct_visible_action_or_reaction` with an exact
label in `visual_actions`/`physical_interactions` and at least two valid temporal
samples creates a local canonical action ID. Prose, subtitle guesses, unsupported
labels, generic presence and `unclear` do not create action evidence. Concrete
action utility without qualified evidence continues to REVIEW under v1 policy.

## Immutable storage and resume

V1 records remain byte-for-byte at `semantic_observations/v1/events/`. Normal
production reuses completed valid legacy observations without treating them as
v2 or automatically paying to upgrade them.

V2 records live at `semantic_observations/v2/events/<event>/<input-fingerprint>.json`.
They persist the raw observation, full input evidence/catalog, event input
identity, source hash, profile, prompt revision and provider provenance. Changed
inputs produce additional historical records, never overwrite old records.
Policy readers prefer a validated matching v2 record and otherwise read v1.

Responses are persisted under `v2/responses/<event>/<fingerprint>/` before local
validation. Identity/sample/action errors receive actionable feedback. The
semantic validation budget is two `generate()` attempts per input fingerprint,
including attempts already saved before interruption. A valid saved response
recovers canonical persistence without another request. Completed compatible
records are free on resume. Transport retries are separately bounded by provider
configuration (OpenAI defaults to two transport attempts per generate).

## Explicit targeted recovery

Read-only planning, without production or provider construction:

```sh
movie-broll observe plan-temporal-enrichment input/<episode>
movie-broll observe enrich-temporal input/<episode>
```

Only the authoritative final store's VALIDATED `broll_policy_v1` REVIEW events
with exactly `insufficient_observation` qualify. Existing KEEP/REJECT records
are not selected. Before any paid work, replay cached policy, reuse valid v2
evidence, and locally exclude established high context dependency, which cannot
qualify under the existing policy regardless of better completeness evidence.
This necessary-condition exclusion is recorded in the plan; it never changes
the old observation or pretends v1's earlier uncertainty branch returned REJECT.
Unknown context/utility remains eligible for actual clarification.

Explicit execution, only after authorization:

```sh
movie-broll observe enrich-temporal input/<episode> --execute --max-events 5
```

The execution entry point verifies source identity, validates sampling bounds,
and writes only immutable v2 observations/responses. It returns replayed policy
results; it does not change canonical events, group shots, render media, approve
QA or resume production. Completed enrichment records are removed from the next
paid queue. An interrupted queue proceeds in authoritative event order.

E03 read-only planning at implementation time: **72 eligible**, **31 excluded
locally for high context**, and **72 KEEP/REJECT events untouched**. The 72 include
59 useful low/medium-context observations plus 13 whose context still needs
clarification. One successful first attempt each would require 72 generate
calls; the semantic validation maximum is 144. With default OpenAI transport
retry configuration the worst-case transport attempt ceiling is 288. No calls
were made and no E03 records were enriched during implementation.

## Failure diagnostics and completeness provenance

Reusable-state violations have separate diagnostics for raw action completeness,
utility (`useful_state` required), missing substantive visible states, and temporal
support. False/unclear raw completeness is valid; true is a violation.
The structured response schema describes the same conditional contract as the
prompt and local validator. Blank state descriptions do not establish a state.

The compatibility adapter adds `temporal_completeness` with `moment_status`,
`raw_action_or_moment_complete`, and `effective_action_or_moment_complete`.
`apply_policy_result()` preserves this distinction in editorial metadata. The
legacy observation field supplied to policy contains the effective value, while
the immutable V2 observation retains the raw value. Policy rules are unchanged.

New raw attempt records include input evidence, event input identity, source hash
and observation fingerprint before validation. Historical response files and
diagnostics are not rewritten during recovery. Feedback is revalidated in memory.
Recovery charges no additional attempt and retains cumulative provenance from all
saved paid attempts, even when the first response is selected.

Batch processing deliberately fails closed after the two-attempt event budget is
exhausted. Earlier completed records remain reusable; later events are not called.
Restart reuses completed records and revalidates saved responses before considering
a provider request. A still-invalid exhausted event stops again without payment.
