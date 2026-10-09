# E03 batched visual-utility execution: offline forensic audit

No provider was called, resolution was not rerun, and no media was rendered.
The full saved response, decoded observations, exact original error lists,
request identities, event fingerprints, and raw-archive SHA256 checksums are
in `e03-visual-utility-forensics.json` beside this report.

## Findings

The successful HTTP request was
`77c6c0141916ecd8a2b4eb82f0450ccd44c2a3f8547a6b97072c5a78076fe648-2`.
It requested eight events and returned eight JSON strings in the expected
`{"events": [...]}` envelope. All event IDs matched exactly and occurred once.
Every observation passed the original Pydantic event schema; none was discarded
by the parser. All required fields were present, and no single-event envelope
was used. The saved artifact contains the SDK's parsed response, not a complete
raw HTTP transcript.

All eight bodies returned an empty `shot_focus_plan`, despite known technical
shots and existing saved focus evidence. This is a shared generation-contract
failure, not eight independent ambiguous visual judgments. The inherited base
prompt allowed empty evidence, while the new batch prompt did not explicitly
state the inherited one-directive-per-shot requirement. The event JSON schema
was supplied as context, but only the envelope was transport-enforced.

Seven bodies additionally violated the inherited reusable-state/useful-state
summary invariant. One of these claimed raw action completeness for a reusable
state and used an observed-action label absent from its structured action lists.
Three bodies cited temporal samples outside their suggested local ranges.
These additional failures remain real validation failures: no semantic labels,
action completion, temporal ranges, or citations were rewritten to accept them.

| Event | Original validation errors (shared focus error omitted here) |
|---|---|
| VE_ce865311d8a14107 | reusable_state_requires_useful_state_utility |
| VE_5323966610e1583a | reusable_state_requires_useful_state_utility; insufficient_local_temporal_support |
| VE_1fdc590d300b260f | reusable_state_raw_action_completeness_must_be_false_or_unclear; reusable_state_requires_useful_state_utility; action_not_grounded_in_visible_structured_action |
| VE_ea6d3919c585fe80 | reusable_state_requires_useful_state_utility |
| VE_f6a1dc77dc7bccf9 | none beyond one_focus_directive_per_technical_shot |
| VE_cf8f5537477c7c29 | reusable_state_requires_useful_state_utility |
| VE_fddbf5f0841708fa | reusable_state_requires_useful_state_utility; insufficient_local_temporal_support |
| VE_3666fcaafc03e618 | reusable_state_requires_useful_state_utility; insufficient_local_temporal_support |

Each event had exactly one semantic response attempt, one timed-out transport
dispatch, and one successful transport dispatch. Each appeared in the successful
response. Individual fingerprints and complete decoded bodies are in the JSON
report. The original errors and response archives were preserved byte-for-byte.

## Retry and timeout defects

`dispatch()` previously used `len(previous) >= MAX_SEMANTIC_ATTEMPTS` for the
subset dispatch ceiling. The first timeout plus the one received invalid batch
therefore prevented the second semantic attempt for all eight events. The event
attempt counter itself was one, not two; the subset gate incorrectly treated a
transport attempt as if it exhausted the semantic retry allowance. The recursive
retry never reached the provider, so its feedback was never delivered.

The `structured_output_invalid` exception path also committed an empty batch
as eight missing-event attempts. `_attempt_count()` counted all attempt files,
so envelope errors or missing members could exhaust semantic allowances. Request
budget exhaustion also incorrectly appeared as VALIDATION_BLOCKED.

The first timeout was request `77c6...-1`, followed by successful retry
`77c6...-2` of the same eight-event batch. The other timeout was
`d19cd87c20d37be9c1478009b0afa79372cdbb1966e685abe53cadb0f73829ce-1`,
a separate batch containing eight different events. Read timeout settings for
these two passes were 90 seconds and 300 seconds; logs show 90.7s and 300.2s.
Both durable request records correctly retain TRANSPORT_DEFERRED, reason timeout,
retryable true, and one transport attempt. Neither generated semantic attempt
records. The historical archives did not persist the underlying exception class
or timeout configuration, so the exact historical HTTP/client timeout subclass
cannot be proven from those artifacts. Future records now preserve both.

Request identity, members and IN_FLIGHT status were written before sending;
request and HTTP reservations were also persisted before dispatch. Timeouts are
durable. Completed canonical records are reused independently of regrouping.
A timed-out request has no provider-side completion or billing proof: retrying
could be billed again. Local request IDs are not server-side idempotency keys.
The existing cumulative 40-request ceiling remains in force; three requests
were already made before this audit. No execution budget was reset.

## Fixes and recovery

- Separate per-event semantic responses, missing-member responses, envelope
  failures, and transport dispatch limits; all retain bounded durable controls.
- Envelope failures become BATCH_CONTRACT_BLOCKED without per-event semantic
  exhaustion. Transport-only exhaustion stays TRANSPORT_DEFERRED. Request-budget
  exhaustion becomes REQUEST_BUDGET_DEFERRED.
- Batch prompt v2 explicitly states inherited focus, summary, action-label and
  temporal citation invariants. Batch request identity records that prompt
  revision; existing immutable event input fingerprints and valid caches stay
  reusable.
- Validation feedback is scoped to the event, includes detailed legacy errors,
  and is reconstructed from durable attempts on resume.
- Offline recovery can reuse an existing focus plan only for an empty returned
  plan, only if the complete resulting observation passes every existing
  validator. Field provenance includes the saved observation fingerprint.
- Offline planning unwraps persisted active-picture metadata to use the same
  geometry as execution; otherwise a valid completed cache was missed.

Exactly one event, VE_f6a1dc77dc7bccf9, was recovered offline from the successful
saved response plus its existing compatible focus plan. All remaining claims
were preserved. The seven other observations remain invalid without further
semantic assessment. No original error archive or raw response was rewritten.

A separate `policy_v2_offline_recovery.json` records the current projection;
original execution summaries remain historical. Stable asset-window plans now
contain 18 windows. Current counts: new VALID 1, KEEP 18, editorial REVIEW 0,
REJECT 21, VALIDATION_BLOCKED 7, PROVIDER_BLOCKED 1, unresolved nonquarantined 135
(including the seven validation-blocked events). No rendering is authorized or
performed. Resolution can resume with the persisted 40-request cap, retaining
all completed siblings; old scripts asserting exactly 136 unresolved events
must use the current offline plan instead.

## Verification

Focused tests: 143 passed. Full pytest: 645 passed, 1 skipped.
`git diff --check`: passed. Synthetic test providers only; real provider calls: 0.
Historical request archives have identical SHA256 checksums. Source and existing
rendered-media size/mtime inventory is unchanged. Durable HTTP counter remains
3/40; one completed observation is now recognized by the offline planner, which
reports 135 outstanding logical observations. Regression coverage includes valid
8-member batches, reordered IDs, malformed members/envelopes, SDK parse errors,
timeout persistence before sending, semantic retries after timeout, durable resume
feedback, focus-only offline recovery, contradictory evidence rejection, and
active-picture cache compatibility.
