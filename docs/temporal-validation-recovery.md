# Temporal V2 validation recovery

`TEMPORAL_VALIDATION_RECOVERY_REVISION` in `temporal_semantics.py` is an explicit,
reviewed authorization boundary. Its initial value, `temporal_validation_recovery_v1`,
adopts the existing response archives and grants no additional attempts. Ordinary
validator, diagnostic, or implementation edits must leave this value unchanged.
Only an explicitly reviewed revision change authorizes another bounded budget.

The observation fingerprint identifies source movie, event range/technical shots,
contact sheet pixels, active picture, narrative context, schema/prompt versions,
provider compatibility overrides, evidence profile and exact input evidence.
Completed valid records use this fingerprint, independently of recovery revision.

The validation execution identity adds configured provider, model, and recovery
revision. Each execution can save at most two semantic responses across operator
reruns. Returning to an exhausted execution identity stays blocked. Original v1
response files retain their existing locations; their saved provider/model
provenance determines which identity adopts them. Other identities use
`responses/<event>/<observation-fingerprint>/executions/<execution-fingerprint>/`.
Unknown offline provider objects conservatively adopt existing provenance; real
providers expose `identifier` and `model`. Explicit provider/model compatibility
can identify custom adapters or pools.

Raw parsed responses, exact evidence/context, fingerprint/profile, expected and
returned identity, revision, attempt number, usage/provenance and transport request
count are saved before local validation. Diagnostics then add error codes, paths
and actionable messages. Historical attempts are revalidated in memory and never
rewritten on resume. A response saved before diagnostic/completed persistence can
be recovered without another call. Valid completed observations remain reusable
when the revision changes.

Provider-owned transport retries remain distinct from semantic response attempts:
`response.attempts` counts transport requests, while a successfully returned
parsed response consumes one validation attempt. A thrown transport exception
produces no parsed response and propagates without recording a semantic failure.
This mechanism does not change the provider's transport retry policy.

The batch is fail-soft for terminal per-event outcomes. Semantic budget exhaustion
persists `BLOCKED_VALIDATION` and continues to the next event. Provider input rejection
(`invalid_prompt` / `content_policy_violation`) persists `PROVIDER_BLOCKED`, consumes
zero additional semantic attempts, and continues without rewriting or retrying the
input. Terminal records contain allowlisted HTTP/error metadata and fingerprints;
no exception text, request headers or credentials are saved. Existing v1 facts and
completed v2 records remain immutable. Blocked outcomes are REVIEW, never KEEP.

Provider rejection cache identity includes observation fingerprint, exact prompt,
context and contact-sheet fingerprint, and configured provider/model. It excludes
validation recovery revision: a validator revision cannot repay rejected input.
Legitimate input/profile/provider/model changes create separately identified records.
Semantic validation budgets remain two parsed responses per execution identity.
Transport retries retain the provider's bounded per-call policy; failed calls are
journaled under `transport_failures`, outside semantic response numbering, and may
stop/resume the batch. No additional cross-rerun transport cap is imposed.
Batch reports completed, review, provider_blocked, validation_blocked, remaining,
and COMPLETE / COMPLETE_WITH_REVIEW / PARTIAL_PROVIDER. Completed counts include
validated observations of every policy outcome; review counts completed REVIEW
observations, with blocked counts reported separately. Structural integrity errors
still propagate. Enrichment never updates rendering manifests; rendering continues
to require validated KEEP decisions.

A revision change does not authorize execution by itself. The operator must
separately authorize temporal enrichment. The active revision is `temporal_validation_recovery_v2`, authorized because
deterministic feedback now identifies the exact reusable-state/utility
contradiction. Existing archives remain assigned to v1. No enrichment was executed
as part of this authorization.

## Valid uncertainty and offline recovery

Validation distinguishes unknown evidence from inconsistent assertions. An
`unclear` moment permits raw action completeness `false` or `unclear`: knowing
that an action is incomplete need not establish whether a reusable moment exists.
It still cannot assert raw completeness `true`. Complete actions and reusable
states retain their temporal support, grounding and consistency requirements.
Schema-valid unknown utility, context, action evidence and focus classifications
remain evidence, not retry triggers. The unchanged `broll_policy_v1` decides
REVIEW/KEEP/REJECT; its hard exclusions and generic-dialogue rejection precede
its uncertainty branch. Uncertainty never creates KEEP by itself.

`recover_saved_response_offline(run, event, attempt_path)` validates archived
source/event identity, evidence fingerprint/profile and prompt version, then
revalidates the original response and calls normal immutable persistence with
archived evidence and cumulative provenance. It has no provider, rendering or
enrichment dependency. Archives and historical diagnostics are not rewritten.
The recovery revision is unchanged; this fix grants no additional retry budget.

Offline E03 recovery of `VE_1561653b3d1d05d3` used attempt 0002 unchanged:
`moment_status=unclear`, raw completeness `false`, utility `generic_dialogue_only`,
context `high`, no canonical actions, five canonical temporal samples. Attempt
0001 remains invalid because reusable_state requires useful_state utility.
Two semantic attempts consumed three transport requests (7,428 total tokens,
$0.00067944 cumulative recorded cost). Existing policy rejects attempt 0002 for
generic dialogue without independent visual signal. Before recovery: 15 completed
(3 KEEP, 2 REVIEW, 10 REJECT), 57 remaining. After recovery: 16 completed
(3 KEEP, 2 REVIEW, 11 REJECT), 56 remaining, with no saved responses for those
remaining events. First-attempt success therefore needs 56 additional generate
calls. Recovery made zero provider calls and all 375 existing observation/archive
JSON files remained byte-identical. No production/enrichment execution occurred.

## Historical rejection without an archive

`quarantine_reported_rejection_offline` adds a new diagnostic when the old runner
failed before recording a request. It explicitly leaves the exact request fingerprint
unknown. Its conservative input snapshot covers source/event identity, persisted
narrative, subtitles, technical shots, active picture and metadata, prompt/profile,
and configured provider/model/compatibility. Matching input is skipped before even
producing a contact sheet. Validator revisions cannot release it; legitimate input
or provider/model revisions produce separate identities. No historical response is
rewritten and no semantic attempt is invented.

E03's reported HTTP 400 invalid_prompt is assigned by first unresolved enrichment
order to `VE_84a221dc116d7896`; no archived request confirms that attribution. A new
explicit quarantine records that basis for openai / gpt-6-luna. At diagnosis: 66
completed V2 observations, one provider-blocked event, five remaining. Creating this
diagnostic made no provider calls and left all existing run JSON files unchanged.
