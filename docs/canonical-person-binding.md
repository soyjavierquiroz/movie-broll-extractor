# Canonical person binding and controlled regeneration

`semantic_observations.canonical_evidence_catalog()` produces `id =
<technical-shot-id>:<shot-local-person-id>` and records the owner separately.
`broll_pilot._person_candidates()` and `finalization._binding_candidates()`
assign P1/P2/... at the shot reference sample using the same deterministic
person-detector ordering. These IDs are scoped to a shot, not global characters.
Previously `build_shot_crop_plan()` passed the canonical string unchanged to
`_bound_single_subject_samples()`, which compared it with a bare local P ID.

`resolve_shot_person_id()` validates the complete ID, checks exact shot ownership,
and requires exactly one local candidate with finite, positive reference geometry.
Malformed, wrong-owner, missing, duplicated, and overlapping indistinguishable
candidates fail explicitly. Saved legacy bare P IDs remain supported only within
the directive's current shot. There is no position-based retry for explicit binding.
Canonical `target_person_ids` remain unchanged; `resolved_local_person_ids` is a
separate planner field. Unresolved bindings can have diagnostic crop pixels, but
remain review-required and cannot claim semantic validation.

The existing midpoint seed and forward/backward geometry tracking remain per shot.
Ambiguous continuity is rejected. Multiple target IDs use all participant tracks,
including directives whose primary focus is woman/man. Simultaneous/secondary
participant contracts with only one ID remain unresolved. Collapsed participant
tracks fail; complete tracks target their union. The existing crop width, margins,
face/head constraints and interaction checks remain authoritative. A union too wide
for the crop remains review/failed QA. The vertical metadata override cannot collapse
required interactions to one person.

Observation-only policy projection now retains the observed people facts in
`event.semantic_people`, outside the horizontal/semantic pixel fingerprint inputs.
`2_plus` remains a lower bound: the exact people count is null rather than invented.
The canonical visible-person IDs and count evidence remain available in metadata.
Finalization also preserves existing top-level narrative segment IDs if the legacy
nested narrative block is absent; it does not synthesize narrative summaries.
Rendition and aggregate semantic validation reflect subject/interaction retention,
explicit binding failures, clipping, hard QA failures and post-render audit evidence.
Review packages can carry false semantic validation without becoming publishable.

## Cache boundary

Only `REFRAME_ALGORITHM_VERSION` changes, from
`phase-f-face-head-constrained-reframe-v3` to
`phase-f-canonical-person-bound-reframe-v4`. This changes `reframe_fingerprint()`
and invalidates vertical render/validation/thumbnail/final-package reuse. Focus
schema, detector version, face/head constraint version, observation schema/prompt/
fingerprints, temporal evidence v2, policy versions/results, Narrative Map,
technical shots, Visual Events and source/active-picture evidence remain reusable.
The new people metadata projection is excluded from the existing horizontal-v2
candidate fingerprint; canonical focus IDs are not rewritten.

## Later authorized regeneration

For an affected saved event, load authoritative source media, canonical shot/event
ranges, active-picture evidence and saved observation/temporal/policy evidence.
Replay local projection; rebuild each shot's crop plan; render the vertical MP4
from the source; rerun vertical QA and post-render subject/face/head audits; rebuild
the vertical thumbnail, package metadata and vertical/finalization ledger records.
Reuse verified horizontal MP4/thumbnail provenance. No semantic re-observation,
temporal enrichment, Narrative Map rebuilding or policy reevaluation version change
is required. Preserve historical media/hashes before replacing any package through
a separately authorized controlled workflow. This change itself performs none of
those operations and does not approve an existing asset.
