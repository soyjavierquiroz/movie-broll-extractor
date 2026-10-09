# Policy V2 evidence compatibility audit

E03 remains unmigrated. Offline result: **KEEP 17 / REVIEW 136 / REJECT 21 / PROVIDER_BLOCKED 1** over 175 events. **17 planned windows, zero multi-window events. 11 of the original 147 REVIEWs resolve locally. 136 initial new semantic observations remain targeted; zero real provider calls occurred.** The blocked event is excluded from that plan. These are evidence-based results, not a forecast or a target asset count. Full production migration is **not safe yet** because unresolved semantic assessments and quarantine remain.

The [machine-readable audit](reports/broll-policy-v2-e03-compatibility.json) includes the baseline, updated replay, and all 147 original REVIEWs with field provenance and unresolved requirements. Classification is partial: a category can be resolved locally while temporal usability remains unresolved. Thus “136 need new assessments” does **not** mean 136 have no saved images or observations.

## Original 147 REVIEW breakdown

| Exact baseline reason | Count | Resolution |
| --- | ---: | --- |
| `missing_temporal_visual_moments` | 83 | Category/context often map from original observations; no cited temporal usability assessment exists |
| `local_moment_or_state_usability_unresolved` | 31 | Saved temporal content exists, but action completion or independent state usability is unresolved |
| `legacy_generic_class_did_not_assess_visual_reuse` | 19 | Conversation presence maps category, never strength/usability; additional assessment still needed |
| `asset_cut_continuity_requires_visual_assessment` | 11 | Resolved locally using one cited within-shot state span |
| `visual_vs_narrative_context_unresolved` | 3 | Saved high/unclear context does not establish low/medium visual dependency |

Provider-blocked: one additional event, outside these 147. No quarantine is released.

## Exact remaining requirement combinations

These mutually exclusive rows sum to 136; abbreviations refer to the precise requirement keys in the JSON audit.

| Requirements | Events |
| --- | ---: |
| Temporal support + context interpretation + action/state usability | 46 |
| Action/state usability | 45 |
| Temporal support + action/state usability | 28 |
| Context interpretation + action/state usability | 2 |
| Utility category + temporal support + action/state usability | 2 |
| Utility category + temporal support + context interpretation + action/state usability | 7 |
| Utility category + context interpretation + action/state usability | 3 |
| Context interpretation | 3 |

Overlapping individual requirements: `missing_temporal_support` **83**, `missing_context_dependency_interpretation` **61**, `missing_action_or_sustained_state_assessment` **133**, `missing_moment_utility_category` **12**. No final render-window assessment is requested from a provider. Semantic range and continuity support may still need assessment before deterministic planning is possible.

## Evidence sources and limits

| Requirement | Reusable sources | When a new semantic assessment is required |
| --- | --- | --- |
| Moment category, including conversation | Original semantic observation (1), Temporal V2 (2): explicit ontology aliases and conversation presence | Unknown/contradictory utility cannot establish a useful moment |
| Therapy/session/discussion/listening/group conversation | Existing positive conversation evidence maps to broad `conversation_scene`; explicit reaction/nonverbal/interaction utility aliases remain available | No subtype is inferred from subtitles, event hints, or vague prose; category mapping alone never asserts usefulness |
| Temporal support | Temporal V2 citations (2), stored sample identities/ranges (4) | Original one-reference-per-shot evidence or an unobserved deterministic plan does not prove duration or transitions |
| Sustained-state usability | Explicit Temporal V2 `reusable_state` plus `useful_state`, visible states, false/unclear raw completion (2) | Mere presence, movement, or a truncated action does not establish an independently reusable state |
| Local action | Explicit complete-action status and beginning/development/completion citations (2); native v2 local moments | Reaction/interaction labels alone do not prove an action is whole |
| Context dependency | Existing low/medium dependency (1 or 2) safely implies low/medium visual dependency | High/unclear historical context cannot be separated into narrative versus visual without evidence |
| Asset-window eligibility | Event range, shot boundaries and supported continuity (3) plus temporal citations (2, 4) | Only the missing semantic scope/continuity needs assessment; exact render-window selection stays local |
| Ambiguous content | Existing utility/context/completeness judgments and saved descriptions remain inspectable (1, 2) | No keyword-derived strength, invented action, or deterministic motion threshold substitutes for semantic evidence (5) |
| Provider-blocked | Saved identity-bound quarantine | Remains blocked; no automatic retry or release |

Sources numbered as requested: (1) original observation, (2) Temporal V2 observation, (3) deterministic event metadata, (4) stored temporal samples, (5) new provider observation. Source media, technical-shot identities, visual signals, intrusive-text data, descriptions, focus directives, utilities, reactions, interactions, and v1 decisions remain available. Technical signals and v1 KEEP reasons cannot replace missing semantic temporal judgments. Latest valid Temporal V2 observations take precedence over older observations: an older optimistic field cannot override a later unresolved assessment.

## Local compatibility and windows

`policy_compatibility.project` returns a separate partial contract with `derived_from`, source fields, original observation fingerprint, and per-moment sample/event provenance. It never writes or mutates an observation. Category aliases cover reactions, physical/nonverbal interaction, movement, objects, and composition. Of the original 147 REVIEWs, 135 have a locally mapped utility category, including 52 broad conversation categories; 86 have compatible low/medium context, 64 have cited temporal support, and 14 have an explicit usable action/state assessment. These field counts overlap and do not imply 135 locally resolved events. Generic dialogue plus positive conversation presence maps to `conversation_scene` without inventing useful strength or sustained-state usability. Explicit Temporal V2 useful states and complete actions map to usable moments. Unsupported and contradictory fields stay missing.

For a useful state spanning unverified cuts, the planner chooses the longest supported cited span within one shot, deterministically breaking ties by earliest start. It selects one representative window for that state rather than treating camera angles as distinct needs. Native independently evidenced moments can yield multiple windows. The planner preserves complete actions whole, keeps coherent 5–20 second moments, and selects a central maximum-20-second state window. Short moments and long whole actions carry explicit duration exceptions. It never pads, tiles, invents transitions, or proves completion from event endpoints. Historical state reuse keys are event-scoped because cross-event semantic equivalence was not assessed.

## Producer handoff

`asset_window_handoff.producer_candidates` is a pure, explicit handoff API. It materializes zero/one/multiple finalization candidates from a KEEP decision, without rendering, writing artifacts, or altering source events. Each selected frame range owns a `producer_window_id`: a 96-bit deterministic digest of version, source Visual Event ID, start frame, and exclusive end frame. It receives a separate candidate/ledger identity. Duplicate frame windows collapse; different moments are selected by policy before handoff. Boundaries and technical-shot focus directives are clipped/filtered for that window.

Package IDs use the original sparse timeline ordinal plus the window digest (`<movie-code><ordinal>w<digest>`), never selection order or completion order. Resume reuses that identity and stored slug. Failures/reviews cannot renumber other assets; registry conflicts and ID collisions fail explicitly. Source Visual Event ID and included canonical technical-shot IDs survive in final metadata. Registry entries retain source event, window identity, frame bounds, and ordinal. `movie-broll close` checks source event, slug, and window identity while retaining compatibility with old single-event entries. Finalization refuses an implicit provider semantic refresh for incompatible window focus evidence. Review reconciliation looks up QA and reframe plans by the separate window ledger identity. For selected subranges, event-wide descriptions/actions stay in source evidence rather than becoming unsupported assertions about the window.

Default production and legacy `apply_cached_policy` remain guarded on v1. This task supplies the explicit, testable window handoff; it does not enable or execute E03 migration, rendering, production, or close. Unresolved assessment/quarantine must be handled before a full migration.

## Cache and safety

All **175 original observations**, **71 valid completed Temporal V2 records**, deterministic plans, stored images, response archives, source media, source fingerprints, technical-shot catalogs, narrative evidence and provider provenance remain reusable. Policy projection is separate from immutable provider contracts. Existing packages remain associated with their v1 identities; window packages have separate identities and require their own render validation when later authorized. No episode artifact is written by replay or compatibility projection.

Files changed in this task:

- `src/movie_broll/policy_compatibility.py`
- `src/movie_broll/asset_window_handoff.py`
- `src/movie_broll/broll_policy_v2.py`
- `src/movie_broll/finalization.py`
- `src/movie_broll/closure.py`
- `tests/test_broll_policy_v2.py`
- `docs/broll-policy-v2.md`
- `docs/policy-v2-evidence-compatibility.md`
- `docs/reports/broll-policy-v2-e03-compatibility.json`

Pre-existing workspace changes were preserved. No provider, temporal enrichment, production, rendering, human approval, E01/E02, Atlas, commit/push, or reset/clean/stash/restore action was executed.

Validation on the final implementation: focused policy/temporal/original-observation/finalization/closure suite **208 passed**; full pytest **616 passed, 1 skipped**; `git diff --check` **passed**. Synthetic policy tests forbid socket connections. Tests cover ontology mapping, unsupported/contradictory evidence, provenance, deterministic durations/actions/states/multiple shots, distinct windows, duplicate suppression, identity stability across reversed resume order, source/registry/window matching, and window-specific QA reconciliation. A SHA-256 comparison of all 551 non-media E03 files before/after the offline audit was unchanged; no source or output media was written. The saved replay exactly matches the final read-only replay. **Real provider calls: 0. Safe to migrate E03: NO.**
