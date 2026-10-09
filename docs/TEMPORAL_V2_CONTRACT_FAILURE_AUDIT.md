# Temporal V2 contract failure audit

ROOT CAUSE: The error combined three conditions. It does not require raw
completeness true: it rejects true, wrong utility, or absent visible states. Both
responses violated the explicit prompt requirement `visual_utility_kind=useful_state`
for `moment_status=reusable_state`. Their false/unclear completeness was correct.

FAILURE CLASS: MODEL_OUTPUT_ERROR. The diagnostic was misleading; there was no
validator requirement to call a reusable state a complete action. The schema
previously enumerated independent fields without explaining their relationship;
it now describes that relationship. No semantically invalid response is promoted.

SAVED RESPONSE: Two paid attempts exist under the paths below. No V2 completed
record exists for this event. These extracts preserve exact values. Original
files include shot focus directives and visible-person IDs as well.

## 0001.json

`runs/mi-otra-yo-s03e03/semantic_observations/v2/responses/VE_3dc28f49d13b80fe/7603a4c29c2cd1aae2f6fdae27f2c424e7062432f1fb7078a66e68133b838a95/0001.json`

```json
{
  "created_at": "2026-10-08T23:32:05.838347Z",
  "diagnostic": {
    "errors": [
      "reusable_state_not_complete_action"
    ],
    "event_id_present": true,
    "expected_event_id": "VE_3dc28f49d13b80fe",
    "returned_event_id": "VE_3dc28f49d13b80fe"
  },
  "cumulative_provider_provenance": {
    "cost_usd": 0.00079137,
    "model": "gpt-6-luna",
    "provider": "openai",
    "request_count": 1,
    "usage": {
      "cached_tokens": 1407,
      "prompt_tokens": 5325,
      "response_tokens": 771,
      "thinking_tokens": 0,
      "total_tokens": 6096
    }
  },
  "provider_trace": [
    {
      "attempt": 1,
      "model": "gpt-6-luna",
      "provider": "openai",
      "status": "COMPLETE"
    }
  ],
  "response": {
    "event_id": "VE_3dc28f49d13b80fe",
    "represented_shot_ids": [
      "FULL_SHOT_0038",
      "FULL_SHOT_0039",
      "FULL_SHOT_0040"
    ],
    "moment_status": "reusable_state",
    "action_or_moment_complete": "false",
    "visual_utility_kind": "strong_nonverbal_interaction",
    "conversation_visual_signal": "strong_body_language",
    "context_dependency": "low",
    "visible_states": [
      "Several women are seated close together outdoors, with colorful clothing and head coverings.",
      "A woman in sunglasses faces the group; women turn toward one another and appear to speak."
    ],
    "temporal_support_sample_ids": [
      "SAMPLE_01",
      "SAMPLE_03",
      "SAMPLE_05",
      "SAMPLE_06",
      "SAMPLE_08",
      "SAMPLE_10",
      "SAMPLE_11",
      "SAMPLE_13",
      "SAMPLE_15"
    ],
    "visual_actions": [
      "Several women converse while seated together outdoors.",
      "A woman in sunglasses gestures expressively toward another woman.",
      "A woman leans toward a companion and makes animated facial expressions."
    ],
    "observed_actions": [
      {
        "canonical_label": "A woman in sunglasses gestures expressively toward another woman.",
        "evidence_type": "distinct_visible_action_or_reaction",
        "sample_ids": [
          "SAMPLE_07",
          "SAMPLE_08",
          "SAMPLE_09"
        ]
      },
      {
        "canonical_label": "A woman leans toward a companion and makes animated facial expressions.",
        "evidence_type": "distinct_visible_action_or_reaction",
        "sample_ids": [
          "SAMPLE_12",
          "SAMPLE_13",
          "SAMPLE_14"
        ]
      }
    ],
    "action_evidence_ids": []
  }
}
```

## 0002.json

`runs/mi-otra-yo-s03e03/semantic_observations/v2/responses/VE_3dc28f49d13b80fe/7603a4c29c2cd1aae2f6fdae27f2c424e7062432f1fb7078a66e68133b838a95/0002.json`

```json
{
  "created_at": "2026-10-08T23:32:11.837740Z",
  "diagnostic": {
    "errors": [
      "reusable_state_not_complete_action"
    ],
    "event_id_present": true,
    "expected_event_id": "VE_3dc28f49d13b80fe",
    "returned_event_id": "VE_3dc28f49d13b80fe"
  },
  "cumulative_provider_provenance": {
    "cost_usd": 0.00151704,
    "model": "gpt-6-luna",
    "provider": "openai",
    "request_count": 2,
    "usage": {
      "cached_tokens": 2814,
      "prompt_tokens": 10718,
      "response_tokens": 1397,
      "thinking_tokens": 0,
      "total_tokens": 12115
    }
  },
  "provider_trace": [
    {
      "attempt": 1,
      "model": "gpt-6-luna",
      "provider": "openai",
      "status": "COMPLETE"
    }
  ],
  "response": {
    "event_id": "VE_3dc28f49d13b80fe",
    "represented_shot_ids": [
      "FULL_SHOT_0038",
      "FULL_SHOT_0039",
      "FULL_SHOT_0040"
    ],
    "moment_status": "reusable_state",
    "action_or_moment_complete": "unclear",
    "visual_utility_kind": "generic_dialogue_only",
    "conversation_visual_signal": "generic_dialogue_only",
    "context_dependency": "medium",
    "visible_states": [
      "Several women are gathered outdoors, seated close together and facing one another.",
      "In the later shots, two women lean toward one another while the surrounding group remains nearby."
    ],
    "temporal_support_sample_ids": [
      "SAMPLE_01",
      "SAMPLE_03",
      "SAMPLE_05",
      "SAMPLE_06",
      "SAMPLE_08",
      "SAMPLE_10",
      "SAMPLE_11",
      "SAMPLE_13",
      "SAMPLE_15"
    ],
    "visual_actions": [],
    "observed_actions": [],
    "action_evidence_ids": []
  }
}
```

INPUT EVIDENCE: Failed attempt files did not persist input evidence, request
context, contact sheet, or canonical people catalog. The directory fingerprint
identifies the input but cannot reconstruct it. Event ranges and deterministic
sample_plan remain available: fps 24, frames [4037,4385), shots FULL_SHOT_0038,
FULL_SHOT_0039, FULL_SHOT_0040. Reconstructed sample frames are
4037,4058,4079,4100,4121,4122,4150,4179,4208,4237,4238,4274,4311,4347,4384.
This is reconstructed metadata, not the original saved visual evidence.
Read-only offline validation confirms the utility violation in both responses;
full historical people/pixel grounding cannot be reproduced from the failed
attempts alone. The utility violation independently prevents recovery.

FIX: temporal_semantics.TemporalObservationResult describes conditional semantics;
validate_response separates errors and rejects blank states; policy_projection
exposes raw/effective completeness metadata; observe_events persists attempt input
evidence, revalidates feedback without rewriting historical files, and retains
all paid-attempt provenance on offline recovery. semantic_observations.apply_policy_result
materializes completeness metadata. broll_policy_v1 logic remains unchanged.

RAW VS EFFECTIVE COMPLETENESS: Raw V2 action_or_moment_complete is true only for
complete_action, false for incomplete_action, false/unclear for reusable_state,
and unclear for unclear. Only a validated reusable state is projected to effective
action_or_moment_complete=true for the legacy moment branch. Immutable raw
observations are unchanged. temporal_completeness records both values explicitly.

OFFLINE RECOVERY: NO for these E03 responses. Both still violate useful_state
utility; no field is repaired or inferred from prose. Synthetic regression tests
verify that a valid saved response with an old rejection diagnostic is recovered
without provider calls or rewriting diagnostics. This is recovery machinery
coverage, not a claim that the old validator actually rejected that valid fixture.

ENRICHMENT STATE: Five completed records before failure, all validated read-only
against current event identity: VE_fc73488efeb90cb3 (reusable_state),
VE_f6a1dc77dc7bccf9 (unclear), VE_cf8f5537477c7c29 (unclear),
VE_7103dc3ef60622a0 (unclear), VE_fddbf5f0841708fa (unclear). All remain reusable
as observations; unclear does not imply KEEP. The batch stopped after two invalid
responses for this event, not after the first provider response. Fail-fast is
retained: diagnostics are durable, prior successes are reused, later events are
unattempted, and restarting an exhausted invalid event makes no new call.

REAL PROVIDER CALLS DURING FIX: 0.

E03: Not resumed. No E03 artifacts or existing decisions modified.

TESTS: Focused temporal/semantic tests: 39 passed. Full pytest: 530 passed,
1 skipped (531 collected), exit 0 on final implementation. git diff --check:
passed. Fixtures use synthetic responses and Offline providers only. Coverage
includes complete/incomplete/unclear/state semantics, false/unclear raw state
completeness, missing temporal support, raw/effective projection, legacy policy,
historical diagnostic recovery, no duplicate call, invalid identity, interruption,
batch failure/resume, and generic event IDs without episode-specific branches.
