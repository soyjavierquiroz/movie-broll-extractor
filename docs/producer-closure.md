# Producer lifecycle

`movie-broll run input/<episode>` defaults to automatic Narrative Mapper V3.
Set `NARRATIVE_PROVIDER_MODE=external` for the existing manual chunk/inbox workflow.
API mode uses `NARRATIVE_PROVIDER=openai` (default) and `NARRATIVE_MODEL`, falling
back to `OPENAI_MODEL` and the existing OpenAI default. `OPENAI_API_KEY` supplies
credentials. Existing `OPENAI_REASONING_EFFORT` and
`OPENAI_CONNECT_TIMEOUT_SECONDS`, `OPENAI_READ_TIMEOUT_SECONDS`,
`OPENAI_WRITE_TIMEOUT_SECONDS`, `OPENAI_POOL_TIMEOUT_SECONDS` configure requests.
The narrative runner owns two validation attempts and up to two transient retries;
SDK retries are disabled. It never selects Gemini as fallback. The explicit legacy
`narrative run` Gemini diagnostic command remains available.

Chunk checkpoints fingerprint input content (including cue IDs/timestamps), V3
prompt content/version, schema content, window/overlap, and production profile.
Provider/model are recorded as provenance. Valid matching maps are reused at zero
requests. Invalid output and exhausted provider retries stop before downstream
production; rerunning resumes. Consolidation requires every chunk to be valid.

Inspect the IDs reported by `movie-broll status input/<episode>`. Record a decision:

```
movie-broll review-vertical approve --run <episode> --asset-id m007 --reason "Visually inspected" --reviewer operator
movie-broll review-vertical approve --run <episode> --asset-id m007 --override-qa --reason "Inspected and accepted the crop" --reviewer operator
movie-broll review-vertical reject --run <episode> --asset-id m007 --reason "Crop unusable"
movie-broll close input/<episode>
```

The override is explicit and requires a reason. Automatic REVIEW, hard failures,
soft warnings, and review reason remain unchanged. Structural identity, event,
metadata, checksum, missing-member and media-decode failures cannot be overridden.
Approval records are immutable and repeat approvals validate/reuse them.

Closure uses only local producer state and makes zero semantic calls. It includes
automatic PASS and human APPROVED REVIEW, blocks pending reviews and broken approved
packages, and excludes rejected packages. JSON `asset.id` is authoritative, including
sparse IDs. Registry identity and source-event correspondence are checked.

`runs/<episode>/final-export/` contains exactly five files per approved asset:
`<id>-<slug>.json`, `<id>-<slug>.mp4`, `v<id>-<slug>.mp4`,
`<id>-<slug>.jpg`, `v<id>-<slug>.jpg`, plus
`E<episode-number>_FINAL_APPROVAL_MANIFEST.json` (zero padded to at least two digits).
Metadata is copied; media uses hardlinks where possible, otherwise copies.

`producer_final_approval_manifest_v1` records episode identity, approval totals,
exact producer ID membership, all five filenames, original approval provenance,
computed SHA256 checksums, and existing source timeline. No semantic fields are
invented. Closure probes dimensions/orientation and positive finite MP4 duration,
fully decodes video and thumbnails, checks supplied checksums and sizes, and verifies
physical membership. A matching existing handoff is validated and reused. A changed
or contaminated handoff blocks rather than silently rebuilding or deleting files.
Successful closure persists `handoff_state.json` outside final-export.

Source reset commands retain their existing behavior. Release remains a separate
operator action after external SAFE TO RELEASE approval.
