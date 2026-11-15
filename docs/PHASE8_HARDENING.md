# Phase 8: delivery hardening

The validated v1 possession CSV and frozen inference system are preserved. Packaging
now keeps the participant's existing handoff instructions byte for byte on repeated
runs. This prevents a rerun from removing locally recorded form links and deadline
details. The document and integrity manifest are refreshed normally. A conflicting
team name or a manifest already marked submitted fails before package writes, and
the validator log directory is created when absent.

Two synthetic regression tests exercise the complete packaging entry point with a
tiny invented CSV. They check document refresh, manifest digests, unchanged CSV and
handoff bytes, and rejection before writes for team/submission-status conflicts.
The official in-process format validator is used; the subprocess invocation and
frozen model check are mocked only in these synthetic tests.

The technical document now includes the pinned clone command and fixes duplicated
wording in the geometry-reliability explanation. These edits change no model claim,
weight, feature, decoder or possession label. Current `main` contains the packaging
fix; `inference-v1` remains the tested inference-source snapshot.

## Verification

Both frozen suites passed **82 tests**. The exact local Kaizen package was regenerated
twice. CSV bytes and the existing private handoff were identical before and after;
the regenerated document matches the revised public report, its manifest matches both
upload files, and both official validators pass. The status-only submission audit
also passed, including frozen source/model/input checks and saved independent replay
comparisons. Restricted details and the pre-update package backup remain local.

Linux reproduction remains unverified. The local Docker daemon answered as Linux
ARM, but its image store returned a storage I/O error on enumeration. No inference
container could be tested. The previously completed macOS arm64 fresh-public-clone
reproduction remains the verified platform. This is not evidence about Linux x86-64.

## Eligibility and next step

Eligibility is **unresolved** under the published 3–5-member rule. The participant
reported solo registration; no organizer exception or approved team change has been
provided. A private organizer inquiry is prepared at
`outputs/phase8/ORGANIZER_INQUIRY.md`; it has not been sent. Registration correspondence
and personal details stay out of the public repository.

Technical hardening is complete; the participant must settle advancement eligibility
with the organizers. Phase 9 remains a separate authorized task: run the bounded
external-only reliability experiment in the [improvement plan](SUBMISSION_REVIEW.md).
Do not use target predictions/video or the already-open final external games to
select its settings. No new fitting or target quality evaluation occurred in Phase 8.
No form submission or organizer message was sent.
