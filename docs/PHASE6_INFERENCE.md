# Phase 6: frozen evaluation and local inference

## Freeze and access boundaries

`configs/phase6_frozen.json` records the Phase 5 selected manifest, both model hashes,
the Phase 3 comparator, decoder configurations, dependency lock and original feature
sources. It was written before acquiring the final holdouts. The fixed test settings
use two reserved games, four corruption conditions and seed `2026 + 2000 + 10*i + half`.
The structured corruption retains its existing additional seed offset of 5000.

`configs/phase6_execution_frozen.json` records the tested execution code before
opening final-holdout performance. Every inference/evaluation process verifies both
records. The scripts refuse changed hashes. The original training/development loaders
still reject final-holdout IDs. A separate guarded loader provides test-only access,
verifies publisher checksums and prepared-file provenance, and performs no fitting.

The copied Phase 5 manifest is a historical snapshot: its unseen-holdout and
no-target-inference fields describe the state at selection. Phase 6 outcomes are
recorded separately; those historical fields are preserved to keep the freeze intact.

The final external comparison uses the frozen Phase 3, Phase 4 and selected motion
fusion checkpoints on identical observations. Inputs and labels remain separate.
All conditions and both halves are evaluated; no best condition or model is selected
from final-test scores. Features and predictions are cached separately with integrity
hashes, allowing an independent checkpoint reload to reproduce probabilities, labels
and official metrics without another fit.

The inference wrapper first reproduced all existing Phase 5 development metrics.
That is an implementation regression check; development was already used for model
selection and remains separate from the final-test claim.

## Final external results

| Frozen model | Mean spellF1, clean/moderate/severe | Structured-noise spellF1 |
|---|---:|---:|
| Phase 3 | 0.5328 | 0.4283 |
| Phase 4 | 0.5607 | 0.4554 |
| Selected motion fusion | **0.5901** | **0.5309** |

The primary mean averages two unseen matches and three fixed input conditions;
the six measurements are not six independent games. The selected system improves
on Phase 4 in every condition of both matches. No model was fitted, selected or
retuned from these results. Full per-match/per-condition results and diagnostics are
in [phase6_results.json](phase6_results.json).

Pooled below-two-second recall at IoU 0.5 is 30.8% clean, 13.3% severe and 19.0%
structured, versus 30.8%, 10.3% and 12.8% for Phase 4. Long-spell recall under
structured noise improves from 90.6% to 94.5%. Brief possessions remain weak despite
the overall improvement. Both final games use the same external provider and its
ownership convention; this does not measure transfer to the competition camera
or establish a competition score.

## Tactical adapter

The model reads only the tactical metadata XML and ATD text. XML order supplies team
membership and slot order, matching the pinned organizer loader: first team is H,
second is A. Slots are not treated as persistent player identities. The third
observation field is not used; movement features are recomputed from coordinates
using the same code as external training.

ATD record `k` becomes video/submission frame `k-1`. No additional tracking-counter
offset is applied. The pinned organizer contract supplies 199,319 frames at 25 fps,
the corrected 105.3-by-68-metre pitch, and inclusive half windows 11185–90651 and
110808–190197. The coordinate adapter multiplies normalized x/y by pitch length/width.
It preserves missing coordinates and positions outside the field instead of clipping
them or inventing observations.

Each half gets a separate feature extraction, temporal-network pass and Viterbi decode.
No smoothing, velocity derivative, possession carry or neural context crosses halftime.
Predictions inside the windows are H/A/D. Other frames receive X using the organizer's
published ungraded windows. Smart tags, homography, broadcast tracking and video do
not enter this selected system; no unvalidated input fusion is added during inference.

CSV writing is atomic and canonical: UTF-8, LF line endings, exact `frame,label`
header and contiguous zero-based frames. Both the pinned validation function and
the original validator CLI check the saved file. Validator details stay in ignored
local reports, and the controller sees fixed PASS/FAIL messages only.

The complete CLI runs twice in independent processes, each loading checkpoints and
reading the raw input again. It recomputes features and predictions and compares
the resulting CSV bytes, input hashes, freeze hashes and validation results. This
checks deterministic reproduction on the current host and locked environment;
cross-platform byte identity is not established by this check.

## Reproduction

The earlier phases' external preparations and local checkpoints are prerequisites.
All data operations and model inference run locally. No hosted model API is used.

```sh
uv sync --frozen --extra temporal
uv run --frozen --extra temporal python -m unittest discover -s tests -v
uv run --frozen --extra temporal python scripts/freeze_phase6.py
uv run --frozen --extra temporal python scripts/acquire_phase6.py
uv run --frozen --extra temporal python scripts/freeze_phase6_execution.py
LOKY_MAX_CPU_COUNT=2 uv run --frozen --extra temporal python scripts/audit_frozen_predictor.py
LOKY_MAX_CPU_COUNT=2 uv run --frozen --extra temporal python scripts/run_phase6_holdouts.py
LOKY_MAX_CPU_COUNT=2 uv run --frozen --extra temporal python scripts/run_phase6_holdouts.py --replay
LOKY_MAX_CPU_COUNT=2 uv run --frozen --extra temporal python scripts/audit_target_replay.py
```

The acquisition command needs network access and verifies the original publisher
checksums even when using the existing mirror. Completed raw tracking is compressed
losslessly after preparation. On a repeat run with final-test results already present,
use `--replay`; the first evaluation record is retained.

For a single prediction run, the tactical bundle defaults to `data/` and can be moved
with `MTAH_DATA_DIR` or `--data-dir`. Output must remain under ignored `outputs/`:

```sh
LOKY_MAX_CPU_COUNT=2 uv run --frozen --extra temporal python scripts/infer_target.py
```

The submission candidate is `outputs/phase6/submission.csv`. Detailed input hashes,
prediction hashes, local validator logs, feature caches and checkpoints remain local.
Public documentation contains external aggregate metrics and coarse target-check
statuses only. No target possession labels, manual corrections or video comparison
are used. Team naming, technical-document delivery and fresh-clone/checkpoint
distribution are Phase 7 packaging work.

## Completed checks

All 75 synthetic tests passed. Both independent target runs generated valid full-length
CSVs, passed the pinned validator function and CLI, and produced identical bytes and
provenance reports. Only coarse check statuses are published; the prediction file
and detailed reports remain local. The target runs overlapped the final external
transfer because the frozen model made them independent of holdout outcomes.

Independent checkpoint reloads also exactly reproduced final-holdout probabilities,
labels and all recorded official metrics. Both model and execution freezes stayed
unchanged throughout evaluation and inference.

Phase 7 must package all three checkpoint files required by the freeze guard, including
the Phase 3 comparator, along with the source and instructions. The current check is
local process reproduction; a fresh-checkout reproduction and distribution route for
the external-trained checkpoints still need to be completed in Phase 7.
