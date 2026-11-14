# Phase record

## Phase 0 — local asset preparation (complete)

Consolidated supplied assets by content hash, removed verified redundant copies,
and added Git exclusions. Seven structural audit checks passed. Detailed target
measurements remain local. Neither supplied camera video was present.

## Phase 1 — reproducible foundation

Implemented:

* Python 3.12 environment with pinned NumPy/pandas and a transitive `uv.lock`.
* Selected official scorer, validator, baseline and loaders pinned to upstream commit
  `f8f2629ff1e0985f81d7e59d8413678514ee1530`. Nine original-file blob hashes are
  checked automatically; the MIT licence is retained.
* A streaming ATD parser with explicit frame-index conversion, metadata-derived
  team membership, preserved NaNs and generic parse errors.
* Sixteen synthetic tests covering scoring semantics, false stoppage fragmentation,
  same-team restarts, halftime isolation, abstentions, invalid CSVs, the official
  validator CLI, coordinate scaling, missing data and source integrity.
* A staged-file publication gate, CI and phase-by-phase agent instructions.

Verification completed locally on Python 3.12.4:

* Frozen dependency installation succeeded (NumPy 2.2.6, pandas 2.2.3).
* All 16 synthetic tests passed in the locked environment.
* All nine vendored upstream blob hashes match the pinned source.
* Both complete local tracking files passed the new streaming parser.
* All seven existing structural asset checks passed again.
* The staged public-file gate passed; restricted inputs and local reports are excluded.

CI is configured to repeat synthetic checks on GitHub; its remote result is separate
from these completed local checks.

These checks establish input/evaluation mechanics. No possession model has been
trained, no target prediction generated and no public target-label score consulted.

## Phase 2 — external validation foundation (complete)

Implemented:

* Publisher-backed CC BY 4.0 catalogue with 21 original checksums and explicit DFL/paper credit.
* Frozen four/one/two match split; guards reject target IDs and silent repartitioning.
* Resumable mirror transfers with complete-file publisher checksum verification.
* Sportec metadata/geometry/label adapter with role-based teams, masked unknown labels,
  preserved missing observations, separate half timelines and no label fields in inputs.
* Separate local prepared input/label files with raw/prepared integrity provenance.
* Deterministic moderate/severe gap, noise and track-fragment corruption.
* Exact official external scoring and a training-only untrained diagnostic comparator.

Verification:

* 29 synthetic tests passed, including split isolation, label masks, coordinate mapping,
  duplicate rejection, deterministic corruption and halftime scoring boundaries.
* All seven real metadata adapters passed.
* All 14 real metadata/event files match publisher sizes and checksums.
* The complete J03WMX pilot tracking file matches the publisher checksum; both halves
  prepared successfully (145,967 total frames, no unknown status/ownership labels).
* The real-data corruption/scoring integration passed in all three fixed scenarios.

Untrained nearest-player diagnostic on the external training pilot:

| Inputs | spellF1 | macroF1 | Predicted possessions |
|---|---:|---:|---:|
| Clean | 0.1043 | 0.3826 | 2,240 |
| Moderate corruption | 0.0550 | 0.4289 | 3,775 |
| Severe corruption | 0.0111 | 0.4142 | 5,560 |

The provider labels contain 511 possessions. This is a training-pilot diagnostic,
not a competition score or model-selection experiment. Random missingness can improve
macroF1 by accidentally predicting dead ball while severely worsening spell recovery.
This supports prioritizing dedicated stoppage inference and coherent temporal decoding.

Acquisition limitation: one complete tracking match is locally verified; the other
six tracking files have not been downloaded. All seven metadata/event pairs are
verified. The frozen catalogue and downloader cover the entire dataset. Remaining
tracking acquisition is an explicit Phase 3 prerequisite, not an unattended job.

No training, target inference, public target-label scoring or final-holdout performance
inspection occurred. The pilot is a training match. Provider ownership remains a proxy
for the competition control definition; the adapter does not establish their equivalence.

## Supplemental archive review — complete

Checked six newly supplied ZIPs locally after Phase 2. Their seven members were
SHA-256-identical to the canonical tactical/broadcast assets. Removed the redundant
archives after rechecking both archived and retained content. No new asset or video
was present. The detailed manifest remains in ignored local reports. This inventory
check does not authorize or start Phase 3; the model plan is unchanged.

## Phase 3 — pending user instruction

Finish the remaining full external tracking transfers explicitly, verify and prepare
training/development data, then implement the first control/dead-ball emissions and
three-state decoder. All seven metadata/event pairs are already available. The pilot
validates the external workflow; it does not replace development evaluation. Do not
leave acquisition running after this phase or inspect holdout performance during tuning.

The initial corruption suite covers moderate/severe random interval loss, positional
noise and same-team track fragments. Add crowd/boundary-dependent loss and false
observations as further external ablations. Do not tune those mechanisms to target outputs.

Each phase ends with checks, plan updates and a GitHub push, followed by a stop.
The next phase begins only when the user asks to continue.
