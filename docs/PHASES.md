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

## Phase 3 — trained baseline and temporal decoding (complete)

* Acquired, publisher-verified and prepared all four training matches plus development.
  Verified raw tracking was compressed losslessly to stay within available disk space.
  The two final holdout tracking files remain deferred.
* Implemented geometry-only control/movement features, separate boosted alive/dead
  and ownership models, team symmetry, and exact three-state Viterbi decoding.
* Fitted 340,374 sampled training examples, including clean/moderate/severe inputs.
* Selected penalties on the development match from ten declared candidates. Compared
  against raw nearest-player and the original official smooth50 implementation.
* Added publisher-checked partial-range resume, acquisition/training locks, prepared
  provenance enforcement, checkpoint replay and a class/short-spell audit.

Verification: 45 synthetic tests passed; full pilot feature extraction passed; all
five required matches passed raw/prepared integrity checks. The checkpoint replayed
all development scores exactly. No holdout score or competition prediction was used.

| External development spellF1 | Clean | Moderate | Severe | Mean |
|---|---:|---:|---:|---:|
| Official smooth50 | 0.3171 | 0.2707 | 0.1260 | 0.2379 |
| Boosted argmax | 0.2680 | 0.1386 | 0.0476 | 0.1514 |
| Boosted Viterbi | 0.5260 | 0.5139 | 0.4378 | 0.4926 |

The selected pipeline is boosted Viterbi, with turnover/dead-transition penalties
5/40. Mean macroF1 is 0.8887. The development match was used for selection; these
are proxy-label validation results, not an independent test or competition score.
Full aggregates and method details: [Phase 3 model](PHASE3_MODEL.md) and
[results](phase3_results.json).

Main finding: temporal decoding suppresses fragmentation, but brief ownership spells
are often lost. Below-two-second recall at IoU 0.5 is only 7.3% clean and 0% severe;
above-15-second recall exceeds 93%. More fixed smoothing is not the next priority.

## Phase 4 — control evidence and adaptive boundaries (complete)

* Audited external training-game short spells against timestamp-aligned event evidence;
  preserved all original labels and recorded uncertainty in the control definition.
* Tested frame weighting, capped inverse-duration weighting, and a control/adaptive
  boundary model in four whole-match training folds. No fold model saw its held-out match.
* Added geometry-only crowd/camera occlusion, duplicate tracks and false ball detections.
* Selected `control_adaptive` before the development comparison. It improved primary
  spellF1 in all four folds, from 0.4710 mean for the frame baseline to 0.5153.
* Verified 56 tests, frozen Phase 3 replay and byte-identical development labels from
  two independent Phase 4 checkpoint loads. Final holdouts remain sealed.

| Development spellF1 | Clean | Moderate | Severe | Mean, original scenes |
|---|---:|---:|---:|---:|
| Frozen Phase 3 | 0.5260 | 0.5139 | 0.4378 | 0.4926 |
| Phase 4 | 0.6024 | 0.5600 | 0.4495 | 0.5373 |

Structured-noise spellF1 improves from 0.4283 to 0.4498. Clean short-spell recall
improves from 7.3% to 16.9%, but remains weak; long-spell recall under structured
noise falls from 91.9% to 85.5%. These external selection-set results do not establish
a competition score. See [method](PHASE4_MODEL.md) and [aggregates](phase4_results.json).

The semantic audit found control-supporting event overlap inside 63 of 600 brief
training spells and 626 of 685 long spells. Event timing/completeness limits the
interpretation, so no labels were removed. Only 14 short spells had no stride-five
training sample; changing sampling resolution alone is unlikely to fix the weakness.

## Phase 5 — reliability and player-motion fusion (complete)

* Added geometry reliability, conditional-logit tempering and conservative motion fusion.
* Trained a 19,169-parameter, team-symmetric temporal network from scratch on player
  movement only. PyTorch is optional, with CPU wheels configured for Linux CI.
* Compared four fixed policies in four whole-match folds. Each fold refitted the tree
  and neural models without its evaluation match; frozen Phase 4 scores replayed exactly.
* Selected motion fusion before development: mean CV spellF1 rises from 0.5153 to
  0.5436, with wins in all four folds. Structured-noise spellF1 rises from 0.3979
  to 0.4694, and pooled long-spell recall rises from 79.2% to 83.3%.
* Reliability-only variants did not meet the declared gain requirement. Model settings,
  eight training epochs and adoption thresholds were fixed before these comparisons.

| Development spellF1 | Clean | Moderate | Severe | Mean, original scenes |
|---|---:|---:|---:|---:|
| Frozen Phase 4 | 0.6024 | 0.5600 | 0.4495 | 0.5373 |
| Selected motion fusion | 0.6058 | 0.5704 | 0.4982 | 0.5581 |

Structured development spellF1 rises from 0.4498 to 0.5111; long-spell recall stays
at 85.5%. Severe short-spell recall improves from 5.6% to 8.1%, while clean recall
stays at 16.9%. Brief possessions and provider-to-competition transfer remain concerns.
The development match has been reused across phases and is not an independent test.

All 67 synthetic tests passed. Reloaded checkpoints exactly replayed out-of-fold
probabilities, decoded labels, all policy metrics and the development comparison.
Training-match metadata, cache/source hashes and dependency integrity were checked.
The selected checkpoint/source manifest, experiment settings and external aggregates
are public; detailed arrays and model weights stay
local. See [method](PHASE5_MODEL.md), [results](phase5_results.json) and
[selection manifest](../configs/phase5_selected.json). No final holdout or competition
inference was run.

## Phase 6 — frozen final tests and local inference (complete)

* Froze selected weights, decoder settings, feature sources and dependencies before
  acquiring final holdouts; froze tested execution code before opening their performance.
* Acquired, publisher-verified, prepared and losslessly compressed J03WQQ/J03WR9.
  A separate final-test loader preserves the original training/development guards.
* Evaluated three frozen models on both halves and all four fixed conditions.
  No model selection, fitting or parameter changes followed final-test scores.
* Added the tactical ATD adapter, full-length CSV generation, private validator logs,
  raw-input integrity checks and independent-process byte reproduction.

| Final external model | Mean spellF1, original scenes | Structured spellF1 |
|---|---:|---:|
| Phase 3 | 0.5328 | 0.4283 |
| Phase 4 | 0.5607 | 0.4554 |
| Selected motion fusion | 0.5901 | 0.5309 |

The selected model improves every match/condition comparison with Phase 4. Pooled
structured short-spell recall improves from 12.8% to 19.0%, and long-spell recall
from 90.6% to 94.5%. Short spells remain weak. Two same-provider external tests do
not establish accuracy on the competition camera or its possession convention.

All 75 synthetic tests passed. The inference wrapper replayed existing development
scores exactly, and reloaded checkpoints replayed all final-holdout probabilities,
labels and metrics exactly. Two independent complete target runs produced identical
CSV bytes; both pinned validator function/CLI checks passed. The CSV has the required
199,319 rows. Only coarse target statuses are published; predictions and detailed
reports remain local. No target labels, manual corrections or video comparison were used.

The validated candidate is `outputs/phase6/submission.csv`. Target checks overlapped
the remaining external transfer under the already frozen model. See
[method and reproduction](PHASE6_INFERENCE.md), [aggregates](phase6_results.json),
[selection freeze](../configs/phase6_frozen.json) and
[execution freeze](../configs/phase6_execution_frozen.json).

## Phase 7 — packaging and clean public reproduction (complete)

* Created `Kaizen.csv` and `Kaizen.md` using the participant's registered team name.
  The CSV bytes match the Phase 6 candidate exactly and the exact upload file passes
  both pinned validation interfaces. The document covers method, assets, labelled-data
  provenance/amount, possession semantics, selection, limits and reproduction.
* Published only the externally trained checkpoints/card/licence as `models-v1`, with
  CC BY 4.0 attribution and pinned archive/member hashes. No competition data or
  predictions were uploaded. The archive's anonymous public download hash matches.
* Published tested source as `inference-v1`. A fresh anonymous public clone began
  without data/models/environment, installed a separate locked environment, downloaded
  public weights and reproduced the candidate CSV and provenance exactly.
* Added strict checkpoint installation and private team-package tooling. All 80
  synthetic tests passed. Target details remain local and frozen runtime/weights
  were unchanged. Verification is macOS arm64; Linux CI uses synthetic data only.

The private upload files are under `outputs/phase7/upload/`, with a manifest and
handoff instructions. The public [delivery record](PHASE7_DELIVERY.md) and
[technical report](TECHNICAL_REPORT.md) document completed checks and their scope.
All implementation phases are complete. The participant reviews the document and
submits the form; no form submission was made by the tools.

Each phase ends with checks, plan updates and a GitHub push, followed by a stop.
The next phase begins only when the user asks to continue.
