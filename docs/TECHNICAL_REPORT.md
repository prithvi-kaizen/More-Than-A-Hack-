# Possession inference from unreliable tracking with a player-motion expert

## Approach

Ball visibility and proximity alone are unreliable indicators of controlled possession.
A pass can move the ball away from its team, a challenge can bring both teams close,
and missing observations can resemble a stoppage. The submitted system combines
geometry-based classifiers, a small player-motion temporal network and exact
three-state Viterbi decoding. The motion branch contributes most when ball geometry
is unreliable. Every component runs locally with NumPy, scikit-learn and PyTorch.

On two external games reserved until the system was frozen, mean official spellF1
was 0.5901 across clean/moderate/severe input conditions, compared with 0.5607 for
the preceding tree model. Under structured corruption, scores were 0.5309 and
0.4554. These tests support the added motion branch within the external provider;
competition-game accuracy has not been measured.

## Inputs and supervision

Inference uses the tactical ATD metadata and raw tracking only. Team membership
comes from the metadata; track slots are not treated as persistent player identities.
Normalized coordinates are converted to metres using the organizer's published pitch
dimensions. Missing observations and positions outside the field are preserved.
ATD record k maps directly to video frame k−1. Each published half window is processed
independently, so feature context and decoding cannot cross halftime.

Homography is unnecessary for these already normalized pitch coordinates. Smart
tags and broadcast fusion were deferred because their reliability and alignment
lacked matching external validation. No supplied video was available locally or
used. No target possession labels, manually corrected tracking, human-produced
event data about the provided game, output-versus-video checks or target tuning
entered the method. Codex assisted implementation and documentation; numerical
fitting and inference run in local processes.

The only labelled source is IDSSE, released by Deutsche Fußball Liga (DFL) under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its
[publisher catalogue](https://doi.org/10.6084/m9.figshare.28196177) and
[Bassek et al. (2025)](https://doi.org/10.1038/s41597-025-04505-y) describe the data.
Sportec/DFL supplies the control-based BallPossession and active/inactive BallStatus
fields. The collection includes operator quality control; the paper does not identify
the individual annotator or algorithm for every possession frame. This project
created no new human labels. External human-annotated events were used only for a
training-game semantic audit, never as prediction inputs or relabelling rules.

| Use | Complete match IDs |
|---|---|
| Fit and four-fold architecture comparison | J03WMX, J03WN1, J03WOH, J03WOY |
| Development selection | J03WPY |
| Final tests after freezing | J03WQQ, J03WR9 |

The four fitting games contain eight halves and 567,278 provider-labelled frames at
25 Hz: 301,977 H/A frames and 265,301 D frames. There are 113,458 stride-five samples
per input view. The final tree model uses four views, giving 453,832 examples for
the alive/dead head; ownership fitting uses live frames and team-exchanged copies.
The temporal model uses the same games and four views at 5 Hz, with overlapping
windows. Augmented views are repeated observations, not additional independent games.
All halves/windows/views of an evaluation match stay outside its fitted model.
The released final checkpoints were refitted only on the four fitting games.

BallStatus 0 maps to D. For active frames, BallPossession 1/2 maps to H/A. Unknown
ownership is masked from team loss, and unknown status from alive/dead loss. Dead
status overrides ownership. The adapter preserves provider labels and never infers
dead ball from an event-record gap. Original publisher sizes/checksums and separate
input/label archives enforce provenance; exact counts are in `docs/training_coverage.json`.

## Model and possession rules

The tree system has separate histogram-boosted alive/dead and conditional H/A heads.
It uses 42 alive/dead features and 21 ownership features, including team proximity,
player movement, boundary distance, visibility history, relative velocity and control
compatibility. Ownership features reverse sign when teams are exchanged. Training
and prediction enforce that symmetry. Capped partial inverse-duration weights favor
brief spells without letting them dominate the loss. A separate 84-feature boundary
head is supervised near live H/A changes, not stoppages. Its score is transition
evidence rather than a calibrated probability.

The player-motion expert receives no ball coordinates. For each team, pooled position,
velocity, speed and 4×3 occupancy statistics produce 27 values plus missingness
indicators. Team differences and sums give 108 inputs. A 19,169-parameter temporal
convolutional network uses 32 channels and three residual kernel-five layers with
dilations 1, 2 and 4. Comparing each input with exchanged teams enforces antisymmetric
ownership logits. Per-frame channel normalization makes chunked inference stable.
No published pretrained weights are used.

Training runs eight fixed AdamW epochs with learning rate 0.001, weight decay 0.001,
batch size 64 and gradient clipping at 1. At 5 Hz, 64-sample windows have stride 32;
internal window edges are masked by the 14-sample context radius. D/unknown frames
do not contribute ownership loss. The receptive field spans 29 samples, or 5.6 seconds
between endpoints. Inference uses 512-sample chunks with context halos and interpolates
logits back to 25 Hz within each half.

Geometry reliability q is zero for a missing ball. A centered 0.08-second trajectory
residual and extreme-speed penalty reduce confidence in suspect detections, using
fixed scales of 3 metres and 15 m/s above 45 m/s. Unsupported visible edges receive
q=0.5. Conditional tree ownership logits are scaled by 0.5+0.5q. The motion model
then receives probability weight 0.35(1−q). Dead-ball emission mass is preserved.

Exact Viterbi decoding jointly chooses H/A/D labels. The H/A switch cost starts at 5
and is reduced by at most 75% using reliability-weighted boundary evidence; transitions
to/from D cost 40. There is no minimum spell duration. During passes or briefly loose
balls, context and switching cost encourage continuity until contrary control evidence
accumulates. A nearby opponent or single touch does not mechanically trigger a switch.
Challenges are resolved from the same learned control evidence, with no separate
contested state or manual correction. Stoppages and restarts are inferred by the
alive/dead head and joint decoder, not whistle detection or hand-specified restart times.
Missing ball data alone does not force D. These are learned approximations to control,
so deflections, aerial duels and very short controlled touches can still be wrong.

Inside the organizer's half windows the system emits H/A/D. It emits X outside them.
Both feature extraction and decoding restart at each half. Offline future context is
used; this is not a streaming real-time system.

## Selection and evidence

The official metric averages spellF1 over IoU thresholds 0.3–0.7. It rewards recovery
of complete team possessions and penalizes fragmentation; D splits spells. The
organizer's original metric code is pinned and used directly.

The first decoder's 5/40 penalties were selected on development from ten declared
candidates. Subsequent structural choices used whole-match folds within the four
fitting games before a development check. Phase 5 compared the prior tree system,
reliability-only changes and motion fusion. Adoption required a primary gain of at
least 0.005, wins in at least three folds, and no structured spellF1 or long-spell
recall regression above 0.01. Epochs and thresholds were fixed before this comparison.
Motion fusion won all four folds. The same development game was reused across phases;
its scores are selection results, not independent tests.

| Evaluation | Prior tree model | Selected motion fusion |
|---|---:|---:|
| Training-match CV, original three conditions | 0.5153 | 0.5436 |
| Development, original three conditions | 0.5373 | 0.5581 |
| Final two games, original three conditions | 0.5607 | 0.5901 |
| Final two games, structured corruption | 0.4554 | 0.5309 |

Clean/moderate/severe conditions share labels and differ only in seeded input gaps,
noise and track fragments. Structured corruption additionally introduces crowd/edge
occlusion, partial camera coverage, false ball detections and duplicate tracks. No
labels enter corruption functions. Camera visibility is simulated using ball position,
so the player-only branch can indirectly observe ball-conditioned visibility patterns.
This approximation is disclosed; synthetic stress is not a measured camera-error model.

Both final games, all halves and all conditions were evaluated after weight/code
freezes, with no later fitting or selection. Motion fusion improved every condition
of both matches. Pooled sub-two-second recall at IoU 0.5 was 30.8% clean, 13.3% severe
and 19.0% structured; long-spell recall under structured noise was 94.5%. Corrupted
conditions of one game are correlated observations. With only two same-provider
final games, these results do not establish another provider's performance or ranking.

The main limitations are brief possessions, uncertain control semantics at touches,
missing or systematically misplaced balls, and transfer from external multicamera
tracking to the competition feed. Real direction changes can also lower geometry
reliability. Broader external control annotations and actual camera outages would
be useful future tests; no such additions were introduced after freezing.

## Reproduction and artifacts

The public repository is
https://github.com/prithvi-kaizen/More-Than-A-Hack-. The `models-v1` release contains
the three externally trained checkpoints required by the frozen guard, plus a model
card and CC BY 4.0 attribution. Competition data and predictions are excluded.

```sh
uv sync --frozen --extra temporal
uv run --frozen --extra temporal python scripts/download_checkpoints.py
LOKY_MAX_CPU_COUNT=2 uv run --frozen --extra temporal python scripts/infer_target.py \
  --data-dir /path/to/organizer-bundle
```

The bundle needs only `tactical/BAR-ATH_tactical_atd.xml` and
`tactical/BAR-ATH_tactical_atd.txt`. The default output is
`outputs/phase6/submission.csv`. Downloads verify archive/member SHA-256 digests;
inference verifies the original model/configuration/source/dependency freezes before
and after prediction. No training data or credentials are needed for released-model
inference. Python 3.12 and the locked `temporal` dependencies are required; the
supplied CLI uses CPU inference with two threads.

The saved 199,319-row CSV is checked by both the pinned validator function and CLI.
Two independent full processes reread inputs, reloaded models and generated identical
CSV bytes. Final-test checkpoints also exactly replayed probabilities, labels and
metrics. These checks establish file validity and local numerical reproduction, not
target accuracy. Cross-platform byte identity requires separate verification.

Before uploading, read this document and confirm the team details and claims. The
team-named Markdown/CSV package is generated from this report and the unchanged
validated candidate; the registered team uses the organizer's emailed submission form.
