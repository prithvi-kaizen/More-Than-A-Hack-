# Phase 9 external reliability experiment

Protocol authored before the Phase 9 evaluation. Executable settings are in
`configs/phase9.json`; `scripts/run_phase9.py` freezes their hashes, all new model/run
source and the reused fold-checkpoint hashes before reading experiment outcomes.
Any change after that freeze aborts rather than silently resuming a different run.

## Single hypothesis and fixed challenger

The existing geometry score can trust a smoothly misplaced ball. A small validity
classifier may identify those cases from player proximity and relative movement,
then shift ownership evidence toward the existing player-motion expert. This phase
changes only q: `min(original_q, calibrated_validity)`. Frozen v1 ownership/dead-ball
models, motion checkpoints, fusion cap and Viterbi penalties remain fixed.

Validity features are ball visibility, original q, two nearest player distances,
counts within 2 and 5 metres, maximum velocity/proximity control compatibility, ball
speed, field-boundary distance and visible-player count. They are invariant to team
exchange and slot order. They contain no possession labels or original clean ball
position at prediction time. The reference ball is used only to construct the
external training target: observed error at most 2 metres, on frames where the
original ball is known. Unknown original ball positions are excluded. This reference
is a provider measurement, not a proof of physical ball accuracy.

The HGB validity classifier uses 80 iterations, seven leaves, minimum leaf size 80,
L2=1, no early stopping and seed 2026. In each held-out match fold, the last remaining
training match calibrates a sigmoid logistic regression (C=1) on classifier logits;
the other two fit HGB. All halves, corruptions and windows of the held-out match are
excluded from both fitting and calibration. Only the original four training matches
are available to these APIs. Existing Phase 5 tree/TCN folds, each trained without
its corresponding held-out match, are hash-checked and reused without refitting.

Fit data uses clean/moderate/severe/sustained-error views, every 25th frame, with
fixed seeds 3101 and 3203 (clean once). Natural passes, aerial geometry and unknowns
are retained without possession-label filtering. No new control labels are made.

## Stress conditions and controls

Five evaluation seeds (4101, 4201, 4301, 4409, 4507) are disjoint from fit seeds. The
actual RNG seed additionally includes a fixed match/half offset. Conditions are
clean, moderate, severe, original structured corruption, sustained false ball tracks,
2/8/20-second ball outages, independent camera crops, and ball-centred camera crops.
All transforms take inputs only. These fixed distributions are experimental stress
tests, not estimates of the competition camera's error process.

Sustained errors add a 10–30-metre vector in a random direction for 2–8 seconds at
six Poisson starts per minute. These offsets may overlap or go outside the pitch.
Outages start at three per minute. Camera crops last eight seconds and hide players
outside a 50-metre horizontal view. Independent centres are uniform over allowed
field centres; paired ball crops use the original ball's x at block start when
known. Same-team slot permutations every block and two brief duplicate slots per
team are included in both camera conditions. These camera tests leave the ball
input unchanged to isolate the player-visibility shortcut; they are not full camera
simulations. Clean repeats are identical and computed once, not five independent
observations. Source code specifies clipping at half ends and missingness handling.

Four fixed policies are scored: v1, challenger, v1 with motion weight zero and
challenger with motion weight zero. The last two are diagnostic ablations only;
they are ineligible for adoption. Compare independent-camera and ball-centred-camera
results explicitly; a gain requiring ball-centred visibility fails the adoption gate.

## Endpoints and decision

Use the original official spellF1 at every IoU threshold, with half isolation.
Report condition/match means, worst-condition performance, short/long spell recall,
false D runs intersecting true possessions, and restart/turnover timing diagnostics.
The latter use same-pair one-to-one nearest-first matching within 25 frames, reporting
support/matches and summed absolute/signed latency; this diagnostic is distinct from
the official scorer's transition tie-break. No bootstrap is requested for individual
stress runs. Matches are the sampling unit; seed repeats are correlated.

Each of ten conditions and five seeds receives equal weight within each match; four
match means receive equal weight. The exact gates are mean spellF1 gain >=0.01,
wins in at least three matches, no match-average or condition-average loss >0.01,
no pooled long-spell recall loss >0.01, positive sustained-error gain, and positive
independent-camera gain. This conservative independent-camera gate can reject an
otherwise useful model with a neutral camera result. Gates are decision thresholds,
not a statistical significance claim or prediction of ranking.

If CV passes, fit HGB on the first three original training matches and calibrate on
the fourth, then run one fixed comparison on development J03WPY. Apply the same
gain/regression gates, with one winning development match required. No rescue tuning.
If CV fails, skip development and retain v1; leave its development labels unopened
by this experiment. J03WQQ/J03WR9 and competition inputs/labels/video are never read.

Outputs/checkpoints, row-level arrays and replay artifacts stay under ignored local
directories. Publish aggregate external evidence, protocol/settings and code only.
The final upload CSV and frozen inference release remain v1. Any recommended
challenger would still require Phase 10 independent confirmation and a separate
release. No boundary or alive/dead model is added within this phase.

## Execution repair, 17 November

The first parallel scheduler allowed workers to rewrite the shared protocol record,
causing a temporary-file replacement race. Two folds finished and two remained
incomplete. The parent now performs the protocol write; workers only verify the
record and frozen hashes. The old scheduler source/hash and execution record are
retained locally, alongside the repaired scheduler's separate execution record.
Completed predictions are reused and their official scores are replayed by the
canonical runner. A synthetic regression test checks read-only worker verification.
Model code, experiment settings, fit/calibration partitions and adoption gates were
not changed. This is a documented execution repair, not a restarted parameter search.
