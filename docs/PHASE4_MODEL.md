# Phase 4: control evidence and adaptive turnover boundaries

The selected model improves the frozen Phase 3 baseline using geometry-derived
control features, capped short-spell weighting and a learned turnover-boundary head.
Variants were selected by four leave-one-training-match-out folds before one
comparison on the existing development match. Both final holdouts remain sealed.

## External semantic audit

The audit reads only the four external training games. It validates the 25 Hz
tracking UTC timebase, maps event times through that timebase and counts same-team
event overlap. No possession labels are modified. Control-supporting categories are
intentional plays, claims/interceptions and ground duels explicitly marked as a
possession change. Aerial duels and generic touches are counted separately. A shot
alone is not counted as decisive evidence of controlled acquisition.

Of 600 training spells under two seconds, 63 have a control-supporting event inside
and 104 have one within ±12 frames. For spells lasting at least five seconds, 626
of 685 have such an event inside. Timestamp uncertainty and incomplete event logging
prevent treating absence as evidence that control did not occur. This audit supports
conservative weighting and preserving the provider benchmark, not removing short labels.

An additional descriptive audit after CV found 35 spells shorter than 0.2 seconds,
214 lasting 0.2–1 seconds and 351 lasting 1–2 seconds. Only 14 of the 600 short spells
have no selected frame under the stride-five training sampler. Sampling resolution
alone therefore cannot account for the much larger short-spell failure rate. This
posthoc audit was not used to choose the Phase 4 model.

## Model and validation protocol

`configs/phase4.json` fixes the candidate family, seeds, scenarios and settings before
evaluation. Three candidates use identical match splits and augmented data:

* **frame:** Phase 3 feature/head architecture, trained with the added structured noise.
* **balanced:** the same features with partially inverse-duration ownership weights.
* **control_adaptive:** weighted ownership with additional control-motion evidence and
  a learned team-invariant turnover-boundary score.

Weights are proportional to `sqrt(5 / max(duration_seconds, 0.2))`, scaled to unit
mean over live frames within each half and clipped to [0.25, 4]. Normalization uses
bounded rescaling so the cap remains enforced. Alive/dead training is unweighted.
The weights reduce length bias without assigning equal total weight to every spell.

Additional ownership features use nearest-player/ball relative velocity, a proximity
and velocity compatibility score, ball-speed changes and short-term proximity trends.
There are 21 ownership features, retaining the original 15 as a prefix. Team exchanges
negate ownership features; boundary inputs are invariant. No ball-status label,
provider ownership flag, event annotation, match identity or frame index enters
prediction features.

The boundary head sees 84 invariant features and predicts a score near live H/A
changes, with a ±5-frame supervision window and positive-class weight 20. Its score
is not a calibrated transition probability. It lowers only H/A switch cost from 5
toward a minimum of 1.25. Dead-ball transition cost stays 40. No minimum duration is
imposed, and all processing resets between halves.

The four-fold comparison holds out every scene and both halves of one whole training
match. Binning and all fitted classifiers use only the other three matches. A
training-loader guard rejects development, target and final-holdout IDs. Deterministic
feature caches contain geometry features separately from labels and supervision.
Fold models share only an alive/dead head fitted on that fold's training data.

Primary selection uses mean spellF1 over clean/moderate/severe conditions. A candidate
must also avoid a structured-noise regression larger than 0.02 against the fold-trained
frame baseline. Exact primary-score ties use macroF1. The selected variant is then
refit on all four training games and compared once on J03WPY. No alternate Phase 4
variant is chosen from development results.

Structured corruption adds crowd/boundary ball occlusion, temporary partial camera
coverage, duplicate track fragments and false ball detections to the previous severe
scenario. It uses geometry only and never changes labels. All variants see the same
seeded corruptions. These simulations are stress tests, not an estimate of the
competition feed's error distribution.

## Results

Four-fold training-match comparison:

| Variant | Mean spellF1, original scenes | Mean macroF1 | Structured spellF1 |
|---|---:|---:|---:|
| frame | 0.4710 | 0.8724 | 0.3786 |
| balanced | 0.4768 | 0.8727 | 0.3875 |
| control_adaptive | 0.5153 | 0.8825 | 0.3979 |

The control/adaptive variant improved the primary score in all four folds. Capped
weighting alone had a much smaller effect. Control features and adaptive transitions
were tested as a bundle, so their individual contributions are not isolated.

Separate development comparison:

| Inputs | Frozen Phase 3 spellF1 | Phase 4 spellF1 |
|---|---:|---:|
| clean | 0.5260 | 0.6024 |
| moderate | 0.5139 | 0.5600 |
| severe | 0.4378 | 0.4495 |
| structured | 0.4283 | 0.4498 |

Mean development spellF1 over the original three scenes rises from **0.4926 to
0.5373**. The recommended pipeline is Phase 4. These are model-selection/validation
results on provider labels, not independent test or competition results.

Clean short-spell recall at IoU 0.5 improves from 7.3% to 16.9%; severe-corruption
short-spell recall improves from 0% to 5.6%. This remains a major limitation.
Under structured noise, long-spell recall falls from 91.9% to 85.5%, despite better
overall spellF1. That tradeoff warrants reliability-aware boundary decisions in the
next phase. The current evidence does not establish which component caused it.

All 56 synthetic tests passed. They cover capped weights, label/feature separation,
structured corruption, symmetry, exact dynamic-decoder optimality, frozen-decoder
compatibility and checkpoint serialization. The frozen Phase 3 development scores
replayed exactly. Two independently loaded Phase 4 checkpoints produced identical
probabilities, boundary scores and label bytes on every development condition, and
replayed all recorded metrics. Aggregate results are in [phase4_results.json](phase4_results.json).
Detailed caches, checkpoints and intermediate arrays remain local.

## Reproduce

```sh
uv sync --frozen
LOKY_MAX_CPU_COUNT=2 uv run --frozen python scripts/audit_external_spells.py
LOKY_MAX_CPU_COUNT=2 uv run --frozen python scripts/audit_sampling_resolution.py
LOKY_MAX_CPU_COUNT=2 uv run --frozen python scripts/run_phase4.py
LOKY_MAX_CPU_COUNT=2 uv run --frozen python scripts/audit_phase4_checkpoint.py
```

Phase 3 data preparation and its checkpoint are prerequisites. Model selection and
full provenance remain in `outputs/phase4/report.json`; the selected checkpoint is
`outputs/phase4/model.pkl`. Published settings are in `configs/phase4_selected.json`.
Source and cache hashes allow deterministic reuse without a new download. A file
lock prevents overlapping Phase 4 runs.

## Decisions for Phase 5

Keep Phase 4 as the new comparator and preserve Phase 3 as a fallback. Test geometry
reliability gating before increasing model size: suspect false detections should not
create strong turnover evidence. An optional player-motion temporal model should
be compared using the same match-separated protocol and retained only if it adds
robust value. Public pretrained weights must have training-match provenance that
excludes our reserved holdouts; otherwise retrain within the frozen split.

Duration priors and tag fusion remain deferred. Additional persistence needs a clear
short-spell guard, and the external catalogue has no matching Metrica automatic-tag
source. Event-derived or label-derived phase proxies cannot establish the reliability
of the supplied automatic tags. The original provider benchmark remains unchanged.
Final holdout evaluation and competition inference belong to later phases.
