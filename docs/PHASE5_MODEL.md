# Phase 5: reliability and player-motion experiments

## Protocol

This phase tests whether unreliable ball observations should weaken turnover evidence,
and whether a small player-only temporal expert supplies useful ownership information.
The comparator is the selected Phase 4 control/adaptive-boundary model. The same
four external training matches, development match and corruption seeds are retained.
The two final holdouts and competition game are excluded from these experiments.

The settings in `configs/phase5.json` are fixed before the comparison. Four policies
are evaluated in leave-one-training-match-out folds. Both halves and every corruption
of the held-out match stay outside fitting. Each fold refits the Phase 4 tree model
and trains a new temporal model on the other three matches. The frozen Phase 4 fold
scores must replay before selection proceeds.

| Policy | Ownership emissions | Adaptive boundary evidence |
|---|---|---|
| `phase4` | Original tree probabilities | Original boundary score |
| `reliable_boundary` | Original tree probabilities | Boundary score times geometry reliability |
| `reliable_tempered` | Conditional ownership logits softened when reliability is low | Reliability-weighted score |
| `motion_fusion` | Softened tree ownership blended with player-motion ownership | Reliability-weighted score |

Geometry reliability uses only ball observations: missing coordinates give zero;
otherwise a centered 0.08-second trajectory residual and an extreme-speed penalty
reduce confidence in isolated detections. The fixed residual scale is 3 metres and
the speed penalty begins above 45 metres/second with a 15-metre/second scale.
An observed frame without enough neighboring observations receives reliability 0.5.
These are fixed physical priors, not calibrated probabilities. Real changes of ball
direction and legitimate fast motion can also reduce this score.

With reliability `q`, tempering multiplies the conditional ownership logit by
`0.5 + 0.5*q`. Motion fusion uses weight `0.35*(1-q)`. All policies preserve the
tree model's dead-ball probability before decoding, although the joint decoder can
still change the final alive/dead sequence. The decoder penalties remain 5 and 40,
and the maximum boundary reduction remains 0.75. All processing resets at halftime.

A new policy must improve mean clean/moderate/severe CV spellF1 by at least 0.005,
win at least three of four folds, and lose no more than 0.01 in either structured-noise
spellF1 or pooled long-spell recall at IoU 0.5. Long spells are at least 15 seconds.
Eligible policies are ordered by primary spellF1, then macroF1. Only the selected
policy is compared with Phase 4 on development; rejected policies are not compared
there. Development must improve primary spellF1 and satisfy the same structured
guards. If no policy qualifies, Phase 4 is retained.

## Player-motion model

The temporal expert receives no ball coordinates. For each team, player positions,
velocities, speed statistics and a 4-by-3 occupancy grid produce 27 values plus
27 missingness indicators. Team differences and sums give 108 inputs. Pooling
ignores player-slot ordering. Physical pitch dimensions and fixed speed scales
normalize the features; no scaler is fitted across evaluation matches.

A 19,169-parameter, 32-channel temporal convolutional network has three residual kernel-five layers
with dilations 1, 2 and 4. Layer normalization operates across channels independently
at each time, so inference chunks do not change normalization statistics. Subtracting
the score for exchanged teams enforces exact ownership antisymmetry. No pretrained
weights are used. PyTorch is an optional dependency, with a CPU wheel index on Linux.

Features are sampled every fifth frame, giving 5 Hz. The receptive field is 29 samples,
covering 5.6 seconds between its endpoints. Training uses 64-sample windows with
stride 32, masks internal window edges by the 14-sample context radius, and never
joins halves. Unknown/dead ownership is masked out of binary loss. Capped Phase 4
spell weights remain unchanged. Training runs eight fixed AdamW epochs with batch
size 64, learning rate 0.001, weight decay 0.001 and gradient clipping at 1. There
is no selection of epochs on validation data.

Inference uses overlapping halos around 512-sample chunks and interpolates logits
back to 25 Hz within each half. Interpolation does not add temporal information.
Synthetic checks compare chunked inference against full-sequence inference, check
team symmetry, and verify deterministic optimization and checkpoint reloads.

The structured camera simulation is ball-conditioned, so simulated player visibility
can indirectly depend on ball position. The expert's feature API itself has no ball
input. Simulated robustness does not establish performance on real camera outages.

The decoder penalties were selected during Phase 3 using the same development match.
That match has also been checked after later phases, so its results are not an
independent generalization estimate. The four-fold comparison isolates each new
fitted model from its evaluation match, but has only four matches from one provider.
Provider ownership labels can differ from the competition's control definition.
The sealed final holdouts are needed before drawing stronger performance conclusions.

## Four-fold results

| Policy | Mean spellF1, original scenes | Structured spellF1 | Structured long-spell recall |
|---|---:|---:|---:|
| Phase 4 | 0.5153 | 0.3979 | 79.2% |
| Reliable boundary | 0.5167 | 0.3994 | 79.2% |
| Reliable tempered | 0.5157 | 0.3969 | 78.1% |
| Motion fusion | **0.5436** | **0.4694** | **83.3%** |

Motion fusion improves the primary score in all four folds and passes every declared
guard. Mean spellF1 improves by 0.0283, while structured-noise spellF1 improves by
0.0714. Reliability changes alone do not meet the minimum gain. The temporal fusion
bundle is selected before inspecting its development performance. These comparisons
do not isolate the contribution of each part of that bundle.

## Development result and decision

| Inputs | Frozen Phase 4 spellF1 | Selected motion fusion spellF1 |
|---|---:|---:|
| Clean | 0.6024 | 0.6058 |
| Moderate | 0.5600 | 0.5704 |
| Severe | 0.4495 | 0.4982 |
| Structured | 0.4498 | 0.5111 |

Mean spellF1 over the original three scenarios rises from **0.5373 to 0.5581**.
Structured long-spell recall stays at 85.5%, satisfying the guard. Clean short-spell
recall remains 16.9%; severe short-spell recall rises from 5.6% to 8.1%, and structured
short-spell recall rises from 8.1% to 10.5%. Brief possessions remain the main weakness.
The modest drop in moderate-scene alive/dead F1 (0.9853 to 0.9849) also illustrates
that unchanged dead-ball emissions do not guarantee identical decoded stoppages.

The selected system is Phase 4 plus motion fusion. The baseline is retained as a
recorded comparator. Aggregate results are in [phase5_results.json](phase5_results.json)
and the selected code/checkpoint manifest is in
[`configs/phase5_selected.json`](../configs/phase5_selected.json). No final-holdout
evaluation, target inference or target-video comparison occurred.

All 67 synthetic tests passed. Reloaded checkpoints reproduced every saved out-of-fold
tree probability, boundary score and neural ownership probability exactly. The audit
also reproduced decoded labels and all metrics for all four policies across every
fold/scene, then replayed the development comparison. Checkpoint training-match
metadata, prepared/cache provenance and source/dependency hashes were verified.

## Reproduction

Phase 4 prepared external data, feature caches and checkpoint are prerequisites.
All array operations, fitting and checkpoint evaluation run locally.

```sh
uv sync --frozen --extra temporal
LOKY_MAX_CPU_COUNT=2 uv run --frozen --extra temporal python -m unittest discover -s tests -v
LOKY_MAX_CPU_COUNT=2 uv run --frozen --extra temporal python scripts/run_phase5.py
LOKY_MAX_CPU_COUNT=2 uv run --frozen --extra temporal python scripts/audit_phase5_checkpoint.py
```

Detailed caches, out-of-fold probabilities and checkpoints stay in ignored
`outputs/phase5/`. Cache records bind inputs to verified prepared sources and the
feature/scenario code. The report records code, checkpoint and dependency-lock
hashes. A process lock prevents overlapping experiment runs.
