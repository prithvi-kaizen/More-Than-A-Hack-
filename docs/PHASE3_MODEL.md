# First trained possession model

Phase 3 fits separate alive/dead and live-team emission models on the frozen four
IDSSE training matches. HistGradientBoostingClassifier (scikit-learn 1.7.2) provides
nonlinear tabular predictions and supports missing features. Parameters are fixed
in source/config before development evaluation: 120 iterations, 15 leaves,
80 samples per leaf, L2 regularization 1, no internal random early-stopping split.
No competition data, public labels or target video are used in fitting or selection.

## Inputs and features

The feature API accepts only player x/y, ball x/y, H/A membership, pitch dimensions
and FPS, separately per half. The 42 alive/dead features include distance to pitch
boundaries, ball visibility, time since observation, nearest-player distance,
collective movement, spread and temporal context. The 15 ownership features include
relative team proximity, crowding, movement/spread differences, previous reliable
geometric ownership evidence and centred evidence over 0.4–4 seconds.

Player speeds above 12 m/s and ball speeds above 60 m/s are masked as unreliable
motion features; positions are preserved. Those broad fixed plausibility limits
are not calibrated to the target. Motion estimates use observed continuous intervals
and smoothing rather than trusting unverified provider speed units. Off-pitch ball
positions remain available as stoppage evidence. No raw match time, frame number,
person ID, provider alive flag or provider ownership label enters model features.

Team-order symmetry is explicit. Alive features are invariant to exchanging H/A.
Ownership features change sign, training includes both signs with opposite labels,
and probability symmetrization enforces opposite ownership under a team exchange.
Permuting track columns consistently does not change features. Temporal identity
switches still matter and are tested through input corruption.

Future context is allowed because the submission is offline. All context is bounded
inside a half; no feature state carries between halves. Missing ball observations
are a masked observation condition, not a dead-ball label.

## Training and selection

Training extracts features at the original 25 Hz and samples every fifth frame.
The clean, moderate and severe corruption scenarios have equal exposure. Corruption
acts only on inputs and uses local seeded generators. Both class heads mask unknown
supervision; team fitting uses only known live H/A frames. All training data come
from registered training matches, with checked raw/prepared provenance.

Joint output probabilities are `P(H)=P(live)P(H|live)`,
`P(A)=P(live)(1-P(H|live))`, `P(D)=1-P(live)`.
They are not claimed to be calibrated probabilities under the competition domain.

Viterbi maximizes summed log emissions minus transition penalties over H/A/D.
H/A turnover and dead-ball transition penalties are selected from the ten pairs
declared in `configs/phase3.json`. The objective is mean exact official spellF1 on
the development match across clean/moderate/severe inputs. The zero-penalty candidate
is the framewise argmax ablation. No duration terms, tag fusion or hard minimum
possession length are added in this phase. Penalty tuning is not an unbiased test
result; the two final holdout matches remain unused.

For comparison, the same development inputs also run the untrained nearest-player
baseline and the official smoothing implementation with its published ±50-frame
window. Exact primary-score ties use mean development macroF1. The report retains
every candidate, per-scenario spellF1/macroF1 and spell
counts, training provenance, feature/source hashes and model checksum. It tests
checkpoint roundtrip equality. Scores estimate recovery of the IDSSE convention,
not competition performance.

## Run locally

```sh
uv sync --frozen
uv run --frozen python scripts/acquire_phase3.py
LOKY_MAX_CPU_COUNT=2 uv run --frozen python scripts/train_phase3.py
LOKY_MAX_CPU_COUNT=2 uv run --frozen python scripts/audit_phase3.py
```

Acquisition downloads training/development files only, prepares each match, and
compresses verified tracking XML to reduce peak disk use. Gzip reads verify original
uncompressed size/checksum. The model data loader rejects holdout/target access and
unexpected fields in prepared inputs. The training command checks all required data
before fitting. Optional detailed stdout/stderr can be redirected into ignored
`local_reports/`. Detailed training artifacts remain in ignored `outputs/phase3/`.

Use only checkpoints produced locally by this command: pickle loading executes code
and is not a generic external checkpoint loader. Reproduction depends on the locked
environment, frozen data split, seeds and source versions. No competition prediction
is produced by Phase 3. Target inference and final submission packaging are later phases.

## Measured development results

Trained on 340,374 sampled external frames across four training matches and three
input scenarios. Evaluation uses the separate J03WPY development match; its scores
were used for decoder selection and are not an unbiased held-out test estimate.

| Method | Clean spellF1 | Moderate | Severe | Mean |
|---|---:|---:|---:|---:|
| Nearest player | 0.0895 | 0.0412 | 0.0087 | 0.0465 |
| Official smooth50 | 0.3171 | 0.2707 | 0.1260 | 0.2379 |
| Boosted argmax | 0.2680 | 0.1386 | 0.0476 | 0.1514 |
| Boosted Viterbi | 0.5260 | 0.5139 | 0.4378 | 0.4926 |

Selected turnover penalty: **5**; dead-ball transition penalty: **40**. Mean macroF1
is **0.8887**. The boosted/Viterbi pipeline is the recommended Phase 3 baseline.
The checkpoint and all development predictions replayed with identical metrics.
Aggregate results and all candidate scores are in [phase3_results.json](phase3_results.json).
The selected settings are in `configs/phase3_selected.json`; the trained binary
remains local in `outputs/phase3/model.pkl` and is regenerated by the training command.

Alive/dead F1 is 0.9968 clean, 0.9867 moderate and 0.9542 severe. Short ownership spells
are the main weakness: at IoU 0.5, recall below 2 seconds is 7.3%, 7.3% and 0.0%,
compared with 95.2%, 93.5% and 93.5% for spells lasting at least 15 seconds. There are
421 reference possessions, but the selected decoder predicts 274/278/266. Counts are
diagnostics, not a target to force. Clean transition F1 rises from 0.5282 at ±12
frames to 0.6208 at ±25, indicating boundary timing remains relevant as well.

Phase 4 should prioritize short-spell/control semantics, segment-balanced ownership
training and data-dependent transition evidence. Additional fixed smoothing or a
hard minimum duration would work against the observed weakness. Keep this model as
a frozen comparison; use training-match cross-validation for further architecture
choices to reduce repeated optimization on the single development game. Preserve
original provider-label benchmarks even if an additional control-aligned proxy is
introduced; do not redefine labels to improve the reported score. Inspect external
event evidence before assuming every brief provider ownership spell meets the
competition's controlled-possession definition.

Both final holdouts and all competition outputs remain unevaluated. These results
cannot establish a competition rank or predict its hidden score. Four training and
one development tracking files are verified/prepared; the two holdout tracking
files remain intentionally deferred. Raw tracking occupies about 257 MiB compressed,
with about 81 MiB of prepared arrays at completion.

The native Hugging Face CLI 0.35.3 completed the final two transfers when repeated
HTTP range requests stalled. Both routes were checked against the same original
publisher checksums before preparation. This optional acquisition route does not
change training inputs or model dependencies; the scripted curl route remains
available. Acquisition and training now have single-process locks.
