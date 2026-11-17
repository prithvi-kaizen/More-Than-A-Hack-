# More Than a Hack 2027: research and implementation plan

Originally prepared 14 November 2026; status updated 17 November 2026. Phases 0–9
are complete and the Kaizen submission is packaged and locally reproduced. The
original design sections below are historical proposals; the phase outcome sections
record what was actually built. No competition accuracy or rank has been measured.

The current [submission review and improvement plan](../docs/SUBMISSION_REVIEW.md)
supersedes earlier prospective next-step statements. It prioritizes eligibility and
delivery hardening, then a bounded external-only reliability experiment, then an
independently checked release decision. Phase 9 rejected its challenger under the
unchanged adoption gates; retain v1. Eligibility still requires an organizer response.
Phase 10 delivery closure has not started and requires separate user authorization.

## Recommendation

Build an offline, uncertainty-aware possession segmenter with three complementary sources of evidence: local ball control, collective player movement, and automatic tactical tags. Decode a coherent sequence with explicit possession and stoppage durations. Train and choose all parameters on other matches, with realistic tracking corruption. Keep a simpler tracking-only model as the submission fallback.

The central hypothesis is that reliable changes of state matter more than precise reconstruction of every missing ball position. A player touching the ball, a team controlling it, and a ball being alive are different facts. Model those differences explicitly.

No credible probability of a top-20 finish can be calculated from the available information. There is no live leaderboard, and beating the published baseline does not establish a qualifying score. This plan prioritizes improvements that directly address the metric and documented failure modes.

## 1. Review scope and constraints

Reviewed the public challenge, data, evaluation, rules, submission, FAQ and overview documents; the metric implementation, baseline, loaders, validator, evaluator and starter notebook content; and relevant primary research and external data documentation. Public reference files are saved in `research/reference/` with their upstream MIT licence. Repository tree observed: `f8f2629ff1e0985f81d7e59d8413678514ee1530`; files were fetched from main during this review, so pin a commit explicitly when implementation begins.

The supplied Drive downloads are now available locally. They have been consolidated by SHA-256 into `data/tactical/` and `data/broadcast/`. The retained bundle contains tactical tracking metadata, raw tracking, homography and Smart Tagging, plus broadcast tracking metadata, raw tracking and homography. No tactical or broadcast MP4 was found among the supplied files. Duplicate loose copies and ZIP containers were deleted only after their retained contents were hash-verified. One copy of the supplied README is retained locally at `data/BUNDLE_README.md`; the project root now has implementation instructions. No additional terms document was identified. Detailed inventories and audit measurements stay in ignored `local_reports/`; they are not pasted into the chat. This review does not include visual inspection or prediction-versus-video checking. Numerical data characteristics elsewhere in this plan remain attributed to public organizer documentation, unless explicitly identified as a local structural check.

The [FAQ](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/FAQ.md) prohibits disclosing restricted match materials or extracts to hosted LLMs. Codex will author code and launch local processes; those processes read the local files directly. Restricted inputs, arrays, plots, predictions and detailed diagnostics must not appear in tool output or model context. Local execution alone does not provide that separation if a command prints the data. The execution contract in Section 7 defines the boundary. The signed terms still need to be checked by the team.

Your supplied rules are stricter than parts of the repository: use external data only for training and parameter tuning. Follow that stricter interpretation. No target-video fine-tuning, target pseudo-label training, target self-supervised adaptation, target-derived threshold calibration, output-versus-video review, manually corrected tracks, or match-specific event feeds. Automatic inference with frozen models and automatic schema/invariant checks remain in scope. Do not use public-window scores to select architectures, thresholds, ensembles or checkpoints.

The starter notebook contains suggestions to inspect errors against video and sweep parameters on the public window. Those suggestions are not part of this plan because they conflict with your supplied restrictions and the repository's no-tuning rule.

## 2. What the competition actually rewards

The deliverable is a tactical-video-indexed CSV with 199,319 rows and columns `frame,label`. Frames run from 0 through 199318. Labels are H, A, D and X. Submit the CSV, a reproducible public repository and a short technical document. The published deadline is 17 November 2026; its exact cutoff time and timezone are not stated in the reviewed submission page. Confirm them from the registration communication and aim to be ready on November 16. [Submission contract](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md)

The main score is mean possession-spell F1 at temporal IoU thresholds 0.3, 0.4, 0.5, 0.6 and 0.7. Only maximal H/A runs count as possessions; D separates runs even when the same team restarts. Matching is same-team, greedy in descending IoU, one-to-one—not Hungarian matching and not average precision. The score at each threshold is `2 TP / (N_pred + N_true)`. Macro-F1 is the first tie-break; the evaluator then uses loose-tolerance transition F1, absolute latency and filename. Use the original scorer, not an approximate reimplementation. [Metric code](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/evaluation/possession_metrics.py), [ranking code](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/evaluation/run_evaluation.py)

This suggests three objectives: avoid spurious splits; preserve genuine short possessions; locate transitions accurately. A large smoothing window only addresses the first and can damage the other two.

Analytical illustration, not a benchmark: a true 20-second H possession split by a one-second false D into two 9.5-second predictions gives each prediction IoU 0.475. At thresholds 0.3 and 0.4 only one can match, giving F1 2/3; at the other three thresholds neither matches. Mean spellF1 for this isolated example is 0.267 despite 95% frame accuracy. Conversely, a two-second spell shifted by 0.4 seconds has IoU 1.6/2.4 = 0.667 and misses the strictest threshold. Both fragmentation and timing matter.

X is not a useful new latent football state. For temporal scoring the implementation forward-fills X inside each graded run, back-filling at its head; macro-F1 penalizes X. Explicitly emitting the same filled labels preserves the temporal result and cannot worsen macro-F1. Prefer resolved H/A/D predictions inside graded halves and X outside. Retain uncertainty in a separate local file. [Abstention implementation](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/evaluation/possession_metrics.py)

The published full-match baseline to beat is 0.2433 spellF1, but it is only a reference floor. Public-window scores reverse the ranking of two smoothing variants; that is evidence against using those five minutes for model selection. [Evaluation guidance](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md)

## 3. Data engineering decisions

| Asset | Intended use | Critical handling |
|---|---|---|
| Tactical ATD | Main geometric and motion evidence | Missing values remain missing; 46 slots are track fragments, not 46 people |
| Tactical metadata | Team membership and schema | Verify H/A mapping from metadata; do not infer teams from attacking direction |
| Smart Tagging | Soft team and restart evidence | Defensive tags imply the opponent has the ball; N/A means no team evidence |
| Tactical homography | Optional frozen visual inference | Maps normalized pitch coordinates to pixels, not metres to pixels |
| Tactical video | Not supplied; deferred optional local inference | No training, tuning or prediction review against it |
| Broadcast assets | Tracking and homography available; video not supplied | Deferred: separate timeline; no constant-offset assumption |

The organizer reports 41% ball missingness on graded frames. Missingness is structured around occlusion, aerial play and crowds; random point deletion alone is an inadequate training augmentation. Tactical tags are 638 phase intervals rather than frame-exact possession labels. All throw-ins have N/A team tags. Broadcast homography is usable for only 65.8% of frames. These are reasons to make the tactical pipeline self-sufficient. [Data documentation](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md)

Alignment contract:

* Delivered ATD: `video_frame = frame_id - 1`; verify exactly one occurrence of every ID from 1 through 199319.
* Smart tags: `round(seconds * 25)`, with half-open intervals in the loader.
* Homography: already zero-based video frames.
* Half windows: inclusive 11185–90651 and 110808–190197. Reset temporal inference between them.
* The large offsets in `games.py` concern the separate counter used to derive the truth. They must not be applied to delivered ATD. This is explicitly clarified in [load_atd.py](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/starter/load_atd.py), rather than an unresolved requirement to estimate offsets.
* The second graded interval starts before play restarts. Do not initialize it as alive merely because grading has begun. Infer restart from allowed inputs; keep published kickoff landmarks for pipeline checks, not extra hand-coded possession labels.

Convert normalized tracking coordinates to metres with declared pitch dimensions before computing distances. Preserve the normalized representation for homography. Use physical time for temporal windows so external 5/10/25 Hz data are comparable. Upsampling does not create timing information.

Quality checks must cover duplicate/nonmonotonic frames, slot count, team assignment, finite values, missingness masks, malformed tags, unit consistency and interval clipping. Treat out-of-pitch ball positions as potential evidence, not automatically invalid coordinates. Compute derivatives only within sufficiently observed, continuous tracklets; identity swaps otherwise manufacture enormous accelerations.

## 4. External data and literature

| Resource | Role | Qualification |
|---|---|---|
| [IDSSE / Sportec official repository](https://github.com/spoho-datascience/idsse-data) | First choice for external training and validation | Seven matches with synchronized tracking/events; explicitly CC BY 4.0, requiring DFL credit and paper citation |
| [SkillCorner Open Data](https://github.com/SkillCorner/opendata) | Cross-provider stress test and possible additional training | Repository carries MIT licence; preserve its notice and verify any file-specific terms. Provides possession fields, detection flags and phases; label conventions differ |
| [Metrica sample data](https://github.com/metrica-sports/sample-data) | Optional same-provider comparison | Public availability and acknowledgement guidance are not a substitute for checking reuse rights; do not make it a critical dependency until resolved |

External possession fields are not automatically interchangeable with the competition labels. Build an adapter with documented semantics: dead while not in play; live ownership persists through passes and loose balls until controlled opposition possession; no dead-ball ownership. Use independent ball-status fields where available. Do not label every gap between event records as D. Where labels remain uncertain, mask their supervision or audit/annotate only external matches. Automatic labels should be identified as such, not called human ground truth.

Important research connections:

* [Individual ball possession in soccer (2017)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0179953) distinguishes contact, control and team-level possession. It supports separating these concepts; its reported performance and access to a running flag do not establish performance here.
* [Automatic event detection in football using tracking data (2022)](https://link.springer.com/article/10.1007/s12283-022-00381-6) uses local ball/player geometry and trajectory changes to detect control. Borrow these features, while remembering that contact alone does not prove controlled opposition possession.
* [Using Temporal Convolutional Networks to estimate ball possession in soccer games (2023)](https://www.sciencedirect.com/science/article/pii/S0957417423002816) motivates a compact temporal neural candidate. Only the abstract was accessible in this review; do not claim to reproduce its architecture or numerical results.
* [Ball Radar (KDD 2023)](https://github.com/hyunsungkim-ds/ballradar) combines set-based representations and temporal inference for possession and ball trajectories. Useful principle: player movement contains information when the ball disappears. Avoid turning a single imputed trajectory into certain evidence.
* [TranSPORTmer (ACCV 2024)](https://arxiv.org/abs/2410.17785) jointly treats trajectory inference and states including uncontrolled/out-of-play. Its state taxonomy needs a competition-specific mapping. It is a research reference, not a verified ready-to-deploy dependency here.
* [PathCRF (KDD 2026)](https://arxiv.org/abs/2602.12080), with [official code/checkpoints](https://github.com/hyunsungkim-ds/pathcrf), uses player-only structured possession paths. Crucially, its experiment uses known in-play episodes and downsampling to 5 Hz. It cannot simply supply a full-match dead-ball detector. Code is MPL-2.0; check checkpoint scope and provenance separately. Its event metrics are not this challenge's spellF1.

Before using pretrained checkpoints, record their training matches and split. Never report performance on a checkpoint's training matches as held-out validation. Use its documented held-out game, a genuinely separate provider, or retrain within our split. Exclude any dataset containing the competition match, regardless of vendor.

## 5. Proposed model, in implementation order

### 5.1 First runnable system: geometry, control confirmation, stoppages

Implement a CPU-friendly emission model and a three-state temporal decoder before training a large model. Logistic regression or gradient-boosted trees over temporal features are sensible candidates. Fit on external data only. This yields a usable fallback and isolates whether a neural model adds anything.

For each team and frame compute nearest and second-nearest player-ball distances, opponent distance margin, ball speed, local direction change, player-ball relative velocity, short-window proximity persistence, and observation quality. Add team-level speed quantiles, movement coherence, player spatial spread and boundary proximity. Maintain masks for every derived feature. A stationary player near a stationary ball is not proof of live possession.

A control proposal should combine proximity with motion compatibility and evidence of an intentional next action or continued team control. Distinguish a ball passing close to a player from a genuine reception. Do not impose a universal two-second dwell time: it would erase controlled one-touch play. Confirmation may use future frames because submission is offline; locate the boundary retrospectively at the inferred acquisition time rather than at confirmation time.

For loose balls retain last controlled team as a latent ownership state. Do not transfer ownership just because an opponent is temporarily closest. Future reception provides evidence about the event, but do not assign the receiver's team to the entire preceding flight of an intercepted pass.

### 5.2 Explicit alive/dead inference

Predict dead-ball probability separately from team ownership, then combine them in the temporal decoder. Signals include reliable pitch-boundary crossing, static ball plus collective slowdown, restart formations, motion resumption, and set-piece tags. Fouls and offside can stop play while the ball remains inside the pitch; a boundary-only detector is incomplete. A goalkeeper holding the ball or slow buildup can still be live; a slow-motion-only detector is incomplete.

Set-piece intervals nominate a search region for a restart. They do not mean the entire interval is D, and their start/end is not automatically the exact restart frame. Within the region use frozen change-point logic over ball and player movement. If the ball is missing, reduce confidence and use the other evidence. Do not estimate a systematic tag time offset from target-video observations.

This component is first priority because false D fragments live possessions, while missing D merges possessions across same-team restarts.

### 5.3 Tactical tags as a fallible sensor

Implement an explicit semantic map. Build Up, Progression, Creating Chances and Attacking Transition support the named team's ownership. Defensive phases and Defensive Transition support the opposing team. Set Pieces support a possible restart and, when known, the restarting team. N/A supplies no team vote. Goal tags support a stoppage hypothesis, not a fixed ownership interval.

Use capped, grouped evidence rather than multiplying every active tag as an independent likelihood. Attacking and defensive tags may describe the same underlying phase, and tags may share upstream errors with tracking. Contradictory tags should weaken the prior. Taper interval boundaries using parameters chosen externally; do not treat coarse phase boundaries as exact possession switches.

There may be no external dataset with the identical Smart Tagging generator. Use external analogous phases and explicitly corrupted synthetic tag intervals for stress testing, but call that a proxy calibration. If the tag component helps only when given nearly perfect external tags, omit it or keep its fixed weight small. Do not learn target tag reliability by matching target predictions or public truth.

### 5.4 Structured temporal decoding

Start with H/A/D and data-dependent transition penalties. Expand only if external experiments support it, for example to `H_control`, `H_loose`, `A_control`, `A_loose`, and `D`; both H substates map to H. Loose substates retain the last owner. Do not add an output contested class.

A semi-Markov score for segmentation S is:

`score(S) = sum_segments [sum_t emission(t, state) + duration_prior(state, length)] + sum_boundaries transition(previous, next, boundary_features)`.

Fit the emissions, duration distributions and transition terms on external matches. A conditional scoring model need not pretend highly correlated frame observations are independent. Duration priors should be soft and broad: short possessions and long stoppages are legitimate. Avoid a hard minimum duration that makes brief real possessions impossible.

Viterbi recurrence for a segment ending at t with state k:

`V[t,k] = max_(s<t,j!=k) { V[s,j] + segment_score(s,t,k) + transition_score(j,k,s) }`.

Use prefix sums for segment emissions. For tractability, start with a linear-chain decoder; introduce semi-Markov durations on a 5 Hz proposal grid or sparse boundary candidates, then refine candidate boundaries at 25 Hz. Preserve the ability to discover short spells using dense local proposals; do not quantize the final CSV to seconds. Include periodic fallback candidates so missed visual events cannot make a transition impossible. Reset at halftime. Test the optimized decoder against exhaustive enumeration on tiny synthetic sequences.

State confidence should drop when observations become unreliable; missingness must not be interpreted as direct evidence for D. Marginalize or neutralize missing ball-derived features. Long gaps should widen uncertainty rather than generate a confident straight-line ball trajectory.

### 5.5 Optional neural branch: player-motion evidence

Build a small set encoder per frame, team pooling, and a bidirectional temporal convolutional network. Inputs: positions, valid velocities, team membership, missingness masks and short-track quality. Outputs: ownership, alive/dead and boundary probabilities. Optional auxiliary tasks can predict control or masked ball location, using external data only.

The set representation protects against arbitrary slot order. Temporal per-player processing still needs continuity; track permutations are not solved merely by attention. Prefer team-pooled temporal features for the first version. If adopting PathCRF, use short automatic tracklet association and validate under slot fragmentation. Treat its output as one live-ownership expert; supply our own dead-ball model and test without oracle in-play boundaries.

Do not make new ball detection, full player re-identification, broadcast synchronization or a large video-language model a dependency of the first submission.

### 5.6 Fusion and the research extension

Fuse component probabilities before one final decoder. Fit weights and reliability gating on held-out external predictions, including missing-data and tag-corruption conditions. Do not majority-vote hard frame labels from several complete segmenters; mismatched boundaries can create new tiny spells.

Optional extension: generate a small set of plausible whole segmentations and choose one maximizing expected spellF1 under posterior sequence samples. This is a minimum-Bayes-risk decision rule aligned with the competition's non-additive metric. It is an experiment, not a proven improvement. Posterior misspecification can make it worse. Compare against ordinary MAP decoding using the exact scorer on untouched external matches and retain it only if gains are robust.

The useful novelty is the combination of observation reliability, control-aware transitions, explicit stoppages and metric-aware decisions—not a claim that any constituent algorithm is new.

## 6. Training and validation protocol

1. Freeze an external match manifest before experimentation. For seven IDSSE matches, a practical starting split is four training, one development and two final holdout matches. Keep all windows and halves of each match in the same split. If reusing published weights, align the split with their actual training history or retrain.
2. Build and audit external label adapters before model comparison. Include dead intervals in the alive/dead dataset; do not silently use a research loader that discards stoppages. Separate genuinely observed ball status from inferred proxy labels.
3. Corrupt only input features, retaining clean labels: contiguous ball gaps, boundary/airborne/crowding-dependent loss, positional noise, ball outliers, duplicate players, slot swaps, player dropout and timing noise. Randomize tag omission, wrong teams, overlapping correlated tags and boundary jitter.
4. Choose broad corruption ranges using external evidence and physical reasoning. The published target missingness is a reason to include severe corruption, not a licence to optimize a missingness generator to this match. Do not adjust model parameters after viewing target outputs.
5. Train simple emissions first. Compare a compact neural model only after the external harness is trustworthy. Fit probability calibration on development matches, not on training residuals or target data.
6. Select on exact mean spellF1 across clean and corrupted external conditions, with per-match results, per-IoU precision/recall and macro-F1. Prefer improvements consistent across games and failure conditions. Use block/match-level uncertainty; adjacent frames are not independent samples.
7. Freeze code, config, checkpoint and random seeds; record hashes. Run the target automatically. Public ground truth may support a permitted one-time pipeline correctness check, not model selection. No output-versus-video review.
8. If a schema/indexing defect is discovered, document the deterministic correction and rerun. A poor public score alone does not authorize trying a different threshold. Do not keep revisiting final external holdouts during development either.

Suggested ablations, in this order:

| Experiment | Question |
|---|---|
| Nearest team with published smoothing | Can the loader and baseline reproduce expected mechanics? |
| Control features + three-state decoder | Does structured ownership reduce false turnovers? |
| Add alive/dead model | Does improved stoppage handling reduce splits and merges? |
| Add duration terms | Do soft duration priors help without deleting short spells? |
| Add tactical priors | Is the gain robust to correlated and inaccurate tags? |
| Add player-motion expert | Does it recover missing-ball regions under fragmentation? |
| Add metric-aware sequence selection | Does decision optimization improve held-out spellF1? |

Report short-spell recall, split/merge counts, onset/offset error, dead-ball precision/recall, performance by ball-gap length and stability under corruption. Possession count is a diagnostic, not a target statistic to force to an expected value.

## 7. Local implementation structure

```text
src/
  io/                 # tactical ATD, tags, external providers
  features/           # control, collective movement, quality masks
  models/             # emissions, optional set/temporal network
  decoding/           # transitions, durations, Viterbi, refinement
  validation/         # external splits, corruption, scorer adapter
scripts/
  audit_inputs.py
  prepare_external.py
  train.py
  evaluate_external.py
  predict.py
configs/
  external_split.json
  frozen_final.yaml
tests/                # index, semantics, decoder, output contract
artifacts/            # model/config hashes, permitted model files
data/                 # ignored; never public
outputs/              # ignored except expressly permitted submission
TECHNICAL_REPORT.md
```

This is a proposed layout; the modelling modules have not been implemented. `scripts/organize_local_assets.py` and `scripts/audit_local_assets.py` now implement local consolidation and structural auditing. `.gitignore` excludes competition data, ZIPs, videos, outputs, external data and local reports. Public reference files under research are not a complete runnable clone.

Define a common frame schema with time, team membership, player observations/masks, ball observation/mask, quality features and tag evidence. The training entry point should refuse the competition match identifier and path. Inference should not import a target ground-truth loader. Put all tunable values in one config with provenance, rather than hiding them in preprocessing.

Keep detailed intermediate arrays local: observation quality, three-state emissions, candidate transitions and decoded confidence. Do not publish them if they reconstruct restricted tracking. Exact reproducibility requires fixed seeds, dependency versions, deterministic inference, model/config hashes and a documented single command. Re-run from a clean environment and compare the final CSV SHA-256.

Focused tests: one-based/zero-based conversion; tag-side inversion; NaN handling; metres versus normalized coordinates; same-team passes; uncontrolled deflection; controlled one-touch turnover; missing ball during live play; H–D–H yielding two possessions; halftime reset; short-spell preservation; decoder equivalence on tiny inputs; exact row count and labels. These are synthetic/external fixtures, never hand annotations of the competition match.

### Codex local execution contract

Codex can write, review and run the pipeline without receiving the restricted data. Apply this boundary throughout implementation:

1. **Model-visible material:** source code, public specifications, synthetic fixtures, dependency information, generic errors, and allowlisted operational statuses. Public/external evaluation summaries may be surfaced when their licences permit it.
2. **Local-only material:** competition XML/TXT/CSV contents, video/audio, per-frame features, inferred events, predictions, tracking plots, target distributions, and detailed error logs. Store these in `data/`, `outputs/` or `local_reports/`, all ignored by Git.
3. **Output discipline:** target-data commands print only stage names and fixed PASS/FAIL or completion messages. Redirect detailed stdout/stderr to local log files. Do not use `head`, `cat`, dataframe previews, notebook displays or image tools on restricted assets or derived outputs. Parser exceptions can contain input rows: suppress their strings and provide generic error codes.
4. **Debugging:** reproduce parsing and model failures using synthetic fixtures with the same public schema. Local code can inspect a failing record and classify the defect, but should emit only an allowlisted error category. Do not ask Codex to inspect a saved raw traceback unless it has first been sanitized locally.
5. **Network boundary:** acquire dependencies and licensed external training data separately. Target inference uses local checkpoints, no hosted inference APIs, experiment tracking uploads, cloud notebook mounts or automatic telemetry carrying artifacts. Offline flags are useful but do not themselves enforce network isolation; use host-level network restrictions where available and inspect dependencies for upload integrations.
6. **Training separation:** the training process reads only an external-data manifest. Reject target paths, match identifiers and target input hashes. Normalization, calibration, transition penalties, duration priors and fusion weights are all frozen from external training/development. Loading the target locally does not make target fitting permissible.
7. **Reproducibility:** a frozen configuration and checkpoint generate the CSV automatically. Run the official validator with detailed output captured locally and expose only its exit status. Compare independently generated CSV hashes locally. Keep submission CSV and detailed reports out of the public repository unless expressly permitted.
8. **Git checks:** before publishing, run a local staged-file gate that rejects restricted inputs, archives, videos, derived outputs and symlinks pointing into data folders. `.gitignore` is a safeguard, not a replacement for reviewing the staged file list.

The first audit script performs streaming checks of index continuity, observation shape, metadata membership, numeric parseability, homography structure and tag interval bounds. It does not validate tracking accuracy, possession correctness or model quality. Detailed measurements are saved locally and not used to select target-specific thresholds.

### Codex implementation task sequence

| Task | Concrete deliverable | Acceptance condition |
|---|---|---|
| 0. Local inputs | Canonical data folders, duplicate manifest, structural audit | Hash-preserved unique files; schema checks pass; no data contents printed |
| 1. Reproducible environment | Pinned official code/dependencies and synthetic fixtures | Scorer and loader tests run without target data in context |
| 2. External validation | Licensed downloads, match split, label adapter, corruption harness | Held-out matches isolated; no target training inputs |
| 3. Baseline system | Control features, dead-ball emissions, three-state decoder | Valid full-length external predictions and exact scorer results |
| 4. Structured improvements | Duration terms, boundary refinement, cautious tag fusion | Consistent external improvement with short-spell checks |
| 5. Optional neural model | Player-motion branch and externally calibrated fusion | Gain under missing-ball and track-fragmentation stress tests |
| 6. Freeze and infer | Immutable config/checkpoint and local target prediction | No parameter changes based on target output or public scores |
| 7. Package | Validated team CSV, public code and technical document | Clean reproduction gives identical CSV; restricted artifacts excluded |

Task 0 is complete: all seven structural audit checks passed and all retained archive-member hashes were reverified after cleanup. These checks establish structural usability, not tracking accuracy. Phase 1 implements Task 1: Python 3.12 with locked dependencies, official source pinned by commit/blob hashes, a streaming parser, synthetic metric/input tests, a staged-artifact gate and CI. See `docs/PHASES.md` for verification outcomes. Phase 2 implements Task 2: a frozen external catalogue and split, licensed acquisition, a Sportec adapter with explicit unknown-label masks, deterministic corruption and exact period-isolated scoring. Real-data readiness and transfer limitations are recorded in `docs/PHASES.md`; Tasks 3–7 require subsequent user authorization. The supplied tactical files are sufficient for the recommended first system, so missing videos do not block implementation. Broadcast fusion remains deferred. No additional model complexity is justified merely by the presence of the second tracking feed.

## 8. Schedule for the remaining time

This schedule assumes work begins November 14, using India time for team planning. It is an execution budget, not a claim that the official deadline is midnight IST. A CPU is sufficient for the first model; GPU access is optional for the neural branch. Benchmark actual runtime before committing to it.

| Window | Required outcome | Stop/defer criterion |
|---|---|---|
| Nov 14, first 4 hours | Asset consolidation complete; finish structural audit, review terms locally, freeze external split | Do not start broad model search before valid external labels exist |
| Nov 14, next 4–8 hours | Geometry/control features, dead-ball model, runnable external evaluation | Retain simple model if neural setup threatens core delivery |
| Nov 15 | Structured decoder, corruption tests, tag semantics, ablations | Drop additions that do not help across external matches |
| Nov 16, first half | Optional player-motion branch; freeze selected model | Do not introduce broadcast synchronization this late |
| Nov 16, second half | Deterministic target inference, validator, clean reproduction, technical document | Resolve packaging and data disclosure issues before cosmetic work |
| Nov 17 before confirmed cutoff | Final validation and all three deliverables submitted | Reserve substantial upload/reproduction buffer |

Implementation will be Codex-driven, with local scripts performing all data operations. Use the staged tasks below rather than assuming three simultaneous human implementers. Human team members remain responsible for registration, licence/terms review and submission requirements. No sub-agents were launched in this review.

## 9. Risks and decision rules

| Risk | Response |
|---|---|
| External labels define possession differently | Explicit semantic adapter, external audit, report residual uncertainty |
| Clean external tracking overstates accuracy | Structured corruption and a separate provider evaluation |
| Dead-ball labels unavailable or filtered out | Recover provider status where supported; otherwise use audited external annotations, never assume every gap is dead |
| Track IDs unstable | Team-level pooling, local tracklet continuity, fragmentation augmentation |
| Tags duplicate geometric evidence | Cap/group their contributions and stress-test conflicts |
| Duration model removes short spells | Soft priors, dense boundary proposals, short-spell recall report |
| Pretrained model has hidden validation leakage | Match-level training manifest and unseen-match tests |
| Novel decoder adds complexity without gains | Keep standard Viterbi as the default |
| Insufficient development time | Submit the validated classical/structured pipeline |

Final selection should answer: does the added component improve exact spellF1 on genuinely unseen external games under realistic corruption, without harming short spells or dead-ball handling? If not, leave it out. A simpler reproducible system with strong temporal logic is the preferred submission over an unvalidated large model.

## 10. Phase execution agreement

Complete one phase at a time. At its end, run its acceptance checks, record results and limitations in `docs/PHASES.md`, revise this plan from the evidence, and commit/push public files to `prithvi-kaizen/More-Than-A-Hack-`. Stop all work after reporting that phase. Start the next phase only after the user asks to continue. Do not interpret the original overall implementation request as authorization to cross this boundary.

Phase 1 establishes mechanics, not model quality. Its synthetic score checks cannot establish expected competition performance. Phase 2 will prepare licensed external data, freeze match-level splits, validate alive/dead and ownership semantics, and implement corruption-based external evaluation. Phase 2 should first prove external dead-ball labels are usable before fitting any possession model.

### Phase 2 decisions

The publisher catalogue—not an earlier third-party mirror description—defines the seven external IDs: J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ and J03WR9. Four training matches, one development match and two holdouts are frozen in `configs/external_split.json` and enforced in code. J03WPF/J03WQF mentioned in earlier research sources are not part of this catalogue.

The first real training match is the end-to-end pilot. All seven metadata/event pairs are available locally. Remaining full tracking transfers are explicit prerequisites for Phase 3, rather than background jobs that run across the user's phase boundary. The downloader preserves completed ranges and verifies whole-file publisher checksums. Acquisition of roughly 2.62 GB is materially slower than anticipated on the original host; the mirror is supported without relaxing integrity checks.

Sportec ball status and ownership are separate supervision fields. Missing status maps to unknown, not dead; live unknown ownership retains alive supervision while masking the team label. These are provider labels with a potential convention mismatch to the competition. Inputs and labels are saved separately. Model quality has not yet been estimated. The initial corruption suite tests gaps, noise and track fragmentation; extend it with structured camera/crowd/boundary loss and tag errors during later model ablations.

On continuation, finish and verify the remaining selected training/development tracking files before fitting the first emissions model. Do not substitute training-match diagnostic results for development performance; do not evaluate final holdouts during tuning.

Pilot verification: J03WMX full tracking passed the original publisher checksum. Both halves prepared successfully, with 145,967 total frames and complete provider status/ownership labels. The untrained nearest-player comparator gives spellF1 0.1043 clean, 0.0550 moderate corruption and 0.0111 severe corruption; macroF1 rises from 0.3826 clean to 0.4289/0.4142 because some false dead-ball guesses happen to be correct. These external training-pilot diagnostics justify keeping spellF1 as the selection objective and prioritizing alive/dead inference and temporal coherence. They do not establish competition accuracy or justify choosing a hyperparameter. Six full tracking files remain to acquire in Phase 3; no acquisition continues after this phase stop.

Supplemental archive check after Phase 2: six additional Drive ZIPs contained only SHA-256-verified copies of the seven canonical assets. Redundant ZIPs were removed after re-verification. No videos or new inputs were added, so the asset inventory and Phase 3 prerequisites are unchanged. This check did not begin Phase 3.

### Phase 3 results and revised priorities

Phase 3 is complete: all four training matches and the development match are now
verified/prepared. The remaining two holdout tracking files are deferred; they are
not required for development fitting. Disk constraints were handled with lossless
raw-XML compression and original-byte checksum verification. Native mirror transfers
were adopted only after their publisher hashes matched. Acquisition/training locks
prevent overlapping retries. Detailed source provenance and the model binary stay local.

The first model uses 42 alive/dead and 15 ownership features, separate histogram
boosted classifiers and exact H/A/D Viterbi decoding. It trained on 340,374 sampled
external examples across three corruption scenarios. Selected turnover/dead penalties
are 5/40. Mean development spellF1 is 0.4926 versus 0.2379 for official smooth50;
clean/moderate/severe scores are 0.5260/0.5139/0.4378. Mean macroF1 is 0.8887.
These are selection-set results on IDSSE provider labels, not a competition estimate.
All 45 synthetic tests and full development replay passed. See `docs/PHASE3_MODEL.md`
and `docs/phase3_results.json` for aggregates and limitations.

The audit changes Phase 4 priorities. Alive/dead F1 is strong (0.9968 clean and
0.9542 severe), whereas short ownership spells are frequently missed: below-two-second
recall at IoU 0.5 is 7.3% clean and 0% severe, versus more than 93% for spells over
15 seconds. First check whether those short provider spells represent controlled
possessions using external event evidence. Preserve the original label benchmark
rather than redefining it for a higher score. Then test capped inverse-duration
sample weighting, boundary-sensitive ownership emissions and adaptive transitions;
avoid simply increasing smoothing or imposing a minimum duration. Use four training
matches for cross-validation of further architectural choices and keep the two final
holdouts sealed. Tag fusion remains an optional, separately stress-tested prior.

No competition inference, target-label tuning or target-video comparison occurred.
Phase 4 requires a new user instruction after the Phase 3 GitHub update and stop.

### Phase 4 results and Phase 5 direction

Phase 4 is complete. An external training-only event audit found control-supporting
annotations inside 63/600 spells shorter than two seconds and 626/685 spells at least
five seconds long. Timestamp uncertainty and incomplete logging prevent equating
uncorroborated spells with lack of control. Original labels were preserved. Capped
partial inverse-duration weights and a geometry-based control/adaptive-boundary model
were compared against frame weighting in four leave-one-training-match-out folds.

The selected control/adaptive bundle improved mean original-scene CV spellF1 from
0.4710 to 0.5153, with gains in all four folds and a better structured-noise score.
Only after selection was it refit and checked on the development match. Mean spellF1
rose from the frozen Phase 3 result of 0.4926 to 0.5373; clean/moderate/severe scores
are 0.6024/0.5600/0.4495. Structured-noise spellF1 rose from 0.4283 to 0.4498.
All 56 tests and independent checkpoint probability/label-byte/metric replay passed.
These remain external provider-label validation results, not competition estimates.

Short-spell recall improved but remains low: 16.9% clean and 5.6% severe at IoU 0.5.
Long-spell recall under structured noise fell from 91.9% to 85.5%, a material tradeoff.
A posthoc training-label audit found only 14/600 short spells with no stride-five
sample, so resolution alone is not the main explanation. Phase 5 should first test
reliability-aware boundary gating and then an optional player-motion temporal model.
Keep Phase 4 as the new comparator and Phase 3 as a fallback; do not adopt a larger
model without match-separated evidence. Any pretrained model must exclude reserved
holdout matches from its training provenance or be retrained within our split.

Soft duration terms and tag fusion were deferred: more persistence needs a short-spell
guard, and no external source here reproduces Metrica's automatic tags. Synthetic
phase priors alone would not establish their reliability. No competition inference
or final-holdout evaluation occurred. Phase 5 requires the user's next instruction.

### Phase 5 results and selection

The four-policy comparison selected a small player-motion temporal expert with
reliability-weighted fusion. It was trained from scratch inside each training fold;
no published weights or reserved holdout matches entered fitting. A fixed symmetric
19,169-parameter TCN uses pooled player geometry/movement, masks dead/unknown ownership
and runs eight epochs. Its ownership contribution is capped at 0.35 and increases
only as ball-geometry reliability falls. The Phase 4 alive/dead emissions and decoder
settings are retained. Reliability-only changes did not meet the 0.005 gain requirement.

Mean original-scene CV spellF1 rises from 0.5153 to 0.5436, with all four folds won.
Structured-noise spellF1 rises from 0.3979 to 0.4694 and long-spell recall from 79.2%
to 83.3%. The selected policy was then refitted on all four training matches and
compared once on development. Mean development spellF1 improves from 0.5373 to
0.5581; clean/moderate/severe scores are 0.6058/0.5704/0.4982. Structured spellF1
improves from 0.4498 to 0.5111 while long-spell recall remains 85.5%, passing the
declared guards. The selected system is Phase 4 plus motion fusion, identified in
`configs/phase5_selected.json`. Full experiment details are in `docs/PHASE5_MODEL.md`.

All 67 synthetic tests passed, including deterministic real optimization, team/slot
symmetry, chunk halos, loss masks and leakage/adoption guards. Frozen Phase 4 fold
and development scores replayed exactly. Short spells remain weak: clean recall is
unchanged at 16.9%, while severe recall rises from 5.6% to 8.1%. Repeated development
use and single-provider simulated corruptions limit the generalization claim. The
final holdouts, target inference and target-video comparison remain untouched.

Independent checkpoint reloads also exactly reproduced saved fold probabilities,
decoded labels, every policy's fold metrics and the development comparison. The
audit verified training-match metadata, raw/prepared/cache provenance and source/
dependency hashes. Detailed probabilities and checkpoint files remain local.

Phase 6 should now validate and freeze this selected system, rather than add another
model family or tune toward the development game. Report the final holdout results
honestly, retain the recorded baselines, and proceed with automatic local inference
checks. The next phase needs a new user instruction after the Phase 5 GitHub update.

### Phase 6 execution requirements

Phase 6 starts only after the Phase 5 report, checks and GitHub update, and a new
user instruction. Freeze the Phase 5 selected configuration, checkpoint hashes,
feature/decoder source hashes and dependency lock before opening final-holdout
performance. Retain the Phase 4 and Phase 3 checkpoints as recorded comparators.

Acquire and publisher-verify J03WQQ/J03WR9 tracking locally, with lossless raw-file
compression if needed for disk space. Keep their inputs and provider labels separate.
Evaluate the frozen system on clean/moderate/severe/structured conditions with the
official scorer and short/long-spell diagnostics. Report these independent results
without changing thresholds, model weights, fusion settings or epochs in response.
Any implementation defect that invalidates an evaluation must be fixed and documented
as such, rather than treated as an opportunity for further model selection.

Then implement the local tactical-feed inference adapter. Validate coordinate units,
team membership, frame rate and half windows using metadata and automatic structural
checks. Match the trained feature/time conventions and reset temporal state between
halves. Verify selected checkpoint/source hashes before inference. Generate the
full organizer-format CSV locally, enforce the required row count and label alphabet,
and run the pinned official validator. Repeat inference from independent checkpoint
loads and require identical CSV bytes. No target output, public possession labels or
video comparison may guide model changes. Detailed predictions remain local.

Do not add tag fusion, duration constraints or camera fusion during final inference:
none has the required external validation. Publish source code, external aggregate
results and the revised phase record; stop again before Phase 7 packaging/submission.

### Phase 6 results and Phase 7 requirements

Phase 6 is complete. Selection artifacts were frozen before acquiring the two reserved
games, and execution code was frozen before scoring them. Both publisher-verified
holdouts were prepared and compressed losslessly. A separate final-test loader keeps
training access unchanged. Frozen Phase 3/Phase 4/motion-fusion mean spellF1 over
clean/moderate/severe conditions is 0.5328/0.5607/0.5901; structured-noise spellF1 is
0.4283/0.4554/0.5309. Motion fusion improves every condition of each held-out game.
Structured short-spell recall improves from Phase 4's 12.8% to 19.0%, and long-spell
recall from 90.6% to 94.5%. No fitting, selection or parameter changes followed these
independent results. Both tests still share the external provider's possession
convention, so competition transfer remains unmeasured.

The tactical adapter uses organizer-published frame/pitch conventions, converts ATD
record k to video frame k-1, preserves missing/out-of-field observations and processes
halves independently. It uses only tactical metadata/tracking. The complete inference
CLI ran twice in independent processes, reread raw inputs and reloaded checkpoints,
generated identical CSV bytes and passed both pinned validation interfaces. Detailed
predictions, hashes and logs stay local; no target label/video comparison or tuning
occurred. Target inference overlapped the second external download because its model
was already frozen and independent of holdout outcomes.

All 75 synthetic tests passed. The new wrapper reproduced existing development scores;
independent checkpoint reloads reproduced every final-holdout probability, decoded
label and metric. The candidate is `outputs/phase6/submission.csv`; public method,
aggregates and hashes for the frozen records are documented in `docs/PHASE6_INFERENCE.md`
and `docs/phase6_results.json`.

Phase 7 should concentrate on delivery. Obtain the team's chosen name, create the
team-named CSV and short technical document, and settle a reproducible distribution
route for the external-trained checkpoints. The freeze guard currently requires all
three checkpoint files, including the Phase 3 comparator. Verify a fresh-checkout
prediction run against the exact candidate bytes, validate the final upload file,
and keep competition data/derived predictions outside public Git history. Explain
the provider-label/competition-definition gap, missing-ball behavior and remaining
short-spell limitations in the technical document. Do not introduce another model
family or tune against the now-open final tests. The original form submission stays
with the team unless explicitly authorized. Phase 7 requires the user's next call
after the Phase 6 GitHub update and stop.

### Phase 7 delivery and final state

The participant supplied the registered name Kaizen. The exact validated candidate
was copied to `outputs/phase7/upload/Kaizen.csv`; a team-named Markdown technical
document was generated from `docs/TECHNICAL_REPORT.md`. Both validation interfaces
pass on the exact upload CSV and its bytes match Phase 6. Private instructions
include the organizer's form link and confirmed cutoff. The form has not been submitted.

Checkpoint distribution is resolved through the public `models-v1` GitHub release:
three externally trained checkpoint files plus model card/licence, CC BY 4.0 with
DFL/Bassek et al. attribution. The 597,573-byte compressed archive and every member
have pinned SHA-256 hashes. The installer rejects changed/extra/missing/duplicate/
linked members and conflicting destinations before installing files. Competition
data, predictions and detailed reports are excluded from public Git history and assets.

The public `inference-v1` source tag, commit da7343eae56abe2f2b72fb88da19fbad5e14993d,
passed a genuinely fresh anonymous clone test. No data, models, environment or feature
cache existed there. A separate locked environment and anonymous public checkpoint
download supported a full local inference run against the original bundle; candidate
CSV bytes, input/freeze provenance and official validation matched exactly. The test
environment was macOS arm64, Python 3.12.4, NumPy 2.2.6, scikit-learn 1.7.2 and
PyTorch 2.8.0. Other platforms' target-data byte identity remains unverified. All
80 synthetic tests passed; subsequent delivery documentation leaves tested runtime
files, model weights and frozen records unchanged.

Implementation is complete. Preserve the current selection and artifacts. The next
action belongs to the participant: review the technical document, verify registration
details and upload the named CSV/Markdown with the public repository URL via the
organizer's form. Do not start another modelling phase or use final-test/target
outputs for tuning. Future work would need a new task and a new evaluation protocol.

### Phase 8 delivery hardening and next experiment

The user authorized Phase 8 after the submission review. Repackaging now preserves
existing private handoff instructions, checks team/submission-status conflicts before
writes, and creates the validator-log directory when absent. Two synthetic end-to-end
regression tests cover reruns and rejected overwrites. Both frozen suites passed 82
tests. The exact package was backed up and regenerated twice: candidate CSV/handoff
bytes stayed identical, the revised technical document and manifest agree, and the
official validators and status-only local audit passed. The report now includes the
pinned clone command and corrects duplicated reliability wording.

The Docker daemon answered as Linux ARM, but image enumeration failed with an
existing storage I/O error. Linux target-byte reproduction remains unverified; the
completed macOS arm64 replay is the supported reproduction path. No model/runtime
changes, Docker storage repair or target-quality inspection occurred.

Eligibility remains unresolved: the user reported one registered member, and no
organizer-approved exception/team change has been supplied. A concise inquiry and
status record are private in `outputs/phase8/`; no message was sent. The participant
must resolve eligibility independently of the technical experiment.

Phase 9 is next only on a new user authorization. Follow the bounded reliability
experiment and predeclared adoption gates in `docs/SUBMISSION_REVIEW.md`, preserve
v1, and keep the opened final external games out of selection. Do not expand into
several challenger architectures or target-based tuning. Stop after Phase 8's checks,
plan update and GitHub push. See `docs/PHASE8_HARDENING.md` for the completed scope.

### Phase 9 outcome and Phase 10 scope

Completed the single externally trained reliability challenger on 17 November 2026.
The original four training games were held out in turn; validity fitting used two
other games and sigmoid calibration used the third. Ten conditions and five fixed
evaluation seeds produced 200 comparison cells (184 distinct input views). The
existing tree/TCN folds were reused without refitting. The experiment did not read
competition inputs/labels/video or the already-open final external games.

Mean stress spellF1 improved from 0.483609 to 0.489890, winning all four matches;
sustained false-ball gain was +0.024778 and independent-camera gain +0.006062.
Long recall rose 0.003542; short recall fell slightly. All regression and targeted
condition gates passed, but mean gain 0.006281 missed the fixed 0.01 requirement.
Reject for adoption, skip development as declared, and preserve v1. These repeated
training-match CV results are not independent tests or target accuracy estimates.

The parallel scheduler's shared protocol-write race was repaired with parent-only
writes and read-only worker verification. Original/repaired execution hashes are
retained. No frozen experiment source, model setting, partition or selection gate
was changed. Both full frozen suites passed 92 tests; the independent audit verified
hashes, all saved official spell scores at each IoU, historical clean fold scores and
aggregate arithmetic. Source, protocol, aggregate results and the retained/rejected
decision are in `docs/PHASE9_RELIABILITY.md` and `docs/phase9_results.json`.

Phase 10 should now close delivery using v1. No new boundary model, alive/dead head,
parameter rescue or independent-data acquisition is required to preserve the already
tested submission. Review the exact refreshed Kaizen document/CSV, confirm public
reproduction access and resolve team-size eligibility with the organizer. Actual form
submission remains with the participant unless expressly delegated. Any further
reliability research needs new independent data and belongs after this submission.
Stop after Phase 9 verification and GitHub push; Phase 10 needs the next user call.
