# Submission review and improvement plan

Reviewed 15 November 2026. Candidate: `models-v1` / `inference-v1`, team Kaizen.

Update 17 November: [Phase 8 technical hardening](PHASE8_HARDENING.md) and the
[Phase 9 reliability comparison](PHASE9_RELIABILITY.md) are complete. The package
overwrite is fixed and the document is updated. The challenger gained 0.006281 but
missed the predeclared 0.01 requirement, so retain v1. Linux reproduction remains
unverified because Docker storage failed. Eligibility remains unresolved. Phase 10
should close delivery on v1; the findings and original proposals below retain their
historical review context.

**Assessment: technically ready within the checked delivery scope; advancement
eligibility needs resolution; top-20 competitiveness is unmeasured.** Preserve the
validated candidate. The review found useful improvement opportunities, but no
demonstrated error requiring immediate replacement of its predictions.

The ranking is decided by private full-match **spellF1**, with macroF1 as the first
tie-break. The report, originality and model size do not earn additional ranking
points. There is no live leaderboard or known top-20 cutoff. An external score of
0.5901 cannot be compared directly with the organizer's 0.2433 target-match baseline:
the games, tracking quality and reference labels differ.
[Evaluation](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md),
[competition overview](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/OVERVIEW.md).

## 1. Eligibility comes first

The published rules require **3–5 members** and explicitly make teams with fewer
than three ineligible for the Innovation Challenge and Final Pitch. A solo entrant
must obtain an organizer-approved team change or a written exception. Do not assume
that a scored CSV establishes eligibility or that adding names without registration
and consent resolves it. The rules do not settle every detail of solo leaderboard
participation; ask the organizers to distinguish scoring from advancement.
[Rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md).

The participant should contact the registration organizer now to confirm whether
two or more eligible teammates can join before the deadline and what registration
updates are required. No message has been sent by this review. Keep registration
correspondence and member details private.

## 2. What was checked

* Read the current organizer rules, challenge, evaluation, submission, FAQ and
  overview; inspected the pinned metric/validator and the implemented frame/team/
  coordinate adapter, feature extraction, model fusion, decoder, split guards,
  corruption, checkpoint installation and package/reproduction tooling.
* Reran both frozen unittest commands: **80 tests passed in each run**.
* Checked the exact upload CSV with an independent streaming CSV reader and both
  official validation interfaces. Header, frame numbering, length, allowed labels
  and the algorithm's inside/outside-half convention passed.
* Rehashed the original local inputs, frozen runtime/models, upload manifest and
  document. Compared the upload CSV with both independent Phase 6 outputs and the
  saved fresh-public-checkout output: identical bytes and provenance.
* Recomputed six published final-test means and four pooled recall figures from the
  per-match aggregate records. Confirmed all eight motion-versus-tree comparisons
  improve spellF1. This is independent aggregation, not a new raw-label evaluation.
* Checked reachable Git history for restricted directory names and artifact
  extensions, and confirmed the checkpoint release remains public. This path check
  is not a forensic content scan of every historical blob or an audit of signed terms.
* Reproduced two robustness limitations with synthetic coordinates/probabilities.
  No competition labels, video, output-quality inspection or model fitting was used.

Repeat the local artifact checks with:

```sh
uv run --frozen --extra temporal python scripts/review_submission.py
```

This is a local delivery audit, requiring the existing private package and prior
replay outputs. It is not the fresh-clone inference command. It prints fixed statuses
and retains failure/validator details locally. See [reproduction](REPRODUCTION.md)
for the public inference path.

### Scope and remaining coverage

Counts below describe the listed review units, not a probability of correctness.
Zero observed defects does not mean that every possible input/platform was tested.

| Submission quality category | Observed defects | Assessment |
|---|---|---|
| Required delivery and eligibility | 1 / 5 | CSV, public code, technical document and recorded reproduction pass; the team-size requirement needs resolution for a solo entrant. |
| Document completeness | 0 / 6 | Approach, assets, training data, possession rules, limitations and reproduction are covered. Minor wording/pinning improvements are proposed below. |
| Visual/interactive behavior | N/A | Deliverables are CSV, Markdown and a CLI, not a dashboard. |

| Analytical correctness category | Observed defects | Assessment |
|---|---|---|
| Controlling definitions | 0 / 3 | Official spellF1, label semantics and frame contract traced to organizer sources/code. |
| Numerical aggregation | 0 / 10 | Six means and four pooled recall values agree; underlying raw-label evaluation was not repeated. |
| Within-chart agreement | N/A | No charts in the submission report. |
| Source explanations | 0 / 4 | Organizer rules, external data provenance, model selection and reproduction sources are identified. Signed enrollment terms were not independently audited. |
| Cross-artifact agreement | 0 / 4 | Upload CSV, report, frozen models and reproduction outputs agree. Historical plan wording was clarified in this review. |
| Engineering controls | 0 / 6 | Schema, frame mapping, half isolation, split guards, freeze hashes and artifact integrity checked; new Linux target reproduction was not performed. |
| Conclusion support | 0 / 3 | External gain, local reproducibility and lack of a target-rank estimate are supported. Generalization remains uncertain. |

## 3. Findings that matter for the score

**P1: Generalization evidence is narrow.** Four training matches, one reused development
match and two final matches all come from one provider. The corruption scripts use
one deterministic realization per match/half/scenario; they do not establish robustness
across camera-error distributions. The simulated camera crop follows the true external
ball, allowing visibility to carry a ball-location cue. This is disclosed, but an
independent-camera-mask test is missing. A new model must not reuse the already-open
final matches as a tuning set or present them as fresh independent evidence.

**P1: Smooth false balls evade the reliability gate.** In `geometry_reliability`,
adding a constant spatial offset leaves speed and midpoint residual unchanged.
A synthetic straight trajectory shifted by 25 m and 15 m still receives q=1 away
from the edges. Then the motion fusion weight `0.35*(1-q)` is zero. Existing stress
adds mostly 1–3-frame false-ball detections; it does not test sustained plausible
false tracks adequately. This proves a gate limitation, not a target error. Proposed
fix: train an external-only reliability estimator with sustained false tracks and
ball/player compatibility, taking care not to mistake legitimate passes for errors.

**P1: Short possessions and transitions during outages remain weak.** Final external
sub-two-second recall at IoU 0.5 is 60/195 clean (30.8%), 26/195 severe (13.3%) and
37/195 structured (19.0%). With a missing ball, q=0 also zeros the adaptive boundary
signal. Ownership emissions can still cause a transition, but lose that boundary
assistance. A player-only boundary head is worth testing; more blanket smoothing or
a minimum possession duration is a poor default. These recall numbers are diagnostic,
not the full official metric, and provider short-touch semantics remain uncertain.

**P1: Alive/dead has no separate motion fallback.** Fusion preserves the tree's D
emission mass; the network predicts conditional ownership only. Joint decoding can
still change D labels indirectly, but the neural branch supplies no direct stoppage
evidence. Audit false D splits and missed same-team restarts on external matches
before deciding to add a player-motion alive/dead head. Its benefit is unproven.

**P2: Delivery hardening is worthwhile.** Exact target reproduction is established
on macOS arm64, not Linux x86-64. Synthetic Linux CI does not establish byte identity
on the target. Also, `package_submission.py` rewrites `UPLOAD_INSTRUCTIONS.md` with a
generic message, losing separately added form/deadline details on rerun. Preserve
that handoff before repackaging. Put the pinned clone command directly in the short
technical document and remove the repeated word in the reliability paragraph.

The final external breakdown confirms where motion fusion helps:

| Condition, mean across two final matches | Prior tree | Current candidate | Difference |
|---|---:|---:|---:|
| Clean | 0.6238 | 0.6279 | +0.0041 |
| Moderate | 0.5967 | 0.6140 | +0.0173 |
| Severe | 0.4615 | 0.5282 | +0.0667 |
| Structured | 0.4554 | 0.5309 | +0.0755 |

Source: [per-match external results](phase6_results.json). Conditions from the same
match are correlated; eight wins are not eight independent match replications.

## 4. Improvement plan: three optional phases

Phases 8–9 are complete. Phase 9 did not adopt its challenger, so Phase 10 should
close the existing v1 delivery rather than seek new modelling improvements. Each new
phase requires the participant's next authorization, ends with verification and
a GitHub update, and stops. Preserve v1
as the submission fallback throughout. Time estimates are planning budgets, not
measured runtime or promised score gains.

### Phase 8: eligibility and delivery hardening (first priority; 1–3 hours of work)

1. Participant obtains organizer guidance on team registration/eligibility. Organizer
   response time is outside the work estimate. Do not silently presume an exception.
2. Preserve private handoff instructions during packaging; add a synthetic rerun test.
   Clarify the technical document's pinned source/environment commands. Regenerate
   the document/manifest while requiring the candidate CSV hash to remain identical.
3. If a functioning local Linux VM/container is available, run the released pipeline
   with the local restricted bundle, capture logs privately and compare CSV bytes.
   Do not move match data to hosted CI or claim Linux identity from synthetic tests.
   If unavailable, retain the explicit macOS reproduction instructions and limitation.
4. Rerun the audit, both test suites and staged-file gate, commit/push, then stop.

**Exit:** exact upload files validated; source/runtime documented; handoff preserved;
eligibility outcome explicitly recorded privately. An unresolved team requirement
remains an advancement blocker even if every technical check passes.

### Phase 9: external stress diagnosis and one bounded challenger (6–12 hours)

1. Write a new experiment manifest before observing its results. Keep J03WQQ/J03WR9
   sealed to this experiment; their v1 findings stay historical. Use whole-match folds
   within the original four training games, with all halves/views grouped together.
   Treat repeated CV and J03WPY as selection evidence, never independent tests.
2. Use five declared corruption seeds per evaluation condition, held separate from
   training-corruption seeds. Add sustained false ball tracks, 2/8/20-second outages,
   ball-independent partial camera views, slot fragmentation and duplicated tracks.
   Predeclare distributions from external data/physical reasoning; do not estimate
   or tune them from competition outputs. Fix a probability-combination/decoder rule
   before testing rather than selecting separate target-frame winners.
3. Compare only v1 and **one reliability-aware challenger** first. Train a small
   validity classifier on externally corrupted versus original geometry, with explicit
   visibility, trajectory and player-compatibility evidence. Fit calibration only
   inside training folds. Preserve ordinary passes and aerial play in the training
   examples. Freeze the candidate settings before the validation comparisons.
4. Report official spellF1 at every IoU threshold, per-match and worst-condition
   differences, short-spell recall, long-spell recall, false D fragmentation and
   restart/turnover timing. Include crop-mask and motion-branch ablations to expose
   visibility shortcuts. Use matches as the statistical unit; seed repetitions do
   not turn four games into twenty independent games.
5. Only if external diagnosis supports it, propose a separate next experiment for
   a player-only boundary or alive/dead head. Do not fold both into this challenger
   or launch a larger architecture search within the same phase.

**Proposed adoption gates, to lock before running:** mean matched-condition spellF1
gain at least 0.01; improvement in at least 3/4 match folds; no match-average or
condition-average loss above 0.01; no pooled long-spell recall loss above one
percentage point; and improvement on the specific sustained-error condition the
change targets. Reject a candidate that wins only through a camera-mask shortcut.
These are engineering decision thresholds, not significance levels or top-20 cutoffs.
Run one fixed development comparison after CV selection; no threshold rescue search.

**Exit:** a documented accept/reject recommendation, unchanged v1 fallback, all
experiments and rejected variants retained. No competition inference/model selection
occurs in this phase. Stop after checks and the GitHub update.

### Phase 10: independent confirmation and release decision (2–4 hours plus data access)

1. Prefer new licensed, organizer-accessible external matches, ideally another tracking
   provider, selected before opening outcomes. Record annotation semantics, rights,
   checksums and split. No suitable new source has yet been verified; data access is
   a real dependency, not an assumed completed step. Never rename the old holdouts
   to make them look unseen.
2. Freeze v2 weights, preprocessing, calibration, decoder, dependencies and release
   code before this test. Evaluate once; do not repair poor results by retuning on it.
   An audit of an actual implementation defect must be recorded separately.
3. If new independent confirmation is unavailable before the freeze deadline, the
   conservative recommendation is to submit v1. A CV-selected challenger would have
   weaker confirmation; do not present it as independently verified or automatically
   substitute it for the current candidate.
4. For an adopted v2, generate the target CSV locally only after selection; perform
   automatic schema/invariant checks and two-process plus fresh-checkout byte replay.
   Publish separate immutable-by-hash source/model versions and a complete document.
   Never choose v1 versus v2 by target labels, video or apparent prediction plausibility.
5. Review the exact upload files and submit one response through the official form.
   Registration and actual form submission remain participant actions unless expressly
   delegated. Save the submission receipt and submitted file hashes privately.

**Schedule:** aim to freeze modelling by **16 November, 18:00 IST**, leaving time for
reproduction and submission. The organizer's previously verified email cutoff is
**17 November, 23:59 CET = 18 November, 04:29 IST**. If eligibility or experiment
work overruns, prioritize the valid frozen submission and accurate eligibility status.

## 5. Ideas to defer

Do not spend the remaining time on a presentation, larger transformer, detector
rebuild, ad hoc smart-tag rules or broadcast alignment without external validation.
Smart tags and the second feed are legitimate potential evidence, but no matching
external reliability/alignment benchmark currently supports their fusion weights.
More inputs and novelty are not independently scored.

Two research directions support the existing approach rather than a rushed rewrite:
[PathCRF](https://arxiv.org/abs/2602.12080) models possession paths from player
trajectories with a constrained CRF and Viterbi; adapting its player-level formulation
would require checking identity assumptions, data rights and compatibility with the
fragmented ATD slots. [MS-TCN](https://arxiv.org/abs/1903.01945) uses temporal
classification and smoothing to reduce over-segmentation on action datasets. Its
reported tasks are not this competition; importing extra smoothing without short-spell
guards could worsen the weakness already measured here. Neither paper supplies a
transferable top-20 score estimate.

**Decision:** resolve eligibility, protect the reproducible submission, and test the
specific reliability weakness before buying complexity. No review can certify a
top-20 or first-place finish without the private evaluation and competing entries.
