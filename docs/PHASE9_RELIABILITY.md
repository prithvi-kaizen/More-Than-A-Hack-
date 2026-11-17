# Phase 9: reliability challenger evaluated and rejected for adoption

Completed 17 November 2026. **Retain the frozen v1 submission.** The challenger
improved the external stress-test mean from 0.483609 to 0.489890, a gain of 0.006281.
It won all four match folds and passed every regression/targeted-condition guard,
but missed the predeclared minimum mean gain of 0.01. That requirement was not
relaxed after seeing the result. Development was therefore skipped, as declared.

## What was tested

A ten-feature histogram-boosted ball-validity classifier and sigmoid calibration
lowered the existing reliability score via `min(q, calibrated_validity)`. The tree
ownership/alive heads, player-motion TCN, fusion cap and decoder stayed frozen.
This was one fixed challenger, not a parameter sweep. Validity training targets
compare automatically corrupted coordinates with their original external provider
measurements; no new manual labels were created.

Each of the original four training games was held out in turn. The validity model
fit two remaining matches and calibrated on the third. Existing verified Phase 5
tree/TCN folds were reused without refitting and excluded their corresponding held-out
match. Calibration examples and corruption seeds were separate from evaluation.
All external data are the same CC BY 4.0 IDSSE/DFL source already documented in the
technical report. The original four fitting games total 567,278 provider frames;
the validity experiment samples every 25th frame from seven corrupted/clean views
per half. These repeated views are not additional independent matches.
This produces 158,858 sampled validity examples across the four games before fold
partitioning; they are reused across folds, not 158,858 independent observations.

Ten conditions and five seeds yielded 200 match/condition/seed comparisons with four
fixed policies. Identical clean repeats reduce this to 184 distinct input views;
only **four independent match groups** are represented. This is reused training-match
CV with synthetic corruption, not independent confirmation or a target score. Full
settings: [protocol](PHASE9_PROTOCOL.md), [configuration](../configs/phase9.json).

## External results

Means below give each match and seed equal weight within each condition. The overall
mean weights the ten conditions equally; it is a different endpoint from Phase 6's
clean/moderate/severe mean on two final games and must not be compared directly.

| Condition | v1 spellF1 | Challenger | Difference |
|---|---:|---:|---:|
| Clean | 0.5855 | 0.5934 | +0.0079 |
| Moderate | 0.5478 | 0.5492 | +0.0013 |
| Severe | 0.4722 | 0.4725 | +0.0003 |
| Original structured | 0.4626 | 0.4629 | +0.0003 |
| Sustained false ball | 0.4251 | 0.4499 | +0.0248 |
| Two-second outages | 0.5786 | 0.5855 | +0.0069 |
| Eight-second outages | 0.5072 | 0.5115 | +0.0043 |
| Twenty-second outages | 0.3992 | 0.4008 | +0.0016 |
| Independent camera crop | 0.3735 | 0.3796 | +0.0061 |
| Ball-centred camera crop | 0.4843 | 0.4938 | +0.0095 |
| **Overall** | **0.4836** | **0.4899** | **+0.0063** |

Per-match overall gains were +0.0050, +0.0019, +0.0106 and +0.0076. Pooled long-spell
recall rose from 80.91% to 81.26%; short-spell recall slightly fell from 17.06% to
16.94%. False predicted D runs inside true possessions averaged 157.35 versus
157.59 per match/condition/seed cell. These D-run errors include edge overlaps and
are not all internal fragmentation events. The challenger does not directly repair
D emissions or coordinates.

Without the motion branch, the respective overall scores were 0.4650 and 0.4670.
The positive reliability gain on independent crops means the measured improvement
does not require a ball-following crop. Both camera tests remain idealized visibility
simulations. Their differing absolute scores show that the simulated visibility
mechanism materially affects difficulty; they do not establish actual camera transfer.

All five IoU-threshold scores, short/long recall, false-D and restart/turnover timing
diagnostics are retained in the [aggregate results](phase9_results.json). Repeated
support counts in that file include the same reference possessions across conditions
and seeds; they are not independent sample sizes. Timing is a same-pair, nearest-first
25-frame diagnostic, distinct from the official transition tie-break.

## Verification and execution repair

Both full frozen suites passed **92 tests**. Synthetic tests cover deterministic
input-only corruption, independent crop masks, sustained errors, validity-label
unknowns, slot invariance, fitting/calibration isolation, checkpoint probability
round trips, official scoring, adoption guards and read-only worker verification.

The initial parallel scheduler had a shared-protocol temporary-file race. Two folds
finished and two remained incomplete. The repaired scheduler gives protocol writes
to the parent only. Old/new scheduler hashes and local execution records are retained;
frozen model/run sources, settings and selection gates did not change. The canonical
runner verified cached metric equality, reloaded validity checkpoints against feature
probes and completed the prescribed comparisons. A separate audit verified all saved
artifact/source hashes, independently reran official spell scoring at all five IoUs,
recomputed aggregate arithmetic, and exactly replayed historical clean v1 fold scores.
These checks passed. They do not constitute a second independent dataset evaluation.

Predictions, detailed arrays, checkpoints and raw/prepared external inputs remain
local. Only code, protocol and aggregate external evidence are published. Neither
competition inputs/labels/video nor the already-open final external matches entered
the experiment. Original v1 inference source, weights and candidate CSV are unchanged.

To reproduce the analysis locally after the prerequisite Phase 5 external preparation
and fold checkpoints:

```sh
uv run --frozen --extra temporal python scripts/run_phase9.py
uv run --frozen --extra temporal python scripts/audit_phase9.py
uv run --frozen python scripts/publish_phase9_summary.py
```

The canonical runner can execute the complete experiment serially; the parallel
scheduler is optional and requires prepared training caches. When parallel execution
records are present, the audit additionally verifies their scheduler hashes. These research
commands are separate from the public released-weight inference path and do not
regenerate or replace the upload CSV.

## Updated decision

This experiment supports a modest targeted reliability benefit, while failing the
declared adoption requirement. Do not introduce a new boundary/stop model or lower
the threshold to rescue this prototype. Preserve v1 and its existing independent
external-test/reproduction evidence. Phase 10 should close delivery with the selected
v1 artifacts, registration/eligibility clarification and final file review. A larger
research programme can revisit the prototype with new independent data after the
competition; it is not a reason to delay the validated submission. Phase 10 still
requires the participant's next instruction. No organizer message or form submission
was sent during Phase 9.
