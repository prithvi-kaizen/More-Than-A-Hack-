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

## Phase 2 — pending user instruction

Prepare external training/development/holdout manifests and licensed data access;
implement provider adapters and input corruption; evaluate label availability and
semantics before training. Keep complete matches together. Verify dead-ball status
is actually available rather than inferring it from gaps in event logs.

Changes to the remaining plan from Phase 1: the local streaming parser works on both
provided tracking sources, so loader replacement is unnecessary. External label
quality and definition alignment are now the next dependency. Continue with the
classical structured pipeline before investing in a player-motion neural model.

Each phase ends with checks, plan updates and a GitHub push, followed by a stop.
The next phase begins only when the user asks to continue.
