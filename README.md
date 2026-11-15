# More Than a Hack: possession inference

Local implementation for the Metrica Sports More Than a Hack 2027 possession challenge.
The frozen player-motion fusion model has been tested on two final external games
and generates a validated local competition CSV. Externally trained checkpoints,
technical documentation and reproduction tools are now available.

## Setup and verification

Install [uv](https://docs.astral.sh/uv/) and run:

```sh
uv sync --frozen --extra temporal
uv run --frozen --extra temporal python -m unittest discover -s tests -v
```

Python 3.12 is required. `uv.lock` pins all dependencies. Tests use synthetic data and
do not require competition inputs, public labels or a network connection after setup.
CI runs the same synthetic checks on Linux.
The temporal extra installs PyTorch; the earlier tree-only system can use the core
environment without that extra.

The [external-data contract](docs/EXTERNAL_DATA.md) documents the frozen match splits,
licensed downloads, label masks and reproducible corruption checks implemented in Phase 2.

The [Phase 3 model](docs/PHASE3_MODEL.md) documents the first trained control/dead-ball
emissions, temporal decoder and external development-selection procedure.

Mean external development spellF1 is 0.5581 across three input scenarios, compared
with 0.5373 for Phase 4, 0.4926 for Phase 3 and 0.2379 for official smooth50. These
selection-set results use IDSSE provider labels and do not estimate a competition score. Brief possessions
and unreliable geometry remain limitations. See the [Phase 5 method](docs/PHASE5_MODEL.md)
for the four-match comparison, robustness guards and reproduction commands. Motion
fusion improved every training-match fold. The [selected manifest](configs/phase5_selected.json)
identifies the local checkpoints and source hashes.

Final external holdout mean spellF1 is **0.5901**, compared with 0.5607 for Phase 4;
structured-noise spellF1 is 0.5309 versus 0.4554. The selected model was unchanged
after these tests. See [Phase 6](docs/PHASE6_INFERENCE.md) for exact conditions,
limitations, frozen artifacts and reproduction commands.

Download the public externally trained checkpoints, then generate the CSV with the
locally supplied tactical bundle:

```sh
uv run --frozen --extra temporal python scripts/download_checkpoints.py
LOKY_MAX_CPU_COUNT=2 uv run --frozen --extra temporal python scripts/infer_target.py
```

The candidate stays in `outputs/phase6/submission.csv`. Two independent full runs
produced identical CSV bytes and passed the pinned official validator. These checks
establish format and local reproduction; competition prediction accuracy is unmeasured.
See [reproduction instructions](docs/REPRODUCTION.md), the
[technical document](docs/TECHNICAL_REPORT.md), and the
[public model release](https://github.com/prithvi-kaizen/More-Than-A-Hack-/releases/tag/models-v1).
The upload package uses the team's exact registered name and stays local.
For **Kaizen**, the final upload files are `outputs/phase7/upload/Kaizen.csv` and
`outputs/phase7/upload/Kaizen.md`. A fresh anonymous public checkout, separate locked
environment and public weight download reproduced the candidate exactly on macOS arm64.
See the [delivery record](docs/PHASE7_DELIVERY.md). Form submission remains pending.

Place the separately licensed competition bundle in `data/tactical/` and
`data/broadcast/`, retaining the organizer filenames. To check local inputs:

```sh
uv run --frozen python scripts/audit_local_assets.py
uv run --frozen python scripts/check_local_inputs.py
```

These commands expose structural PASS/FAIL statuses only. Detailed audit measurements
stay in ignored `local_reports/`. The supplied bundle README is retained locally as
`data/BUNDLE_README.md`.

## Development

The [implementation plan](research/IMPLEMENTATION_PLAN.md) describes the model and
external-only validation protocol. [Phase results](docs/PHASES.md) record completed
work and the next authorized boundary. Work stops after each phase and GitHub update
until the user asks to continue.

`src/mtah/inputs.py` reads tracking without assuming that slots are persistent people.
`src/mtah/official.py` imports the organizer's metric directly. `vendor/mtah/` holds
selected official files pinned by commit and Git blob hashes in `UPSTREAM.json`.
The vendored baseline and loader command-line tools are reference tools: use our
status-only wrappers on restricted assets. No public ground-truth labels are bundled.

Before publishing, stage public files explicitly and run:

```sh
uv run --frozen python scripts/check_staged_files.py
```

The gate rejects restricted paths, symlinks, unexpected file types and oversized files.
Review staged source/document changes too: a filename gate cannot detect every data
extract embedded inside otherwise allowed text. Inputs, derived predictions, plots,
detailed logs and model inspection stay local. Codex authors code and launches local
processes; restricted records must never appear in its tool output.

## Attribution

Official challenge code/documentation: [Metrica Sports and FC Barcelona Innovation
Hub](https://github.com/metrica-sports/more-than-a-hack-2027), MIT; see
[upstream licence](vendor/mtah/LICENSE). Competition assets have separate terms and
are excluded from this repository. Local project code has no additional licence
grant at this stage.
