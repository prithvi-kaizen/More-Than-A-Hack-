# Reproduce the submission

The `inference-v1` tag pins the source used for the clean-checkout test. Model weights
are separate, immutable-by-hash `models-v1` release assets. These files were fitted
only on the four licensed external training games; no competition data or predictions
are distributed. See [model card](MODEL_CARD.md), [checkpoint licence](MODEL_LICENSE.md)
and [technical report](TECHNICAL_REPORT.md).

## Inference from released weights

Requirements: Git, [uv](https://docs.astral.sh/uv/), curl, Python 3.12 and the organizer's
locally supplied tactical metadata/tracking. uv can locate/install the pinned Python
version. The `temporal` extra supplies the locked CPU/PyTorch environment. No training
dataset, hosted inference service or GitHub credential is required for this path.

```sh
git clone --branch inference-v1 --single-branch https://github.com/prithvi-kaizen/More-Than-A-Hack-.git
cd More-Than-A-Hack-
uv sync --frozen --extra temporal
uv run --frozen --extra temporal python scripts/download_checkpoints.py
LOKY_MAX_CPU_COUNT=2 uv run --frozen --extra temporal python scripts/infer_target.py \
  --data-dir /absolute/path/to/organizer-bundle
```

The data directory must contain:

```text
tactical/BAR-ATH_tactical_atd.xml
tactical/BAR-ATH_tactical_atd.txt
```

The full-length candidate is `outputs/phase6/submission.csv`. The CLI verifies the
frozen checkpoint/configuration/source/dependency hashes, rereads the local tracking,
generates predictions and runs both original validation interfaces. Detailed validator
output and input/prediction hashes remain in ignored local files. CLI output is
status-only. `MTAH_DATA_DIR` is an alternative to `--data-dir`.

The download script verifies the exact compressed archive SHA-256, every member,
the complete allowlisted file set and each installed checkpoint. It rejects symlinks,
extra files, changed bytes and conflicting existing models before installing anything.
All three checkpoints are required by the frozen guard, including the earlier Phase 3
comparator. They total about 1.3 MB before compression. This path does not retrain.

## Validation and local upload package

```sh
uv run --frozen --extra temporal python -m unittest discover -s tests -v
LOKY_MAX_CPU_COUNT=2 uv run --frozen --extra temporal python scripts/audit_target_replay.py
uv run --frozen --extra temporal python scripts/package_submission.py --team-name Kaizen
```

The team-name argument must be the exact registered name. It creates the team-named
CSV and Markdown technical document under `outputs/phase7/upload/`, verifies that
the CSV bytes match the validated candidate, and validates the exact packaged file.
Do not upload the manifest or local checkpoints as the possession file. Use the
organizer's emailed form to upload the CSV/document and supply the public repository
URL. The scripts do not submit that form.

On current `main`, repackaging preserves any existing `UPLOAD_INSTRUCTIONS.md`
byte for byte, including locally added form links, deadlines and registration notes.
It refreshes the technical document and integrity manifest while preserving candidate
CSV bytes. A different team name or an existing manifest marked submitted is rejected
before package writes. Review those cases explicitly rather than overwriting an
existing team's package. This hardening is newer than the `inference-v1` tooling
snapshot; use current `main` for packaging and the pinned tag for inference replay.

To test a genuinely empty checkout from an existing working copy with a validated
reference CSV, run:

```sh
LOKY_MAX_CPU_COUNT=2 uv run --frozen --extra temporal python scripts/audit_clean_checkout.py
```

This clones the public tag anonymously into a new ignored directory, asserts that
no data, weights, local reports or environment exist, installs a separate locked
environment, downloads the public checkpoint release and runs inference against
the local organizer bundle. It compares CSV bytes and provenance with the original.
It refuses an existing destination directory. This test may need additional disk
space for a second environment. It makes no prediction-quality comparison.

## Training and evaluation history

Source, fixed settings and split/checksum metadata for fitting are retained in the
repository. Dataset acquisition and model selection are documented in the Phase 3–6
reports. Use released weights to reproduce the submitted file; retraining on a
different machine can yield a different checkpoint hash and is not silently accepted
as the submitted model. Final holdout scores must not be used for further selection.

The current reproduction claim is limited to the recorded operating system,
architecture and locked software environment. The synthetic CI suite runs on Linux;
that does not establish cross-platform byte identity on the competition game.

The public `inference-v1` checkout test passed on macOS arm64 with Python 3.12.4,
NumPy 2.2.6, scikit-learn 1.7.2 and PyTorch 2.8.0. It installed weights from the
public release in a separate environment and generated identical CSV bytes and
provenance. See [delivery evidence](PHASE7_DELIVERY.md). The final Kaizen upload
document includes this completed check; the tag retains the original tooling snapshot.

Phase 8 rechecked local Docker availability. The Linux ARM daemon answered its info
request, but image enumeration failed with a container storage I/O error, preventing
a container reproduction run. Linux target byte identity remains unverified. No
runtime changes or Docker storage repair were performed. See [Phase 8](PHASE8_HARDENING.md).
