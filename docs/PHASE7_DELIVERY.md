# Phase 7: Kaizen delivery

The registered team name supplied by the participant is **Kaizen**. The completed
private upload package contains `Kaizen.csv` and `Kaizen.md`, plus local instructions
and an integrity manifest, under `outputs/phase7/upload/`. The CSV is a byte-preserving
copy of the validated Phase 6 candidate. The Markdown document is generated from
the public [technical report](TECHNICAL_REPORT.md) with the team name attached.
The exact packaged CSV passed both pinned validation interfaces. No labels changed.

## Public reproduction assets

* Source: [`inference-v1`](https://github.com/prithvi-kaizen/More-Than-A-Hack-/tree/inference-v1),
  commit `da7343eae56abe2f2b72fb88da19fbad5e14993d`.
* Checkpoints: [`models-v1`](https://github.com/prithvi-kaizen/More-Than-A-Hack-/releases/tag/models-v1).
  The archive contains exactly three external-trained model files and two attribution
  documents. Archive and member hashes are pinned in `configs/checkpoint_release.json`.
* Method: [technical report](TECHNICAL_REPORT.md), [model card](MODEL_CARD.md),
  [checkpoint licence](MODEL_LICENSE.md), [reproduction guide](REPRODUCTION.md).

The checkpoints are released under CC BY 4.0 with DFL/Bassek et al. attribution.
No competition input, prediction, feature cache or validator detail is included in
the public archive. Its unauthenticated download was verified against the pinned hash.

## Completed verification

All 80 synthetic tests passed, including archive/member integrity, complete file-set
validation, conflicting destinations and team-name handling. Changed, extra, missing,
duplicate and linked archive members are rejected before model installation.

A fresh anonymous clone of the public source tag started without data, model files,
local reports or a virtual environment. It installed a separate locked environment,
downloaded the public checkpoint release without credentials, and generated predictions
using only the original local tactical bundle. CSV bytes, input/freeze provenance and
official validation matched the original candidate exactly. No training data or prior
feature cache was copied into that checkout.

The recorded environment is macOS arm64, Python 3.12.4, NumPy 2.2.6, scikit-learn 1.7.2
and PyTorch 2.8.0. Linux CI checks synthetic cases; cross-platform target-data byte
identity has not been established. Final documentation updates leave the tested
inference source, model weights and Phase 6 frozen records unchanged.

The public check summary is [phase7_delivery.json](phase7_delivery.json). Target hashes,
private reports, prediction files and the clean checkout remain in ignored directories.

## Handoff

The team's local instructions include the organizer's submission link and the cutoff
from the submission email. The participant should review the technical document and
upload the two named files, supplying the public repository URL. Form submission has
not been performed. No further fitting, selection or prediction changes are planned.
