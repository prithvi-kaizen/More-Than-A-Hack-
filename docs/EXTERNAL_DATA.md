# External dataset contract

IDSSE / Sportec is the primary external source. The publisher catalogue is
https://doi.org/10.6084/m9.figshare.28196177; the official project is
https://github.com/spoho-datascience/idsse-data. Catalogue licensing is CC BY 4.0.
Credit Deutsche Fußball Liga (DFL) and Bassek, Rein, Weber and Memmert (2025),
*An integrated dataset of spatiotemporal and event data in elite soccer*,
Scientific Data 12, 195, https://doi.org/10.1038/s41597-025-04505-y.

`configs/external_split.json` freezes 21 publisher file names, sizes, download URLs
and MD5 checksums before any model fitting or evaluation-driven selection:

| Split | Match IDs |
|---|---|
| Training | J03WMX, J03WN1, J03WOH, J03WOY |
| Development | J03WPY |
| Final holdout | J03WQQ, J03WR9 |

All windows and both halves of a match belong to the same split. Development may
select hyperparameters; final holdout scores remain sealed until the selected model
is frozen. Reusing a published checkpoint requires reconciling its training history
with these splits first. The manifest guard rejects target match IDs, duplicate
assignments and silent repartitioning.

## Acquisition and preparation

```sh
uv run --frozen python scripts/download_external.py --mirror --small-only
uv run --frozen python scripts/download_external.py --mirror --match J03WMX --workers 24
uv run --frozen python scripts/prepare_external.py --match J03WMX
uv run --frozen python scripts/check_external_pipeline.py --match J03WMX
```

Without `--match`, download/preparation covers all seven catalogue matches. The
mirror is pinned to Hugging Face revision `a715a38` of `pysport/idsse-data`, but
the original publisher's full-file MD5 and byte count remain the acceptance gate.
Parallel range transfers can resume completed chunks; files become usable only
after assembly and whole-file verification. Remove failed partial chunks and retry
if a full-file integrity check fails. The default host is Figshare; `--mirror`
selects the faster observed route. No integrity check or TLS verification is disabled.

Raw tracking totals approximately 2.62 GB before preparation. Check disk space for
download parts, assembly and derived arrays. Downloads run explicitly, not in CI.
All raw files, prepared arrays and detailed audits are under ignored directories.
CI uses synthetic fixtures and requires no dataset credentials or downloads.

## Label mapping and masks

The mapping follows the Sportec schema, also documented by the PySport/kloppy
[deserializer](https://github.com/PySport/kloppy/blob/master/kloppy/infra/serializers/tracking/sportec/deserializer.py).

| BallStatus | BallPossession | Ownership label | Alive supervision |
|---|---|---|---|
| 0 | any | D | 0 |
| 1 | 1 | H | 1 |
| 1 | 2 | A | 1 |
| 1 | missing/other | X (masked) | 1 |
| missing/other | any | X (masked) | -1 (masked) |

Dead status overrides ownership. Missing ownership is never silently mapped to A.
Event-record gaps do not supply dead-ball labels. No future-owner backfill or target
public labels are used. These are provider-derived supervision labels, not newly
hand-labelled data and not demonstrated equivalents of the challenge's controlled
possession definition. In particular, contested-ball and short-touch conventions
may differ. External scores measure recovery of this provider convention.

The adapter uses explicit home/guest roles rather than XML ordering. Coordinates
are centred metres in the raw Sportec file; shift by half the metadata pitch
dimensions to obtain a fixed pitch origin. Player identities are external metadata
IDs, not competition track slots. Do not use provider speed fields until their
units are verified; the initial adapter stores x/y only. No dynamic attack-direction
rotation is applied. No labels enter coordinate normalization.

Frame IDs remain intact inside each half. The union of valid player and ball ranges
defines a dense half timeline; missing records remain NaN/X. Halves are parsed and
corrupted separately. Scoring inserts an ungraded separator so same-team spells can
never merge across halftime. Duplicate observations fail rather than overwrite data.

Prepared inputs and labels are separate NPZ files. Model inputs contain player x/y,
ball x/y, team membership, frame IDs and FPS. Labels and alive status stay in separate
files. Provenance records hashes of raw and prepared files. A future training loader
must enforce these hashes and split membership; the Phase 2 diagnostic already
checks prepared integrity and refuses development/holdout matches.

## Stress testing and diagnostic comparator

`corrupt_inputs` accepts geometry and team membership only, copies inputs, and uses
a local seeded generator. It adds contiguous ball/player gaps, positional noise and
same-team fragment swaps. Labels are never passed to the transform. This first
corruption suite is a reproducible starting point, not a claim to match the target
noise distribution. Crowd/boundary-conditioned missingness, false ball detections,
duplicate tracks and coarse-tag noise are further ablations for later phases.

`check_external_pipeline.py` runs an untrained nearest-player comparator, then repeats
it on moderate and severe corrupted training inputs, with fixed seeds. It records exact official macroF1/spellF1 locally.
Its purpose is pipeline verification; no model or threshold is selected from it.
An oracle equal to the provider labels scores perfectly by construction and is not
evidence of useful model accuracy.
