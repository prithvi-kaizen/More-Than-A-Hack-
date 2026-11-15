# Frozen possession checkpoints

The `models-v1` release contains the exact checkpoints used by the frozen Phase 6
pipeline. They support offline football possession inference from team-tagged player
and ball coordinates at 25 Hz. Prediction uses H/A/D states; the target adapter adds
X outside the organizer's published half windows.

| Archive member | Purpose |
|---|---|
| `outputs/phase3/model.pkl` | Earlier histogram-boosted comparator; also required by the freeze guard |
| `outputs/phase4/model.pkl` | Selected alive/dead, ownership and adaptive-boundary tree heads |
| `outputs/phase5/motion.pt` | 19,169-parameter player-motion temporal expert |

Only four licensed external IDSSE games entered fitting: J03WMX, J03WN1, J03WOH and
J03WOY. Every window, half and corruption of an evaluation match was excluded from
its training fold. The final checkpoints use those four complete games. J03WPY was
used for development selection; J03WQQ/J03WR9 were opened after model/code freezes
for final tests. Their labels were not used to refit the released models.

Supervision is the Sportec/DFL provider's released BallStatus and BallPossession
fields, with unknown ownership/status masked appropriately. No new human labels
were created by this project. The temporal network was initialized and trained
from scratch. No competition-game input, label, video, public possession window or
human-produced event data about that game entered training or calibration.

The selected model obtained mean spellF1 0.5901 across two final external games and
clean/moderate/severe conditions, versus 0.5607 for Phase 4. Structured-noise spellF1
was 0.5309 versus 0.4554. These are same-provider external tests, not competition
accuracy estimates. Short spells and transfer from provider labels to a control-based
definition remain limitations. See the phase reports for complete selection history.

The archive contains only these checkpoints and this card/licence. It contains no
competition tracking, video, target predictions, prepared data or feature arrays.
Use the pinned SHA-256 hashes in `configs/checkpoint_release.json` and the frozen
records before loading the files. The tree checkpoints use Python pickle and should
be loaded only after successful verification of these published trusted artifacts.

Inference requires the locked Python 3.12 environment with the `temporal` extra.
Use CPU inference, two threads and the supplied CLI to reproduce the tested setup.
Models were fitted on the project's macOS host; exact reproduction on other operating
systems or CPU architectures must be checked separately. Checkpoint licence and
external-data attribution are in `MODEL_LICENSE.md` in this archive.
