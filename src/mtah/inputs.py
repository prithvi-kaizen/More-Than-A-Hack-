"""Streaming ATD parsing with generic errors safe to expose to the controller."""
from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET


class InputError(ValueError):
    pass


@dataclass(frozen=True)
class TrackingFrame:
    video_frame: int
    players: tuple[tuple[float, float, float], ...]
    ball: tuple[float, float, float]


def video_frame_of(frame_id: int) -> int:
    if not isinstance(frame_id, int) or isinstance(frame_id, bool) or frame_id < 1:
        raise InputError('INVALID_FRAME_ID')
    return frame_id - 1


def _observation(value: str) -> tuple[float, float, float]:
    try:
        fields = tuple(float(x) for x in value.split(','))
        if len(fields) != 3 or any(math.isinf(x) for x in fields):
            raise ValueError
        return fields
    except (ValueError, OverflowError):
        raise InputError('INVALID_OBSERVATION') from None


def iter_tracking(path: Path, slots: int, expected_frames: int):
    count = 0
    try:
        with path.open() as stream:
            for count, line in enumerate(stream, 1):
                parts = line.strip().split(':')
                if len(parts) != 3:
                    raise InputError('INVALID_RECORD_SHAPE')
                try:
                    index = int(parts[0])
                except ValueError:
                    raise InputError('INVALID_FRAME_ID') from None
                if index != count or count > expected_frames:
                    raise InputError('NONCONTIGUOUS_FRAME_IDS')
                players = tuple(_observation(v) for v in parts[1].split(';'))
                if len(players) != slots:
                    raise InputError('INVALID_SLOT_COUNT')
                yield TrackingFrame(video_frame_of(index), players, _observation(parts[2]))
        if count != expected_frames:
            raise InputError('FRAME_COUNT_MISMATCH')
    except (OSError, UnicodeError):
        raise InputError('UNREADABLE_TRACKING') from None


def team_membership(path: Path) -> tuple[str, ...]:
    try:
        root = ET.parse(path).getroot()
        teams = list(root.iter('Team'))
        ids = [t.get('id') for t in teams]
        players = list(root.iter('Player'))
        if len(ids) != 2 or None in ids or len(set(ids)) != 2 or not players:
            raise InputError('INVALID_METADATA')
        if any(p.get('teamId') not in ids for p in players):
            raise InputError('UNKNOWN_TEAM')
        return tuple('H' if p.get('teamId') == ids[0] else 'A' for p in players)
    except (OSError, ET.ParseError):
        raise InputError('UNREADABLE_METADATA') from None


def position_metres(observation, pitch=(105.3, 68.0)):
    # Out-of-bounds positions are preserved as potential dead-ball evidence.
    return observation[0] * pitch[0], observation[1] * pitch[1]
