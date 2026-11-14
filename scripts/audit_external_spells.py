"""Training-only event corroboration audit; never edits provider labels."""
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah.external import framesets, manifest, SECTIONS
from mtah.prepared import verify_prepared, read_half
from mtah.official import metrics as pm


def event_evidence(event, team_map):
    values = []
    for node in event.iter():
        team = node.get('Team')
        category = None
        if node.tag == 'Play': category = 'intentional_play'
        elif node.tag == 'BallClaiming': category = 'claim_or_interception'
        elif node.tag == 'ShotAtGoal': category = 'shot'
        elif node.tag == 'OtherBallAction': category = 'other_touch'
        elif node.tag == 'TacklingGame':
            team = node.get('WinnerTeam')
            if node.get('Type') == 'air': category = 'aerial_duel'
            elif node.get('PossessionChange') == 'true': category = 'ground_turnover'
        if category and team in team_map:
            values.append((category, team_map[team]))
    return values


def main():
    config = manifest(ROOT / 'configs/external_split.json')
    report = {}
    for match in config['split']['train']:
        folder, _, _ = verify_prepared(ROOT, config, match, 'train')
        entries = {e['kind']: e for e in config['files'] if e['match_id'] == match}
        raw = ROOT / 'external_data/idsse/raw'
        metadata = ET.parse(raw / entries['metadata']['name']).getroot()
        team_map = {t.get('TeamId'): 'H' if t.get('Role') == 'home' else 'A'
                    for t in metadata.findall('.//Teams/Team')}
        positions = raw / entries['positions']['name']
        if not positions.exists(): positions = positions.with_suffix('.xml.gz')
        anchors = {}
        for fs in framesets(positions):
            if fs.get('TeamId') != 'BALL': continue
            frames = fs.findall('Frame')
            first = frames[0]
            n0, t0 = int(first.get('N')), datetime.fromisoformat(first.get('T'))
            for frame in frames:
                expected = (datetime.fromisoformat(frame.get('T'))-t0).total_seconds()*25
                if abs(expected-(int(frame.get('N'))-n0)) > 0.01:
                    raise ValueError('EXTERNAL_TIMEBASE_NONLINEAR')
            anchors[SECTIONS[fs.get('GameSection')]] = (n0, int(frames[-1].get('N')), t0)
        events = defaultdict(list)
        outside = 0
        for event in ET.parse(raw / entries['events']['name']).iter('Event'):
            when = datetime.fromisoformat(event.get('EventTime'))
            matched = False
            for half, (first, last, start) in anchors.items():
                frame = first+round((when-start).total_seconds()*25)
                if first <= frame <= last:
                    for category, team in event_evidence(event, team_map):
                        events[half].append((frame, category, team))
                    matched = True
                    break
            outside += not matched
        bins = defaultdict(Counter)
        for half in (1, 2):
            inputs, labels, _ = read_half(folder, half)
            codes = pm.label_codes(labels)
            for a, b, code in pm.possession_spells(codes, pm.valid_runs(codes)):
                key = 'under_2s' if b-a < 50 else '2_to_5s' if b-a < 125 else 'at_least_5s'
                stats = bins[key]; stats['spells'] += 1
                side = pm.LABELS[code]
                start, stop = int(inputs['frame_ids'][a]), int(inputs['frame_ids'][b-1])
                strict = {cat for f, cat, team in events[half] if team == side and start <= f <= stop}
                nearby = {cat for f, cat, team in events[half] if team == side and start-12 <= f <= stop+12}
                stats['with_any_same_team_event_inside'] += bool(strict)
                stats['with_intentional_play_inside'] += 'intentional_play' in strict
                stats['with_control_support_inside'] += bool(strict & {'intentional_play', 'claim_or_interception', 'ground_turnover'})
                stats['with_control_support_nearby'] += bool(nearby & {'intentional_play', 'claim_or_interception', 'ground_turnover'})
                stats['aerial_or_other_touch_inside'] += bool(strict & {'aerial_duel', 'other_touch'})
        report[match] = dict(duration_bins={k: dict(v) for k, v in bins.items()}, events_outside_periods=outside)
        print(match+'_external_event_audit: PASS', flush=True)
    report['limitations'] = ('Event times are approximate; temporal overlap is corroboration, not proof of controlled possession. '
        'Missing events are not negative evidence. No labels were changed. No development or holdout events were inspected.')
    out = ROOT / 'outputs/phase4'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'event_audit.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__': main()
