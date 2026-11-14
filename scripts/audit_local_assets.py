"""Audit restricted inputs locally, exposing only structural pass/fail results."""
from pathlib import Path
import csv
import json
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def audit():
    checks = {}
    details = {}
    for source, expected in [('tactical', 199319), ('broadcast', 164488)]:
        base = ROOT / 'data' / source / ('BAR-ATH_' + source)
        meta = ET.parse(str(base) + '_atd.xml').getroot()
        players = list(meta.iter('Player'))
        teams = list(meta.iter('Team'))
        team_ids = {t.get('id') for t in teams}
        checks[source + '_metadata'] = bool(players) and len(teams) == 2 and all(p.get('teamId') in team_ids for p in players)
        sequential = True
        shape_ok = True
        missing_ball = 0
        count = 0
        with open(str(base) + '_atd.txt') as stream:
            for count, line in enumerate(stream, 1):
                parts = line.strip().split(':')
                if len(parts) != 3:
                    shape_ok = False
                    continue
                sequential &= int(parts[0]) == count
                observations = parts[1].split(';')
                shape_ok &= len(observations) == len(players)
                for observation in observations + [parts[2]]:
                    fields = observation.split(',')
                    shape_ok &= len(fields) == 3
                    values = [float(x) for x in fields]
                    shape_ok &= all(not math.isinf(x) for x in values)
                missing_ball += any(not math.isfinite(float(x)) for x in parts[2].split(',')[:2])
        checks[source + '_tracking_index_and_shape'] = count == expected and sequential and shape_ok
        details[source] = {'tracking_rows': count, 'slots': len(players), 'missing_ball_rows': missing_ball}
        hom_ok = True
        count = 0
        with open(str(base) + '_homography.csv', newline='') as stream:
            for count, row in enumerate(csv.DictReader(stream), 1):
                hom_ok &= int(row['frame']) == count - 1
                valid = int(row['valid'])
                values = [float(row['h' + str(i) + str(j)]) for i in range(3) for j in range(3)]
                hom_ok &= valid in (0, 1)
                if valid:
                    hom_ok &= all(math.isfinite(v) for v in values)
        checks[source + '_homography_structure'] = hom_ok and count == expected
    tags = ET.parse(ROOT / 'data/tactical/BAR-ATH_tactical_smart_tagging.xml').getroot()
    instances = list(tags.iter('instance'))
    checks['tactical_tags_structure'] = bool(instances) and all(
        0 <= float(i.findtext('start')) <= float(i.findtext('end')) <= 199319 / 25
        and bool(i.findtext('code')) for i in instances)
    details['tag_instances'] = len(instances)
    details['video_present'] = {source: (ROOT / 'data' / source / ('BAR-ATH_' + source + '.mp4')).exists()
                                for source in ('tactical', 'broadcast')}
    (ROOT / 'local_reports').mkdir(exist_ok=True)
    (ROOT / 'local_reports/asset_audit.json').write_text(json.dumps({'checks': checks, 'details': details}, indent=2))
    for name, passed in checks.items():
        print(name + ': ' + ('PASS' if passed else 'FAIL'))
    print('Detailed measurements retained locally; no input records emitted.')
    return all(checks.values())


if __name__ == '__main__':
    try:
        success = audit()
    except Exception:
        # Do not surface exception strings: parsers can include restricted input text.
        print('Audit could not complete. Inspect parser diagnostics locally without exposing input records.')
        raise SystemExit(1)
    raise SystemExit(0 if success else 1)
