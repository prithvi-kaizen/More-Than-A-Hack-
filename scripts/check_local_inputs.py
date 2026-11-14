"""Exercise the pipeline parser locally and print fixed status messages only."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah.inputs import iter_tracking, team_membership


def main():
    for source, expected in [('tactical', 199319), ('broadcast', 164488)]:
        base = ROOT / 'data' / source / ('BAR-ATH_' + source)
        try:
            teams = team_membership(Path(str(base) + '_atd.xml'))
            for _ in iter_tracking(Path(str(base) + '_atd.txt'), len(teams), expected):
                pass
        except Exception:
            print(source + '_pipeline_parser: FAIL')
            return 1
        print(source + '_pipeline_parser: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
