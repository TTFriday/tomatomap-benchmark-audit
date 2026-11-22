"""Audit split assignment separately from actual camera-specific training."""
from pathlib import Path
import argparse
import csv
ROOT = Path(__file__).resolve().parents[1]
CAMS = ['cam1_bottom45', 'cam2_horizontal90', 'cam3_tilttop135', 'cam4_vertical180']

def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def save(path, values):
    if not values:
        raise ValueError('No rows to write')
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader()
        w.writerows(values)
    print(Path(path).name, len(values), flush=True)

def audit(work, output):
    index = {r['name']: r for r in rows(work / 'index.csv')}
    out = []
    for file in sorted((work / 'splits').glob('*.csv')):
        rs = rows(file)
        tr_all = [r for r in rs if r['split'] == 'train']
        tr_used = [r for r in tr_all if r['cam'] == CAMS[1]]
        for scope, train in [('all_camera_assignment', tr_all), ('actual_cam2_training', tr_used)]:
            pl = {r['plant_id'] for r in train}
            se = {r['image_set_id'] for r in train}
            for cam in [CAMS[1]]:
                te = [r for r in rs if r['split'] == 'test' and r['cam'] == cam]
                n = len(te)
                out.append(dict(protocol=file.stem, scope=scope, test_camera=cam,
                                n_train_images=len(train), n_test_images=n,
                                n_test_plants=len({r['plant_id'] for r in te}),
                                pct_test_plant_in_train=100 * sum(r['plant_id'] in pl for r in te) / n,
                                pct_test_set_in_train=100 * sum(r['image_set_id'] in se for r in te) / n))
    save(output / 'T15_training_scope.csv', out)
    for proto in ['per_set', 'per_plant']:
        rs = rows(work / 'splits' / (proto + '.csv'))
        sets = [{r['image_set_id'] for r in rs if r['split'] == 'test' and r['cam'] == cam} for cam in CAMS]
        assert all(s == sets[0] for s in sets), proto
    actual = [r for r in out if r['scope'] == 'actual_cam2_training']
    assert all(r['pct_test_set_in_train'] == 0 for r in actual)
    print('Cam2 test images have no simultaneous sibling in actual training.', flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=ROOT / "work" / "tm_yolo")
    p.add_argument("--output", type=Path, default=ROOT / "work" / "tm_yolo" / "tables")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    audit(a.work, a.output)
