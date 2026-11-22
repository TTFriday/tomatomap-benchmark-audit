"""Construct reverse stage holdout from (plant,date) units. Sort once by stage, CRC32 salt, plant and date; reverse the same order with int(n*0.70)/int(n*0.90) cuts. Reverse test is in forward train; training units overlap by 40%. These protocols are not independent repetitions. Source data remain read-only."""
import argparse
import csv
import json
import os
import sys
import zlib
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
from tm_config import load_config, read_image_list, relative_path, OUTPUT_TABLES
CFG = load_config()
WORK = CFG['work_root']
IMG_DIR = os.path.join(WORK, 'images')
LBL_DIR = os.path.join(WORK, 'labels')
SPLIT_DIR = os.path.join(WORK, 'splits')
LIST_DIR = os.path.join(WORK, 'lists')
DST_BROKEN = os.path.join(WORK, 'tm_broken')
PROTO = 'stage_holdout_rev'
BASE = 'stage_holdout'
CAMS = ('cam1_bottom45', 'cam2_horizontal90', 'cam3_tilttop135', 'cam4_vertical180')
SEED = int(CFG['seed'])
RATIO = tuple((float(x) for x in CFG['ratio']))
assert SEED == 888 and RATIO == (0.7, 0.2, 0.1), 'config seed/ratio must match (888, 0.70/0.20/0.10): %r %r' % (SEED, RATIO)
CLASSES = [c for c in open(os.path.join(WORK, 'classes.txt'), encoding='utf-8').read().split('\n') if c.strip()]
assert len(CLASSES) == 7, CLASSES

def rd_csv(p):
    with open(p, encoding='utf-8') as f:
        return list(csv.DictReader(f))

def prior_of(names_of_cam):
    import numpy as np
    c = np.zeros(len(CLASSES))
    for (cd, ns) in names_of_cam.items():
        d = os.path.join(LBL_DIR, cd)
        for nm in ns:
            with open(os.path.join(d, nm[:-4] + '.txt'), encoding='utf-8') as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        c[int(ln.split()[0])] += 1
    return c / max(1.0, c.sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    base_rows = rd_csv(os.path.join(SPLIT_DIR, BASE + '.csv'))
    assert base_rows, BASE + '%s.csv is empty'
    print('[1] Read units from %s.csv (%d rows)' % (BASE, len(base_rows)))
    units = {}
    for r in base_rows:
        k = (r['plant_id'], r['date'])
        st = int(r['bbch_stage'])
        if k in units and units[k] != st:
            raise AssertionError('Unit %s spans stages %d and %d' % (k, units[k], st))
        units[k] = st
    print('    units=%d each has one BBCH stage; range=%d-%d' % (len(units), min(units.values()), max(units.values())))

    def key(k):
        salt = zlib.crc32(('%s|%s|%s' % (SEED, k[0], k[1])).encode()) % 1000003
        return (units[k], salt, str(k[0]), str(k[1]))
    uk = sorted(units, key=key)
    n = len(uk)
    (c1, c2) = (int(n * RATIO[0]), int(n * (RATIO[0] + RATIO[1])))
    sp_unit = {}
    for (i, k) in enumerate(reversed(uk)):
        sp_unit[k] = 'train' if i < c1 else 'val' if i < c2 else 'test'
    idx = rd_csv(os.path.join(WORK, 'index.csv'))
    sp = {}
    for r in idx:
        sp[r['name']] = sp_unit[r['plant_id'], r['date']]
    assert len(sp) == len(idx), (len(sp), len(idx))
    cnt = {s: sum((1 for r in idx if sp[r['name']] == s)) for s in ('train', 'val', 'test')}
    print('[2] Reverse assignments: %r (%d images)' % (cnt, len(idx)))
    assert sum(cnt.values()) == len(idx)
    by_set = {}
    for r in idx:
        by_set.setdefault(r['image_set_id'], []).append(r['name'])
    rev_test_units = {k for (k, s) in sp_unit.items() if s == 'test'}
    rev_tr_units = {k for (k, s) in sp_unit.items() if s == 'train'}
    unit_role_fwd = {}
    for r in base_rows:
        unit_role_fwd.setdefault((r['plant_id'], r['date']), set()).add(r['split'])
    for (k, v) in unit_role_fwd.items():
        assert len(v) == 1, 'Forward unit %s occurs in multiple splits: %s' % (k, v)
    unit_role_fwd = {k: next(iter(v)) for (k, v) in unit_role_fwd.items()}
    assert set(unit_role_fwd) == set(units), (len(unit_role_fwd), len(units))
    overlap = len(rev_tr_units & {k for (k, s) in unit_role_fwd.items() if s == 'train'}) / max(1.0, n)
    print('    Forward/reverse train overlap=%.1f%%; reverse test entirely in forward train: %s' % (100.0 * overlap, all((unit_role_fwd[k] == 'train' for k in rev_test_units))))
    assert all((unit_role_fwd[k] == 'train' for k in rev_test_units)), 'Reverse test must be contained in forward train (earliest 10%% vs latest 10%%).'
    if a.dry_run:
        print('[dry-run] No files written')
        return 0
    p = os.path.join(SPLIT_DIR, PROTO + '.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['name', 'cam', 'plant_id', 'image_set_id', 'date', 'bbch_stage', 'split'])
        for r in idx:
            wr.writerow([r['name'], r['camdir'], r['plant_id'], r['image_set_id'], r['date'], r['bbch_stage'], sp[r['name']]])
    print('[3] Wrote %s' % os.path.relpath(p, ROOT))
    n_list = 0
    for cd in CAMS:
        rows = [r for r in idx if r['camdir'] == cd]
        for s in ('train', 'val', 'test'):
            lines = [os.path.join(IMG_DIR, cd, r['name']).replace('\\', '/') for r in rows if sp[r['name']] == s]
            q = os.path.join(LIST_DIR, '%s__%s__%s.txt' % (PROTO, cd, s))
            open(q, 'w').write('\n'.join(lines) + ('\n' if lines else ''))
            n_list += 1
    print('[4] Wrote %d image lists' % n_list)
    names_yaml = '\n'.join(('  %d: %s' % (i, c) for (i, c) in enumerate(CLASSES)))
    y = ['path: ' + WORK.replace('\\', '/'), 'train: ' + (LIST_DIR + '/%s__cam2_horizontal90__train.txt' % PROTO).replace('\\', '/'), 'val: ' + (LIST_DIR + '/%s__cam2_horizontal90__val.txt' % PROTO).replace('\\', '/'), 'nc: %d' % len(CLASSES), 'test:']
    for cd in ('cam1_bottom45', 'cam3_tilttop135', 'cam4_vertical180'):
        y.append('  - ' + (LIST_DIR + '/%s__%s__test.txt' % (PROTO, cd)).replace('\\', '/'))
    y.append('names:')
    y.append(names_yaml)
    yp = os.path.join(WORK, 'tm_%s.yaml' % PROTO)
    open(yp, 'w').write('\n'.join(y) + '\n')
    import yaml
    with open(os.path.join(WORK, 'tm_%s.yaml' % BASE), encoding='utf-8') as stream:
        ref = yaml.safe_load(stream)
    with open(yp, encoding='utf-8') as stream:
        mine = yaml.safe_load(stream)
    same_shape = set(ref) == set(mine) and ref['nc'] == mine['nc'] and ref['names'] == mine['names']
    print('[5] Wrote %s (same structure as tm_%s.yaml: %s)' % (os.path.relpath(yp, ROOT), BASE, same_shape))
    assert same_shape, 'YAML structure differs from %s' % BASE
    if os.path.isdir(os.path.join(DST_BROKEN, 'labels', 'cam2_horizontal90')):
        bl = os.path.join(DST_BROKEN, 'lists')
        os.makedirs(bl, exist_ok=True)
        W = WORK.replace('\\', '/') + '/images/'
        D = DST_BROKEN.replace('\\', '/') + '/images/'
        for cd in CAMS:
            for s in ('train', 'val', 'test'):
                src = os.path.join(LIST_DIR, '%s__%s__%s.txt' % (PROTO, cd, s))
                lines = [ln.replace(W, D) for ln in open(src, encoding='utf-8').read().splitlines() if ln.strip()]
                open(os.path.join(bl, '%s__%s__%s.txt' % (PROTO, cd, s)), 'w').write('\n'.join(lines) + ('\n' if lines else ''))
        Db = DST_BROKEN.replace('\\', '/')
        yb = ['path: ' + Db, 'train: ' + '%s/lists/%s__cam2_horizontal90__train.txt' % (Db, PROTO), 'val: ' + '%s/lists/%s__cam2_horizontal90__val.txt' % (Db, PROTO), 'nc: %d' % len(CLASSES), 'test:']
        for cd in ('cam1_bottom45', 'cam3_tilttop135', 'cam4_vertical180'):
            yb.append('  - ' + '%s/lists/%s__%s__test.txt' % (Db, PROTO, cd))
        yb.append('names:')
        yb.append(names_yaml)
        open(os.path.join(WORK, 'tm_broken__%s.yaml' % PROTO), 'w').write('\n'.join(yb) + '\n')
        print('[6] Wrote raw-convention lists and tm_broken__%s.yaml' % PROTO)
    else:
        print('[6] Raw-convention label mirror unavailable; skip')
    tr = [r for r in idx if sp[r['name']] == 'train']
    te = [r for r in idx if sp[r['name']] == 'test']
    va = [r for r in idx if sp[r['name']] == 'val']
    tr_names = {r['name'] for r in tr}
    pl_tr = {r['plant_id'] for r in tr}
    se_tr = {r['image_set_id'] for r in tr}
    da_tr = {r['date'] for r in tr}
    st_tr = {int(r['bbch_stage']) for r in tr}
    pct_plant = 100.0 * sum((r['plant_id'] in pl_tr for r in te)) / max(1, len(te))
    pct_set = 100.0 * sum((r['image_set_id'] in se_tr for r in te)) / max(1, len(te))
    near = sum((any((q in tr_names for q in by_set[f['image_set_id']] if q != f['name'])) for f in te))
    shared_days = len(da_tr & {r['date'] for r in te})
    shared_bbch = len(st_tr & {int(r['bbch_stage']) for r in te})
    pct_novel = 100.0 * sum((int(r['bbch_stage']) not in st_tr for r in te)) / max(1, len(te))
    import numpy as np
    pa = prior_of(_by_cam(tr))
    pb = prior_of(_by_cam(te))
    tv = 0.5 * float(np.abs(pa - pb).sum())
    st_a = np.array([int(r['bbch_stage']) for r in tr], dtype=float)
    st_b = np.array([int(r['bbch_stage']) for r in te], dtype=float)
    q = np.linspace(0.05, 0.95, 19)
    w1 = float(np.mean(np.abs(np.quantile(st_a, q) - np.quantile(st_b, q))))
    row = [PROTO, len(tr), len(va), len(te), round(pct_plant, 1), round(pct_set, 1), round(100.0 * near / max(1, len(te)), 1), shared_days, shared_bbch, round(pct_novel, 1), round(tv, 4), round(w1, 2), '%.1f/%.1f' % (float(np.median(st_a)), float(np.median(st_b)))]
    lp = os.path.join(WORK, 'leakage.csv')
    old = [ln for ln in open(lp, encoding='utf-8').read().splitlines()]
    hdr = old[0].split(',')
    assert hdr[0] == 'protocol' and len(hdr) == 13, hdr
    body = [ln.split(',')[0] for ln in old[1:]]
    keep = [ln for ln in old[1:] if ln.split(',')[0] != PROTO]
    new = [old[0]] + keep + [','.join((str(x) for x in row))]
    assert all((k in new for k in keep)), 'Existing protocol rows must be preserved verbatim'
    with open(lp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new) + '\n')
    cur = open(lp, encoding='utf-8').read().splitlines()
    assert [ln for ln in cur if ln.split(',')[0] != PROTO] == [old[0]] + keep, 'Existing non-reverse rows must remain unchanged and in order'
    print('[7] Added leakage.csv row %s: %s' % (PROTO, row))
    print('    preserved rows=%d total rows=%d' % (len(keep), len(new) - 1))
    print('Next: python scripts/tm_train.py --protocol %s --model yolo11s --imgsz 800 --epochs 60 --batch 16' % PROTO)
    return 0

def _by_cam(rows):
    d = {}
    for r in rows:
        d.setdefault(r['camdir'], set()).add(r['name'])
    return d
if __name__ == '__main__':
    sys.exit(main())
