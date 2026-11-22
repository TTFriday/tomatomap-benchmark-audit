"""Collect per-run annotation cross-evaluations into a 2x2 table. Missing weights/split fields use historical best.pt/val defaults, marked code_default. Ordered basename SHA1 checks list identity within a run, not image-content identity."""
import argparse
import glob
import json
import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
from tm_config import load_config, read_image_list, relative_path, OUTPUT_TABLES
CFG = load_config()
WORK = CFG['work_root']
OUT = str(OUTPUT_TABLES / 'T14_xcheck2x2.csv')
COLS = ['trained_labels', 'eval_labels', 'proto_trained', 'run', 'epochs', 'imgsz', 'weights', 'split', 'caliber_src', 'n_eval_img', 'val_list_digest', 'P', 'R', 'mAP50', 'mAP50_95', 'mAP50_vs_own_diag', 'per_class']

def trained_labels_of(proto):
    return 'broken' if proto.startswith('broken__') else 'fixed'

def eval_labels_of(label_set):
    return 'broken' if label_set.startswith('broken') else 'fixed'

def n_lines(path):
    try:
        with open(path, encoding='utf-8') as f:
            return sum((1 for l in f if l.strip()))
    except Exception:
        return None

def val_digest(yaml_path):
    import hashlib
    try:
        vp = None
        for line in open(yaml_path, encoding='utf-8'):
            if line.split(':')[0].strip() == 'val':
                vp = line.split(':', 1)[1].strip().strip('\'"')
                break
        if vp is None:
            return (None, None)
        if not os.path.isabs(vp):
            vp = os.path.join(os.path.dirname(yaml_path), vp)
        names = [os.path.basename(l.strip()) for l in open(vp, encoding='utf-8') if l.strip()]
        if not names:
            return (None, 0)
        return (hashlib.sha1('\n'.join(names).encode('utf-8')).hexdigest()[:16], len(names))
    except Exception:
        return (None, None)

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(os.path.join(WORK, 'runs', '*', '*', 'xcheck.json')))
    if not files:
        sys.exit('No xcheck.json under local runs; first run tm_xcheck.py')
    rows = []
    for p in files:
        run_dir = os.path.dirname(p)
        man_path = os.path.join(run_dir, 'manifest.json')
        if not os.path.exists(man_path):
            print('Missing manifest.json; skip %s' % run_dir)
            continue
        man = json.load(open(man_path, encoding='utf-8'))
        proto = man['protocol']
        base = proto[len('broken__'):] if proto.startswith('broken__') else proto
        for r in json.load(open(p, encoding='utf-8')):
            (w, s) = (r.get('weights'), r.get('split'))
            el = eval_labels_of(r['label_set'])
            yaml_name = 'tm_broken__%s.yaml' % base if el == 'broken' else 'tm_%s.yaml' % base
            (dig, nev) = val_digest(os.path.join(WORK, yaml_name))
            rows.append(dict(trained_labels=trained_labels_of(proto), eval_labels=el, proto_trained=proto, run=os.path.basename(run_dir), epochs=man.get('epochs'), imgsz=r.get('imgsz', man.get('imgsz')), weights=w or 'best.pt', split=s or 'val', caliber_src='json' if w and s else 'code_default', n_eval_img=nev, val_list_digest=dig, P=r['P'], R=r['R'], mAP50=r['mAP50'], mAP50_95=r['mAP50_95'], per_class=json.dumps(r.get('per_class', {}), ensure_ascii=False, sort_keys=True)))
    diag = {(x['run'], x['trained_labels']): x['mAP50'] for x in rows if x['eval_labels'] == x['trained_labels']}
    for x in rows:
        d = diag.get((x['run'], x['trained_labels']))
        x['mAP50_vs_own_diag'] = round(1.0 - x['mAP50'] / d, 6) if d else None
    import csv
    with open(OUT, 'w', encoding='utf-8', newline='') as f:
        wcsv = csv.DictWriter(f, fieldnames=COLS)
        wcsv.writeheader()
        for x in sorted(rows, key=lambda y: (y['trained_labels'], y['proto_trained'], y['eval_labels'])):
            wcsv.writerow({k: x[k] for k in COLS})
    print('-> %s (%d rows / %d runs)' % (os.path.relpath(OUT, ROOT), len(rows), len(files)))
    for x in sorted(rows, key=lambda y: (str(y['proto_trained']), str(y['epochs']), y['eval_labels'])):
        print('  %-20s e%-4s trained=%-6s eval=%-6s %-8s %-3s n=%-5s mAP50=%.4f relative_difference=%s digest=%s' % (x['proto_trained'], x['epochs'], x['trained_labels'], x['eval_labels'], x['weights'], x['split'], x['n_eval_img'], x['mAP50'], '%.2f%%' % (100 * x['mAP50_vs_own_diag']) if x['mAP50_vs_own_diag'] is not None else '—', x['val_list_digest']))
    import collections
    byrun = collections.defaultdict(list)
    for x in rows:
        byrun[x['run']].append(x)
    bad = []
    for (run, xs) in sorted(byrun.items()):
        dg = {x['val_list_digest'] for x in xs if x['val_list_digest']}
        if len(dg) > 1:
            bad.append('%s：%s' % (run, sorted(dg)))
        elif len(dg) == 0:
            bad.append('%s: val list unavailable for both cells' % run)
    if bad:
        print('[Warning] Within-run list digests disagree: ' + '；'.join(bad))
        print('    Cells cannot form a same-image 2x2 row; inspect list membership.')
    else:
        print('[Check] Each run has identical ordered names for repaired/raw evaluation cells.')
    n_cell = sum((1 for x in rows if x['trained_labels'] != x['eval_labels']))
    print('[Completeness] Cross-evaluation cells=%d; a full 2x2 also requires broken-trained runs.' % n_cell)
if __name__ == '__main__':
    main()
