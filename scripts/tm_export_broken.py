"""Export the raw-box convention comparison. Write raw [x,y,w,h] unchanged as YOLO centers; repaired labels use xc=x+w/2, yc=y+h/2. Reuse images, splits, class order and augmentation. The comparison concerns annotation conventions."""
import argparse
import csv
import json
import os
import shutil
import subprocess
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
DST = os.path.join(WORK, 'tm_broken')
SRC = CFG['src_root']
CAMS = ['cam1_bottom45', 'cam2_horizontal90', 'cam3_tilttop135', 'cam4_vertical180']
PROTOS = ['random_img', 'per_set', 'per_plant', 'temporal_fwd', 'temporal_rev', 'stage_strat', 'stage_holdout']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--copy-images', action='store_true', help='Copy local images instead of linking directories')
    args = parser.parse_args()
    samples = json.load(open(os.path.join(SRC, 'samples.json'), encoding='utf-8'))['samples']
    det = [s for s in samples if 'det' in (s.get('tags') or [])]
    raw = {}
    for s in det:
        nm = os.path.basename(s['filepath']).rsplit('.', 1)[0] + '.jpg'
        rows = []
        g = s.get('ground_truth')
        for d in g.get('detections') or [] if isinstance(g, dict) else []:
            b = d.get('bounding_box') or []
            if len(b) < 4:
                continue
            lab = d.get('label')
            lab = lab.get('value') if isinstance(lab, dict) else lab
            rows.append((lab, float(b[0]), float(b[1]), float(b[2]), float(b[3])))
        raw[nm] = rows
    print('[1] Read raw annotations for %d images' % len(raw))
    classes = [c for c in open(os.path.join(WORK, 'classes.txt'), encoding='utf-8').read().split('\n') if c.strip()]
    cls_idx = {c: i for (i, c) in enumerate(classes)}
    assert len(classes) == 7, classes
    index = list(csv.DictReader(open(os.path.join(WORK, 'index.csv'), encoding='utf-8')))
    print('[2] Reuse index.csv: %d rows' % len(index))
    print('[3] Link local image directories ...')
    os.makedirs(os.path.join(DST, 'images'), exist_ok=True)
    for c in CAMS:
        link = os.path.normpath(os.path.join(DST, 'images', c))
        tgt = os.path.normpath(os.path.join(WORK, 'images', c))
        if os.path.isdir(link):
            print('    Exists: %s' % c)
            continue
        if args.copy_images:
            shutil.copytree(tgt, link)
        elif os.name == 'nt':
            result = subprocess.run('mklink /J "%s" "%s"' % (link, tgt), shell=True, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError('Could not create junction; use --copy-images')
        else:
            os.symlink(tgt, link, target_is_directory=True)
        if not os.path.isdir(link):
            raise RuntimeError('Image directory is unavailable: %s' % link)
    print('[4] Write raw-convention labels ...')
    n_box = n_clip = n_unknown = 0
    for c in CAMS:
        os.makedirs(os.path.join(DST, 'labels', c), exist_ok=True)
    for row in index:
        (nm, cd) = (row['name'], row['camdir'])
        out = []
        for (lab, a, b_, w, h) in raw.get(nm, []):
            ci = cls_idx.get(lab)
            if ci is None:
                n_unknown += 1
                continue
            if a - w / 2.0 < -1e-09 or b_ - h / 2.0 < -1e-09:
                n_clip += 1
            out.append('%d %.6f %.6f %.6f %.6f' % (ci, a, b_, w, h))
        with open(os.path.join(DST, 'labels', cd, nm[:-4] + '.txt'), 'w') as f:
            f.write('\n'.join(out) + ('\n' if out else ''))
        n_box += len(out)
    print('    boxes=%d left/top exceedances under center interpretation=%d (%.2f%%)' % (n_box, n_clip, 100.0 * n_clip / max(1, n_box)))
    print('[5] Write raw-convention dataset YAML and lists ...')
    lst = os.path.join(DST, 'lists')
    os.makedirs(lst, exist_ok=True)
    n_list = 0
    for p in PROTOS:
        for cd in CAMS:
            for sp in ('train', 'val', 'test'):
                src_l = os.path.join(WORK, 'lists', '%s__%s__%s.txt' % (p, cd, sp))
                if not os.path.exists(src_l):
                    continue
                lines = [ln.replace(WORK.replace('\\', '/') + '/images/', DST.replace('\\', '/') + '/images/') for ln in open(src_l, encoding='utf-8').read().splitlines() if ln.strip()]
                open(os.path.join(lst, '%s__%s__%s.txt' % (p, cd, sp)), 'w').write('\n'.join(lines) + ('\n' if lines else ''))
                n_list += 1
        W = WORK.replace('\\', '/')
        D = DST.replace('\\', '/')
        y = ['path: ' + D, 'train: ' + '%s/lists/%s__cam2_horizontal90__train.txt' % (D, p), 'val: ' + '%s/lists/%s__cam2_horizontal90__val.txt' % (D, p), 'nc: 7', 'test:']
        for cd in ('cam1_bottom45', 'cam3_tilttop135', 'cam4_vertical180'):
            y.append('  - ' + '%s/lists/%s__%s__test.txt' % (D, p, cd))
        y.append('names:')
        for (i, cname) in enumerate(classes):
            y.append('  %d: %s' % (i, cname))
        open(os.path.join(WORK, 'tm_broken__%s.yaml' % p), 'w').write('\n'.join(y) + '\n')
    print('    image lists=%d dataset YAML files=%d' % (n_list, len(PROTOS)))
    print('[6] Compare raw and repaired labels for the same image')
    for (nm, cd) in [(index[0]['name'], index[0]['camdir']), (index[len(index) // 2]['name'], index[len(index) // 2]['camdir'])]:
        a = open(os.path.join(DST, 'labels', cd, nm[:-4] + '.txt')).read().splitlines()
        b = open(os.path.join(WORK, 'labels', cd, nm[:-4] + '.txt')).read().splitlines()
        print('    %s' % nm)
        for i in range(min(2, len(a), len(b))):
            print('      broken=%s\n      fixed =%s' % (a[i], b[i]))
    print('Completed -> %s' % DST)
    print('Run: python scripts/tm_train.py --protocol broken__per_plant --batch 16')
    return 0
if __name__ == '__main__':
    sys.exit(main())
