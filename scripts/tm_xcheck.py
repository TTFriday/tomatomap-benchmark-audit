"""Cross-evaluate a checkpoint against repaired and raw annotation conventions. Explicit dataset filenames avoid doubled broken__ prefixes for broken-trained runs. Paired cells must use identical image lists; check membership before changing evaluation split."""
import argparse
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

def set_name(yaml_name, proto):
    stem = os.path.basename(yaml_name)
    stem = stem[:-5] if stem.endswith('.yaml') else stem
    if not stem.startswith('tm_'):
        return stem
    core = stem[3:]
    base = proto[len('broken__'):] if proto.startswith('broken__') else proto
    if core.startswith('broken__'):
        return 'broken(raw)' if core[8:] == base else 'broken(%s)' % core[8:]
    return 'fixed(repaired)' if core == base else 'fixed(%s)' % core

def per_class(bb, model):
    import numpy as np
    names = getattr(bb, 'names', None) or model.names
    keys = sorted(names)
    arr = np.asarray(bb.maps, dtype=float)
    return {str(names[k]): round(float(v), 4) for (k, v) in zip(keys, arr) if np.isfinite(v)}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--imgsz', type=int, default=0)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--weights', default='best.pt')
    ap.add_argument('--split', default='val', help='Check image-list identity before changing split')
    ap.add_argument('--data', default='', help='Comma-separated dataset YAML filenames')
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    with open(os.path.join(run, 'manifest.json'), encoding='utf-8') as stream:
        man = json.load(stream)
    proto = man['protocol']
    imgsz = a.imgsz or man['imgsz']
    weights = os.path.join(run, 'weights', a.weights)
    assert os.path.exists(weights), weights
    from ultralytics import YOLO
    m = YOLO(weights)
    rows = []
    base = proto[len('broken__'):] if proto.startswith('broken__') else proto
    cells = [x.strip() for x in a.data.split(',') if x.strip()] or ['tm_%s.yaml' % base, 'tm_broken__%s.yaml' % base]
    for yaml_name in cells:
        label_set = set_name(yaml_name, proto)
        data = os.path.join(WORK, yaml_name)
        if not os.path.exists(data):
            print('Skip %s: missing %s' % (label_set, data))
            continue
        print('=' * 64)
        print('%s  <-  %s' % (label_set, data))
        print('=' * 64)
        r = m.val(data=data, imgsz=imgsz, batch=a.batch, workers=a.workers, device='0', split=a.split, verbose=True, save_json=False, plots=False, project=os.path.join(run, 'eval_xcheck'), name=label_set, exist_ok=True)
        bb = r.box
        rows.append(dict(label_set=label_set, run=os.path.basename(run), proto=proto, weights=os.path.basename(weights), split=a.split, imgsz=imgsz, P=round(bb.mp, 4), R=round(bb.mr, 4), mAP50=round(bb.map50, 4), mAP50_95=round(bb.map, 4), per_class=per_class(bb, m)))
        print('  >>> P=%.4f R=%.4f mAP50=%.4f mAP50-95=%.4f' % (bb.mp, bb.mr, bb.map50, bb.map))
    out = os.path.join(run, 'xcheck.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print('\n' + '=' * 64)
    hi = next((x for x in rows if x['label_set'].startswith('fixed')), None)
    lo = next((x for x in rows if x['label_set'].startswith('broken')), None)
    if hi and lo:
        d50 = hi['mAP50'] - lo['mAP50']
        d = hi['mAP50_95'] - lo['mAP50_95']
        print('Same checkpoint (%s / %s): mAP50 %.4f(repaired) to %.4f(raw), absolute difference %.4f (%.1f%%)' % (hi['weights'], hi['split'], hi['mAP50'], lo['mAP50'], d50, 100 * d50 / max(1e-09, hi['mAP50'])))
        print('          mAP50-95 %.4f to %.4f, absolute difference %.4f' % (hi['mAP50_95'], lo['mAP50_95'], d))
    print('-> %s' % out)
if __name__ == '__main__':
    main()
