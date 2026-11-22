"""Compare exported centre-format boxes with upstream annotation files.

Inputs may be directories or ZIP archives containing normalized YOLO label files.
Matching is order independent and class constrained. Degenerate boxes are checked
numerically but are excluded from IoU summaries.
"""
from pathlib import Path
import argparse
import csv
import json
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]


def parse(text):
    values = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 5:
            values.append([int(float(fields[0]))] + [float(x) for x in fields[1:]])
    return np.asarray(values, dtype=float).reshape(-1, 5)


def labels(path):
    path = Path(path)
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.lower().endswith('.txt') and Path(name).name.lower() != 'classes.txt':
                    yield Path(name).stem, parse(archive.read(name).decode('utf-8-sig'))
    else:
        for file in sorted(path.rglob('*.txt')):
            if file.name.lower() != 'classes.txt':
                yield file.stem, parse(file.read_text(encoding='utf-8-sig'))


def compare(a, b, tolerance):
    if len(a) != len(b):
        return False, float('nan'), np.empty(0)
    if not len(a):
        return True, 0.0, np.empty(0)
    cost = np.abs(a[:, None, 1:] - b[None, :, 1:]).max(axis=2)
    cost += 10 * (a[:, None, 0] != b[None, :, 0])
    i, j = linear_sum_assignment(cost)
    aa, bb = a[i], b[j]
    deviation = float(np.abs(aa[:, 1:] - bb[:, 1:]).max())
    good = bool(np.array_equal(aa[:, 0], bb[:, 0]) and deviation <= tolerance)
    keep = (aa[:, 3:] > 0).all(axis=1) & (bb[:, 3:] > 0).all(axis=1)
    aa, bb = aa[keep, 1:], bb[keep, 1:]
    lower = np.maximum(aa[:, :2] - aa[:, 2:] / 2, bb[:, :2] - bb[:, 2:] / 2)
    upper = np.minimum(aa[:, :2] + aa[:, 2:] / 2, bb[:, :2] + bb[:, 2:] / 2)
    intersection = np.maximum(upper - lower, 0).prod(axis=1)
    union = aa[:, 2:].prod(axis=1) + bb[:, 2:].prod(axis=1) - intersection
    return good, deviation, intersection / union


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, action='append', required=True)
    parser.add_argument('--export', type=Path, default=ROOT / 'work' / 'tm_yolo' / 'labels')
    parser.add_argument('--tolerance', type=float, default=2e-6)
    parser.add_argument('--output', type=Path, default=ROOT / 'work' / 'tm_yolo' / 'tables' / 'annotation_verification.csv')
    args = parser.parse_args()
    local = dict(labels(args.export))
    output, overlaps = [], []
    for source in args.upstream:
        for stem, original in labels(source):
            found = stem in local
            match, deviation, ious = compare(local[stem], original, args.tolerance) if found else (False, float('nan'), np.empty(0))
            overlaps.extend(ious.tolist())
            output.append(dict(image_stem=stem, n_upstream_boxes=len(original), local_found=found,
                               numeric_agreement=match, max_coordinate_deviation=deviation,
                               n_nondegenerate_ious=len(ious), median_iou=float(np.median(ious)) if len(ious) else float('nan')))
    if not output:
        raise SystemExit('No upstream label records found')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps(dict(n_files=len(output), n_upstream_boxes=sum(r['n_upstream_boxes'] for r in output),
                          n_agree=sum(r['numeric_agreement'] for r in output),
                          median_nondegenerate_iou=float(np.median(overlaps)) if overlaps else None), indent=2))
    if not all(r['numeric_agreement'] for r in output):
        raise SystemExit('At least one annotation file is missing or fails the numeric tolerance')


if __name__ == '__main__':
    main()
