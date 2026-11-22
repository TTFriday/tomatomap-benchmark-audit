"""Check all five paired prediction archives without images, weights or resampling."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import tm_boot as B
from tm_config import ROOT, WORK

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    with (ROOT / 'tables' / 'T7_paired_ap.csv').open(encoding='utf-8-sig') as stream:
        expected = {(int(row['n_img_paired']), row['protocol']): row for row in csv.DictReader(stream)}
    reports = []
    aligned = {}
    for (n_images, epochs, protocols) in [(592, 60, ['random_img', 'per_set', 'per_plant']), (1014, 200, ['random_img', 'per_plant'])]:
        list_name = 'paired_subset__cam2__%d.txt' % n_images
        archived_names = [Path(line).name for line in (WORK / 'paired' / list_name).read_text(encoding='utf-8').splitlines()]
        for protocol in protocols:
            stem = B.stem_of(protocol, epochs, 'last.pt', list_name)
            (store, meta) = B.load(stem)
            names = store['im_name']
            assert names.tolist() == archived_names, 'Archive/list order mismatch: ' + stem
            assert len(names) == n_images and meta['n_img'] == n_images
            assert meta['epochs_run'] >= meta['epochs_manifest']
            previous = aligned.get(n_images)
            if previous is not None:
                for key in ('im_name', 'goff', 'tcls'):
                    assert np.array_equal(previous[key], store[key]), 'Paired GT mismatch: ' + key
            aligned[n_images] = store
            plants = B.plant_of(names, str(WORK / 'index.csv'))
            assert len(set(plants.tolist())) == 31
            (ap50, ap95, map50, map95) = B.ap7(store, np.arange(n_images))
            assert np.isfinite(ap50).all() and np.isfinite(ap95).all()
            error = max(abs(map50 - meta['lib_mAP50']), abs(map95 - meta['lib_mAP50_95']))
            assert error < B.GATE, 'Library AP gate failed: ' + stem
            table = expected[n_images, protocol]
            assert abs(map50 - float(table['mAP50'])) <= 5e-05
            assert abs(map95 - float(table['mAP50_95'])) <= 5e-05
            reports.append({'archive': 'work/tm_yolo/boot/' + stem + '.npz', 'n_images': n_images, 'n_plants': 31, 'mAP50': round(map50, 6), 'mAP50_95': round(map95, 6), 'library_error': round(error, 9)})
    print(json.dumps({'passed': len(reports), 'archives': reports}, indent=2))
if __name__ == '__main__':
    main()
