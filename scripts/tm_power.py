"""Scan within-plant image counts with fixed models and 31 plants. Seeded permutations retain first m images, or all if fewer. Recompute AP and leave-one-plant jackknife; bootstrap is optional. Each level uses its own fix7 anchors; skip levels lacking GT classes. The 592/1014 operating points differ in model, size and epochs and are not nested (48 images of 592 absent from 1014); SE ratio 0.0079/0.0232=0.34 is not an image-count effect. Fit mean SE squared across seeds to floor squared+c/m. A nonpositive intercept does not establish a floor. Check all-level archived delta/jackknife values within 0.00005."""
import argparse
import collections
import csv
import json
import os
import sys
import time
import numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import tm_boot as B
from tm_config import OUTPUT_TABLES
(WORK, OUT) = (B.WORK, str(OUTPUT_TABLES))
KEYS = B.KEYS

class NoAnchor(Exception):
    pass

def select(clusters_all, perm_of_plant, m):
    picks = [c[pm][:m] if m else c[pm] for (c, pm) in zip(clusters_all, perm_of_plant)]
    cl_store = [np.sort(c, kind='stable') for c in picks]
    sizes = [len(c) for c in picks]
    idx = np.sort(np.concatenate(picks), kind='stable')
    lvl = [np.searchsorted(idx, c) for c in cl_store]
    assert sum((len(c) for c in lvl)) == len(idx) == sum(sizes)
    assert np.array_equal(np.sort(np.concatenate(lvl)), np.arange(len(idx)))
    for (c, l) in zip(cl_store, lvl):
        assert np.array_equal(idx[l], c), 'Subset positions and archive indices must align'
    return (idx, lvl, cl_store, sizes)

def one_level(stores, GTm, idx, lvl, cl_store, n_c, tcrit, boot_iters, boot_seed):
    (pts, LOO) = ({}, {})
    (gt_min, loo_absent) = ([], 0)
    for (p, st, meta) in stores:
        (v50, v95, s50, s95) = B.ap7(st, idx)
        if np.isnan(v50).any():
            raise NoAnchor('%s: selected full subset lacks %d GT classes (%s)' % (p, int(np.isnan(v50).sum()), ','.join((str(i) for i in np.where(np.isnan(v50))[0]))))
        pts[p] = dict(v50=v50, v95=v95, p50=s50, p95=s95, f50=B.fix7(v50, v50), f95=B.fix7(v95, v95))
        G = GTm[p][idx]
        gt_min.append(int(G.sum(0).min()))
        loo = np.zeros((n_c, len(KEYS)))
        for j in range(n_c):
            keep = np.setdiff1d(idx, idx[lvl[j]], assume_unique=True)
            (w50, w95, t50, t95) = B.ap7(st, keep)
            loo[j] = [t50, t95, B.fix7(w50, pts[p]['v50']), B.fix7(w95, pts[p]['v95'])]
            mask = np.ones(len(idx), bool)
            mask[lvl[j]] = False
            loo_absent += int((G[mask].sum(0) == 0).sum())
        LOO[p] = loo
    (pp, qq) = (stores[0][0], stores[1][0])
    fq = np.array([pts[pp][k] for k in KEYS]) - np.array([pts[qq][k] for k in KEYS])
    (jm, jse, jlo, jhi) = B.pseudo_ci(fq, LOO[pp] - LOO[qq], tcrit)
    arm_jse = {p: B.pseudo_ci(np.array([pts[p][k] for k in KEYS]), LOO[p], tcrit)[1] for (p, _, _) in stores}
    out = dict(f50_pp=pts[pp]['f50'], f50_qq=pts[qq]['f50'], delta=fq[KEYS.index('f50')], jk_delta=jm[KEYS.index('f50')], jk_se=jse[KEYS.index('f50')], jk_lo=jlo[KEYS.index('f50')], jk_hi=jhi[KEYS.index('f50')], delta_present=fq[0], jk_se_present=float(jse[0]), jk_se_arm_pp=float(arm_jse[pp][KEYS.index('f50')]), jk_se_arm_qq=float(arm_jse[qq][KEYS.index('f50')]), gt_min=min(gt_min), loo_absent=loo_absent, boot_se=None, iters=0)
    if boot_iters:
        rng = np.random.default_rng(boot_seed)
        (d, a1, a2) = ([], [], [])
        for _ in range(boot_iters):
            pick = rng.integers(0, n_c, n_c)
            bidx = np.sort(np.concatenate([cl_store[k] for k in pick]), kind='stable')
            f = {}
            for (p, st, _) in stores:
                (w50, w95, _, _) = B.ap7(st, bidx)
                f[p] = B.fix7(w50, pts[p]['v50'])
            d.append(f[pp] - f[qq])
            a1.append(f[pp])
            a2.append(f[qq])
        d = np.asarray(d)
        out.update(boot_se=float(d.std(ddof=1)), iters=boot_iters, boot_lo=float(np.percentile(d, 2.5)), boot_hi=float(np.percentile(d, 97.5)), boot_prob_le_0=float((d <= 0).mean()), boot_se_arm_pp=float(np.std(a1, ddof=1)), boot_se_arm_qq=float(np.std(a2, ddof=1)))
    return out

def main():
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True, help='Archived image list filename; no image inference, only record slicing')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--weights', default='last.pt')
    ap.add_argument('--proto', action='append', help='Two arms in subtraction order; default random_img then per_plant')
    ap.add_argument('--ms', default='2,4,8,16,32', help='Retained images per plant; all appended automatically')
    ap.add_argument('--seeds', default='20261201,20261202,20261203')
    ap.add_argument('--boot-iters', type=int, default=200, help='0=jackknife only')
    ap.add_argument('--boot-seed', type=int, default=20261117)
    ap.add_argument('--index', default=os.path.join(WORK, 'index.csv'))
    ap.add_argument('--check-pairs', default='T8_boot1014_pairs.csv', help='Archived paired table checked at all level; none disables check')
    ap.add_argument('--out-prefix', default='T13_power')
    a = ap.parse_args()
    protos = a.proto or ['random_img', 'per_plant']
    assert len(protos) == 2, 'Exactly two arms required, got %d' % len(protos)
    rows = []
    stems = [B.stem_of(p, a.epochs, a.weights, a.list) for p in protos]
    stores = []
    print('[Archives] list=%s weights=%s epochs=%d arms=%s' % (os.path.basename(a.list), a.weights, a.epochs, ' vs '.join(protos)))
    for (p, s) in zip(protos, stems):
        (st, meta) = B.load(s)
        stores.append((p, st, meta))
        print('  %-13s n_img=%-5d collection_delta=%.6f library_mAP50=%.4f' % (p, meta['n_img'], meta['gate_abs'], meta['lib_mAP50']))
    names = [x[1]['im_name'] for x in stores]
    assert np.array_equal(names[0], names[1]), 'Arms have different image order'
    n_img = len(names[0])
    pl = B.plant_of(names[0], a.index)
    plants = sorted(set(pl.tolist()))
    clusters_all = [np.flatnonzero(pl == k) for k in plants]
    n_c = len(clusters_all)
    assert n_c >= 20, 'Only %d plants; require >=20' % n_c
    from scipy import stats as sps
    tcrit = float(sps.t.ppf(0.975, n_c - 1))
    mde_fac = tcrit + float(sps.t.ppf(0.8, n_c - 1))
    GTm = {p: B.gt_matrix(st) for (p, st, _) in stores}
    sizes_full = np.array([len(c) for c in clusters_all])
    print('[Design] images=%d plants=%d within_plant mean=%.2f median=%.1f max=%d LOO_calls_per_arm=%d/level' % (n_img, n_c, sizes_full.mean(), float(np.median(sizes_full)), sizes_full.max(), n_c))
    ms = [int(x) for x in a.ms.split(',') if x.strip()] + [None]
    seeds = [int(x) for x in a.seeds.split(',')]
    sk = collections.defaultdict(list)
    booted = set()
    chk = None
    for m in ms:
        lab0 = 'all' if m is None else m
        for (si, sd) in enumerate(seeds):
            if m is None and si:
                continue
            t0 = time.time()
            rs = np.random.default_rng(sd)
            perms = [rs.permutation(len(c)) for c in clusters_all]
            (idx, lvl, cl_store, sz) = select(clusters_all, perms, m)
            want_boot = a.boot_iters and lab0 not in booted
            try:
                r = one_level(stores, GTm, idx, lvl, cl_store, n_c, tcrit, a.boot_iters if want_boot else 0, a.boot_seed)
            except NoAnchor as e:
                sk[lab0].append('seed%d：%s' % (sd, e))
                print('  m=%-4s skipped (undefined fix7 anchor): %s' % (lab0, e), flush=True)
                continue
            if want_boot:
                booted.add(lab0)
            keep = np.array(sz)
            lab = lab0
            row = dict(list=os.path.basename(a.list), epochs=a.epochs, weights=a.weights, pair='%s_minus_%s' % (protos[0], protos[1]), metric='mAP50_fix7', m_per_plant=lab, n_img_sel=int(len(idx)), mean_imgs_per_plant=round(float(keep.mean()), 2), min_imgs_per_plant=int(keep.min()), n_cluster=n_c, t_df=n_c - 1, sel_seed=sd, point_a=round(r['f50_pp'], 4), point_b=round(r['f50_qq'], 4), delta=round(r['delta'], 4), jk_delta=round(r['jk_delta'], 4), jk_se=round(float(r['jk_se']), 4), jk_ci_lo=round(float(r['jk_lo']), 4), jk_ci_hi=round(float(r['jk_hi']), 4), boot_se=round(float(r['boot_se']), 4) if r['boot_se'] is not None else None, se_ratio=round(float(r['jk_se'] / r['boot_se']), 3) if r['boot_se'] else None, jk_se_arm_a=round(float(r['jk_se_arm_pp']), 4), jk_se_arm_b=round(float(r['jk_se_arm_qq']), 4), delta_present=round(float(r['delta_present']), 4), jk_se_present=round(float(r['jk_se_present']), 4), gt_min_class=r['gt_min'], loo_absent_cls=r['loo_absent'], sd_per_plant=round(float(r['jk_se'] * np.sqrt(n_c) / (n_c - 1.0)), 5), mde_fac=round(mde_fac, 3), mde_80pct=round(float(mde_fac * float(r['jk_se'])), 4), iters=r['iters'])
            if row['boot_se'] is not None and (not 0.2 <= row['se_ratio'] <= 5.0):
                raise SystemExit('m=%s jackknife/bootstrap SE ratio %.2f outside gate' % (lab, row['se_ratio']))
            rows.append(row)
            print('  m=%-4s n=%-5d within_plant=%.2f delta=%+.4f jk_se=%.4f%s min_GT=%d LOO_absent_classes=%d %.0f s' % (lab, len(idx), keep.mean(), row['delta'], row['jk_se'], ' boot_se=%.4f' % row['boot_se'] if row['boot_se'] is not None else '', row['gt_min_class'], row['loo_absent_cls'], time.time() - t0), flush=True)
        if m is None:
            chk = [r for r in rows if r['m_per_plant'] == 'all' and r['sel_seed'] == seeds[0]]
    checked = None
    if a.check_pairs and a.check_pairs.lower() != 'none':
        path = os.path.join(ROOT, 'tables', a.check_pairs)
        if not os.path.exists(path):
            raise SystemExit('Missing archived paired table %s' % a.check_pairs)
        pub = [r for r in csv.DictReader(open(path, encoding='utf-8')) if r['pair'] == '%s_minus_%s' % (protos[0], protos[1]) and r['metric'] == 'mAP50_fix7']
        assert len(pub) == 1 and len(chk) == 1, 'Archived paired or all-level row is not unique'
        (p0, c0) = (pub[0], chk[0])
        bad = [(k, p0[k], c0[k]) for k in ('delta', 'jk_delta', 'jk_se') if abs(float(p0[k]) - float(c0[k])) > 5e-05]
        if bad:
            raise SystemExit('All level fails archived paired reproduction: %s' % bad)
        checked = {k: p0[k] for k in ('delta', 'jk_delta', 'jk_se', 'boot_se')}
        print('[Check] all matches %s (delta/jk_delta/jk_se differences <=5e-05)' % a.check_pairs)
    fit = None
    lv = {}
    for r in rows:
        if r['m_per_plant'] == 'all':
            continue
        lv.setdefault(int(r['m_per_plant']), []).append(float(r['jk_se']) ** 2)
    if len(lv) >= 3:
        ks = sorted(lv)
        xs = np.array([1.0 / k for k in ks])
        ys = np.array([float(np.mean(lv[k])) for k in ks])
        A = np.vstack([np.ones_like(xs), xs]).T
        (coef, *_) = np.linalg.lstsq(A, ys, rcond=None)
        (floor2, c) = (float(coef[0]), float(coef[1]))
        pred = A @ coef
        ss = float(((ys - ys.mean()) ** 2).sum())
        fit = dict(n_levels=len(ks), ms=ks, seeds_per_level=[len(lv[k]) for k in ks], intercept_floor2=round(floor2, 8), slope_c=round(c, 8), floor_se=round(float(np.sqrt(floor2)), 4) if floor2 > 0 else None, floor_mde_80pct=round(float(mde_fac * np.sqrt(floor2)), 4) if floor2 > 0 else None, floor_se_note='Positive intercept: extrapolated precision floor' if floor2 > 0 else 'Nonpositive intercept: no demonstrated floor over observed m; mean of largest two observed levels is a plateau upper bound', plateau_se=round(float(np.sqrt(np.mean(ys[-2:]))), 4), plateau_mde_80pct=round(float(mde_fac * np.sqrt(np.mean(ys[-2:]))), 4), r2=round(1.0 - float(((ys - pred) ** 2).sum()) / ss, 3) if ss > 0 else None)
        print('[Fit, level means] SE^2=%.6f+%.6f/m floor_SE=%s plateau_SE=%.4f (MDE80=%.4f) R^2=%s levels=%d seeds/level=%d' % (floor2, c, '%.4f' % fit['floor_se'] if fit['floor_se'] else 'nonpositive (no established plateau)', fit['plateau_se'], fit['plateau_mde_80pct'], fit['r2'], fit['n_levels'], fit['seeds_per_level'][0]))
    skip_rep = {str(k): v for (k, v) in sk.items()}
    if skip_rep:
        print('[Missing anchors] ' + ' ｜ '.join(('m=%s: %d/%d selection seeds have undefined fix7 anchors' % (k, len(v), len(seeds)) for (k, v) in sorted(skip_rep.items(), key=lambda kv: (kv[0] == 'all', int(kv[0] if kv[0] != 'all' else 0))))), flush=True)
    if not rows:
        raise SystemExit('Every level lacks anchors; increase --ms or use another list')
    with open(os.path.join(OUT, a.out_prefix + '.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('  Wrote %s (%d rows)' % (os.path.relpath(os.path.join(OUT, a.out_prefix + '.csv'), ROOT).replace('\\', '/'), len(rows)))
    meta = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), list=os.path.basename(a.list), epochs=a.epochs, weights=a.weights, protos=protos, n_img=n_img, n_cluster=n_c, t_df=n_c - 1, mde_fac=round(mde_fac, 4), tcrit_975=round(tcrit, 4), gt_per_class_full=[int(x) for x in GTm[protos[0]].sum(0)], imgs_with_gt_full=[int(x) for x in (GTm[protos[0]] > 0).sum(0)], imgs_per_plant_full=[int(x) for x in sizes_full], ms=['all' if m is None else m for m in ms], sel_seeds=seeds, boot_iters=a.boot_iters, boot_seed=a.boot_seed, index=os.path.basename(a.index), npz=[os.path.relpath(os.path.join(B.BOOT, s + '.npz'), ROOT).replace('\\', '/') for s in stems], reproduce_check=checked, skipped_no_anchor=skip_rep, fit=fit)
    with open(os.path.join(OUT, a.out_prefix + '_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
    print('  Wrote %s' % os.path.relpath(os.path.join(OUT, a.out_prefix + '_meta.json'), ROOT).replace('\\', '/'))
if __name__ == '__main__':
    main()
