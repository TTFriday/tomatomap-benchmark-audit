"""Offline paired cluster inference with archived Ultralytics AP records. Resample plant_id with replacement, retaining all images of sampled plants; image clustering is an independence comparison. Paired 592/1014 subsets each contain 31 plants. present averages GT-present classes; fix7 averages seven classes, replacing absent-class AP with full-subset anchors. Use identical paired indices. Report percentile bootstrap, leave-one-cluster jackknife+t(df=n_cluster-1), and pairwise cluster sign flips. Prefer jackknife+t below 50 clusters. AP is nonadditive: globally recompute leave-one AP. Reject full-subset error >=0.002 and jackknife/bootstrap SE ratios outside [0.2,5]. Jackknife pseudo_j=n*theta-(n-1)*loo_j; SE=sqrt((n-1)/n*sum((loo-mean(loo))**2))=sd(pseudo)/sqrt(n); CI=mean(pseudo)+/-tcrit*SE."""
import argparse
import csv
import itertools
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
import tm_dump_pred as D
import tm_paired as TP
(WORK, BOOT, CLASSES, GATE) = (D.WORK, D.BOOT, D.CLASSES, D.GATE)
from tm_config import OUTPUT_TABLES
OUT = str(OUTPUT_TABLES)
NC = len(CLASSES)
METRICS = (('mAP50_present', 'p50'), ('mAP50_95_present', 'p95'), ('mAP50_fix7', 'f50'), ('mAP50_95_fix7', 'f95'))
KEYS = [k for (_, k) in METRICS]
SMALL_CLUSTER = 50

def stem_of(proto, epochs, weights, list_name, tag=''):
    tag0 = os.path.splitext(os.path.basename(weights))[0]
    lst = os.path.splitext(os.path.basename(list_name))[0]
    return '%s__e%d__%s__%s%s' % (proto, int(epochs), tag0, lst, '__' + tag if tag else '')

def load(stem):
    (npz, mf) = (os.path.join(BOOT, stem + '.npz'), os.path.join(BOOT, stem + '.json'))
    if not os.path.exists(npz):
        raise SystemExit('Missing archive %s; collect with tm_dump_pred.py using the same arguments' % npz)
    st = dict(np.load(npz))
    meta = json.load(open(mf, encoding='utf-8'))
    assert meta['gate_abs'] < GATE, '%s collection gate failed (delta=%.4f); reject inference' % (stem, meta['gate_abs'])
    n = len(st['im_name'])
    assert st['off'].size == n + 1 and st['goff'].size == n + 1
    assert st['tp'].shape == (st['conf'].size, D.IOUV.size)
    return (st, meta)

def plant_of(names, index_csv):
    col = {}
    with open(index_csv, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            (k, v) = (r['name'], r['plant_id'])
            assert col.get(k, v) == v, 'index.csv maps %s to multiple plant_id values' % k
            col[k] = v
    miss = [x for x in names if x not in col]
    assert not miss, 'Image names absent from index.csv: %s; plant clustering unavailable' % miss[:3]
    return np.array([col[x] for x in names], dtype='<U16')

def ap7(store_or_parts, idx=None):
    parts = store_or_parts if isinstance(store_or_parts, list) else [(store_or_parts, idx)]
    (tp, conf, pcl, tcl) = D.cat_slices(parts)
    (all_ap, ci, _) = D.ap_table(tp, conf, pcl, tcl)
    v50 = np.full(NC, np.nan)
    v95 = np.full(NC, np.nan)
    if all_ap.size:
        ci = np.asarray(ci, dtype=int).ravel()
        v50[ci] = all_ap[:, 0]
        v95[ci] = all_ap.mean(1)
    return (v50, v95, float(np.nanmean(v50)), float(np.nanmean(v95)))

def fix7(v, anchor):
    return float(np.where(np.isnan(v), anchor, v).mean())

def gt_matrix(store):
    n = len(store['im_name'])
    g = np.zeros((n, NC))
    for i in range(n):
        t = store['tcls'][store['goff'][i]:store['goff'][i + 1]].astype(int)
        if t.size:
            np.add.at(g[i], t, 1)
    assert np.allclose(g.sum(0), np.bincount(store['tcls'].astype(int), minlength=NC)), 'GT decomposition does not align'
    return g

def pseudo_ci(theta_full, loo, tcrit):
    loo = np.asarray(loo, float)
    n = loo.shape[0]
    ps = n * np.asarray(theta_full, float) - (n - 1.0) * loo
    se = np.sqrt((n - 1.0) / n * ((loo - loo.mean(0)) ** 2).sum(0))
    return (ps.mean(0), se, ps.mean(0) - tcrit * se, ps.mean(0) + tcrit * se)

def pct(x, lo=2.5, hi=97.5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return (float('nan'), float('nan'), float('nan'), int(x.size))
    return (float(np.percentile(x, lo)), float(np.percentile(x, hi)), float(x.std(ddof=1)), int(x.size))

def boot_indices(clusters, rng):
    pick = rng.integers(0, len(clusters), len(clusters))
    return np.sort(np.concatenate([clusters[k] for k in pick]), kind='stable')

def clusters_of(names, cluster, index_csv, n_img):
    if cluster == 'plant':
        pl = plant_of(names, index_csv)
        cl = [np.flatnonzero(pl == k) for k in sorted(set(pl.tolist()))]
    else:
        cl = [np.array([i]) for i in range(n_img)]
    assert sum((len(c) for c in cl)) == n_img
    assert len(cl) >= 20, 'Only %d clusters; require >=20 or change --cluster' % len(cl)
    return cl

def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('  Wrote %s (%d rows)' % (os.path.relpath(path, ROOT).replace('\\', '/'), len(rows)))

def main():
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('--proto', action='append', help='Repeatable arm; default tm_paired.DEFAULT_PROTOS')
    ap.add_argument('--list', required=True, help='Same image list filename used for prediction archive')
    ap.add_argument('--epochs', type=int, default=60, help='Planned epochs encoded in archive filenames')
    ap.add_argument('--weights', default='last.pt')
    ap.add_argument('--tag', default='', help='Collection tag; must match archive name')
    ap.add_argument('--iters', type=int, default=1000)
    ap.add_argument('--perm', type=int, default=1, help='1=pairwise cluster sign-flip randomization tests')
    ap.add_argument('--perm-iters', type=int, default=0, help='Randomization iterations; 0 uses --iters')
    ap.add_argument('--cluster', choices=['plant', 'image'], default='plant', help='image is the independent-image comparison, not primary cluster inference')
    ap.add_argument('--seed', type=int, default=20261117)
    ap.add_argument('--index', default=os.path.join(WORK, 'index.csv'))
    ap.add_argument('--out-prefix', default='T8_boot')
    a = ap.parse_args()
    protos = a.proto or TP.DEFAULT_PROTOS
    stems = [stem_of(p, a.epochs, a.weights, a.list, a.tag) for p in protos]
    arms = []
    print('[Archives] list=%s weights=%s epochs=%d' % (os.path.basename(a.list), a.weights, a.epochs))
    for (p, s) in zip(protos, stems):
        (st, meta) = load(s)
        arms.append((p, st, meta))
        print('  %-13s n_img=%-5d pred=%-7d gt=%-6d collection_delta=%.6f library_mAP50=%.4f' % (p, meta['n_img'], meta['n_pred'], meta['n_gt'], meta['gate_abs'], meta['lib_mAP50']))
    names = [x[1]['im_name'] for x in arms]
    for (p, q) in zip(protos[1:], names[1:]):
        assert np.array_equal(names[0], q), '%s image order differs from %s; paired inference invalid' % (p, protos[0])
    assert len({m['list_name'] for (_, _, m) in arms}) == 1, 'Arms have different list names'
    assert len({m['weights'] for (_, _, m) in arms}) == 1, 'Arms use different checkpoint types: %s' % [m['weights'] for (_, _, m) in arms]
    n_img = len(names[0])
    clusters = clusters_of(names[0], a.cluster, a.index, n_img)
    n_c = len(clusters)
    n_arm = len(protos)
    n_pair = n_arm * (n_arm - 1) // 2
    extra = 2 * n_pair if a.perm else 0
    print('[Resampling] cluster=%s clusters=%d images=%d largest_cluster=%d arms=%d iters=%d seed=%d AP_calls~%d' % (a.cluster, n_c, n_img, max((len(c) for c in clusters)), n_arm, a.iters, a.seed, a.iters * (n_arm + extra)))
    if a.cluster == 'plant' and n_c < SMALL_CLUSTER:
        print('      clusters=%d < %d: prefer jackknife+t(df=%d); bootstrap tails are secondary.' % (n_c, SMALL_CLUSTER, n_c - 1))
    (pts, full) = ({}, np.arange(n_img))
    for (p, st, meta) in arms:
        (v50, v95, p50, p95) = ap7(st, full)
        assert not np.isnan(v50).any(), '%s full subset lacks GT classes %s; fix7 anchors incomplete' % (p, np.where(np.isnan(v50))[0])
        pts[p] = dict(v50=v50, v95=v95, p50=p50, p95=p95, f50=fix7(v50, v50), f95=fix7(v95, v95))
        d = abs(p50 - meta['lib_mAP50'])
        print('  Full %-13s present=%.6f fix7=%.6f library_difference=%.6f' % (p, p50, pts[p]['f50'], d))
        assert d < GATE, '%s AP differs from library by %.4f >= %.3f; reject inference' % (p, d, GATE)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    S = {p: {k: [] for k in KEYS} for p in protos}
    VC = {p: {'v50': np.full((a.iters, NC), np.nan), 'v95': np.full((a.iters, NC), np.nan)} for p in protos}
    for b in range(a.iters):
        idx = boot_indices(clusters, rng)
        for (p, st, _) in arms:
            (v50, v95, p50, p95) = ap7(st, idx)
            S[p]['p50'].append(p50)
            S[p]['p95'].append(p95)
            S[p]['f50'].append(fix7(v50, pts[p]['v50']))
            S[p]['f95'].append(fix7(v95, pts[p]['v95']))
            VC[p]['v50'][b] = v50
            VC[p]['v95'][b] = v95
        if (b + 1) % 200 == 0:
            print('    bootstrap %d/%d  %.0f s' % (b + 1, a.iters, time.time() - t0), flush=True)
    t1 = time.time()
    GTm = {p: gt_matrix(st) for (p, st, _) in arms}
    print('  GT decomposition: %d arms x %d images, %.0f s' % (n_arm, n_img, time.time() - t1), flush=True)
    from scipy import stats as sps
    tcrit = float(sps.t.ppf(0.975, n_c - 1))
    (JK, LOO, NPRE) = ({}, {}, {})
    all_idx = np.arange(n_img)
    for (p, st, _) in arms:
        loo = np.zeros((n_c, len(KEYS)))
        npre = np.zeros(n_c)
        for (j, cix) in enumerate(clusters):
            keep = np.setdiff1d(all_idx, cix, assume_unique=True)
            (v50, v95, s50, s95) = ap7(st, keep)
            npre[j] = float(np.isfinite(v50).sum())
            loo[j] = [s50, s95, fix7(v50, pts[p]['v50']), fix7(v95, pts[p]['v95'])]
        assert [k for (_, k) in METRICS] == KEYS == ['p50', 'p95', 'f50', 'f95'], 'LOO column order does not match METRICS'
        (LOO[p], NPRE[p]) = (loo, npre)
        JK[p] = pseudo_ci(np.array([pts[p][k] for k in KEYS]), loo, tcrit)
    print('  Leave-one-cluster jackknife: %d clusters x %d arms, %.0f s' % (n_c, n_arm, time.time() - t1), flush=True)
    base = dict(cluster=a.cluster, list=os.path.basename(arms[0][2]['list_name']), n_img=n_img, n_cluster=n_c, t_df=n_c - 1, weights=arms[0][2]['weights'], seed=a.seed)
    rows = []
    warn = []
    for (p, st, meta) in arms:
        (jm, jse, jlo, jhi) = JK[p]
        for (m, (met, key)) in enumerate(METRICS):
            (lo, hi, se, nn) = pct(S[p][key])
            ratio = round(float(jse[m] / se), 3) if se and np.isfinite(se) and (se > 0) else None
            if ratio is not None and (not 0.2 <= ratio <= 5.0):
                warn.append('%s/%s se_ratio=%.2f' % (p, met, ratio))
            rows.append(dict(base, protocol=p, imgsz=meta['imgsz'], epochs=meta['epochs_run'], model=meta['model'], metric=met, point=round(pts[p][key], 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4), boot_se=round(se, 4), jk_point=round(float(jm[m]), 4), jk_se=round(float(jse[m]), 4), jk_ci_lo=round(float(jlo[m]), 4), jk_ci_hi=round(float(jhi[m]), 4), se_ratio=ratio, n_class_present_min=int(NPRE[p].min()), iters=nn, npz=os.path.relpath(os.path.join(BOOT, stems[protos.index(p)] + '.npz'), ROOT).replace('\\', '/')))
    if warn:
        raise SystemExit('Bootstrap/jackknife SE mismatch: %s; investigate before writing tables' % '; '.join(warn))
    write_csv(os.path.join(OUT, a.out_prefix + '_arms.csv'), rows)
    pc = []
    for (p, st, meta) in arms:
        for (i, cn) in enumerate(CLASSES):
            (lo, hi, se, nn) = pct(VC[p]['v50'][:, i])
            s50 = VC[p]['v50'][:, i]
            nimg = int((GTm[p][:, i] > 0).sum())
            pc.append(dict(base, protocol=p, cls=cn, support=int((st['tcls'] == i).sum()), n_img_with_gt=nimg, ap50=round(float(pts[p]['v50'][i]), 4), ap50_ci_lo=round(lo, 4), ap50_ci_hi=round(hi, 4), boot_se=round(se, 4), p_absent=round(float(np.isnan(s50).mean()), 4), ap50_95=round(float(pts[p]['v95'][i]), 4)))
    write_csv(os.path.join(OUT, a.out_prefix + '_perclass.csv'), pc)
    dr = []
    for (p, q) in itertools.combinations(protos, 2):
        fq = np.array([pts[p][k] for k in KEYS]) - np.array([pts[q][k] for k in KEYS])
        (jm, jse, jlo, jhi) = pseudo_ci(fq, LOO[p] - LOO[q], tcrit)
        for (m, (met, key)) in enumerate(METRICS):
            d = np.asarray(S[p][key], float) - np.asarray(S[q][key], float)
            (lo, hi, se, nn) = pct(d)
            ratio = round(float(jse[m] / se), 3) if se and np.isfinite(se) and (se > 0) else None
            dr.append(dict(base, pair='%s_minus_%s' % (p, q), metric=met, delta=round(fq[m], 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4), boot_se=round(se, 4), jk_delta=round(float(jm[m]), 4), jk_se=round(float(jse[m]), 4), jk_ci_lo=round(float(jlo[m]), 4), jk_ci_hi=round(float(jhi[m]), 4), se_ratio=ratio, boot_prob_le_0=round(float(np.mean(d <= 0)), 4), iters=nn))
    if dr:
        write_csv(os.path.join(OUT, a.out_prefix + '_pairs.csv'), dr)
    else:
        print('  Single arm: no paired contrasts (%s_pairs.csv not written)' % a.out_prefix)
    for r in dr:
        if r['metric'] in ('mAP50_fix7', 'mAP50_present'):
            print('    Δ %-28s %-15s = %+.4f  boot95%% [%+.4f, %+.4f]  jk95%% [%+.4f, %+.4f]  P(Δ≤0)=%.3f' % (r['pair'], r['metric'], r['delta'], r['ci_lo'], r['ci_hi'], r['jk_ci_lo'], r['jk_ci_hi'], r['boot_prob_le_0']))
    store_of = {p: st for (p, st, _) in arms}
    perm_info = None

    def perm_pair(pp, qq, seed_i):
        (P, Q) = (store_of[pp], store_of[qq])
        k = a.perm_iters or a.iters
        rng2 = np.random.default_rng(seed_i)
        obs = pts[pp]['f50'] - pts[qq]['f50']
        (n_ge, t2) = (0, time.time())
        for b in range(k):
            flip = rng2.integers(0, 2, n_c).astype(bool)
            ia = np.concatenate([clusters[j] for j in range(n_c) if not flip[j]]) if (~flip).any() else np.zeros(0, int)
            ib = np.concatenate([clusters[j] for j in range(n_c) if flip[j]]) if flip.any() else np.zeros(0, int)
            av50 = ap7([(P, ia), (Q, ib)])[0]
            bv50 = ap7([(Q, ia), (P, ib)])[0]
            d = fix7(av50, pts[pp]['v50']) - fix7(bv50, pts[qq]['v50'])
            if abs(d) >= abs(obs) - 1e-12:
                n_ge += 1
            if (b + 1) % 500 == 0:
                print('    perm %s−%s %d/%d  %.0f s' % (pp, qq, b + 1, k, time.time() - t2), flush=True)
        return (obs, (n_ge + 1) / (k + 1), k, round(time.time() - t2, 1))
    if a.perm and n_arm >= 2:
        rows_pm = []
        for (i, (p, q)) in enumerate(itertools.combinations(protos, 2)):
            sd = a.seed + 1 + i
            (obs, pv, k, dt) = perm_pair(p, q, sd)
            rows_pm.append(dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), pair='%s_minus_%s' % (p, q), metric='mAP50_fix7', list=os.path.basename(arms[0][2]['list_name']), cluster=a.cluster, n_img=n_img, n_cluster=n_c, t_df=n_c - 1, iters=k, seed=sd, seconds=dt, observed_delta=round(obs, 4), perm_p_two_sided=round(pv, 5), note='Null: image-level arm records are exchangeable within %s clusters; p=(count(|delta*|>=|delta_obs|)+1)/(k+1)' % a.cluster))
            print('  Sign flip %s - %s: delta=%+.4f two-sided p=%.4f (%d iterations, seed=%d, %.0f s)' % (p, q, obs, pv, k, sd, dt), flush=True)
        write_csv(os.path.join(OUT, a.out_prefix + '_perm.csv'), rows_pm)
        perm_info = dict(iters=k, seeds=[a.seed + 1 + i for i in range(len(rows_pm))], pairs=[r['pair'] for r in rows_pm], p={r['pair']: r['perm_p_two_sided'] for r in rows_pm})
    elif a.perm:
        print('  Single arm: no randomization contrast; skip.')
    run_info = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), cluster=a.cluster, iters=a.iters, seed=a.seed, list=os.path.basename(a.list), arms=list(protos), npz_stems=stems, n_img=n_img, n_cluster=n_c, t_df=n_c - 1, t_crit=round(tcrit, 4), gate=GATE, perm=perm_info, seconds=round(time.time() - t0, 1), caliber='present=macro AP over GT-present classes; fix7=seven classes with absent-class full-subset anchors; jk=leave-one-cluster jackknife+t(df=n_cluster-1), preferred with few clusters')
    for path in (os.path.join(BOOT, '%s__%s.json' % (a.out_prefix, os.path.splitext(os.path.basename(a.list))[0])), os.path.join(OUT, a.out_prefix + '_meta.json')):
        json.dump(run_info, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('Finished, %.0f s' % (time.time() - t0))
if __name__ == '__main__':
    main()
