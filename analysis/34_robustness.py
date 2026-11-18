"""Robustness of the descriptive results of step 3: main analysis (auto-posting excluded by the refined rule),
original auto-posting rule, no exclusion, weight 1/n_posts of the community (equal contribution per community).
Weighted Spearman = weighted Pearson correlation of ranks (unweighted ranks)."""
import numpy as np, pandas as pd
from scipy.stats import rankdata
from common import *
q = pd.read_parquet(O+'pools.parquet')
def wq(x, w, qq):
    o = np.argsort(x.values); x, w = x.values[o], w.values[o]; cw = (np.cumsum(w) - .5*w)/w.sum()
    return float(np.interp(qq, cw, x))
def wcorr(a, b, w):
    a, b, w = rankdata(a), rankdata(b), np.asarray(w); ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    return float(np.average((a-ma)*(b-mb), weights=w)/np.sqrt(np.average((a-ma)**2, weights=w)*np.average((b-mb)**2, weights=w)))
rows = []
def add(metric, subset, variant, val, n): rows.append({'metric': metric, 'subset': subset, 'variant': variant, 'value': val, 'n': n})
for variant, rk, ak, weighted in [('main (refined rule)', 'pool_rho', 'pool_ratio', False), ('original rule (7 communities)', 'pool_rho_orig', 'pool_ratio_orig', False),
                                  ('no auto-posting exclusion', 'pool_rho_none', 'pool_ratio_none', False), ('weight 1/n per community', 'pool_rho', 'pool_ratio', True)]:
    r = q[q[rk]].copy(); a = q[q[ak]].copy()
    for d in (r, a): d['w'] = (1/d.groupby('community_vk_id').community_vk_id.transform('size')) if weighted else 1.0
    for k, col in [('R','reposts'),('L','likes'),('C','comments')]:
        r['rho_'+k] = r[col]/r.expo; r['lr_'+k] = np.log((r[col]+.5)/r.expo)
        r['lrw_'+k] = r['lr_'+k] - r.groupby(['community_vk_id', r.modality=='clip'])['lr_'+k].transform('median')  # within block
    for m in MODS:
        g = r[r.modality==m]
        for k in 'RLC': add(f'median rho_{k} x1e3', m, variant, 1e3*wq(g['rho_'+k], g.w, .5), len(g))
        add('share R=0', m, variant, float(np.average(g.reposts==0, weights=g.w)), len(g))
    for blk, g in [('text/image/video', r[r.modality!='clip']), ('clip', r[r.modality=='clip'])]:
        add('spearman within-comm log rho_R~rho_L', blk, variant, wcorr(g.lrw_R, g.lrw_L, g.w), len(g))
        add('spearman within-comm log rho_C~rho_L', blk, variant, wcorr(g.lrw_C, g.lrw_L, g.w), len(g))
    a['lRL'] = np.log((a.reposts+.5)/(a.likes+.5)); a['lCL'] = np.log((a.comments+.5)/(a.likes+.5))
    for m in MODS + ['ALL']:
        g = a if m == 'ALL' else a[a.modality==m]
        pos = g[(g.reposts>0)&(g.likes>0)]; pc = g[(g.comments>0)&(g.likes>0)]
        add('median log(R/L), R,L>0', m, variant, wq(np.log(pos.reposts/pos.likes), pos.w, .5), len(pos))
        add('median log(C/L), C,L>0', m, variant, wq(np.log(pc.comments/pc.likes), pc.w, .5), len(pc))
    fin = a[a.stratum=='finance_business']; pos = fin[(fin.reposts>0)&(fin.likes>0)]
    add('finance: share L>0', 'finance', variant, float(np.average(fin.likes>0, weights=fin.w)), len(fin))
    add('finance: median log(R/L), R,L>0', 'finance', variant, wq(np.log(pos.reposts/pos.likes), pos.w, .5), len(pos))
    if not weighted:
        for col in ['lRL','lCL']:
            tot = a[col].var()
            add(f'variance share between strata {col}', 'ALL', variant, a.groupby('stratum')[col].transform('mean').var()/tot, len(a))
            add(f'variance share between communities {col}', 'ALL', variant, a.groupby('community_vk_id')[col].transform('mean').var()/tot, len(a))
R = pd.DataFrame(rows); R.to_csv(O+'step3_robustness_descriptive.csv', index=False)
print(R.pivot_table(index=['metric','subset'], columns='variant', values='value', sort=False).round(3).to_string())
