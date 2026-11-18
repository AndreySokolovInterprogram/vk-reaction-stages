"""Aggregated data for Figs. 5 and 6. Read only; the main analysis is not recomputed.

Sources (environment variables, see config.example.env):
  EXPORT_DIR   - VK panel export (posts/snapshots/communities);
  ANALYSIS_OUT - results of the main analysis (analysis/run_all.sh): pools.parquet from 30_pools.py
                 (pool pool_rho, exposure expo).
Rules follow analysis/31_step3_descriptive.py (steps 3.2 and 3.3).

Usage:  python figdata/make_figdata.py [output_dir] [--full]
  default output_dir: data/ of the repository.
  Without --full, figdata_shares_posts.csv is the public version: modality and cm_* only (no stratum or community
  number; rows shuffled with a fixed seed), so posts cannot be grouped by community.
  With --full: all columns (modality, stratum, community_rank, log_rho_*, c_*, cm_*).

Output 1, figdata_shares_posts.csv - one row per post of pool_rho (exposure >= 100; text/image/video - post views,
  clip - views of the main video; no auto-posting, no communities with missing V).
  log_rho_X = log((X + 0.5) / expo). Centering within community x block (block: clip / not clip), as in step 3.2:
    c_*   - minus the MEAN,
    cm_*  - minus the MEDIAN (rule of step 3.2; gives 0.40/0.27 and 0.49/0.30 reported in the paper).
Output 2, figdata_accumulation.csv - live posts, eligible, all communities (as in step 3.3).
  Reference value: scheduled getbyid measurement with target_age_h = 72 (as in step 3.3; actual deviation of the
  scheduled measurement from 72 h up to ~4 h). Posts with a missing or zero counter at 72 h are excluded for that
  counter.
  Value at age t:
    t in the panel schedule (1, 3, 6, 12, 24, 48, 72) - scheduled measurement target_age_h = t (as in step 3.3);
    other t (2, 4, 8, 16, 20, 30, 36, 60) - nearest non-null measurement by age (poll or getbyid),
    |age - t| <= 0.6 h (half of the wall-poll step, median step 1.17 h).
  Null measurements are skipped, not replaced by the last non-null value (as in step 3.3).
  share_of_sums = sum of values at t / sum of values at 72 h over posts measured at t; this is frac_of_sums of
  step 3.3 and the quantity reported in the paper (65/63/72/49%).
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(REPO, 'analysis')
if 'ANALYSIS_OUT' in os.environ:
    os.environ['OUT_DIR'] = os.environ['ANALYSIS_OUT']  # pools.parquet is only read
sys.path.insert(0, ANALYSIS)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from common import load, KEY, MODS, O

FULL = '--full' in sys.argv
args = [a for a in sys.argv[1:] if a != '--full']
DEST = args[0] if args else os.path.join(REPO, 'data')
os.makedirs(DEST, exist_ok=True)

# ================= Output 1
q = pd.read_parquet(O + 'pools.parquet')
r = q[q.pool_rho].copy()
r['block'] = np.where(r.modality == 'clip', 'clip', 'tiv')
ranks = {cid: i + 1 for i, cid in enumerate(r[['stratum', 'community_vk_id']].drop_duplicates().sort_values(['stratum', 'community_vk_id']).community_vk_id)}
r['community_rank'] = r.community_vk_id.map(ranks)
grp = r.groupby(['community_vk_id', 'block'])
for k, col in [('R', 'reposts'), ('L', 'likes'), ('C', 'comments')]:
    r['log_rho_' + k] = np.log((r[col] + 0.5) / r.expo)
    r['c_log_rho_' + k] = r['log_rho_' + k] - grp['log_rho_' + k].transform('mean')
    r['cm_log_rho_' + k] = r['log_rho_' + k] - grp['log_rho_' + k].transform('median')
cols = ['modality', 'stratum', 'community_rank'] + [p + 'log_rho_' + k for p in ['', 'c_', 'cm_'] for k in 'RLC']
out1 = r.sort_values(['community_rank', 'modality'])[cols]
if FULL:
    out1.to_csv(DEST + '/figdata_shares_posts.csv', index=False, float_format='%.6g')
else:  # public version: no stratum or community number, rows in random order (fixed seed)
    pub = out1[['modality', 'cm_log_rho_R', 'cm_log_rho_L', 'cm_log_rho_C']].sample(frac=1, random_state=20261118)
    pub.to_csv(DEST + '/figdata_shares_posts.csv', index=False, float_format='%.6g')
print('figdata_shares_posts rows', len(out1), ' communities', out1.community_rank.nunique(), ' by modality', out1.modality.value_counts().to_dict())
rows = []
for name, g in [('text/image/video', out1[out1.modality != 'clip'])] + [(m, out1[out1.modality == m]) for m in ['clip'] + [x for x in MODS if x != 'clip']] + [('all (mixed blocks)', out1)]:
    for a, b in [('R', 'L'), ('C', 'L'), ('R', 'C')]:
        rows.append({'subset': name, 'pair': f'{a}-{b}', 'n': len(g),
                     'spearman_c_mean': spearmanr(g['c_log_rho_' + a], g['c_log_rho_' + b]).statistic,
                     'spearman_cm_median': spearmanr(g['cm_log_rho_' + a], g['cm_log_rho_' + b]).statistic,
                     'spearman_raw': spearmanr(g['log_rho_' + a], g['log_rho_' + b]).statistic})
sp = pd.DataFrame(rows); sp.to_csv(DEST + '/check_spearman.csv', index=False, float_format='%.4f')
print(sp.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

# ================= Output 2
c, p, s = load()
lv = p[(p.origin == 'live') & p.eligible][KEY + ['modality']]
m = s.merge(lv, on=KEY)
m['age_h'] = m.age_seconds / 3600
sched = m[m.target_age_h.notna() & m.likes.notna()].drop_duplicates(KEY + ['target_age_h'], keep='last')  # as in step 3.3
SCHED = [1, 3, 6, 12, 24, 48, 72]
GRID = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 30, 36, 48, 60, 72]
TOL = 0.6
COUNTERS = ['views', 'likes', 'comments', 'reposts']
mod_of = lv.set_index(KEY).modality
out2 = []
for col in COUNTERS:
    W = sched.pivot_table(index=KEY, columns='target_age_h', values=col)
    ref = W[72][W[72].notna() & (W[72] > 0)]
    vals = {}
    for t in GRID:
        if t in SCHED:
            vals[t] = W[t].reindex(ref.index)
        else:
            cand = m[m[col].notna() & ((m.age_h - t).abs() <= TOL)].copy()
            cand['d'] = (cand.age_h - t).abs()
            near = cand.sort_values('d').drop_duplicates(KEY).set_index(KEY)[col]
            vals[t] = near.reindex(ref.index)
    mods = mod_of.reindex(ref.index)
    for t in GRID:
        v = vals[t]
        for mod in ['ALL'] + MODS:
            sel = v.notna() if mod == 'ALL' else v.notna() & (mods == mod)
            f = v[sel] / ref[sel]
            out2.append({'modality': mod, 'counter': col, 'age_h': t,
                         'rule': 'scheduled target_age_h' if t in SCHED else f'nearest snapshot |age-t|<={TOL}h',
                         'n_posts': int(sel.sum()), 'n_posts_72h_pos': int(len(ref) if mod == 'ALL' else (mods == mod).sum()),
                         'median_share': f.median(), 'q25': f.quantile(.25), 'q75': f.quantile(.75), 'mean_share': f.mean(),
                         'share_of_sums': v[sel].sum() / ref[sel].sum() if sel.any() else np.nan})
out2 = pd.DataFrame(out2)
out2.to_csv(DEST + '/figdata_accumulation.csv', index=False, float_format='%.6g')
print(out2[(out2.age_h == 24) & (out2.modality == 'ALL')].to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print(out2[out2.modality == 'ALL'].pivot_table(index='counter', columns='age_h', values='share_of_sums').round(2).to_string())
print(out2[out2.modality == 'ALL'].pivot_table(index='counter', columns='age_h', values='n_posts').to_string())
