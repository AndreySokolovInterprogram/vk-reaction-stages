"""Step 2: delay "published -> first measurement" for posts found by the polls of 12 Nov 10:09 and 11:15 UTC (collection gap)."""
import os
import numpy as np, pandas as pd
from common import *
p = pd.read_parquet(D+'posts.parquet', columns=KEY+['community_vk_id','origin','published_at','first_seen_at','missed_targets'])
s = pd.read_parquet(D+'snapshots.parquet')
c = pd.read_parquet(D+'communities.parquet')[['community_vk_id','stratum','size_bucket']]
lv = p[p.origin=='live'].merge(c, on='community_vk_id')
fs = s.sort_values('measured_at').drop_duplicates(KEY)[KEY+['measured_at','age_seconds','source']].rename(columns={'measured_at':'first_snap_at','age_seconds':'first_snap_age_s','source':'first_snap_source'})
lv = lv.merge(fs, on=KEY, how='left')
m1 = s[s.target_age_h==1]
lv['mark1_snapshot'] = lv.set_index(KEY).index.isin(m1.set_index(KEY).index)
lv['mark1_nonnull'] = lv.set_index(KEY).index.isin(m1[m1.likes.notna()].set_index(KEY).index)
lv['delay_found_min'] = (lv.first_seen_at - lv.published_at).dt.total_seconds()/60
lv['delay_first_snap_min'] = lv.first_snap_age_s/60
lv['mark1_out_of_tolerance'] = lv.delay_found_min > 80  # 1-h mark: 60 min + tolerance max(20 min, 6 min)
tg = pd.read_csv(TECH_GAP) if os.path.exists(TECH_GAP) else pd.DataFrame(columns=KEY)
lv['in_tech_gap_list'] = lv.set_index(KEY).index.isin(tg.set_index(KEY).index)
fsr = lv.first_seen_at.dt.floor('min')
lv['group'] = np.select([fsr==pd.Timestamp('2026-11-12 10:09',tz='UTC'), fsr==pd.Timestamp('2026-11-12 11:15',tz='UTC'),
                         lv.first_seen_at.dt.date.astype(str).isin(['2026-11-10','2026-11-11']),
                         lv.first_seen_at.dt.date.astype(str)=='2026-11-12'],
                        ['poll_12.11_10:09','poll_12.11_11:15','baseline_10-11.11','other_12.11'], 'other')
# published before the previous poll (should have been found earlier)
lv['published_before_prev_poll'] = False
polls = np.sort(lv.first_seen_at.dt.floor('min').unique())
prev = {polls[i]: polls[i-1] for i in range(1, len(polls))}
lv['prev_poll'] = fsr.map(prev)
lv['published_before_prev_poll'] = lv.published_at < lv.prev_poll
def summ(g):
    d = g.delay_found_min
    return pd.Series({'posts': len(g), 'polls': g.first_seen_at.dt.floor('min').nunique(),
        'posts_per_poll': len(g)/g.first_seen_at.dt.floor('min').nunique(),
        'delay_med_min': d.median(), 'delay_p90_min': d.quantile(.9), 'delay_max_min': d.max(),
        'delay_>75min': int((d>75).sum()), 'mark1_out_of_tol_n': int(g.mark1_out_of_tolerance.sum()),
        'mark1_out_of_tol_share': g.mark1_out_of_tolerance.mean(),
        'mark1_snapshot_missing_n': int((~g.mark1_snapshot).sum()), 'mark1_missing_share': (~g.mark1_snapshot).mean(),
        'mark1_nonnull_missing_share': (~g.mark1_nonnull).mean(),
        'published_before_prev_poll_n': int(g.published_before_prev_poll.sum()),
        'in_tech_gap_list_n': int(g.in_tech_gap_list.sum())})
S = lv.groupby('group').apply(summ).T
S.to_csv(O+'step2_summary.csv'); print(S.round(3).to_string())
cols = KEY+['community_vk_id','stratum','size_bucket','published_at','first_seen_at','prev_poll','published_before_prev_poll','first_snap_at','first_snap_source','delay_found_min','delay_first_snap_min','mark1_out_of_tolerance','mark1_snapshot','mark1_nonnull','missed_targets','in_tech_gap_list','group']
lv[lv.group.str.startswith('poll_12.11')][cols].sort_values(['group','published_at']).to_csv(O+'step2_tech_gap_poll_posts.csv', index=False)
b = lv[lv.group=='baseline_10-11.11']
pp = b.groupby(b.first_seen_at.dt.floor('min')).agg(n=('delay_found_min','size'), out=('mark1_out_of_tolerance','sum'))
print('baseline per poll: posts', pp.n.describe().round(1).to_dict()); print('baseline polls with any mark1 out of tol', (pp.out>0).sum(), 'of', len(pp), 'max out per poll', pp.out.max())
g = lv[lv.group=='poll_12.11_11:15']; print('11:15 by size', g.groupby('size_bucket').mark1_out_of_tolerance.agg(['size','sum']).to_dict())
print('11:15 out-of-tol by stratum', g[g.mark1_out_of_tolerance].stratum.value_counts().to_dict())
print('11:15: delay of posts published before 10:09', g[g.published_before_prev_poll].delay_found_min.describe().round(1).to_dict())
print('consistency: out_of_tol but mark1 present', (lv.mark1_out_of_tolerance & lv.mark1_snapshot).sum(), ' in-tol but mark1 missing', (~lv.mark1_out_of_tolerance & ~lv.mark1_snapshot & (lv.published_at < pd.Timestamp('2026-11-12 17:30',tz='UTC'))).sum())
x = lv[~lv.mark1_out_of_tolerance & ~lv.mark1_snapshot & (lv.published_at < pd.Timestamp('2026-11-12 17:30',tz='UTC'))]
print('  those by group', x.group.value_counts().to_dict(), ' in tech gap list', x.in_tech_gap_list.sum())
