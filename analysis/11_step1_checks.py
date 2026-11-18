import numpy as np, pandas as pd
from common import *
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
c, p, s = load()
q = pd.read_parquet(O+'posts_last_snapshot.parquet')
START = panel_start(s); EXP = export_time()
# 1 posts without snaps
nos = q[q.measured_at.isna()]
print('no-snap posts by origin', nos.origin.value_counts().to_dict())
print('  published range', nos.published_at.min(), nos.published_at.max(), 'pinned', nos.is_pinned.sum())
print('  age at first_seen (d) quantiles', ((nos.first_seen_at-nos.published_at).dt.total_seconds()/86400).quantile([0,.5,1]).round(1).to_dict())
print('  first_seen range', nos.first_seen_at.min(), nos.first_seen_at.max())
# 2 views null
w = q[q.measured_at.notna()]
print('V null by origin x modality'); print(pd.crosstab(w.origin, [w.modality], values=w.views.isna(), aggfunc='mean').round(3))
print('V null by last source'); print(w.groupby('source').views.apply(lambda v: v.isna().mean()).round(3))
print('V null by repost'); print(w.groupby('is_repost').views.apply(lambda v: v.isna().mean()).round(3))
cm = w.groupby('community_vk_id').views.apply(lambda v: v.isna().mean())
print('communities with V null share: ==1:', (cm==1).sum(), ' >0.5:', (cm>0.5).sum(), ' 0:', (cm==0).sum(), ' of', len(cm))
vn = w[w.views.isna()]
print('V-null posts concentrated: top10 communities share', vn.community_vk_id.value_counts().head(10).sum()/len(vn))
print('V null vs published year'); print(w.groupby(w.published_at.dt.year).views.apply(lambda v: v.isna().mean()).round(3))
cmv = cm.rename('vnull').reset_index().merge(c[['community_vk_id','stratum','size_bucket','name']], on='community_vk_id')
print(cmv[cmv.vnull==1].groupby(['stratum','size_bucket']).size())
# V null at all snapshots of post vs only some (live)
sl = s.merge(p[KEY+['origin']], on=KEY)
lv = sl[sl.origin=='live'].groupby(KEY).views.agg(lambda v: (v.isna().all(), v.isna().any()))
print('live posts: V null in all snaps', sum(a for a,b in lv), ' in some', sum(b for a,b in lv))
# null likes etc (counters null)
print('snap likes null by source', sl.groupby('source').likes.apply(lambda v: v.isna().sum()).to_dict())
ln = sl[sl.likes.isna()]
print('likes-null snaps: views null too', ln.views.isna().mean(), ' origin', ln.origin.value_counts().to_dict(), ' target', ln.target_age_h.value_counts(dropna=False).head(8).to_dict())
# 3 V<R
vr = w[w.views < w.reposts]
print('V<R posts', len(vr), 'by origin', vr.origin.value_counts().to_dict(), 'modality', vr.modality.value_counts().to_dict(), 'repost', vr.is_repost.sum())
print(vr[['views','reposts','likes','comments','age_d','stratum']].describe().round(1))
print(' communities', vr.community_vk_id.nunique(), ' top5', vr.community_vk_id.value_counts().head(5).to_dict())
print(' published year', vr.published_at.dt.year.value_counts().to_dict())
# 4 live eligibility
lv = p[p.origin=='live']
print('live non-eligible: repost', lv.is_repost.sum(), 'pinned', lv.is_pinned.sum(), 'ads', lv.marked_as_ads.sum())
# 5 72h coverage among live posts old enough
old = lv[lv.published_at <= EXP - pd.Timedelta(hours=72*1.1)]
h72 = sl[(sl.origin=='live') & (sl.target_age_h==72)][KEY].drop_duplicates()
print('live posts published >=79.2h before export:', len(old), ' with 72h mark:', old.set_index(KEY).index.isin(h72.set_index(KEY).index).sum())
for T in [1,3,6,12,24,48,72]:
    tol = max(20/60, 0.1*T)
    el = lv[lv.published_at <= EXP - pd.Timedelta(hours=T+tol)]
    have = sl[(sl.origin=='live') & (sl.target_age_h==T)][KEY].drop_duplicates()
    k = el.set_index(KEY).index.isin(have.set_index(KEY).index).sum()
    print(f'  mark {T:>2}h: due {len(el)}, got {k} ({k/len(el):.1%})')
# fact age of target snaps relative to target
g = sl[sl.target_age_h.notna()].copy(); g['dev_min'] = g.age_seconds/60 - g.target_age_h*60
print('target snaps deviation from target (min)'); print(g.groupby('target_age_h').dev_min.describe(percentiles=[.05,.5,.95]).round(1))
# 6 backfill posts found by poll: retro part with live-like measurement
fp = sl.sort_values('measured_at').drop_duplicates(KEY)
bp = fp[(fp.origin=='backfill') & (fp.source=='poll')]
print('backfill-by-poll: n', len(bp), ' first-snap age (h) quantiles', (bp.age_seconds/3600).quantile([0,.25,.5,.75,1]).round(1).to_dict())
# 7 ext12m 2 snaps?
print('ext12m posts with 2 snaps', (sl[sl.origin=="ext12m"].groupby(KEY).size()==2).sum())
# 8 backfill >90d
o = p[(p.origin=='backfill') & (p.published_at < START - pd.Timedelta(days=90))]
print('backfill >90d: n', len(o), 'pinned', o.is_pinned.sum(), 'modality', o.modality.value_counts().to_dict(), 'by year', o.published_at.dt.year.value_counts().sort_index().to_dict())
print('   published in last 1 day before 90d boundary?', (o.published_at > START - pd.Timedelta(days=91)).sum())
# 9 community_snapshots
cs = pd.read_parquet(D+'community_snapshots.parquet')
print('community snaps per community', cs.groupby('community_vk_id').size().value_counts().to_dict(), 'closed', cs.is_closed.sum(), 'deact', cs.deactivated.notna().sum())
