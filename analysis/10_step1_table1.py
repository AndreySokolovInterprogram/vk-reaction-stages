import json, collections
import numpy as np, pandas as pd
from common import *
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
c, p, s = load()
# --- video types used to detect clips
vt = collections.Counter()
for a in pd.read_parquet(D+'posts.parquet', columns=['attachments']).attachments.dropna():
    for x in json.loads(a):
        if x.get('type') in ('video','clip','short_video') and not x.get('from_repost'):
            vt[(x['type'], x.get('video_type'))] += 1
print('video types:', vt.most_common())
print('posts by origin x modality'); print(pd.crosstab(p.origin, p.modality, margins=True))
print('eligible share by origin'); print(p.groupby('origin').eligible.mean())
print('ext12m non video/clip:', ((p.origin=='ext12m') & ~p.modality.isin(['video','clip'])).sum())
# --- table 1a: number of posts
for nm, pp in [('all', p), ('eligible', p[p.eligible])]:
    t = pp.groupby(['part','stratum','size_bucket','modality']).size().unstack('modality', fill_value=0).reindex(columns=MODS, fill_value=0)
    t['total'] = t.sum(axis=1); t.to_csv(O+f'table1a_posts_{nm}_part_stratum_size_modality.csv')
    for part in ['retro','live']:
        print(f'== {nm} {part}: stratum x modality'); x = pp[pp.part==part]
        print(pd.crosstab(x.stratum, x.modality, margins=True).reindex(columns=MODS+['All']))
        print(pd.crosstab(x.size_bucket, x.modality, margins=True).reindex(index=SIZES+['All'], columns=MODS+['All']))
# --- measurements
sp = s.merge(p[KEY+['part','origin','published_at']], on=KEY, how='left')
print('snapshot sources by origin'); print(pd.crosstab(sp.origin, sp.source, margins=True))
n = sp.groupby(KEY+['origin']).size().rename('n').reset_index()
ns = sp.groupby(KEY+['origin','source']).size().unstack('source', fill_value=0).reset_index()
print('snaps per post by origin'); print(n.groupby('origin').n.describe(percentiles=[.1,.25,.5,.75,.9]))
lv = ns[ns.origin=='live']
print('live: poll/getbyid per post'); print(lv[['poll','getbyid']].describe(percentiles=[.1,.25,.5,.75,.9]))
tg = sp[(sp.origin=='live') & sp.target_age_h.notna()].groupby(KEY).target_age_h.nunique()
print('live: distinct target marks per post'); print(tg.value_counts().sort_index())
print('posts without any snapshot:', (~p.set_index(KEY).index.isin(sp.set_index(KEY).index)).sum())
# --- last measurement and final value
ls = last_snap(s)[KEY+['measured_at','age_seconds','views','likes','reposts','comments','source']]
q = p.merge(ls, on=KEY, how='left')
q['age_d'] = q.age_seconds/86400
q['final'] = q.age_seconds >= 7*86400
print('age at last snapshot (days) by origin'); print(q.groupby('origin').age_d.describe(percentiles=[.01,.05,.25,.5,.75]))
print('final (>=7d) counts'); print(pd.crosstab(q.origin, q.final, margins=True))
print('final (>=7d) eligible'); print(pd.crosstab(q[q.eligible].origin, q[q.eligible].final, margins=True))
print('retro <7d by source of last snap'); print(q[(q.part=='retro') & ~q.final].groupby(['origin','source']).size())
# 72h for live
g72 = sp[(sp.origin=='live') & (sp.target_age_h==72)][KEY].drop_duplicates()
print('live posts with 72h mark:', len(g72), 'of', (p.origin=='live').sum())
# --- shares > 0 and quantiles
def summ(g):
    r = {'posts': len(g)}
    for k, col in [('V','views'),('R','reposts'),('L','likes'),('C','comments')]:
        v = g[col]
        r[f'{k}_na'] = v.isna().mean()
        r[f'{k}>0'] = (v > 0).mean()
        r[f'{k}_q25'], r[f'{k}_med'], r[f'{k}_q75'] = v.quantile([.25,.5,.75]).values
    return pd.Series(r)
rows = []
for keys, sub in [(('part',), ['part']), (('part','modality'), ['part','modality']), (('part','size_bucket'), ['part','size_bucket']), (('part','stratum'), ['part','stratum'])]:
    for nm, qq in [('all', q), ('eligible', q[q.eligible]), ('eligible_final', q[q.eligible & q.final])]:
        t = qq.groupby(sub).apply(summ).reset_index(); t.insert(0, 'pool', nm); t.insert(1,'by','×'.join(keys)); rows.append(t)
T = pd.concat(rows, ignore_index=True); T.to_csv(O+'table1b_shares_quantiles.csv', index=False)
print(T[T.by.isin(['part','part×modality'])].round(3).to_string())
# live at 72h mark
s72 = sp[(sp.origin=='live') & (sp.target_age_h==72) & sp.likes.notna()].drop_duplicates(KEY, keep='last').merge(p[KEY+['modality','eligible']], on=KEY)
t72 = s72[s72.eligible].groupby('modality').apply(summ); t72.loc['ALL'] = summ(s72[s72.eligible])
t72.to_csv(O+'table1c_live_72h_eligible.csv'); print('live @72h eligible'); print(t72.round(3).to_string())
# --- oddities
odd = {}
odd['snap_views_null'] = int(s.views.isna().sum()); odd['snap_views_zero'] = int((s.views==0).sum())
odd['last_views_null_posts'] = int(q.views.isna().sum()); odd['last_views_zero_posts'] = int((q.views==0).sum())
for col in ['likes','reposts','comments']:
    odd[f'last_V<{col}'] = int((q.views < q[col]).sum())
odd['dup_posts'] = int(p.duplicated(KEY).sum()); odd['dup_snap_post_time'] = int(s.duplicated(KEY+['measured_at']).sum())
odd['posts_no_stratum'] = int(p.stratum.isna().sum())
odd['retro_backfill_published_>90d_before_start'] = int(((p.origin=='backfill') & (p.published_at < panel_start(s) - pd.Timedelta(days=90))).sum())
odd['  of_them_pinned'] = int(((p.origin=='backfill') & (p.published_at < panel_start(s) - pd.Timedelta(days=90)) & p.is_pinned).sum())
odd['backfill_found_by_poll(first snap source=poll)'] = int(sp.sort_values('measured_at').drop_duplicates(KEY).query("origin=='backfill' and source=='poll'").shape[0])
odd['deleted_posts'] = int(p.deleted_at.notna().sum()); odd['deleted_live'] = int((p.deleted_at.notna() & (p.origin=='live')).sum())
odd['live_missed_targets>0'] = int(((p.origin=='live') & (p.missed_targets>0)).sum())
odd['live_missed_targets_sum'] = int(p.loc[p.origin=='live','missed_targets'].sum())
# counters decreasing between measurements of the same post
ss = s.sort_values(KEY+['measured_at'])
for col in ['views','likes','reposts','comments']:
    d = ss.groupby(KEY)[col].diff()
    odd[f'decrease_{col}_snaps'] = int((d < 0).sum())
    odd[f'decrease_{col}_posts'] = int(ss.loc[d < 0, KEY].drop_duplicates().shape[0])
# first measurement of live posts: delay after publication
fs = sp[sp.origin=='live'].groupby(KEY).age_seconds.min()/60
odd['live_first_snap_delay_min_median'] = round(fs.median(),1); odd['live_first_snap_delay_min_p95'] = round(fs.quantile(.95),1)
odd['live_first_snap_delay_>80min'] = int((fs > 80).sum())
for k, v in odd.items(): print(f'{k:50s} {v}')
pd.Series(odd).to_csv(O+'step1_oddities.csv', header=['value'])
q.drop(columns=[]).to_parquet(O+'posts_last_snapshot.parquet')
