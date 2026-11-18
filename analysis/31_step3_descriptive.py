"""Steps 3.1-3.4 and 3.6 (descriptive). Clips form a separate block (exposure = main-video views)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from common import *
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
q = pd.read_parquet(O+'pools.parquet')
q['block'] = np.where(q.modality=='clip', 'clip (exposure = video_views)', 'text/image/video')
def qs(v, pre):
    v = v.dropna(); a = v.quantile([.25,.5,.75]).values
    return {f'{pre}_q25': a[0], f'{pre}_med': a[1], f'{pre}_q75': a[2]}
# ---------- 3.1
r = q[q.pool_rho].copy()
for k, col in [('R','reposts'),('L','likes'),('C','comments')]: r['rho_'+k] = r[col]/r.expo
def s31(g):
    d = {'posts': len(g), 'communities': g.community_vk_id.nunique()}
    for k in 'RLC': d.update(qs(g['rho_'+k], 'rho_'+k))
    d['R_zero_share'] = (g.reposts==0).mean(); d['L_zero_share'] = (g.likes==0).mean(); d['C_zero_share'] = (g.comments==0).mean()
    return pd.Series(d)
t31 = pd.concat([r.groupby(['block','stratum','modality']).apply(s31),
                 r.groupby(['block','modality']).apply(s31).assign(stratum='ALL').set_index('stratum', append=True).reorder_levels([0,2,1])])
t31.to_csv(O+'step3_1_rho_by_stratum_modality.csv')
print(t31.xs('ALL', level='stratum').to_string(float_format=lambda x: f'{x:.4g}'))
# ---------- 3.2
def lr(x, v): return np.log((x+0.5)/v)
for k, col in [('R','reposts'),('L','likes'),('C','comments')]: r['lr_'+k] = lr(r[col], r.expo)
for k in 'RLC': r['lrw_'+k] = r['lr_'+k] - r.groupby(['community_vk_id','block'])['lr_'+k].transform('median')  # community median within block
rows = []
def sp(a, b): 
    m = a.notna() & b.notna(); rho = spearmanr(a[m], b[m]).statistic; return rho, int(m.sum())
for blk, g in [('text/image/video', r[r.modality!='clip']), ('clip (exposure = video_views)', r[r.modality=='clip'])] + [(f'mod={m}', r[r.modality==m]) for m in MODS]:
    pos = g[(g.reposts>0)&(g.likes>0)&(g.comments>0)]
    for pair in [('R','L'),('C','L'),('R','C')]:
        a, b = pair
        rows.append({'subset': blk, 'pair': f'log rho_{a} vs log rho_{b}', 'version': 'posts, +0.5', 'spearman': sp(g['lr_'+a], g['lr_'+b])[0], 'n': len(g)})
        rows.append({'subset': blk, 'pair': f'log rho_{a} vs log rho_{b}', 'version': 'within communities (minus median), +0.5', 'spearman': sp(g['lrw_'+a], g['lrw_'+b])[0], 'n': len(g)})
        rows.append({'subset': blk, 'pair': f'log rho_{a} vs log rho_{b}', 'version': 'R,L,C>0 without correction, within communities',
                     'spearman': sp(np.log(pos[{'R':'reposts','L':'likes','C':'comments'}[a]]/pos.expo) - np.log(pos[{'R':'reposts','L':'likes','C':'comments'}[a]]/pos.expo).groupby(pos.community_vk_id).transform('median'),
                                    np.log(pos[{'R':'reposts','L':'likes','C':'comments'}[b]]/pos.expo) - np.log(pos[{'R':'reposts','L':'likes','C':'comments'}[b]]/pos.expo).groupby(pos.community_vk_id).transform('median'))[0], 'n': len(pos)})
# correlations of counters R-L-C (V not needed): pool_ratio, all communities
pr = q[q.pool_ratio].copy()
for k, col in [('R','reposts'),('L','likes'),('C','comments')]:
    pr['lc_'+k] = np.log(pr[col]+0.5); pr['lcw_'+k] = pr['lc_'+k] - pr.groupby('community_vk_id')['lc_'+k].transform('median')
for blk, g in [('all modalities', pr)] + [(f'mod={m}', pr[pr.modality==m]) for m in MODS]:
    for a, b in [('R','L'),('C','L'),('R','C')]:
        rows.append({'subset': blk, 'pair': f'log {a} vs log {b} (counters)', 'version': 'posts', 'spearman': sp(g['lc_'+a], g['lc_'+b])[0], 'n': len(g)})
        rows.append({'subset': blk, 'pair': f'log {a} vs log {b} (counters)', 'version': 'within communities', 'spearman': sp(g['lcw_'+a], g['lcw_'+b])[0], 'n': len(g)})
t32 = pd.DataFrame(rows); t32.to_csv(O+'step3_2_spearman.csv', index=False)
print(t32[t32.subset.isin(['text/image/video','clip (exposure = video_views)','all modalities'])].to_string(index=False, float_format=lambda x: f'{x:.3f}'))
# distribution of log(R/L), log(C/L): pool_ratio, clips included
pr['lRL'] = np.log((pr.reposts+0.5)/(pr.likes+0.5)); pr['lCL'] = np.log((pr.comments+0.5)/(pr.likes+0.5))
def s32(g):
    d = {'posts': len(g), 'L>0_share': (g.likes>0).mean()}
    d.update(qs(g.lRL, 'logRL_adj')); d.update(qs(g.lCL, 'logCL_adj'))
    a = g[(g.reposts>0)&(g.likes>0)]; d['n_R,L>0'] = len(a); d.update(qs(np.log(a.reposts/a.likes), 'logRL_pos'))
    b = g[(g.comments>0)&(g.likes>0)]; d['n_C,L>0'] = len(b); d.update(qs(np.log(b.comments/b.likes), 'logCL_pos'))
    return pd.Series(d)
t32b = pd.concat([pr.groupby(['stratum','modality']).apply(s32), pr.groupby('modality').apply(s32).assign(stratum='ALL').set_index('stratum', append=True).swaplevel(),
                  pr.groupby('stratum').apply(s32).assign(modality='ALL').set_index('modality', append=True)])
t32b.to_csv(O+'step3_2_logRL_logCL_by_stratum_modality.csv')
print(t32b.xs('ALL', level='stratum').round(3).to_string()); print(t32b.xs('ALL', level='modality').round(3).to_string())
# variance of log(R/L) between vs within strata
for col in ['lRL','lCL']:
    tot = pr[col].var(); bs = pr.groupby('stratum')[col].transform('mean').var(); bc = pr.groupby('community_vk_id')[col].transform('mean').var()
    print(f'{col}: variance share between strata {bs/tot:.3f}, between communities {bc/tot:.3f}')
# ---------- 3.3 dynamics (live, eligible, all communities; V where not missing)
c, p, s = load()
lv = p[(p.origin=='live') & p.eligible][KEY+['modality','community_vk_id']]
m = s[s.target_age_h.notna() & s.likes.notna()].merge(lv, on=KEY)
m = m.drop_duplicates(KEY+['target_age_h'], keep='last')
w = {col: m.pivot_table(index=KEY+['modality'], columns='target_age_h', values=col) for col in ['views','reposts','likes','comments']}
rows = []
for col, W in w.items():
    W = W[W[72].notna() & (W[72] > 0)]
    for mod, g in [('ALL', W)] + [(mm, W.xs(mm, level='modality', drop_level=False)) for mm in MODS]:
        for T in [1,3,6,12,24,48]:
            f = (g[T]/g[72]).dropna()
            gg = g[g[T].notna()]
            rows.append({'counter': col, 'modality': mod, 'T_h': T, 'posts_with_72h>0': len(g), 'n': len(f), 'frac_median': f.median(),
                         'frac_q25': f.quantile(.25), 'frac_q75': f.quantile(.75), 'frac_mean': f.mean(), 'frac_of_sums': gg[T].sum()/gg[72].sum()})
t33 = pd.DataFrame(rows); t33.to_csv(O+'step3_3_live_dynamics.csv', index=False)
print(t33.pivot_table(index=['counter','modality'], columns='T_h', values='frac_median').round(2).to_string())
print(t33[t33.T_h==1][['counter','modality','posts_with_72h>0']].to_string(index=False))
# ---------- 3.4 reach
cs = pd.read_parquet(D+'community_snapshots.parquet').groupby('community_vk_id').members_count.median().rename('members')
a = q[q.pool_rho_all].merge(cs, on='community_vk_id'); a['reach'] = a.expo/a.members  # clips: video_views (platform-wide views)
def s34(g):
    d = {'posts': len(g)}; d.update(qs(g.reach, 'V_per_member')); d['share_V>members'] = (g.reach>1).mean(); return pd.Series(d)
t34 = pd.concat([a.groupby(['block','size_bucket']).apply(s34), a.groupby(['block','size_bucket','modality']).apply(s34).reset_index('modality').query("modality!='clip' or block.str.startswith('clip')", engine='python').set_index('modality', append=True)])
t34.to_csv(O+'step3_4_reach_by_size.csv'); print(a.groupby(['block','size_bucket']).apply(s34).round(4).to_string())
# copy_history
rp = q[q.is_repost & ~q.drop_basic].copy()
panel_owners = set(-c.community_vk_id)
rp['src_is_panel'] = rp.repost_source_owner_id.isin(panel_owners)
keys = set(zip(q.vk_owner_id, q.vk_post_id))
rp['src_in_data'] = [ (o, pid) in keys for o, pid in zip(rp.repost_source_owner_id, rp.repost_source_post_id)]
print('reposts (copy_history):', len(rp), ' by part', rp.part.value_counts().to_dict(), ' with copy V non-null', int(rp.views.notna().sum()),
      ' V>0', int((rp.views>0).sum()), ' source = own community', int((rp.repost_source_owner_id==rp.vk_owner_id).sum()),
      ' source = another panel community', int((rp.src_is_panel & (rp.repost_source_owner_id!=rp.vk_owner_id)).sum()), ' source post in data', int(rp.src_in_data.sum()))
src = q.set_index(KEY)[['views','reposts','modality']].rename(columns=lambda x: 'src_'+x)
pairs = rp[rp.src_in_data].merge(src, left_on=['repost_source_owner_id','repost_source_post_id'], right_index=True)
pairs['V_copy_over_V_src'] = pairs.views/pairs.src_views
print('pairs copy/source with both V:', int(pairs[['views','src_views']].notna().all(axis=1).sum()), ' median V_copy/V_src', pairs.V_copy_over_V_src.median())
rs = rp.groupby('part').apply(lambda g: pd.Series({'reposts_posts': len(g), 'copy_V_nonnull': int(g.views.notna().sum()), 'copy_V_med': g.views.median(), 'copy_V_q75': g.views.quantile(.75),
     'src_other_panel_comm': int((g.src_is_panel & (g.repost_source_owner_id!=g.vk_owner_id)).sum()), 'src_same_comm': int((g.repost_source_owner_id==g.vk_owner_id).sum()), 'src_post_in_data': int(g.src_in_data.sum())}))
rs.to_csv(O+'step3_4_copy_history.csv'); print(rs.to_string())
pairs[KEY+['repost_source_owner_id','repost_source_post_id','views','src_views','V_copy_over_V_src']].to_csv(O+'step3_4_copy_source_pairs.csv', index=False)
# ---------- 3.6 viral posts
# Main variant: the official v3 rule on text/image/video only (community median V and stratum 90th percentile of R/V
# computed without clips). Clips: v3 with exposure video_views: condition 1 - video_views >= 3 x community median
# video_views of clips, condition 2 - R/video_views in the top 10% of clips of the stratum (thresholds within clips).
# Pool: retrospective, eligible, in window, exposure > 0, age at the last non-null measurement >= 72 h, no auto-posting
# (refined rule). Robustness: (b) R/V threshold within stratum x modality; main variant with the original auto-posting
# rule and without exclusion; for reference, the official v3 on all modalities with post V.
def v3(d, thr_by, e='views'):
    c1 = d[e] >= 3*d.groupby('community_vk_id')[e].transform('median')
    c2 = d.share >= d.groupby(thr_by).share.transform(lambda x: x.quantile(.9))
    return c1 & c2
base36 = q.eligible & ~q.drop_basic & (q.part=='retro') & (q.age_seconds >= 72*3600)
V0 = q[base36 & (q.views>0)].copy(); V0['share'] = V0.reposts/V0.views          # text/image/video (post V)
v = V0[~V0.autopost & (V0.modality!='clip')].copy()
v['viral_main'] = v3(v, ['stratum'])
v['viral_b_cell'] = v3(v, ['stratum','modality'])
def nc_only(d):
    d = d[d.modality!='clip'].copy(); d['x'] = v3(d, ['stratum']); return d.set_index(KEY).x
v['viral_main_origrule'] = v.set_index(KEY).index.map(nc_only(V0[~V0.autopost_orig])).fillna(False).astype(bool).values
v['viral_main_noexcl'] = v.set_index(KEY).index.map(nc_only(V0)).fillna(False).astype(bool).values
ref = V0[~V0.autopost].copy(); ref['viral_v3_all_mod_ref'] = v3(ref, ['stratum'])
cl = q[base36 & ~q.autopost & (q.modality=='clip') & (q.video_views_main>0)].copy(); cl['share'] = cl.reposts/cl.video_views_main
cl['viral_main'] = v3(cl, ['stratum'], 'video_views_main')
cl['viral_clip_noexcl'] = q[base36 & (q.modality=='clip') & (q.video_views_main>0)].assign(share=lambda d: d.reposts/d.video_views_main).pipe(
    lambda d: v3(d, ['stratum'], 'video_views_main')).reindex(cl.index).values
print('clip v3 pool: clips with video_views>0', len(cl), ' of clips in base pool', int((base36 & ~q.autopost & (q.modality=='clip')).sum()))
cl['viral_main_noexcl'] = cl.viral_clip_noexcl
allv = pd.concat([v, cl], ignore_index=True)
K = ['viral_main','viral_b_cell','viral_main_origrule','viral_main_noexcl']
def tab(d):
    g = d.groupby(['stratum','modality'])
    return pd.DataFrame({'posts': g.size(), **{k: g[k].sum(min_count=1) for k in K}})
t36 = pd.concat([tab(allv), tab(allv.assign(stratum='ALL')),
                 tab(v.assign(stratum='ALL', modality='text+image+video')), tab(allv.assign(stratum='ALL', modality='ALL (with clips)'))])
t36['viral_v3_all_mod_ref'] = ref.groupby(['stratum','modality']).viral_v3_all_mod_ref.sum().reindex(t36.index)
for k in K: t36[k+'_pct'] = 100*t36[k]/t36.posts
clip_rows = t36.index.get_level_values('modality')=='clip'
t36.loc[clip_rows, ['viral_b_cell','viral_main_origrule','viral_b_cell_pct','viral_main_origrule_pct']] = np.nan
t36['note'] = np.where(clip_rows, 'clips: exposure = video_views of the main video (post reactions / platform-wide video views)', '')
t36.to_csv(O+'step3_6_viral_v3.csv')
print(t36.xs('ALL', level='stratum').drop(columns='note').round(3).to_string())
print('viral main text/image/video', int(v.viral_main.sum()), ' clips', int(cl.viral_main.sum()), ' (b) t/i/v', int(v.viral_b_cell.sum()),
      ' orig rule t/i/v', int(v.viral_main_origrule.sum()), ' no excl t/i/v', int(nc_only(V0).sum()), ' clips no excl', int(cl.viral_clip_noexcl.sum()))
pt = allv.pivot_table(index='stratum', columns='modality', values='viral_main', aggfunc='sum').reindex(columns=MODS)
print('viral_main (text/image/video + clip) empty cells (of 40):', int((pt==0).sum().sum()), ' of 30 t/i/v:', int((pt[['text','image','video']]==0).sum().sum()), ' clip:', int((pt['clip']==0).sum()))
print(pt.to_string()); pt.to_csv(O+'step3_6_viral_main_stratum_x_modality.csv')
ptb = v.pivot_table(index='stratum', columns='modality', values='viral_b_cell', aggfunc='sum').reindex(columns=['text','image','video'])
print('viral_b_cell empty cells (of 30):', int((ptb==0).sum().sum()))
cols36 = KEY+['community_vk_id','stratum','modality','views','video_views_main','reposts']+K
allv.reindex(columns=cols36).query('viral_main or viral_b_cell == True').to_csv(O+'step3_6_viral_posts.csv', index=False)
