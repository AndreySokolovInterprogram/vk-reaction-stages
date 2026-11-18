"""share_cta (call to share / giveaway): shares, medians with/without, NB regression of R and L with the indicator,
test of the hypothesis "the food effect on R is explained by giveaways", 30 random food texts (local check only)."""
import warnings; warnings.filterwarnings('ignore')
import re
import numpy as np, pandas as pd, statsmodels.api as sm
from common import *
q = pd.read_parquet(O+'pools.parquet'); T = pd.read_parquet(O+'share_cta_texts.parquet')
a = q[q.pool_ratio].copy(); r = q[q.pool_rho].copy(); n_all = q[q.pool_rho_all].copy()
# 1. share of share_cta posts
sh = a.pivot_table(index='stratum', columns='modality', values='share_cta', aggfunc='mean').reindex(columns=MODS)
sh['ALL'] = a.groupby('stratum').share_cta.mean(); sh.loc['ALL'] = list(a.groupby('modality').share_cta.mean().reindex(MODS)) + [a.share_cta.mean()]
sh.to_csv(O+'step3_cta_share_by_stratum_modality.csv'); print('share_cta (pool_ratio, main)'); print((100*sh).round(1).to_string())
print('n share_cta posts in pool_ratio', int(a.share_cta.sum()), 'of', len(a))
# 2. medians with/without
r['rho_R'] = r.reposts/r.expo; r['rho_L'] = r.likes/r.expo
a['lRL_adj'] = np.log((a.reposts+.5)/(a.likes+.5))
rows = []
for by in [['modality'], ['stratum'], ['stratum','modality']]:
    for keys, g in r.groupby(by + ['share_cta']):
        rows.append(dict(zip(by + ['share_cta'], keys), pool='rho (V>=100)', n=len(g), rho_R_med_x1e3=1e3*g.rho_R.median(), rho_L_med_x1e3=1e3*g.rho_L.median()))
    for keys, g in a.groupby(by + ['share_cta']):
        pos = g[(g.reposts>0)&(g.likes>0)]
        rows.append(dict(zip(by + ['share_cta'], keys), pool='ratio', n=len(g), logRL_adj_med=g.lRL_adj.median(), n_RL_pos=len(pos), logRL_pos_med=np.log(pos.reposts/pos.likes).median() if len(pos) else np.nan))
M = pd.DataFrame(rows); M.to_csv(O+'step3_cta_medians.csv', index=False)
mm = M[M.stratum.isna() & M.modality.notna()] if 'stratum' in M else M
print(M[M['stratum'].isna()].drop(columns=['stratum']).round(4).to_string(index=False))
print(M[M['modality'].isna()].drop(columns=['modality']).round(4).to_string(index=False))
# 3. NB regression of R and L with the share_cta indicator
REF = {'stratum': 'news_society', 'size_bucket': 'M', 'modality': 'image'}
def design(d, fe, extra=()):
    X = []
    for f in fe:
        for l in sorted(set(d[f].unique()) - {REF[f]}): X.append(pd.Series((d[f]==l).astype(float).values, name=f'{f}[{l}]', index=d.index))
    X = pd.concat(X, axis=1)
    for e in extra: X[e] = d[e].astype(float)
    return sm.add_constant(X)
def fit(d, y, fe, extra=()):
    X = design(d, fe, extra); off = np.log(d.expo.values)
    p0 = sm.Poisson(d[y].values, X, offset=off).fit(disp=0, maxiter=200).params
    m = sm.NegativeBinomial(d[y].values, X, loglike_method='nb2', offset=off)
    res = m.fit(start_params=np.append(p0.values, 1.0), method='newton', maxiter=200, disp=0, cov_type='cluster', cov_kwds={'groups': d.community_vk_id.values})
    if not res.mle_retvals.get('converged', True):
        res = m.fit(start_params=res.params.values, method='bfgs', maxiter=2000, disp=0, cov_type='cluster', cov_kwds={'groups': d.community_vk_id.values})
    return res
coef = []
nc = n_all[n_all.modality!='clip']; cl = n_all[n_all.modality=='clip']
specs = [('text/image/video: base', nc, ['stratum','size_bucket','modality'], ()),
         ('text/image/video: + share_cta', nc, ['stratum','size_bucket','modality'], ('share_cta',)),
         ('text/image/video: without share_cta posts', nc[~nc.share_cta], ['stratum','size_bucket','modality'], ()),
         ('clip: base', cl, ['stratum','size_bucket'], ()),
         ('clip: + share_cta', cl, ['stratum','size_bucket'], ('share_cta',))]
for name, d, fe, extra in specs:
    for y in ['reposts','likes']:
        res = fit(d, y, fe, extra); ci = res.conf_int()
        for k in res.params.index:
            coef.append({'model': name, 'counter': y, 'term': k, 'n': len(d), 'coef': res.params[k], 'se_cluster': res.bse[k], 'p': res.pvalues[k],
                         'IRR': np.exp(res.params[k]) if k!='alpha' else np.nan, 'IRR_lo': np.exp(ci.loc[k,0]) if k!='alpha' else np.nan,
                         'IRR_hi': np.exp(ci.loc[k,1]) if k!='alpha' else np.nan, 'value': res.params[k] if k=='alpha' else np.nan, 'converged': res.mle_retvals.get('converged')})
        print(name, y, len(d), 'conv', res.mle_retvals.get('converged'), flush=True)
C = pd.DataFrame(coef); C.to_csv(O+'step3_cta_negbin.csv', index=False)
key = C[C.term.isin(['share_cta','stratum[food_delivery]','stratum[humor_entertainment]','stratum[viral_aggregator]','stratum[finance_business]','alpha'])]
print(key.assign(v=np.where(key.term=='alpha', key.value, key.IRR)).pivot_table(index=['term','counter'], columns='model', values='v', sort=False).round(3).to_string())
print(key[key.term!='alpha'][['model','counter','term','IRR','IRR_lo','IRR_hi','p']].round(3).to_string(index=False))
# 4. food: share of cta and 30 random texts
f = a[a.stratum=='food_delivery']
print('food: share_cta', round(f.share_cta.mean(), 4), ' n', int(f.share_cta.sum()), ' by modality', f.groupby('modality').share_cta.mean().round(3).to_dict())
fc = f[f.share_cta].merge(T[KEY+['text']], on=KEY)
smp = fc.sample(min(30, len(fc)), random_state=20261113)
# masking: VK user mentions [id123|Name] and phone numbers (names of private persons are not needed)
mask = lambda s: re.sub(r'\+?\d[\d\-\s\(\)]{9,}\d', '[phone]', re.sub(r'\[id\d+\|[^\]]*\]', '[id…]', s))
smp['text150'] = smp.text.fillna('').str.replace(r'\s+', ' ', regex=True).map(mask).str[:150]
smp[KEY+['community_vk_id','modality','views','reposts','likes','text150']].to_csv(O+'step3_cta_food_sample30.csv', index=False)
for i, x in enumerate(smp.itertuples(), 1): print(f'{i:2d}. [{x.modality}, R={x.reposts:.0f}, L={x.likes:.0f}] {x.text150}')
print('food cta posts by community (top 8):', fc.community_vk_id.value_counts().head(8).to_dict())
# what drives high rho_R in food (descriptive): food communities by median rho_R, text/image/video
fr = r[(r.stratum=='food_delivery') & (r.modality!='clip')]
top = fr.groupby('community_vk_id').agg(posts=('rho_R','size'), rho_R_med_x1e3=('rho_R', lambda x: 1e3*x.median()), rho_L_med_x1e3=('rho_L', lambda x: 1e3*x.median()), cta_share=('share_cta','mean'))
cc = pd.read_parquet(D+'communities.parquet')[['community_vk_id','screen_name','size_bucket']]
top = top.reset_index().merge(cc).sort_values('rho_R_med_x1e3', ascending=False); top.to_csv(O+'step3_food_communities_rhoR.csv', index=False)
print('food communities by median rho_R (text/image/video):'); print(top.round(3).to_string(index=False))
