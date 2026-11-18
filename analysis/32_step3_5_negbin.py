"""Step 3.5: NB2 regression of each counter (R, L, C) with log-exposure offset; fixed effects stratum + size
(+ modality); errors clustered by community. Exposure expo: text/image/video - post views, clips - views of the main
video (post reactions / platform-wide video views, so rho of clips may be underestimated).
Models: text/image/video (main); clips separately; joint text/image/video/clip; by modality; robustness checks
(V>=100, free log-exposure coefficient, auto-posting rules, weight 1/n, videos with expo_vv = video_views).
Pool: pool_rho_all (retrospective, eligible, final value at >=7 days, non-missing exposure, without communities with
V-null >= 50% or flagged by the refined auto-posting rule, R, L <= exposure)."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, statsmodels.api as sm, time
from common import *
q = pd.read_parquet(O+'pools.parquet')
q['video_own'] = q.main_video.str.split('_').str[0].astype('float').eq(q.vk_owner_id)  # the main video belongs to the community
REF = {'stratum': 'news_society', 'size_bucket': 'M', 'modality': 'image'}
def design(d, fe, free_logv=False, e='expo'):
    X = []
    for f in fe:
        lv = sorted(d[f].unique()); lv.remove(REF[f]) if REF[f] in lv else None
        for l in lv: X.append(pd.Series((d[f]==l).astype(float).values, name=f'{f}[{l}]', index=d.index))
    X = pd.concat(X, axis=1) if X else pd.DataFrame(index=d.index)
    if free_logv: X['log_V'] = np.log(d[e])
    return sm.add_constant(X)
def fit(d, y, fe, free_logv=False, e='expo'):
    X = design(d, fe, free_logv, e); off = None if free_logv else np.log(d[e].values)
    m = sm.NegativeBinomial(d[y].values, X, loglike_method='nb2', offset=off)
    start = None
    try:
        p0 = sm.Poisson(d[y].values, X, offset=off).fit(disp=0, maxiter=200).params
        start = np.append(p0.values, 1.0)
    except Exception: pass
    r = m.fit(start_params=start, method='newton', maxiter=200, disp=0, cov_type='cluster', cov_kwds={'groups': d.community_vk_id.values})
    if not r.mle_retvals.get('converged', True) or not np.isfinite(r.params).all() or not np.isfinite(r.bse).all():
        r = m.fit(start_params=start if (start is not None) else None, method='bfgs', maxiter=2000, disp=0, cov_type='cluster', cov_kwds={'groups': d.community_vk_id.values})
    return r
coef_rows, alpha_rows = [], []
NC, CL = q.modality!='clip', q.modality=='clip'
TIV, TIVC = ['stratum','size_bucket','modality'], ['stratum','size_bucket']
specs = [('main: text/image/video', q.pool_rho_all & NC, TIV, False, 'expo'),
         ('clip only (expo = video_views)', q.pool_rho_all & CL, TIVC, False, 'expo'),
         ('joint: text/image/video/clip', q.pool_rho_all, TIV, False, 'expo')]
specs += [(f'by modality: {m}', q.pool_rho_all & (q.modality==m), TIVC, False, 'expo') for m in ['text','image','video']]
specs += [('sens V>=100: text/image/video', q.pool_rho & NC, TIV, False, 'expo'),
          ('sens V>=100: clip', q.pool_rho & CL, TIVC, False, 'expo'),
          ('free log V: text/image/video', q.pool_rho_all & NC, TIV, True, 'expo'),
          ('free log V: clip', q.pool_rho_all & CL, TIVC, True, 'expo'),
          ('robust video expo = video_views: video only', q.pool_rho_all_vv & (q.modality=='video'), TIVC, False, 'expo_vv'),
          ('robust video expo = video_views: own videos only', q.pool_rho_all_vv & (q.modality=='video') & q.video_own, TIVC, False, 'expo_vv'),
          ('robust video expo = video_views: text/image/video', q.pool_rho_all_vv & NC, TIV, False, 'expo_vv'),
          ('robust orig autopost rule: text/image/video', q.pool_rho_all_orig & NC, TIV, False, 'expo'),
          ('robust orig autopost rule: clip', q.pool_rho_all_orig & CL, TIVC, False, 'expo'),
          ('robust no autopost exclusion: text/image/video', q.pool_rho_all_none & NC, TIV, False, 'expo')]
for name, mask, fe, free, e in specs:
    d = q[mask].copy()
    for y in ['reposts','likes','comments']:
        t0 = time.time(); r = fit(d, y, fe, free, e)
        ci = r.conf_int()
        for k in r.params.index:
            coef_rows.append({'model': name, 'counter': y, 'term': k, 'coef': r.params[k], 'se_cluster': r.bse[k], 'p': r.pvalues[k],
                              'ci_lo': ci.loc[k,0], 'ci_hi': ci.loc[k,1], 'IRR': np.exp(r.params[k]) if k!='alpha' else np.nan})
        a = r.params['alpha']
        alpha_rows.append({'model': name, 'counter': y, 'exposure': e if 'e' in dir() else 'expo', 'n': len(d), 'communities': d.community_vk_id.nunique(), 'mean_y': d[y].mean(), 'zero_share': (d[y]==0).mean(),
                           'alpha': a, 'alpha_se_cluster': r.bse['alpha'], 'alpha_ci_lo': ci.loc['alpha',0], 'alpha_ci_hi': ci.loc['alpha',1],
                           'log_V_coef': r.params.get('log_V', np.nan), 'converged': r.mle_retvals.get('converged'), 'llf': r.llf, 'aic': r.aic, 'sec': round(time.time()-t0,1)})
        print(name, y, len(d), 'alpha=%.3f' % a, 'conv', r.mle_retvals.get('converged'), 'logV=%s' % r.params.get('log_V', '-'), flush=True)
# weight 1/n_posts of the community (equal contribution per community): GLM NB, alpha fixed from the main model
A0 = pd.DataFrame(alpha_rows)
for name, mask, fe, base in [('weighted 1/n: text/image/video', q.pool_rho_all & (q.modality!='clip'), ['stratum','size_bucket','modality'], 'main: text/image/video'),
                             ('weighted 1/n: clip', q.pool_rho_all & (q.modality=='clip'), ['stratum','size_bucket'], 'clip only (expo = video_views)')]:
    d = q[mask].copy(); wts = 1/d.groupby('community_vk_id').community_vk_id.transform('size'); wts = wts/wts.mean()
    X = design(d, fe)
    for y in ['reposts','likes','comments']:
        a = float(A0[(A0.model==base)&(A0.counter==y)].alpha.iloc[0])
        r = sm.GLM(d[y].values, X, family=sm.families.NegativeBinomial(alpha=a), offset=np.log(d.expo.values), var_weights=wts.values).fit(
                cov_type='cluster', cov_kwds={'groups': d.community_vk_id.values})
        ci = r.conf_int()
        for k in r.params.index:
            coef_rows.append({'model': name, 'counter': y, 'term': k, 'coef': r.params[k], 'se_cluster': r.bse[k], 'p': r.pvalues[k],
                              'ci_lo': ci.loc[k,0], 'ci_hi': ci.loc[k,1], 'IRR': np.exp(r.params[k])})
        alpha_rows.append({'model': name, 'counter': y, 'n': len(d), 'communities': d.community_vk_id.nunique(), 'mean_y': d[y].mean(), 'zero_share': (d[y]==0).mean(),
                           'alpha': a, 'alpha_se_cluster': np.nan, 'alpha_ci_lo': np.nan, 'alpha_ci_hi': np.nan, 'log_V_coef': np.nan, 'converged': r.converged,
                           'llf': r.llf, 'aic': np.nan, 'sec': np.nan})
        print(name, y, 'alpha fixed', round(a,3), 'conv', r.converged, flush=True)
pd.DataFrame(coef_rows).to_csv(O+'step3_5_negbin_coefficients.csv', index=False)
A = pd.DataFrame(alpha_rows); A.to_csv(O+'step3_5_negbin_alpha.csv', index=False)
print(A[['model','counter','n','alpha','alpha_ci_lo','alpha_ci_hi','log_V_coef','converged']].round(3).to_string(index=False))
