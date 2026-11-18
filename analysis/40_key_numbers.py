"""Key numbers of a run, for comparing exports (printed to stdout)."""
import json, pandas as pd
from common import *
m = json.load(open(D+'manifest.json')); print('export', m.get('label'), m.get('exported_at'), m.get('rows'))
q = pd.read_parquet(O+'pools.parquet')
for k in ['drop_basic','autopost','autopost_orig','comm_vnull_excluded','bad_RV_LV','pool_ratio','pool_ratio_orig','pool_ratio_none','pool_rho_all','pool_rho_all_orig','pool_rho','pool_rho_orig','pool_rho_none','share_cta']:
    print(f'{k:22s}', int(q[k].sum()))
print('pool_rho by modality', q[q.pool_rho].modality.value_counts().to_dict())
cl = q[q.pool_ratio & (q.modality=='clip')]
print('clips in pool_ratio', len(cl), ' with exposure (video_views)', int(cl.video_views_main.notna().sum()), ' in pool_rho_all', int((q.pool_rho_all & (q.modality=='clip')).sum()),
      ' in pool_rho', int((q.pool_rho & (q.modality=='clip')).sum()), ' vv_status', cl.vv_status.value_counts().to_dict())
r1 = pd.read_csv(O+'step3_1_rho_by_stratum_modality.csv'); r1 = r1[r1.stratum=='ALL']
print((r1.set_index('modality')[['posts','rho_R_med','rho_L_med','rho_C_med','R_zero_share']]*[1,1e3,1e3,1e3,1]).round(3).rename(columns=lambda c: c+('_x1e3' if c.startswith('rho') else '')).to_string())
ap = pd.read_csv(O+'step3_autopost_rule_all_communities.csv'); print('autopost (refined, main)', ap[ap.autopost].screen_name.tolist()); print('autopost (original rule)', ap[ap.autopost_orig].screen_name.tolist())
ex = pd.read_csv(O+'step3_excluded_communities.csv'); print('V-null excluded', ex.screen_name.tolist())
s2 = pd.read_csv(O+'step2_summary.csv', index_col=0); print('step2 mark1 out of tol', s2.loc['mark1_out_of_tol_n'].to_dict())
A = pd.read_csv(O+'step3_5_negbin_alpha.csv'); print(A.pivot_table(index='model', columns='counter', values='alpha', sort=False).round(3).to_string())
v = pd.read_csv(O+'step3_6_viral_v3.csv'); t = v[(v.stratum=='ALL')]; print(t.drop(columns=[x for x in t.columns if x.endswith('_pct')]).to_string(index=False))
cells = v[(v.stratum!='ALL')&(v.modality.isin(['text','image','video']))]; print('empty cells (of 30) main:', int((cells.viral_main==0).sum()), ' (b):', int((cells.viral_b_cell==0).sum()))
cc = v[(v.stratum!='ALL')&(v.modality=='clip')]; print('viral clips by stratum:', cc.set_index('stratum').viral_main.astype(int).to_dict(), ' empty clip cells:', int((cc.viral_main==0).sum()))
s = pd.read_csv(O+'step3_2_spearman.csv'); print(s[(s.subset=='text/image/video')&(s.pair=='log rho_R vs log rho_L')][['version','spearman','n']].to_string(index=False))
c = pd.read_csv(O+'step3_cta_negbin.csv'); k = c[c.term.isin(['share_cta','stratum[food_delivery]'])]
print(k.pivot_table(index=['term','counter'], columns='model', values='IRR', sort=False).round(3).to_string())
