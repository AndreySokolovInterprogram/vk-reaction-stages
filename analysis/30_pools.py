"""Pools for step 3.

Windows: backfill - 90 days before the panel start; an older backfill post is kept if it is a video/clip published
not earlier than 11 Nov 2025 MSK (ext12m window); ext12m - not earlier than 11 Nov 2025 MSK.
Auto-posting (pre-specified rule applied to all communities): a community is excluded from the main analysis if the
share of posts repeating the text (text_norm_hash) of an earlier post of the same community is >= 50%, OR if in any
calendar month (MSK) of the window it posted >= 100 posts/day on average. Denominator: all posts of the community in
the window (retrospective + live); posts without text are not counted in the numerator. Days of a month: intersection
of the month with [11 Nov 2025, export time]; before the 90-day window only video/clip posts were collected, so the
rate there is underestimated (conservative).
share_cta: case-insensitive regular expression on the post's own text (call to share / giveaway).
"""
import re
import numpy as np, pandas as pd
from common import *
c, p, s = load()
q = pd.read_parquet(O+'posts_last_snapshot.parquet')  # last non-null measurement
txt = pd.read_parquet(D+'posts.parquet', columns=KEY+['text', 'text_norm_hash'])
q = q.merge(txt, on=KEY, how='left')
START = panel_start(s); EXP = export_time()
cut90 = START - pd.Timedelta(days=90)
EXT_SINCE = pd.Timestamp('2025-11-11T00:00:00+03:00')
print('panel start', START, ' 90d window starts', cut90, ' export', EXP)
q['no_snap'] = q.measured_at.isna()
in_ext = q.modality.isin(['video', 'clip']) & (q.published_at >= EXT_SINCE)
q['out_of_window'] = ((q.origin == 'backfill') & (q.published_at < cut90) & ~in_ext) | ((q.origin == 'ext12m') & (q.published_at < EXT_SINCE))
old_bf = (q.origin == 'backfill') & (q.published_at < cut90)
print('backfill older than 90d:', int(old_bf.sum()), ' kept as ext12m-window video/clip:', int((old_bf & in_ext).sum()),
      ' of them with snapshot:', int((old_bf & in_ext & ~q.no_snap).sum()))
print('no_snap', int(q.no_snap.sum()), ' out_of_window', int(q.out_of_window.sum()), ' union', int((q.no_snap | q.out_of_window).sum()))
q['drop_basic'] = q.no_snap | q.out_of_window

# ---------- auto-posting
# original rule: repeat = same text_norm_hash as an earlier post of the community.
# refined rule (main): lines occurring >= 20 times in the community (signatures, "subscribe to ...", links) are removed
# first; repeat = same normalized remainder; an empty remainder is not a repeat. Same thresholds.
_URL_RE = re.compile(r"https?://\S+|vk\.com/\S+|\[(?:id|club|public)\d+\|([^\]]*)\]")   # as in the collector
_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)
def norm(x):
    x = _URL_RE.sub(lambda m: m.group(1) or " ", x or "").lower().replace("\u0451", "\u0435")  # yo -> ye
    return " ".join(_NON_WORD_RE.sub(" ", x).split())
LINE_MIN = 20
w = q[~q.out_of_window].sort_values('published_at').copy()   # all posts in the window, including those without measurements
w['repeat_orig'] = w.text_norm_hash.notna() & w.duplicated(['community_vk_id', 'text_norm_hash'])
w['lines'] = w.text.fillna('').map(lambda x: list(dict.fromkeys(l for l in (norm(s) for s in x.splitlines()) if l)))
lc = w[['community_vk_id', 'lines']].explode('lines').dropna().groupby(['community_vk_id', 'lines']).size()
frequent = set(lc[lc >= LINE_MIN].index)
w['rest'] = [' '.join(l for l in ls if (cid, l) not in frequent) for cid, ls in zip(w.community_vk_id, w.lines)]
w['rest'] = w.rest.where(w.rest != '', None)
w['repeat_ref'] = w.rest.notna() & w.duplicated(['community_vk_id', 'rest'])
print('frequent lines removed (>=%d per community):' % LINE_MIN, len(frequent), ' communities with such lines:', len({c for c, _ in frequent}))
w['month'] = w.published_at.dt.tz_convert('Europe/Moscow').dt.strftime('%Y-%m')
lo, hi = EXT_SINCE.tz_convert('Europe/Moscow').normalize(), EXP.tz_convert('Europe/Moscow')
def days_in(m):
    a = pd.Timestamp(m + '-01', tz='Europe/Moscow'); b = a + pd.offsets.MonthBegin(1)
    return max((min(b, hi) - max(a, lo)).total_seconds() / 86400, 1.0)
mm = w.groupby(['community_vk_id', 'month']).size().rename('posts').reset_index()
mm['days'] = mm.month.map(days_in); mm['per_day'] = mm.posts / mm.days
top = mm.sort_values('per_day').groupby('community_vk_id').tail(1).set_index('community_vk_id')
ap = w.groupby('community_vk_id').agg(posts=('repeat_orig', 'size'), repeat_share_orig=('repeat_orig', 'mean'), repeat_share=('repeat_ref', 'mean'),
                                      empty_rest_share=('rest', lambda x: x.isna().mean()))
ap = ap.join(top[['month', 'per_day']].rename(columns={'month': 'max_month', 'per_day': 'max_month_posts_per_day'}))
ap['rule_rate'] = ap.max_month_posts_per_day >= 100
ap['autopost_orig'] = (ap.repeat_share_orig >= .5) | ap.rule_rate
ap['rule_repeat'] = ap.repeat_share >= .5
ap['autopost'] = ap.rule_repeat | ap.rule_rate
ap = ap.reset_index().merge(c[['community_vk_id', 'screen_name', 'name', 'stratum', 'size_bucket']])
ap.sort_values(['autopost', 'autopost_orig', 'repeat_share'], ascending=False).to_csv(O+'step3_autopost_rule_all_communities.csv', index=False)
APC = set(ap.loc[ap.autopost, 'community_vk_id']); APC0 = set(ap.loc[ap.autopost_orig, 'community_vk_id'])
cols = ['screen_name', 'stratum', 'size_bucket', 'posts', 'repeat_share_orig', 'repeat_share', 'empty_rest_share', 'max_month', 'max_month_posts_per_day', 'autopost_orig', 'autopost']
print('AUTOPOST, refined rule (main):'); print(ap[ap.autopost][cols].round(3).to_string(index=False))
print('original rule only (robustness):'); print(ap[ap.autopost_orig & ~ap.autopost][cols].round(3).to_string(index=False))
print('closest to refined threshold (not excluded):'); print(ap[~ap.autopost].sort_values('repeat_share', ascending=False).head(5)[cols].round(3).to_string(index=False))
# examples of repeats under the refined rule
for cid in sorted(APC):
    g = w[(w.community_vk_id == cid) & w.repeat_ref]
    vc = g.rest.value_counts().head(3)
    print(f'-- {ap.set_index("community_vk_id").loc[cid, "screen_name"]}: top repeated remainders', vc.tolist())
    for r_ in vc.index: print('     ', repr(r_[:120]))
q['autopost'] = q.community_vk_id.isin(APC); q['autopost_orig'] = q.community_vk_id.isin(APC0)

# ---------- share_cta
# Russian keywords: repost, make a repost, share, giveaway, raffle, contest, tell your friends
CTA = re.compile(r'\u0440\u0435\u043f\u043e\u0441\u0442|\u0441\u0434\u0435\u043b\u0430\u0439(\u0442\u0435)? \u0440\u0435\u043f\u043e\u0441\u0442|\u043f\u043e\u0434\u0435\u043b\u0438\u0441\u044c|\u043f\u043e\u0434\u0435\u043b\u0438\u0442\u0435\u0441\u044c|\u0440\u043e\u0437\u044b\u0433\u0440\u044b\u0448|\u0440\u0430\u0437\u044b\u0433\u0440\u044b\u0432\u0430|\u043a\u043e\u043d\u043a\u0443\u0440\u0441|\u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0438 \u0434\u0440\u0443\u0437\u044c\u044f\u043c', re.I)
q['share_cta'] = q.text.fillna('').map(lambda t: bool(CTA.search(t)))

# ---------- own video views (video.get)
# main video of a post = first own video attachment (matches first_video of the media queue for 100% of posts);
# true views = last non-zero of video_views / video_views_recheck3 (recheck3 only for complete VK responses);
# unavailable (restricted_recheck3 set), zeros and not returned -> missing.
att = pd.read_parquet(D+'posts.parquet', columns=KEY+['attachments'])
def first_vid(a):
    for x in json.loads(a) if a else []:
        if not x.get('from_repost') and x.get('type') in ('video', 'clip', 'short_video') and x.get('id') is not None:
            return f"{x.get('owner_id')}_{x.get('id')}"
    return None
att['main_video'] = att.attachments.map(first_vid)
q = q.merge(att[KEY+['main_video']], on=KEY, how='left')
if os.path.exists(VIDEO_VIEWS):
    vv = pd.read_csv(VIDEO_VIEWS)
    vv['vid'] = vv.owner_id.astype('int64').astype(str) + '_' + vv.video_id.astype('int64').astype(str)
    r3 = vv.video_views_recheck3.where((vv.response_type_recheck3 == 'full') & (vv.video_views_recheck3 > 0))
    r0 = vv.video_views.where(vv.video_views > 0)
    vv['vv_true'] = r3.fillna(r0).where(vv.restricted_recheck3.isna())
    vv['vv_status'] = np.select([vv.restricted_recheck3.notna(), vv.vv_true.notna(), vv.status == 'not_returned'], ['restricted', 'ok', 'not_returned'], 'zero')
    vv = vv.drop_duplicates(KEY + ['vid'])
    q = q.merge(vv[KEY + ['vid', 'vv_true', 'vv_status']].rename(columns={'vid': 'main_video', 'vv_true': 'video_views_main'}), on=KEY + ['main_video'], how='left')
    q['vv_status'] = q.vv_status.fillna(pd.Series(np.where(q.main_video.notna(), 'no_row', 'no_video'), index=q.index))
else:
    print('VIDEO_VIEWS not found:', VIDEO_VIEWS, '- clips without own exposure')
    q['video_views_main'] = np.nan; q['vv_status'] = 'no_file'
vm = q[q.modality.isin(['video', 'clip']) & ~q.drop_basic & q.eligible & (q.part == 'retro')]
print('video_views of main video, retro eligible in window:'); print(pd.crosstab(vm.modality, vm.vv_status, margins=True))
print('median video_views / post views:', (vm.video_views_main / vm.views).groupby(vm.modality).median().round(3).to_dict())
own = vm.main_video.str.split('_').str[0].astype('float').eq(vm.vk_owner_id)
print('main video owned by the community (share):', own.groupby(vm.modality).mean().round(3).to_dict(),
      ' median vv/V own:', (vm.video_views_main / vm.views)[own].groupby(vm.modality[own]).median().round(2).to_dict(),
      ' foreign:', (vm.video_views_main / vm.views)[~own].groupby(vm.modality[~own]).median().round(2).to_dict())
# exposure: text/image/video - post views; clips - video views. expo_vv - robustness (videos also by video_views)
q['expo'] = np.where(q.modality == 'clip', q.video_views_main, q.views)
q['expo_vv'] = np.where(q.modality.isin(['clip', 'video']), q.video_views_main, q.views)

# ---------- communities with missing V
base = q[q.eligible & ~q.drop_basic & (q.part == 'retro') & q.final]
vn = base.groupby('community_vk_id').agg(posts=('views', 'size'), v_null=('views', lambda v: v.isna().sum()))
vn['v_null_share'] = vn.v_null / vn.posts
exc = vn[vn.v_null_share >= .5].reset_index().merge(c[['community_vk_id', 'screen_name', 'name', 'stratum', 'size_bucket']])
exc['autopost'] = exc.community_vk_id.isin(APC); exc['autopost_orig'] = exc.community_vk_id.isin(APC0)
exc.to_csv(O+'step3_excluded_communities.csv', index=False); print('V-null >=50%:'); print(exc.to_string(index=False))
q['comm_vnull_excluded'] = q.community_vk_id.isin(exc.community_vk_id)

# ---------- pools
# *_none - no auto-posting exclusion; *_orig - original rule (7 communities); no suffix - main (refined rule)
q['pool_ratio_none'] = q.eligible & ~q.drop_basic & (q.part == 'retro') & q.final
# rho pools use exposure expo (clips: video_views of the main video)
pre = q.pool_ratio_none & q.expo.notna() & ~q.comm_vnull_excluded
q['bad_RV_LV'] = pre & ((q.reposts > q.expo) | (q.likes > q.expo))
q['pool_rho_all_none'] = pre & ~q.bad_RV_LV
q['pool_rho_none'] = q.pool_rho_all_none & (q.expo >= 100)
for k in ['pool_ratio', 'pool_rho_all', 'pool_rho']:
    q[k] = q[k + '_none'] & ~q.autopost
    q[k + '_orig'] = q[k + '_none'] & ~q.autopost_orig
# robustness for regular videos: exposure = video_views (expo_vv)
pv = q.pool_ratio_none & q.expo_vv.notna() & ~q.comm_vnull_excluded & ~q.autopost
q['pool_rho_all_vv'] = pv & (q.reposts <= q.expo_vv) & (q.likes <= q.expo_vv)
q['pool_rho_vv'] = q.pool_rho_all_vv & (q.expo_vv >= 100)
print('bad R/L > expo by modality', q[q.bad_RV_LV].modality.value_counts().to_dict(), ' (video with vv:', int((pv & q.modality.eq('video') & ((q.reposts > q.expo_vv) | (q.likes > q.expo_vv))).sum()), ')')
print('R>V or L>V excluded', int(q.bad_RV_LV.sum()), q[q.bad_RV_LV].modality.value_counts().to_dict())
for k in ['pool_ratio', 'pool_rho_all', 'pool_rho']:
    print(f'{k}: main {int(q[k].sum())}  orig rule {int(q[k+"_orig"].sum())}  no exclusion {int(q[k+"_none"].sum())}  by modality (main)', q[q[k]].modality.value_counts().to_dict())
left = q[q.pool_rho].groupby(['stratum', 'size_bucket']).community_vk_id.nunique().unstack().reindex(columns=SIZES)
tot = c.groupby(['stratum', 'size_bucket']).size().unstack().reindex(columns=SIZES)
cells = left.fillna(0).astype(int).astype(str) + ' / ' + tot.astype(str)
cells.to_csv(O+'step3_communities_left_per_cell.csv'); print(cells.to_string())
print('communities in pool_rho', q[q.pool_rho].community_vk_id.nunique(), ' in pool_ratio', q[q.pool_ratio].community_vk_id.nunique())
ss = s.sort_values(KEY + ['measured_at'])
print('decreasing snapshots', {col: int((ss.groupby(KEY)[col].diff() < 0).sum()) for col in ['views', 'likes', 'reposts', 'comments']})
q.drop(columns=['text']).to_parquet(O+'pools.parquet')
q.loc[q.share_cta, KEY + ['community_vk_id', 'stratum', 'modality', 'text']].to_parquet(O+'share_cta_texts.parquet')
