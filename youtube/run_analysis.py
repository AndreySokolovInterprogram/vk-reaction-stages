"""YouTube cross-sectional comparison (Secs. IV-D and VI-F of the paper).

Cleaned version of the analysis run of 2026-11-14 (version 2.0): channel funnel, sample, NB2 regressions of likes
and comments with log views as an offset or with a free coefficient, and the within-channel rank correlation between
log like and comment rates. No database access, no credentials, no channel identifiers in the code: the script reads
the exported tables described in youtube/README.md.

Inputs (environment variables):
  YT_BACKFILL_FILE   required. Backfill export (parquet or csv), one row per video: post_id, channel_id, domain,
                     size_tier, is_short, published_at, age_hours, views, likes, comments, duration_s.
                     Rows: selected channels, is_backfill = true, published_at < 2026-11-12 00:00 UTC.
  YT_CHANNELS_FILE   optional. Selected channels: channel_id (is_selected = true).
  YT_POSTS_FILE      optional. All collected videos of any channel: channel_id, is_backfill, published_at, crawled_at.
                     With both optional files the full channel funnel (selected -> any video -> backfill video ->
                     analytical sample) is reported; without them only the last two steps.
  YT_SNAPSHOT_UTC    optional. Videos crawled after this moment are ignored in the funnel
                     (default 2026-11-14T14:30:00Z, the moment of the reported run).
  YT_OUT             output directory (default youtube/out).
Output: YT_OUT/key_numbers_youtube.txt and YT_OUT/nb2_coefficients.csv (no identifiers).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("YT_OUT", HERE / "out"))
SNAPSHOT = pd.Timestamp(os.environ.get("YT_SNAPSHOT_UTC", "2026-11-14T14:30:00Z"))
CUTOFF = pd.Timestamp("2026-11-12T00:00:00Z")  # end of the 90-day backfill window
MIN_AGE_H, MIN_VIEWS = 168, 100

# Topical domains are stored in Russian in the collector database; map them to the stratum codes of the paper.
DOMAINS = {
    "\u0430\u0432\u0442\u043e": "auto",
    "\u0435\u0434\u0430 \u0438 \u0434\u043e\u0441\u0442\u0430\u0432\u043a\u0430": "food_delivery",
    "\u043a\u0440\u0430\u0441\u043e\u0442\u0430 \u0438 \u0443\u0445\u043e\u0434": "beauty",
    "\u043d\u043e\u0432\u043e\u0441\u0442\u0438 \u0438 \u043e\u0431\u0449\u0435\u0441\u0442\u0432\u043e": "news_society",
    "\u043e\u0431\u0440\u0430\u0437\u043e\u0432\u0430\u043d\u0438\u0435 \u0438 \u043d\u0430\u0443\u043a\u0430": "education_science",
    "\u0441\u043f\u043e\u0440\u0442 \u0438 \u0444\u0438\u0442\u043d\u0435\u0441": "sport_fitness",
    "\u0442\u0435\u0445\u043d\u043e\u043b\u043e\u0433\u0438\u0438 \u0438 \u0433\u0430\u0434\u0436\u0435\u0442\u044b": "tech_gadgets",
    "\u0444\u0438\u043d\u0430\u043d\u0441\u044b \u0438 \u0431\u0438\u0437\u043d\u0435\u0441": "finance_business",
    "\u044e\u043c\u043e\u0440 \u0438 \u0440\u0430\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u044f": "humor_entertainment",
}

RHS = "C(domain, Treatment(reference='auto')) + C(is_short) + C(size_tier, Treatment(reference='S')) + np.log(age_hours)"
MODELS = {  # id: (counter, rhs, offset)
    "1A": ("likes", RHS + " + np.log(duration_s)", True),
    "1B": ("likes", "np.log(views) + " + RHS + " + np.log(duration_s)", False),
    "2A": ("comments", RHS + " + np.log(duration_s)", True),
    "2B": ("comments", "np.log(views) + " + RHS + " + np.log(duration_s)", False),
    "1A_nodur": ("likes", RHS, True),
    "2A_nodur": ("comments", RHS, True),
}
SHORT = "C(is_short)[T.True]"


def read(path):
    p = Path(path)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def utc(s):
    return pd.to_datetime(s, utc=True)


def funnel(bf, lines):
    ch_file, posts_file = os.environ.get("YT_CHANNELS_FILE"), os.environ.get("YT_POSTS_FILE")
    sample_ch = bf.loc[(bf.age_hours >= MIN_AGE_H) & (bf.views >= MIN_VIEWS), "channel_id"].unique()
    if ch_file and posts_file:
        sel = set(read(ch_file).channel_id)
        posts = read(posts_file)
        posts = posts[posts.channel_id.isin(sel) & (utc(posts.crawled_at) <= SNAPSHOT)]
        any_post = set(posts.channel_id)
        backfill = set(posts.loc[posts.is_backfill.astype(bool) & (utc(posts.published_at) < CUTOFF), "channel_id"])
        lines.append(f"channel funnel: selected {len(sel)} -> with any video {len(any_post)} -> "
                     f"with a backfill video {len(backfill)} -> in the analytical sample {len(set(sample_ch) & sel)}")
    else:
        lines.append("channel funnel: YT_CHANNELS_FILE / YT_POSTS_FILE not given; last two steps only")
    lines.append(f"channels with a backfill video (backfill file): {bf.channel_id.nunique()}")
    lines.append(f"channels in the analytical sample (age >= {MIN_AGE_H} h, views >= {MIN_VIEWS}): {len(sample_ch)}")


def fit(model_id, df):
    y, rhs, offset = MODELS[model_id]
    kw = dict(offset=np.log(df["views"])) if offset else {}
    return smf.negativebinomial(f"{y} ~ {rhs}", data=df, **kw).fit(
        maxiter=500, cov_type="cluster", cov_kwds={"groups": df["channel_id"]}, disp=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bf = read(os.environ["YT_BACKFILL_FILE"])
    bf = bf[utc(bf.published_at) < CUTOFF].copy()
    bf["domain"] = bf["domain"].map(lambda d: DOMAINS.get(d, d))
    bf["is_short"] = bf["is_short"].astype(bool)
    lines = [f"backfill videos (published before {CUTOFF.date()}): {len(bf)}"]
    funnel(bf, lines)

    d = bf[(bf.age_hours >= MIN_AGE_H) & (bf.views >= MIN_VIEWS)].copy()
    n_short = int(d.is_short.sum())
    dc = d[d.comments.notna()].copy()
    lines += [f"analytical sample: {len(d)} videos, {d.channel_id.nunique()} channels, {n_short} Shorts",
              f"comments disabled: {len(d) - len(dc)} videos; comment sample {len(dc)}"]

    # NB2 regressions, channel-clustered standard errors
    rows, res = [], {}
    for mid in MODELS:
        r = fit(mid, d if MODELS[mid][0] == "likes" else dc)
        res[mid] = r
        ci = r.conf_int()
        for term in r.params.index:
            rows.append({"model": mid, "term": term, "beta": r.params[term], "se_cluster": r.bse[term],
                         "p": r.pvalues[term], "ci_lo": ci.loc[term, 0], "ci_hi": ci.loc[term, 1],
                         "irr": np.exp(r.params[term]) if term != "alpha" else np.nan, "n": int(r.nobs)})
    pd.DataFrame(rows).to_csv(OUT / "nb2_coefficients.csv", index=False)
    lines += [f"elasticity w.r.t. views (free coefficient): likes {res['1B'].params['np.log(views)']:.3f}, "
              f"comments {res['2B'].params['np.log(views)']:.3f}",
              f"NB2 overdispersion alpha (offset models): likes {res['1A'].params['alpha']:.4f}, "
              f"comments {res['2A'].params['alpha']:.4f}"]
    r = res["2A_nodur"]
    irr, lo, hi = np.exp(r.params[SHORT]), *np.exp(r.conf_int().loc[SHORT])
    lines.append(f"Shorts vs standard videos, comments per view (2A_nodur, no duration): IRR {irr:.3f} "
                 f"[{lo:.3f}, {hi:.3f}], p = {r.pvalues[SHORT]:.4f}, difference {100 * (irr - 1):+.1f}%")

    # within-channel rank correlation of log like and comment rates (videos with L > 0 and C > 0,
    # centred by the median of the channel x format group, groups with >= 2 videos)
    pos = dc[(dc.likes > 0) & (dc.comments > 0)].copy()
    pos["lL"], pos["lC"] = np.log(pos.likes / pos.views), np.log(pos.comments / pos.views)
    g = pos.groupby(["channel_id", "is_short"])
    pos = pos[g.lL.transform("size") >= 2].copy()
    med = pos.groupby(["channel_id", "is_short"])[["lL", "lC"]].transform("median")
    rho = spearmanr(pos.lL - med.lL, pos.lC - med.lC).statistic
    lines.append(f"within-channel Spearman correlation, log like vs log comment rate: {rho:.4f} "
                 f"(n = {len(pos)}, groups = {pos.groupby(['channel_id', 'is_short']).ngroups})")

    (OUT / "key_numbers_youtube.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
