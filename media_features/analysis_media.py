"""Step 6. Effects of feed features u_p on reaction shares.

Inputs (read only):
  data/media_items.parquet                 - post, community, stratum, size, modality, first line;
  out/vlm/vlm_triggers.parquet             - predefined triggers (16 concepts 0/1, Qwen3-VL-4B-Instruct);
  out/embed/emb_joint.npy, emb_ids.parquet - embeddings of feed image + first line (Qwen3-VL-Embedding-2B);
  pools.parquet from analysis/run_all.sh   - counters R, L, C, exposure expo (text/image/video - post views,
                                             clips - views of the main video), pools and auto-posting flags.

Sample: posts of the media slice within pool_rho_all of the main analysis (retrospective, eligible, final value at
>= 7 days, non-missing exposure, without communities with V-null >= 50% or auto-posting under the refined rule,
R, L <= exposure); posts with VLM status=error are dropped.

Outcomes (y): log rho_a = log((n_a + 0.5) / expo), a in {R, L, C} (0.5 correction because of zeros: C = 0 for ~55%
of posts); log(rho_R/rho_L) = log((R + 0.5)/(L + 0.5)), log(rho_C/rho_L) = log((C + 0.5)/(L + 0.5)).
A feature acting only on the shared stages T/O shifts the three log rho equally and leaves the ratios unchanged;
a shift of log(rho_R/rho_L) indicates an effect on the sharing motive, log(rho_C/rho_L) on the share of comments
(including negative ones).

Main model (per trigger x outcome): linear mixed model (statsmodels MixedLM, REML)
    y ~ trigger + C(modality) + C(size S/M/L/XL),  random intercept for community.
Weights as in the main analysis: main estimate unweighted; robustness - weight 1/n posts of the community
(WLS with community-clustered errors; MixedLM does not support weights). BH correction separately per family.
Robustness: (a) weight 1/n; (b) pool pool_rho (exposure >= 100); (c) for log rho_R, rho_L, rho_C - GLM NB2 with
alpha fixed from step 3.5 of the main analysis (R, L, C separately) and GLM Poisson, both with log-expo offset and
community-clustered errors (NB2 with estimated alpha did not converge).
Invariance across strata: trigger coefficient within each stratum (same mixed model), DerSimonian-Laird tau^2, Q, I^2,
heterogeneity p; leave-one-stratum-out transfer: the fixed part (OLS) fitted on 9 strata predicts the 10th - MSE with
the trigger vs without it (transfer gain) and vs the model fitted on all strata (MSE_loso / MSE_full); beta without
the stratum.
Latent triggers: PCA of emb_joint within modality (centered within modality), K leading directions, projections in
SD units; the same analysis within each modality (fixed: direction, size; random: community); for interpretation,
8 posts with extreme projections (first line).
Feature families (BH within family):
  CONFIRMATORY: VLM - face, people_many, animal, food, text_overlay, brand_logo, useful_info (+ child if >= 30
     positives); rules - number_in_title, question_in_title (first line, rules.py: digits 0-9; "?" or a question
     word at the start), call_to_share = share_cta (regular expression of the main analysis on the full text);
  EXPLORATORY: humor, humor_strict (second pass humor_pass2.py, if available), surprise, negative_emotion,
     positive_emotion, controversy;
  SUPPLEMENTARY: call_to_share_vlm (VLM on cover and first line).
  VLM answers for number/question/call_to_share are used only to check agreement with the rules (share, Cohen kappa).

  python analysis_media.py [--k 10] [--emb joint|image] [--quick] [--out out/analysis]
  --vlm / --emb-dir - other feature files (e.g. partial); --min-pos - minimum number of positives per trigger (30).
"""from __future__ import annotations

import argparse
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rules import has_digits, is_question  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(os.environ.get("MEDIA_ROOT", Path(__file__).resolve().parent / "work"))  # working dir: data/, out/, logs/
POOLS = Path(os.environ.get("ANALYSIS_OUT", Path(__file__).resolve().parents[1] / "analysis/out")) / "pools.parquet"  # pools of the main analysis
ALPHA_FILE = POOLS.parent / "step3_5_negbin_alpha.csv"  # NB2 alpha of the main analysis
KEY = ["vk_owner_id", "vk_post_id"]
TRIGGERS = ["face", "people_many", "child", "animal", "food", "text_overlay", "number_in_title", "question_in_title",
            "humor", "surprise", "negative_emotion", "positive_emotion", "useful_info", "call_to_share", "brand_logo",
            "controversy"]  # all 16 VLM concepts (columns of vlm_triggers)
# feature families; BH separately within each family
CONFIRM_VLM = ["face", "people_many", "animal", "food", "text_overlay", "brand_logo", "useful_info"]  # + child if >= 30 positives
CONFIRM_RULE = ["number_in_title", "question_in_title", "call_to_share"]  # first-line rule / share_cta on the full text
EXPLORE = ["humor", "humor_strict", "surprise", "negative_emotion", "positive_emotion", "controversy"]  # exploratory
SUPPL = ["call_to_share_vlm"]  # supplementary: call_to_share by VLM (first line + cover)
AGREE = ["number_in_title", "question_in_title", "call_to_share"]  # agreement rule <-> VLM (*_vlm)
OUTCOMES = ["log_rho_R", "log_rho_L", "log_rho_C", "log_R_over_L", "log_C_over_L"]
COUNTER = {"log_rho_R": "reposts", "log_rho_L": "likes", "log_rho_C": "comments"}
REF = {"modality": "image", "size_bucket": "M"}
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ---------------------------------------------------------------- data
def load(args) -> pd.DataFrame:
    it = pd.read_parquet(ROOT / "data/media_items.parquet")
    q = pd.read_parquet(POOLS, columns=KEY + ["reposts", "likes", "comments", "views", "expo", "pool_rho_all", "pool_rho",
                                             "autopost", "members_at_selection", "share_cta"])
    d = it.merge(q.drop(columns=["members_at_selection"]), on=KEY, how="left")
    log(f"media slice: {len(d)} posts; in pool_rho_all: {int(d.pool_rho_all.sum())}; pool_rho (expo >= 100): {int(d.pool_rho.sum())}")
    v = pd.read_parquet(args.vlm)
    d = d.merge(v[["post_id", "status"] + TRIGGERS], on="post_id", how="left")
    log("VLM status:", d.status.value_counts(dropna=False).to_dict())
    # number_in_title and question_in_title are defined by the first-line rule (rules.py);
    # VLM answers are kept as *_vlm and used only for the agreement check
    for k, f in [("number_in_title", has_digits), ("question_in_title", is_question)]:
        d[k + "_vlm"] = d[k]
        d[k] = d.first_line.fillna("").map(f).astype(int)
    # call_to_share: share_cta regular expression of the main analysis (full post text); VLM version is supplementary
    d["call_to_share_vlm"] = d["call_to_share"]
    d["call_to_share"] = d.share_cta.fillna(False).astype(int)
    # humor_strict: second pass (humor_pass2.py), if available: 1 = humor and "yes" in the second pass
    hp = Path(args.humor2)
    if hp.exists():
        h2 = pd.read_parquet(hp)[["post_id", "humor_strict"]]
        d = d.merge(h2, on="post_id", how="left")
        # humor is a nullable Int8 with <NA> (VLM error / not labelled), so np.where cannot be used;
        # logic: humor=0 -> 0, humor=1 -> answer of pass 2, humor missing -> missing
        hs = d.humor_strict.astype("float64")
        hs[d.humor.eq(0).fillna(False).to_numpy(bool)] = 0.0
        d["humor_strict"] = hs
        d.loc[d.humor.isna(), "humor_strict"] = np.nan
        log(f"humor_strict: {hp} (humor=1 without a second-pass answer: {int((d.humor.eq(1).fillna(False) & d.humor_strict.isna()).sum())})")
    else:
        d["humor_strict"] = np.nan
        log(f"humor_strict: no file {hp}; second pass not run, exploratory humor_strict skipped")
    d = d[d.pool_rho_all.fillna(False).astype(bool)].copy()
    for a, c in [("R", "reposts"), ("L", "likes"), ("C", "comments")]:
        d[f"log_rho_{a}"] = np.log((d[c] + 0.5) / d.expo)
    d["log_R_over_L"] = np.log((d.reposts + 0.5) / (d.likes + 0.5))
    d["log_C_over_L"] = np.log((d.comments + 0.5) / (d.likes + 0.5))
    d["w_inv_n"] = 1 / d.groupby("community_vk_id").post_id.transform("size")
    d["w_inv_n"] /= d.w_inv_n.mean()
    d["modality"] = pd.Categorical(d.modality, ["image", "video", "clip"])
    d["size_bucket"] = pd.Categorical(d.size_bucket, ["M", "S", "L", "XL"])
    d["g"] = d.community_vk_id.astype(str)
    log(f"analysis sample: {len(d)} posts, {d.community_vk_id.nunique()} communities; by modality",
        d.modality.value_counts().to_dict())
    return d


def first_line_rules(d: pd.DataFrame) -> pd.DataFrame:
    """Agreement of VLM with the first-line rule (the rule is the main definition)."""
    rows = []
    for k in AGREE:
        m = d[k + "_vlm"].notna()
        a, b = d.loc[m, k + "_vlm"].astype(int), d.loc[m, k].astype(int)
        po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        rows.append({"concept": k, "n": int(m.sum()), "vlm_share": a.mean(), "rule_share": b.mean(), "agreement": po,
                     "vlm1_rule0": int(((a == 1) & (b == 0)).sum()), "vlm0_rule1": int(((a == 0) & (b == 1)).sum()),
                     "kappa": (po - pe) / (1 - pe) if pe < 1 else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- models
def fit_mixed(df: pd.DataFrame, y: str, x: str, fe: list[str]):
    rhs = " + ".join([x] + [f"C({f}, Treatment('{REF[f]}'))" for f in fe if df[f].nunique() > 1])
    m = smf.mixedlm(f"{y} ~ {rhs}", df, groups=df["g"])
    for method in ("lbfgs", "powell", "cg"):
        try:
            r = m.fit(reml=True, method=method, disp=False)
            if np.isfinite(r.params[x]) and np.isfinite(r.bse[x]):
                return r
        except Exception:  # noqa: BLE001
            continue
    return None


def fit_wls(df: pd.DataFrame, y: str, x: str, fe: list[str]):
    rhs = " + ".join([x] + [f"C({f}, Treatment('{REF[f]}'))" for f in fe if df[f].nunique() > 1])
    return smf.wls(f"{y} ~ {rhs}", df, weights=df.w_inv_n).fit(cov_type="cluster", cov_kwds={"groups": df.community_vk_id.values})


def nb_alpha() -> dict:
    """NB2 alpha of the main analysis (step 3.5, row "main: text/image/video"), fixed rather than estimated
    (estimating alpha on ~9k posts did not converge)."""
    a = pd.read_csv(ALPHA_FILE)
    a = a[a.model == "main: text/image/video"].set_index("counter").alpha
    return {c: float(a[c]) for c in ["reposts", "likes", "comments"]}


NB_ALPHA = nb_alpha()


def fit_glm_count(df: pd.DataFrame, counter: str, x: str, fe: list[str], family: str):
    """GLM with log-expo offset and community-clustered errors: family = 'nb2' (alpha fixed) or 'poisson'."""
    rhs = " + ".join([x] + [f"C({f}, Treatment('{REF[f]}'))" for f in fe if df[f].nunique() > 1])
    fam = sm.families.NegativeBinomial(alpha=NB_ALPHA[counter]) if family == "nb2" else sm.families.Poisson()
    m = smf.glm(f"{counter} ~ {rhs}", df, family=fam, offset=np.log(df.expo.values))
    r = m.fit(cov_type="cluster", cov_kwds={"groups": df.community_vk_id.values}, maxiter=200)
    if not (np.isfinite(r.params[x]) and np.isfinite(r.bse[x])):
        raise ValueError("no finite estimate/SE")
    return r


def row(r, x, **kw):
    if r is None:
        return {**kw, "coef": np.nan, "se": np.nan, "p": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    ci = r.conf_int().loc[x]
    return {**kw, "coef": r.params[x], "se": r.bse[x], "p": r.pvalues[x], "ci_lo": ci.iloc[0], "ci_hi": ci.iloc[1]}


def dersimonian_laird(b: np.ndarray, se: np.ndarray) -> dict:
    ok = np.isfinite(b) & np.isfinite(se) & (se > 0)
    b, se = b[ok], se[ok]
    k = len(b)
    if k < 2:
        return {"k": k, "beta_fe": np.nan, "beta_re": np.nan, "se_re": np.nan, "tau2": np.nan, "tau": np.nan, "Q": np.nan, "I2": np.nan, "p_het": np.nan, "sign_agree": np.nan}
    w = 1 / se**2
    bf = (w * b).sum() / w.sum()
    Q = (w * (b - bf) ** 2).sum()
    c = w.sum() - (w**2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    wr = 1 / (se**2 + tau2)
    br = (wr * b).sum() / wr.sum()
    return {"k": k, "beta_fe": bf, "beta_re": br, "se_re": np.sqrt(1 / wr.sum()), "tau2": tau2, "tau": np.sqrt(tau2), "Q": Q,
            "I2": max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0, "p_het": 1 - stats.chi2.cdf(Q, k - 1),
            "sign_agree": float(np.mean(np.sign(b) == np.sign(br)))}


def ols_design(df, x, fe):
    X = [pd.Series(1.0, index=df.index, name="const")]
    if x is not None:
        X.append(df[x].astype(float).rename(x))
    for f in fe:
        for lv in df[f].cat.categories:
            if lv != REF[f]:
                X.append((df[f] == lv).astype(float).rename(f"{f}[{lv}]"))
    return pd.concat(X, axis=1)


def transfer(df: pd.DataFrame, y: str, x: str, fe: list[str]) -> list[dict]:
    """Leave-one-stratum-out transfer: OLS of the fixed part, fitted on 9 strata, predicts the 10th."""
    out = []
    Xa, Xa0 = ols_design(df, x, fe), ols_design(df, None, fe)
    full = np.linalg.lstsq(Xa.values, df[y].values, rcond=None)[0]
    for s in sorted(df.stratum.unique()):
        te = (df.stratum == s).values
        if df.loc[te, x].nunique() < 2 or df.loc[~te, x].nunique() < 2:
            continue
        b1 = np.linalg.lstsq(Xa.values[~te], df[y].values[~te], rcond=None)[0]
        b0 = np.linalg.lstsq(Xa0.values[~te], df[y].values[~te], rcond=None)[0]
        yt = df[y].values[te]
        e1 = ((yt - Xa.values[te] @ b1) ** 2).mean()
        e0 = ((yt - Xa0.values[te] @ b0) ** 2).mean()
        ef = ((yt - Xa.values[te] @ full) ** 2).mean()
        out.append({"held_out": s, "n_test": int(te.sum()), "beta_loso": b1[1], "beta_full": full[1],
                    "mse_loso_with": e1, "mse_loso_without": e0, "mse_full_with": ef,
                    "transfer_gain": e0 - e1, "mse_ratio_loso_full": e1 / ef})
    return out


def analyse(d: pd.DataFrame, feats: list[str], fe: list[str], family: str, by: str | None, args, out: Path, tag: str):
    """Main model + robustness + strata + transfer for the feature set feats (within groups `by`)."""
    main, rob, strat, het, tr = [], [], [], [], []
    groups = [(None, d)] if by is None else [(k, g) for k, g in d.groupby(by, observed=True)]
    for gk, dg in groups:
        for x in feats:
            dx = dg[dg[x].notna()].copy()
            dx[x] = dx[x].astype(float)
            binary = family != "latent"
            if len(dx) == 0:
                log(f"  skip {x} ({gk}): no data"); continue
            npos = int((dx[x] > 0.5).sum()) if binary else len(dx)
            if binary and (npos < args.min_pos or len(dx) - npos < args.min_pos):
                log(f"  skip {x} ({gk}): positives {npos} of {len(dx)}")
                continue
            base = {"family": family, "group": gk, "feature": x, "n": len(dx), "n_pos": npos if binary else np.nan,
                    "share_pos": npos / len(dx) if binary else np.nan, "communities": dx.g.nunique()}
            t0 = time.time()
            for y in OUTCOMES:
                main.append(row(fit_mixed(dx, y, x, fe), x, **base, outcome=y, model="mixed"))
                if not args.quick:
                    rob.append(row(fit_wls(dx, y, x, fe), x, **base, outcome=y, model="wls_w1n_cluster"))
                    dv = dx[dx.pool_rho.fillna(False).astype(bool)]
                    rob.append(row(fit_mixed(dv, y, x, fe), x, **{**base, "n": len(dv)}, outcome=y, model="mixed_expo_ge100"))
                    if y in COUNTER:
                        # GLM NB2 with the alpha of the main analysis and GLM Poisson;
                        # non-converged fits -> row with NaN (so they can be counted) + log entry
                        for fam, mname in [("nb2", "nb2_fixed_alpha_cluster"), ("poisson", "poisson_offset_cluster")]:
                            try:
                                rob.append(row(fit_glm_count(dx, COUNTER[y], x, fe, fam), x, **base, outcome=y, model=mname))
                            except Exception as e:  # noqa: BLE001
                                rob.append(row(None, x, **base, outcome=y, model=mname))
                                log(f"  {mname} did not converge: {x} {y} {gk}: {type(e).__name__}")
                # strata
                bs, ses = [], []
                for s, ds in dx.groupby("stratum"):
                    if binary and (ds[x].nunique() < 2 or min((ds[x] > .5).sum(), (ds[x] <= .5).sum()) < 5):
                        continue
                    r = fit_mixed(ds, y, x, fe)
                    rr = row(r, x, **base, outcome=y, stratum=s, n_stratum=len(ds))
                    strat.append(rr); bs.append(rr["coef"]); ses.append(rr["se"])
                het.append({**base, "outcome": y, **dersimonian_laird(np.array(bs), np.array(ses))})
                for t in transfer(dx, y, x, fe):
                    tr.append({**base, "outcome": y, **t})
            log(f"  {tag} {gk or ''} {x}: {time.time() - t0:.0f} s")
    M = pd.DataFrame(main)
    if len(M):
        ok = M.p.notna()
        M.loc[ok, "q_bh"] = multipletests(M.loc[ok, "p"], method="fdr_bh")[1]
        M["sig_bh_05"] = M.q_bh < 0.05
    M.to_csv(out / f"{tag}_main_mixed.csv", index=False)
    if rob:
        R = pd.DataFrame(rob)
        for mname, g in R.groupby("model"):
            ok = g.p.notna()
            R.loc[g.index[ok], "q_bh"] = multipletests(g.loc[ok, "p"], method="fdr_bh")[1]
        R.to_csv(out / f"{tag}_robustness.csv", index=False)
    S = pd.DataFrame(strat); S.to_csv(out / f"{tag}_by_stratum.csv", index=False)
    H = pd.DataFrame(het); H.to_csv(out / f"{tag}_heterogeneity_tau.csv", index=False)
    T = pd.DataFrame(tr); T.to_csv(out / f"{tag}_transfer_loso.csv", index=False)
    if len(T):
        Ts = T.groupby(["group", "feature", "outcome"], dropna=False).agg(
            strata=("held_out", "size"), gain_mean=("transfer_gain", "mean"), gain_pos_share=("transfer_gain", lambda v: (v > 0).mean()),
            mse_ratio_loso_full=("mse_ratio_loso_full", "mean"), beta_loso_min=("beta_loso", "min"), beta_loso_max=("beta_loso", "max"),
            beta_full=("beta_full", "first")).reset_index()
        Ts["beta_loso_sign_stable"] = np.sign(Ts.beta_loso_min) == np.sign(Ts.beta_loso_max)
        Ts.to_csv(out / f"{tag}_transfer_summary.csv", index=False)
    return M, H


def summary_table(M: pd.DataFrame, H: pd.DataFrame, keys=("group", "feature")) -> pd.DataFrame:
    if not len(M):
        return M
    keys = list(keys)
    a = M.pivot_table(index=keys, columns="outcome", values="coef", dropna=False).reindex(columns=OUTCOMES)
    q = M.pivot_table(index=keys, columns="outcome", values="q_bh", dropna=False).reindex(columns=OUTCOMES)
    t = H.pivot_table(index=keys, columns="outcome", values="tau", dropna=False).reindex(columns=OUTCOMES)
    s = a.round(3).astype(str) + np.where(q < 0.05, "*", "") + " (τ " + t.round(2).astype(str) + ")"
    return s.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--emb", choices=["joint", "image"], default="joint")
    ap.add_argument("--vlm", default=str(ROOT / "out/vlm/vlm_triggers.parquet"))
    ap.add_argument("--emb-dir", default=str(ROOT / "out/embed"))
    ap.add_argument("--min-pos", type=int, default=30)
    ap.add_argument("--humor2", default=str(ROOT / "out/vlm/humor_pass2.parquet"))
    ap.add_argument("--quick", action="store_true", help="main model, strata and transfer only (no robustness checks)")
    ap.add_argument("--out", default=str(ROOT / "out/analysis"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = load(args)
    log(f"NB2 (robustness): alpha fixed from {ALPHA_FILE}: " + ", ".join(f"{k} {v:.4f}" for k, v in NB_ALPHA.items()))
    d = d[d.status.isin(["ok", "retry_ok"])].copy()
    log(f"after dropping VLM errors: {len(d)}")

    # descriptive: prevalence of all features by modality and stratum
    ALL = TRIGGERS + ["humor_strict", "number_in_title_vlm", "question_in_title_vlm", "call_to_share_vlm"]
    prev = d.groupby(["stratum", "modality"], observed=True)[ALL].mean().round(3)
    prev.to_csv(out / "given_prevalence_by_cell.csv")
    log("feature prevalence by modality (number/question/call_to_share - rules; *_vlm - VLM):\n"
        + d.groupby("modality", observed=True)[ALL].mean().round(3).T.to_string())
    fr = first_line_rules(d); fr.to_csv(out / "given_check_rule_vs_vlm.csv", index=False)
    log("number_in_title / question_in_title: defined by the first-line rule; agreement with VLM (share, kappa):\n"
        + fr.round(3).to_string(index=False))

    # predefined triggers: three families, BH within family
    n_child = int(d.child.fillna(0).astype(int).sum())
    confirm = CONFIRM_VLM + (["child"] if n_child >= args.min_pos else []) + CONFIRM_RULE
    log(f"child: positives {n_child} -> {'in the confirmatory family' if n_child >= args.min_pos else 'NOT included (< %d)' % args.min_pos}")
    explore = [x for x in EXPLORE if d[x].notna().any()]
    for fam, feats, title in [("confirm", confirm, "CONFIRMATORY family"),
                              ("explore", explore, "EXPLORATORY features"),
                              ("suppl", SUPPL, "SUPPLEMENTARY: call_to_share by VLM")]:
        log(f"== {title}: {feats} (all modalities; FE modality + size; random effect - community)")
        M, H = analyse(d, feats, ["modality", "size_bucket"], fam, None, args, out, fam)
        st = summary_table(M, H); st.to_csv(out / f"{fam}_summary.csv", index=False)
        log(f"{title} - coefficients (log scale; * - q_BH < 0.05 within family; tau - spread across strata):\n" + st.to_string(index=False))

    # latent triggers
    log(f"== latent triggers: PCA of emb_{args.emb} within modality, K = {args.k}")
    E = np.load(Path(args.emb_dir) / f"emb_{args.emb}.npy").astype(np.float64)
    ids = pd.read_parquet(Path(args.emb_dir) / "emb_ids.parquet")
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    pos = dict(zip(ids.post_id, ids.row))
    have = d.post_id.isin(pos)
    log(f"embeddings available for {int(have.sum())} of {len(d)}")
    d = d[have].copy()
    pcs = [f"pc{i + 1}" for i in range(args.k)]
    ev_rows, ext_rows = [], []
    allp = pd.read_parquet(ROOT / "data/media_items.parquet")  # PCA on all posts of the modality in the slice (not only the pool)
    allp = allp[allp.post_id.isin(pos)]
    for mod in ["image", "video", "clip"]:
        a = allp[allp.modality == mod]
        X = E[[pos[p] for p in a.post_id]]
        mu = X.mean(0)
        U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
        V = Vt[: args.k]
        ev = S**2 / (S**2).sum()
        sc = (X - mu) @ V.T
        sc = (sc - sc.mean(0)) / sc.std(0)
        # the sign of a direction is arbitrary (SVD); interpret via extreme posts at both poles
        scdf = pd.DataFrame(sc, columns=pcs, index=a.post_id.values)
        for i, c in enumerate(pcs):
            ev_rows.append({"modality": mod, "pc": c, "explained_var": ev[i]})
            o = scdf[c].sort_values()
            for pole, sel in [("neg", o.index[:8]), ("pos", o.index[-8:][::-1])]:
                for p in sel:
                    r = a.set_index("post_id").loc[p]
                    ext_rows.append({"modality": mod, "pc": c, "pole": pole, "post_id": p, "score": o[p], "stratum": r.stratum,
                                     "first_line": r.first_line, "feed_image": r.feed_image})
        m = d.modality == mod
        for c in pcs:
            d.loc[m, c] = scdf.loc[d.loc[m, "post_id"], c].values
    pd.DataFrame(ev_rows).to_csv(out / "latent_pca_explained_var.csv", index=False)
    pd.DataFrame(ext_rows).to_csv(out / "latent_pca_extreme_posts.csv", index=False)
    # correlation of latent directions with predefined triggers (for interpretation)
    cr = []
    for mod, g in d.groupby("modality", observed=True):
        for c in pcs:
            for t in TRIGGERS:
                cr.append({"modality": mod, "pc": c, "trigger": t, "corr": g[c].corr(g[t].astype(float))})
    pd.DataFrame(cr).to_csv(out / "latent_pc_vs_given_corr.csv", index=False)
    ML, HL = analyse(d, pcs, ["size_bucket"], "latent", "modality", args, out, "latent")
    sl = summary_table(ML, HL); sl.to_csv(out / "latent_summary.csv", index=False)
    log("latent (per 1 SD of projection; * - q_BH < 0.05):\n" + sl.to_string(index=False))
    log(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")
    (out / "analysis_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
