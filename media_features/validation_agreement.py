"""Agreement of VLM labels with blinded expert labels (samples validation and validation2).

Nothing is re-run: VLM labels are taken as is.
  Sample 1 (40 random, seed 20261116): manual labels - VDIR/validation/manual_labels_coordinator_2026-11-16.csv;
    VLM - ROOT/out/vlm/vlm_triggers.parquet (same as VDIR/validation/validation.csv).
  Sample 2 (80 = 40 random + 4x10 enriched_<concept>, seed 20261117): manual labels -
    VDIR/validation2/manual_labels_coordinator_validation2_2026-11-17.csv; VLM and source - ROOT/data/validation2_key.csv
    (checked against vlm_triggers.parquet and humor_pass2.parquet; stops on mismatch).
  humor_strict - as in analysis_media.py: humor=0 -> 0, humor=1 -> second-pass answer (humor_pass2.parquet);
    compared with the manual humor label.
Posts with VLM status = error are excluded.

Metrics (VLM against the manual labels as reference):
  agreement_random80.csv  - random part (sample 1 + random part of sample 2): TP, FP, FN, TN, shares of 1, precision
                            and recall with 95% Wilson CI, Cohen kappa. Recall and kappa are reported ONLY here (and by
                            sample below).
  agreement_by_sample.csv - same metrics for sample 1 and the 40 random posts of sample 2 + Fisher exact test for a
                            difference in precision and recall between samples (shift check).
  precision_all120.csv    - precision on all VLM=1 posts (random + enriched) with Wilson CI; for face, people_many,
                            useful_info, humor also "random + own enriched part" (a simple random sample of VLM=1 for
                            the concept; other concepts' enriched parts are biased for it).
  disagreements.csv       - all disagreements: post_id, sample, subsample, modality, concept, VLM, manual label, note.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(os.environ.get("MEDIA_ROOT", Path(__file__).resolve().parent / "work"))  # working dir: data/, out/, logs/
VDIR = Path(os.environ.get("VALIDATION_DIR", ROOT / "validation"))  # directory with validation/ and validation2/ (manual labels)
OUT = Path(os.environ.get("VALIDATION_OUT", ROOT / "out/validation_agreement"))
CONCEPTS = ["face", "people_many", "child", "animal", "food", "text_overlay", "brand_logo", "useful_info", "humor",
            "call_to_share"]
ALL = CONCEPTS + ["humor_strict"]          # humor_strict is compared with manual humor
MANUAL_OF = {c: c for c in CONCEPTS} | {"humor_strict": "humor"}
ENRICH = ["face", "people_many", "useful_info", "humor"]
Z = 1.959964


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + Z**2 / n
    c = (p + Z**2 / (2 * n)) / den
    h = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def kappa(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
    return np.nan if pe == 1 else (po - pe) / (1 - pe)


def metrics(d, c):
    v, m = d[f"vlm_{c}"].astype(int), d[f"man_{MANUAL_OF[c]}"].astype(int)
    tp, fp = int(((v == 1) & (m == 1)).sum()), int(((v == 1) & (m == 0)).sum())
    fn, tn = int(((v == 0) & (m == 1)).sum()), int(((v == 0) & (m == 0)).sum())
    n = tp + fp + fn + tn
    pr, pr_lo, pr_hi = wilson(tp, tp + fp)
    rc, rc_lo, rc_hi = wilson(tp, tp + fn)
    return dict(concept=c, n=n, TP=tp, FP=fp, FN=fn, TN=tn, share_vlm=(tp + fp) / n, share_manual=(tp + fn) / n,
                accuracy=(tp + tn) / n, precision=pr, precision_lo=pr_lo, precision_hi=pr_hi,
                recall=rc, recall_lo=rc_lo, recall_hi=rc_hi, kappa=kappa(tp, fp, fn, tn))


def load():
    v = pd.read_parquet(ROOT / "out/vlm/vlm_triggers.parquet")[["post_id", "status"] + CONCEPTS]
    h2 = pd.read_parquet(ROOT / "out/vlm/humor_pass2.parquet")[["post_id", "humor_strict"]]
    v = v.merge(h2, on="post_id", how="left")
    hs = v.humor_strict.astype("float64")
    hs[v.humor.eq(0).fillna(False).to_numpy(bool)] = 0.0
    v["humor_strict"] = hs

    m1 = pd.read_csv(VDIR / "validation/manual_labels_coordinator_2026-11-16.csv", encoding="utf-8-sig")
    m2 = pd.read_csv(VDIR / "validation2/manual_labels_coordinator_validation2_2026-11-17.csv", encoding="utf-8-sig")
    key = pd.read_csv(ROOT / "data/validation2_key.csv", encoding="utf-8-sig")
    ids1 = pd.read_csv(ROOT / "data/validation_ids.csv")
    assert set(m1.post_id) == set(ids1.post_id) and len(m1) == 40, "sample 1: manual labels != validation_ids"
    assert set(m2.post_id) == set(key.post_id) and len(m2) == 80, "sample 2: manual labels != key"
    assert not set(m1.post_id) & set(m2.post_id)

    # check the key against the VLM output (the key is a copy made at sampling time)
    chk = key.merge(v, on="post_id", suffixes=("", "_v"))
    chk.loc[chk.humor == 0, "humor_strict"] = chk.loc[chk.humor == 0, "humor_strict"].fillna(0)  # empty in the key when humor=0
    bad = {c: int((chk[c].fillna(-1) != chk[f"{c}_v"].fillna(-1)).sum()) for c in ALL}
    bad["status"] = int((chk.status != chk.status_v).sum())
    if any(bad.values()) or len(chk) != 80:
        sys.exit(f"validation2 key differs from the VLM output: {bad}")

    m1 = m1.assign(sample="v1", subsample="random")
    m2 = m2.merge(key[["post_id", "source"]], on="post_id").rename(columns={"source": "subsample"}).assign(sample="v2")
    man = pd.concat([m1, m2], ignore_index=True)
    man = man.rename(columns={c: f"man_{c}" for c in CONCEPTS})
    d = man.merge(v.rename(columns={c: f"vlm_{c}" for c in ALL}), on="post_id", how="left")
    return d


def main():
    d = load()
    err = d[~d.status.isin(["ok", "retry_ok"])]
    print(f"posts {len(d)}; excluded with VLM error: {len(err)} {err[['post_id', 'sample']].values.tolist()}")
    d = d[d.status.isin(["ok", "retry_ok"])].copy()
    assert d[[f"vlm_{c}" for c in ALL] + [f"man_{c}" for c in CONCEPTS]].notna().all().all(), "missing labels"
    rnd = d[d.subsample == "random"]
    print(f"valid {len(d)}: random {len(rnd)} (v1 {int((rnd['sample'] == 'v1').sum())}, "
          f"v2 {int((rnd['sample'] == 'v2').sum())}), enriched {int((d.subsample != 'random').sum())}")
    OUT.mkdir(parents=True, exist_ok=True)

    # 1) random part
    a = pd.DataFrame([metrics(rnd, c) for c in ALL])
    a.to_csv(OUT / "agreement_random80.csv", index=False, encoding="utf-8-sig", float_format="%.4f")

    # 3) by sample + Fisher test
    rows = []
    for s in ["v1", "v2"]:
        for c in ALL:
            rows.append({"sample": s} | metrics(rnd[rnd["sample"] == s], c))
    bs = pd.DataFrame(rows)
    pf = []
    for c in ALL:
        x1, x2 = bs[(bs["sample"] == "v1") & (bs.concept == c)].iloc[0], bs[(bs["sample"] == "v2") & (bs.concept == c)].iloc[0]
        pp = fisher_exact([[x1.TP, x1.FP], [x2.TP, x2.FP]])[1] if (x1.TP + x1.FP) and (x2.TP + x2.FP) else np.nan
        pr = fisher_exact([[x1.TP, x1.FN], [x2.TP, x2.FN]])[1] if (x1.TP + x1.FN) and (x2.TP + x2.FN) else np.nan
        pv = fisher_exact([[x1.TP + x1.FP, x1.FN + x1.TN], [x2.TP + x2.FP, x2.FN + x2.TN]])[1]
        pm = fisher_exact([[x1.TP + x1.FN, x1.FP + x1.TN], [x2.TP + x2.FN, x2.FP + x2.TN]])[1]
        pf.append(dict(concept=c, fisher_p_precision=pp, fisher_p_recall=pr, fisher_p_share_vlm=pv, fisher_p_share_manual=pm))
    bs = bs.merge(pd.DataFrame(pf), on="concept")
    bs.to_csv(OUT / "agreement_by_sample.csv", index=False, encoding="utf-8-sig", float_format="%.4f")

    # 2) precision on all VLM=1
    prow = []
    for c in ALL:
        mc = MANUAL_OF[c]
        vv = d[d[f"vlm_{c}"] == 1]
        tp, k = int((vv[f"man_{mc}"] == 1).sum()), len(vv)
        p, lo, hi = wilson(tp, k)
        r = dict(concept=c, n_vlm1_all=k, TP_all=tp, FP_all=k - tp, precision_all=p, precision_all_lo=lo,
                 precision_all_hi=hi, n_vlm1_enriched=int((vv.subsample != "random").sum()))
        base = c if c in ENRICH else ("humor" if c == "humor_strict" else None)
        if base:
            ww = vv[vv.subsample.isin(["random", f"enriched_{base}"])]
            tp2, k2 = int((ww[f"man_{mc}"] == 1).sum()), len(ww)
            p2, lo2, hi2 = wilson(tp2, k2)
            r |= dict(n_vlm1_random_own=k2, TP_random_own=tp2, precision_random_own=p2,
                      precision_random_own_lo=lo2, precision_random_own_hi=hi2)
        prow.append(r)
    pa = pd.DataFrame(prow)
    pa.to_csv(OUT / "precision_all120.csv", index=False, encoding="utf-8-sig", float_format="%.4f")

    # 4) disagreements
    drows = []
    for c in ALL:
        mc = MANUAL_OF[c]
        x = d[d[f"vlm_{c}"].astype(int) != d[f"man_{mc}"].astype(int)]
        for r in x.itertuples():
            drows.append(dict(concept=c, post_id=r.post_id, sample=r.sample, subsample=r.subsample, modality=r.modality,
                              vlm=int(getattr(r, f"vlm_{c}")), manual=int(getattr(r, f"man_{mc}")),
                              type="FP" if getattr(r, f"vlm_{c}") == 1 else "FN",
                              note="" if pd.isna(r.note) else r.note))
    dis = pd.DataFrame(drows).sort_values(["concept", "type", "sample", "post_id"])
    dis.to_csv(OUT / "disagreements.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n== random part\n", a.round(3).to_string(index=False))
    print("\n== by sample\n", bs.round(3).to_string(index=False))
    print("\n== precision on all VLM=1\n", pa.round(3).to_string(index=False))
    print("\n== disagreements:", dis.groupby(["concept", "type"]).size().unstack(fill_value=0).to_dict())

    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
