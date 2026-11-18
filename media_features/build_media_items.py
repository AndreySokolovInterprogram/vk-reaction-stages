"""Step 1. Input table media_items for the feed features u_p.

One input per post - what is visible in the feed before expanding:
  * image: file of the media slice (post photo from the URL stored by the panel: orig_photo.url, else photo.url -
    first photo);
  * video/clip: the slice has no covers (only mp4 and image jpg) -> first non-blank mp4 frame: the first decoded frame
    within 10 s with mean brightness >= 12 and SD >= 6 (of 255), otherwise the most contrasted frame within 10 s
    (flag frame_rule=fallback_max_std). The frame is saved to data/frames/<key>.jpg.
  * first line: first non-empty line of the post text (VK markup [club1|Name] -> Name), up to 200 characters.
Processing order (`order`): random permutation within each stratum x modality cell (seed 20261115), then round-robin
over cells (rank 0 of all cells, rank 1 of all cells, ...), so an early stop covers cells evenly.
Reads the slice and the export only; writes to data/.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MEDIA_ROOT", Path(__file__).resolve().parent / "work"))  # working dir: data/, out/, logs/
CUT = Path(os.environ["MEDIA_CUT"])      # media slice (image jpg, video/clip mp4)
EXPORT = Path(os.environ["EXPORT_DIR"])  # panel export
QUEUE = Path(os.environ["MEDIA_QUEUE"])  # media queue of the panel (CSV)
POOLS = Path(os.environ.get("ANALYSIS_OUT", Path(__file__).resolve().parents[1] / "analysis/out")) / "pools.parquet"  # pools of the main analysis
DATA = ROOT / "data"
FRAMES = DATA / "frames"
SEED = 20261115
MENTION = re.compile(r"\[(?:id|club|public)\d+\|([^\]]*)\]")


def first_line(t: str | None) -> str:
    for line in (t or "").splitlines():
        line = MENTION.sub(r"\1", line).strip()
        if line:
            return line[:200]
    return ""


def first_frame(src: Path, dst: Path) -> dict:
    import av
    best, best_std = None, -1.0
    with av.open(str(src)) as c:
        st = c.streams.video[0]
        for fr in c.decode(st):
            t = float(fr.time or 0.0)
            if t > 10.0:
                break
            img = fr.to_image()
            g = np.asarray(img.convert("L"), dtype=np.float32)
            m, s = float(g.mean()), float(g.std())
            if m >= 12 and s >= 6:
                img.save(dst, quality=92)
                return {"frame_t": round(t, 3), "frame_rule": "first_nonblank", "frame_mean": round(m, 1), "frame_std": round(s, 1)}
            if s > best_std:
                best, best_std, bt, bm = img, s, t, m
    if best is None:
        return {"frame_t": None, "frame_rule": "no_frame", "frame_mean": None, "frame_std": None}
    best.save(dst, quality=92)
    return {"frame_t": round(bt, 3), "frame_rule": "fallback_max_std", "frame_mean": round(bm, 1), "frame_std": round(best_std, 1)}


def main():
    FRAMES.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(CUT / "meta/files_ok.tsv", sep="\t")
    q = pd.read_csv(QUEUE, usecols=["key", "vk_owner_id", "vk_post_id", "community_vk_id", "stratum", "size_bucket",
                                     "modality", "members_at_selection"])
    d = f.merge(q, on="key", how="left", suffixes=("", "_q"))
    assert d.vk_owner_id.notna().all() and (d.modality == d.modality_q).all()
    txt = pd.read_parquet(EXPORT / "posts.parquet", columns=["vk_owner_id", "vk_post_id", "text"])
    d = d.merge(txt, on=["vk_owner_id", "vk_post_id"], how="left")
    d["first_line"] = d.text.map(first_line)
    d["text_len"] = d.text.fillna("").str.len()
    d["post_id"] = d.key
    d["media_path"] = [str(CUT / p) for p in d.path]
    # order: permutation within cell + round-robin
    rng = np.random.default_rng(SEED)
    d["cell_rank"] = -1
    for cell, idx in d.groupby("cell").groups.items():
        d.loc[idx, "cell_rank"] = rng.permutation(len(idx))
    cells = sorted(d.cell.unique())
    cell_pos = {c: i for i, c in enumerate(rng.permutation(cells))}
    d["cell_pos"] = d.cell.map(cell_pos)
    d = d.sort_values(["cell_rank", "cell_pos"]).reset_index(drop=True)
    d["order"] = np.arange(len(d))

    # frame visible in the feed
    rows = []
    prev = {}
    out_pq = DATA / "media_items.parquet"
    if out_pq.exists():
        prev = pd.read_parquet(out_pq).set_index("post_id").to_dict("index")
    for i, r in d.iterrows():
        if r.modality == "image":
            rows.append({"feed_image": r.media_path, "feed_source": "image_file", "frame_t": None, "frame_rule": None,
                         "frame_mean": None, "frame_std": None})
            continue
        dst = FRAMES / f"{r.key}.jpg"
        if dst.exists() and r.key in prev and prev[r.key].get("frame_rule"):
            p = prev[r.key]
            rows.append({k: p[k] for k in ["feed_image", "feed_source", "frame_t", "frame_rule", "frame_mean", "frame_std"]})
            continue
        try:
            info = first_frame(Path(r.media_path), dst)
        except Exception as e:  # noqa: BLE001
            info = {"frame_t": None, "frame_rule": f"error:{type(e).__name__}", "frame_mean": None, "frame_std": None}
        ok = dst.exists() and not str(info["frame_rule"]).startswith(("error", "no_frame"))
        rows.append({"feed_image": str(dst) if ok else None, "feed_source": "first_frame" if ok else "missing", **info})
        if i % 500 == 0:
            print(f"frames {i}/{len(d)}", flush=True)
    d = pd.concat([d, pd.DataFrame(rows, index=d.index)], axis=1)
    cols = ["order", "post_id", "vk_owner_id", "vk_post_id", "community_vk_id", "stratum", "size_bucket",
            "members_at_selection", "modality", "cell", "cell_rank", "media_path", "feed_image", "feed_source",
            "frame_t", "frame_rule", "frame_mean", "frame_std", "first_line", "text_len", "sha256"]
    d = d[cols]
    d.to_parquet(out_pq, index=False)
    d.to_csv(DATA / "media_items.csv", index=False)

    # counts by cell (+ how many are in the rho pool of the main analysis)
    pools = pd.read_parquet(POOLS, columns=["vk_owner_id", "vk_post_id", "pool_rho_all", "pool_rho"])
    m = d.merge(pools, on=["vk_owner_id", "vk_post_id"], how="left")
    m["has_feed_image"] = m.feed_image.notna()
    m["has_first_line"] = m.first_line.str.len() > 0
    tab = m.groupby(["stratum", "modality"]).agg(posts=("post_id", "size"), with_feed_image=("has_feed_image", "sum"),
                                                  with_first_line=("has_first_line", "sum"),
                                                  in_pool_rho_all=("pool_rho_all", "sum"), in_pool_rho_V100=("pool_rho", "sum"))
    tab.to_csv(DATA / "media_items_by_cell.csv")
    wide = tab.posts.unstack()
    print(wide.to_string())
    print("total", len(d), " cells", d.cell.nunique(), " min/max per cell", int(tab.posts.min()), int(tab.posts.max()))
    print("feed_source", d.feed_source.value_counts().to_dict())
    print("frame_rule", d.frame_rule.value_counts(dropna=False).to_dict())
    print("frame_t median (s)", d.frame_t.median(), " >1s:", int((d.frame_t > 1).sum()))
    print("with first line", int(m.has_first_line.sum()), " by modality", m.groupby("modality").has_first_line.mean().round(3).to_dict())
    print("in pool_rho_all", int(m.pool_rho_all.sum()), " in pool_rho (V>=100)", int(m.pool_rho.sum()))


if __name__ == "__main__":
    sys.exit(main())
