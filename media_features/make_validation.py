"""Sample 1 for blinded manual checking of VLM labels.
40 random posts of the slice (seed 20261116), excluding the 60/100 speed-test posts, balanced by modality (14 image,
13 video, 13 clip; the extra post goes to images). Feed image (the one shown to the VLM) -> <post_id>.jpg, long side
<= 800 px, in VALIDATION_DIR/validation/. validation.csv: post_id, modality, stratum, first line, VLM status and the
16 VLM concepts as is (number/question/call_to_share are VLM answers here, not rules). Concepts are read from the
finished chunks in out/vlm/; while the main run is in progress some rows are empty - rerun later (the sample and
images do not change, only validation.csv is rewritten).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from features import KEYS, ROOT  # noqa: E402

DST = Path(os.environ.get("VALIDATION_DIR", ROOT / "validation")) / "validation"  # images and table for blinded manual labelling
SEED = 20261116
N = {"image": 14, "video": 13, "clip": 13}

it = pd.read_parquet(ROOT / "data/media_items.parquet")
tests = set(open(ROOT / "data/speedtest2_ids.txt").read().split()) | set(open(ROOT / "data/speedtest_ids.txt").read().split())
sel_path = ROOT / "data/validation_ids.csv"
if sel_path.exists():
    sel = pd.read_csv(sel_path)
else:
    rng = np.random.default_rng(SEED)
    pool = it[~it.post_id.isin(tests)]
    parts = []
    for m, n in N.items():
        g = pool[pool.modality == m].sort_values("post_id")
        parts.append(g.iloc[rng.choice(len(g), n, replace=False)])
    sel = pd.concat(parts)[["post_id"]]
    sel.to_csv(sel_path, index=False)
d = sel.merge(it, on="post_id", how="left")
assert len(d) == 40 and not d.post_id.isin(tests).any()

DST.mkdir(parents=True, exist_ok=True)
for r in d.itertuples():
    out = DST / f"{r.post_id}.jpg"
    if not out.exists():
        im = Image.open(r.feed_image).convert("RGB")
        im.thumbnail((800, 800), Image.LANCZOS)
        im.save(out, quality=90)

parts = sorted((ROOT / "out/vlm").glob("part_*.parquet"))
v = pd.concat([pd.read_parquet(p) for p in parts]).drop_duplicates("post_id", keep="last") if parts else pd.DataFrame(columns=["post_id", "status"] + KEYS)
d = d.merge(v[["post_id", "status"] + KEYS].rename(columns={"status": "vlm_status"}), on="post_id", how="left")
cols = ["post_id", "modality", "stratum", "first_line", "vlm_status"] + KEYS
d[cols].sort_values(["modality", "post_id"]).to_csv(DST / "validation.csv", index=False, encoding="utf-8-sig")
print(f"images {len(list(DST.glob('*.jpg')))}; VLM labels available for {int(d.vlm_status.notna().sum())} of 40 -> {DST}/validation.csv")
