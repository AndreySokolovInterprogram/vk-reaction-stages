"""Sample 2 for blinded manual checking of VLM labels.
80 posts of the analysis sample (media slice within pool_rho_all, VLM status != error; = 9,394 posts of
analysis_media.py), excluding speed-test posts (speedtest_ids, speedtest2_ids) and sample 1 (data/validation_ids.csv);
seed 20261117:
  40 random (14 image, 13 video, 13 clip; the extra post goes to images)
  + 10 random each with VLM face=1, people_many=1, useful_info=1, humor=1 (no repeats, in concept order).
The order is shuffled (same generator) so that the source cannot be inferred from the order.
VALIDATION_DIR/validation2/: <post_id>.jpg (feed image, long side <= 800 px) and validation2.csv with ONLY post_id,
modality, stratum, first_line (no VLM labels, no selection source).
VLM labels, humor_strict and the source (random / enriched_<concept>) are kept locally in data/validation2_key.csv.
Rerunning does not change the sample (read from validation2_key.csv); missing images are added.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from features import KEYS, ROOT  # noqa: E402

DST = Path(os.environ.get("VALIDATION_DIR", ROOT / "validation")) / "validation2"  # images and table for blinded manual labelling
POOLS = Path(os.environ.get("ANALYSIS_OUT", Path(__file__).resolve().parents[1] / "analysis/out")) / "pools.parquet"  # pools of the main analysis
KEY_PATH = ROOT / "data/validation2_key.csv"
SEED = 20261117
N_RANDOM = {"image": 14, "video": 13, "clip": 13}
ENRICH = ["face", "people_many", "useful_info", "humor"]
N_ENRICH = 10

it = pd.read_parquet(ROOT / "data/media_items.parquet")
v = pd.read_parquet(ROOT / "out/vlm/vlm_triggers.parquet")[["post_id", "status"] + KEYS]
h2 = pd.read_parquet(ROOT / "out/vlm/humor_pass2.parquet")[["post_id", "humor_strict"]]
q = pd.read_parquet(POOLS, columns=["vk_owner_id", "vk_post_id", "pool_rho_all"])
d = it.merge(q, on=["vk_owner_id", "vk_post_id"], how="left").merge(v, on="post_id", how="left").merge(h2, on="post_id", how="left")
d = d[d.pool_rho_all.fillna(False).astype(bool) & d.status.isin(["ok", "retry_ok"])].copy()
print(f"analysis sample: {len(d)}")

if KEY_PATH.exists():
    key = pd.read_csv(KEY_PATH)
    print(f"sample already exists: {KEY_PATH} ({len(key)} posts), not changed")
else:
    tests = set(open(ROOT / "data/speedtest_ids.txt").read().split()) | set(open(ROOT / "data/speedtest2_ids.txt").read().split())
    first40 = set(pd.read_csv(ROOT / "data/validation_ids.csv").post_id)
    pool = d[~d.post_id.isin(tests | first40)].sort_values("post_id").reset_index(drop=True)
    print(f"pool without speed-test posts ({len(tests)}) and sample 1: {len(pool)}")
    rng = np.random.default_rng(SEED)
    picked, src = [], []
    for m, n in N_RANDOM.items():
        g = pool[pool.modality == m]
        ids = g.post_id.values[rng.choice(len(g), n, replace=False)]
        picked += list(ids); src += ["random"] * n
    for c in ENRICH:
        g = pool[(pool[c] == 1) & ~pool.post_id.isin(picked)]
        ids = g.post_id.values[rng.choice(len(g), N_ENRICH, replace=False)]
        picked += list(ids); src += [f"enriched_{c}"] * N_ENRICH
    sel = pd.DataFrame({"post_id": picked, "source": src})
    assert sel.post_id.is_unique and len(sel) == 80
    sel = sel.iloc[rng.permutation(len(sel))].reset_index(drop=True)
    sel.insert(0, "n", np.arange(1, len(sel) + 1))
    key = sel.merge(d[["post_id", "modality", "stratum", "first_line", "status"] + KEYS + ["humor_strict"]], on="post_id", how="left")
    key.to_csv(KEY_PATH, index=False, encoding="utf-8-sig")
    print(f"key -> {KEY_PATH}")

key = key.sort_values("n")
DST.mkdir(parents=True, exist_ok=True)
img = d.set_index("post_id").feed_image
for p in key.post_id:
    out = DST / f"{p}.jpg"
    if not out.exists():
        im = Image.open(img[p]).convert("RGB")
        im.thumbnail((800, 800), Image.LANCZOS)
        im.save(out, quality=90)
key[["post_id", "modality", "stratum", "first_line"]].to_csv(DST / "validation2.csv", index=False, encoding="utf-8-sig")
print(f"images {len(list(DST.glob('*.jpg')))}; sources: {key.source.value_counts().to_dict()}; "
      f"modalities: {key.modality.value_counts().to_dict()} -> {DST}/validation2.csv")
