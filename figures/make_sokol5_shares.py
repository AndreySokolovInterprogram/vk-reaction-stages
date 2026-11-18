"""Fig. 5. Joint distribution of within-community log reaction shares: (a) log rho_R vs log rho_L,
(b) log rho_C vs log rho_L. Centering: minus the MEDIAN of community x block (block: clips / other), columns cm_*
(rule of step 3.2 of the main analysis). Pool pool_rho (exposure >= 100).
Spearman rank correlation for text/image/video, in parentheses for clips (computed here from the data).
Data: data/figdata_shares_posts.csv (from figdata/make_figdata.py).
SHARES_SUBSET: all (default; points are all posts of the pool, including clips) or tiv (text/image/video only).
"""
import os

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common_style import BLUE, DATA, INK, axes_px, figure_px, save, setup

W, H = 2816, 1346
CMAP = mcolors.LinearSegmentedColormap.from_list("blues", [(0, "#f2f7fd"), (0.3, "#b9d5f7"), (0.7, BLUE), (1, "#104281")])
SUBSET = os.environ.get("SHARES_SUBSET", "all")

setup(7)
d = pd.read_csv(DATA / "figdata_shares_posts.csv")
tiv, clip = d[d.modality != "clip"], d[d.modality == "clip"]
pts = tiv if SUBSET == "tiv" else d

fig = figure_px(W, H)
panels = [("cm_log_rho_R", r"log $\rho_R$, centered", "(a) Reposts vs. likes", 219),
          ("cm_log_rho_C", r"log $\rho_C$, centered", "(b) Comments vs. likes", 1678)]
for col, ylab, title, x0 in panels:
    ax = axes_px(fig, x0, 82, x0 + 1109, 1140, W, H)
    ax.hexbin(pts.cm_log_rho_L, pts[col], gridsize=(56, 28), extent=(-3, 3, -3.5, 3.5), bins="log", mincnt=1,
              cmap=CMAP, linewidths=0.1, edgecolors="face")
    ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
    ax.set_xticks(range(-3, 4), [str(v).replace("-", "−") for v in range(-3, 4)])
    ax.set_yticks(range(-3, 4), [str(v).replace("-", "−") for v in range(-3, 4)])
    ax.set_xlabel(r"log $\rho_L$, centered within community", color=INK)
    ax.set_ylabel(ylab, color=INK, labelpad=2)
    ax.set_title(title, loc="left", color=INK, fontsize=7.5, pad=4)
    r_all = spearmanr(tiv.cm_log_rho_L, tiv[col]).statistic
    r_clip = spearmanr(clip.cm_log_rho_L, clip[col]).statistic
    ax.text(0.04, 0.95, f"Spearman ρ = {r_all:.2f}\n(clips: {r_clip:.2f})", transform=ax.transAxes,
            ha="left", va="top", color=INK, fontsize=7, linespacing=1.0)
    print(col, f"spearman tiv {r_all:.4f} clip {r_clip:.4f}")
save(fig, "sokol5_shares")
