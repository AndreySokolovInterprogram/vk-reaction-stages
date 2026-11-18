"""Fig. 6. Accumulation of counters over the first 72 h of live posts: sum of values at age t as a share of the sum
at 72 h (share_of_sums = frac_of_sums of step 3.3), all modalities. Markers at 24 h (paper: 65/63/72/49%).
Data: data/figdata_accumulation.csv (from figdata/make_figdata.py), modality = ALL.
"""
import pandas as pd

from common_style import BLUE, DATA, GREEN, GREY, GRID, INK, LIGHT, ORANGE, axes_px, figure_px, save, setup

W, H = 1900, 1283
SERIES = [("views", "Views", GREY, "-", 2.0), ("likes", "Likes", BLUE, "-", 1.6),
          ("comments", "Comments", GREEN, (0, (6, 2.5, 1.5, 2.5)), 1.6), ("reposts", "Reposts", ORANGE, (0, (4, 2.5)), 1.6)]

setup(7)
a = pd.read_csv(DATA / "figdata_accumulation.csv")
a = a[a.modality == "ALL"]

fig = figure_px(W, H)
ax = axes_px(fig, 250, 12, 1880, 1075, W, H)
for y in range(20, 101, 20):
    ax.axhline(y, color=GRID, lw=0.8, zorder=0)
ax.axvline(24, color=LIGHT, lw=0.8, zorder=0)
for counter, label, color, ls, lw in SERIES:
    d = a[a.counter == counter].sort_values("age_h")
    ax.plot(d.age_h, 100 * d.share_of_sums, color=color, ls=ls, lw=lw, label=label, zorder=2)
    v24 = 100 * d.loc[d.age_h == 24, "share_of_sums"].iloc[0]
    ax.plot(24, v24, "o", ms=4.2, color=color, zorder=3)
    print(f"{label:9s} 24 h: {v24:.1f} %")
ax.set_xlim(0, 73); ax.set_ylim(0, 102)
ax.set_xticks(range(0, 73, 12)); ax.set_yticks(range(0, 101, 20))
ax.set_xlabel("Post age, hours", color=INK)
ax.set_ylabel("Share of 72-h total, %", color=INK, labelpad=2)
ax.legend(loc="lower right", bbox_to_anchor=(0.985, 0.0), handlelength=2.4, labelspacing=0.45, labelcolor=INK)
save(fig, "sokol6_accumulation")
