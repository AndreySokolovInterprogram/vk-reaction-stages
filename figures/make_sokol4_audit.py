"""Fig. 4. Precision and recall of VLM labels against blinded expert labels (79 random posts), 95% Wilson CI.
Data: data/audit_agreement_random80.csv (from media_features/validation_agreement.py).
Concepts ordered by precision; call_to_share is not shown (the VLM gave no positive answers); humor is exploratory.
Right column: number of positives by expert labels (TP + FN). Dashed line: precision threshold 0.70.
"""
import pandas as pd

from common_style import AXIS, BLUE, DATA, GREY, GRID, INK, ORANGE, axes_px, figure_px, save, setup

LABELS = {"food": "Food", "text_overlay": "On-image text", "people_many": "Group of people (3+)", "animal": "Animal",
          "brand_logo": "Logo, brand", "useful_info": "Practical utility", "face": "Face", "child": "Child",
          "humor": "Humor (exploratory)"}
W, H = 2390, 1483

setup(7)
a = pd.read_csv(DATA / "audit_agreement_random80.csv", encoding="utf-8-sig")
a = a[a.concept.isin(LABELS)].sort_values("precision", ascending=False, kind="stable").reset_index(drop=True)
n_posts = int(a.n.iloc[0])

fig = figure_px(W, H)
ax = axes_px(fig, 603, 122, 2230, 1278, W, H)
for x in (0.25, 0.5, 0.75, 1.0):
    ax.axvline(x, color=GRID, lw=0.8, zorder=0)
ax.axvline(0.70, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
for i, r in a.iterrows():
    ax.plot([r.precision_lo, r.precision_hi], [i - 0.15] * 2, color=BLUE, lw=1.1, solid_capstyle="butt", zorder=2)
    ax.plot(r.precision, i - 0.15, "o", ms=5.6, color=BLUE, zorder=3)
    ax.plot([r.recall_lo, r.recall_hi], [i + 0.15] * 2, color=ORANGE, lw=1.1, solid_capstyle="butt", zorder=2)
    ax.plot(r.recall, i + 0.15, "s", ms=4.4, mfc="white", mec=ORANGE, mew=1.3, zorder=3)
    ax.text(1.072, i - 0.02, f"{int(r.TP + r.FN)}", color=AXIS, ha="left", va="center", transform=ax.transData)
ax.text(1.072, -0.75, "pos.", color=AXIS, ha="left", va="center")
ax.set_yticks(range(len(a)), [LABELS[c] for c in a.concept])
ax.set_ylim(len(a) - 0.44, -0.56)
ax.set_xlim(0, 1.04)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0.00", "0.25", "0.50", "0.75", "1.00"])
ax.tick_params(axis="y", length=2.5)
ax.set_xlabel(f"VLM vs. blinded expert, {n_posts} random posts (95% Wilson CI)", color=INK)
h1, = ax.plot([], [], "o-", color=BLUE, ms=5.6, lw=1.1, label="precision")
h2, = ax.plot([], [], "s-", color=ORANGE, ms=4.4, mfc="white", mec=ORANGE, mew=1.3, lw=1.1, label="recall")
fig.legend(handles=[h1, h2], loc="upper center", bbox_to_anchor=(0.555, 1.0), ncol=2, columnspacing=2.2,
           handlelength=2.4, borderaxespad=0.1, labelcolor=INK)
save(fig, "sokol4_audit")
