"""Fig. 7. Effects of confirmatory features on log rho_R, log rho_L and log(rho_R/rho_L): mixed model (random
intercept for community), 95% CI; filled marker - q < 0.05 after BH correction, open - not significant.
Data: data/effects_confirm_main_mixed.csv (from media_features/analysis_media.py, family confirm).
Features ordered by the effect on log(rho_R/rho_L).
"""
import pandas as pd

from common_style import AXIS, BLUE, DATA, GREY, GRID, INK, ORANGE, axes_px, figure_px, save, setup

LABELS = {"useful_info": "Practical utility", "food": "Food", "text_overlay": "On-image text",
          "call_to_share": "Call to share", "number_in_title": "Number in first line", "brand_logo": "Logo, brand",
          "question_in_title": "Question in first line", "animal": "Animal", "people_many": "Group of people (3+)",
          "face": "Face", "child": "Child"}
PANELS = [("log_rho_R", r"log $\rho_R$  (reposts)", BLUE, (617, 1670)),
          ("log_rho_L", r"log $\rho_L$  (likes)", BLUE, (1755, 2806)),
          ("log_R_over_L", r"log($\rho_R/\rho_L$)  (sharing motive)", ORANGE, (2891, 3943))]
W, H = 3973, 1639
TOP, BOTTOM = 89, 1429

setup(7)
m = pd.read_csv(DATA / "effects_confirm_main_mixed.csv")
m = m[m.feature.isin(LABELS)]
order = m[m.outcome == "log_R_over_L"].sort_values("coef", ascending=False).feature.tolist()

fig = figure_px(W, H)
axes = []
for k, (outcome, title, color, (x0, x1)) in enumerate(PANELS):
    ax = axes_px(fig, x0, TOP, x1, BOTTOM, W, H)
    d = m[m.outcome == outcome].set_index("feature").loc[order]
    for x in (-0.4, -0.2, 0.2, 0.4):
        ax.axvline(x, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=GREY, lw=0.5, zorder=0)
    for i, (f, r) in enumerate(d.iterrows()):
        sig = bool(r.sig_bh_05)
        c = color if sig else GREY
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=c, lw=1.4, solid_capstyle="butt", zorder=2)
        ax.plot(r.coef, i, "o", ms=4.8, color=c, mfc=c if sig else "white", mec=c, mew=1.2, zorder=3)
    ax.set_xlim(-0.418, 0.42)
    ax.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4], ["−0.4", "−0.2", "0.0", "0.2", "0.4"])
    ax.set_ylim(len(order) - 0.5, -0.506)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS[f] for f in order] if k == 0 else [])
    ax.set_title(title, color=INK, fontsize=7, pad=3)
    axes.append(ax)
axes[1].set_xlabel("Effect on the log scale (95% CI)", color=INK)
h1, = axes[2].plot([], [], "o", ms=4.8, color=AXIS, label="q < 0.05 (BH)")
h2, = axes[2].plot([], [], "o", ms=4.8, mfc="white", mec=GREY, mew=1.2, label="not significant")
axes[2].legend(handles=[h1, h2], loc="lower right", bbox_to_anchor=(1.03, 0.02), handletextpad=0.4,
               borderaxespad=0.2, labelcolor=INK)
save(fig, "sokol7_effects")
