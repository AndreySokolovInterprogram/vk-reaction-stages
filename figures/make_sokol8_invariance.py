"""Fig. 8. Invariance of effects on log(rho_R/rho_L) across thematic strata: per-stratum estimates (circles;
vertical jitter for readability only, fixed seed), random-effects mean (DerSimonian-Laird) with 95% CI (diamond)
and tau, the between-stratum SD of the effect.
Shown: features significant for log(rho_R/rho_L) in the main model (q < 0.05, BH), ordered by effect.
Data: data/effects_confirm_by_stratum.csv, data/effects_confirm_heterogeneity_tau.csv,
      data/effects_confirm_main_mixed.csv (from media_features/analysis_media.py).
"""
import numpy as np
import pandas as pd

from common_style import AXIS, DATA, GRID, INK, ORANGE, axes_px, figure_px, save, setup

LABELS = {"useful_info": "Practical utility", "food": "Food", "text_overlay": "On-image text",
          "people_many": "Group of people (3+)", "face": "Face", "child": "Child", "animal": "Animal",
          "brand_logo": "Logo, brand", "call_to_share": "Call to share", "number_in_title": "Number in first line",
          "question_in_title": "Question in first line"}
OUT_ = "log_R_over_L"
W, H = 2260, 1552
JITTER, SEED = 0.17, 20261117

setup(7)
m = pd.read_csv(DATA / "effects_confirm_main_mixed.csv")
order = m[(m.outcome == OUT_) & m.sig_bh_05].sort_values("coef", ascending=False).feature.tolist()
s = pd.read_csv(DATA / "effects_confirm_by_stratum.csv")
s = s[s.outcome == OUT_]
t = pd.read_csv(DATA / "effects_confirm_heterogeneity_tau.csv")
t = t[t.outcome == OUT_].set_index("feature")
n_strata = s.stratum.nunique()

fig = figure_px(W, H)
ax = axes_px(fig, 603, 142, 2230, 1342, W, H)
for x in (-0.6, -0.3, 0.3, 0.6):
    ax.axvline(x, color=GRID, lw=0.8, zorder=0)
ax.axvline(0, color=AXIS, lw=0.8, zorder=0)
rng = np.random.default_rng(SEED)
for i, f in enumerate(order):
    d = s[s.feature == f]
    ax.plot(d.coef, i + rng.uniform(-JITTER, JITTER, len(d)), "o", ms=3.3, mfc="white", mec=AXIS, mew=0.9, zorder=2)
    r = t.loc[f]
    lo, hi = r.beta_re - 1.959964 * r.se_re, r.beta_re + 1.959964 * r.se_re
    ax.plot([lo, hi], [i, i], color=ORANGE, lw=2.2, solid_capstyle="round", zorder=3)
    ax.plot(r.beta_re, i, "D", ms=5.6, color=ORANGE, mec="white", mew=0.7, zorder=4)
    ax.text(0.72, i, f"τ = {r.tau:.2f}", color=AXIS, ha="left", va="center")
ax.set_xlim(-0.749, 0.901)
ax.set_xticks([-0.6, -0.3, 0.0, 0.3, 0.6], ["−0.6", "−0.3", "0.0", "0.3", "0.6"])
ax.set_ylim(len(order) - 0.56, -0.394)
ax.set_yticks(range(len(order)), [LABELS[f] for f in order])
ax.set_xlabel(r"Effect on log($\rho_R/\rho_L$) by thematic stratum", color=INK)
h1, = ax.plot([], [], "o", ms=3.3, mfc="white", mec=AXIS, mew=0.9, label=f"single stratum ({n_strata})")
h2, = ax.plot([], [], "D-", ms=5.6, color=ORANGE, mec="white", mew=0.7, lw=2.2, label="random-effects mean, 95% CI")
fig.legend(handles=[h1, h2], loc="upper center", bbox_to_anchor=(0.56, 1.0), ncol=2, columnspacing=0.6,
           handlelength=2.6, handletextpad=0.5, borderaxespad=0.1, labelcolor=INK, fontsize=6.3)
save(fig, "sokol8_invariance")
