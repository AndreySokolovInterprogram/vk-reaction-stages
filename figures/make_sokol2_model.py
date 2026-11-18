"""Fig. 2. (a) Multistage model of the reaction to a view: shared stages T/O, evaluation, actions (repost, like,
comment) and the paths of feed, evaluation and motive cues; (b) signatures of a feature acting on each stage
(which log shares and ratios shift). Schematic, no data."""
from common_style import AXIS, BLUE, BOXFILL, GREEN, INK, ORANGE, SHADE, save, setup
from schematic import arrow, box, canvas

setup(7)
fig, ax = canvas(2576, 1391, 4371, 2361)
T = 7.5   # text in boxes
C = 6.8   # cue labels

ax.text(37, 35, "(a)", ha="left", va="center", fontsize=9, weight="bold", color=INK)
box(ax, 379, 303, 1290, 685, fc=SHADE, ec="none", lw=0, r=30, zorder=0)
ax.text(834, 653, "shared stages  T / O", ha="center", va="center", fontsize=7, weight="bold", color=BLUE)

box(ax, 543, 42, 1125, 128, "Feed cues: image, first line", ec=BLUE, lw=1.0, r=22, size=C)
box(ax, 1252, 42, 1732, 128, "Evaluation cues", ec=GREEN, lw=1.0, r=22, size=C)
box(ax, 1859, 33, 2340, 102, "Motive cues", ec=ORANGE, lw=1.0, r=22, size=C)

box(ax, 37, 424, 316, 598, "Exposure\n(view)", fc=BOXFILL, ec=AXIS, lw=1.1, r=22, size=T)
box(ax, 416, 424, 796, 572, "Trigger\n$q_T$", ec=BLUE, lw=1.3, r=22, size=T)
box(ax, 872, 424, 1252, 572, "Operate\n$q_O$", ec=BLUE, lw=1.3, r=22, size=T)
box(ax, 1378, 424, 1708, 598, "Test\n$q_+$", ec=GREEN, lw=1.3, r=22, size=T)
box(ax, 1910, 155, 2290, 268, r"Repost  $e_R^+$", ec=ORANGE, lw=1.8, r=22, size=T)
box(ax, 1910, 337, 2290, 450, r"Like  $e_L^+$", ec=ORANGE, lw=1.8, r=22, size=T)
box(ax, 1910, 589, 2290, 702, r"Comment  $e_C^+$, $e_C^-$", ec=ORANGE, lw=1.8, r=22, size=T)

arrow(ax, [(316, 511), (412, 511)])
arrow(ax, [(796, 497), (868, 497)])
arrow(ax, [(1252, 497), (1374, 497)])
ax.plot([1708, 1808], [485, 485], color=AXIS, lw=1.0)
ax.plot([1808, 1808], [211, 615], color=AXIS, lw=1.0)
for y in (211, 394, 615):
    arrow(ax, [(1808, y), (1906, y)])
ax.text(1781, 290, "+", ha="center", va="center", fontsize=T, color=AXIS)
arrow(ax, [(695, 128), (608, 420)], color=BLUE)
arrow(ax, [(975, 128), (1058, 420)], color=BLUE)
arrow(ax, [(1493, 128), (1540, 420)], color=GREEN)
ax.plot([2340, 2378, 2378], [68, 68, 645], color=ORANGE, lw=1.0)
for y in (211, 394, 645):
    arrow(ax, [(2378, y), (2294, y)], color=ORANGE)
arrow(ax, [(1543, 598), (1543, 763), (2100, 763), (2100, 706)], color=GREEN, dashed=True, lw=1.2)
ax.text(1568, 778, "negative evaluation (1 − $q_+$):\ncomment only, $e_R^-$ = $e_L^-$ = 0", ha="left", va="top",
        fontsize=C, color=AXIS, linespacing=1.15)
ax.text(37, 812, r"Identity:  $\rho_R/\rho_L = e_R^+/e_L^+$  — the shared stages cancel in the ratio".replace(
    r"—", "—"), ha="left", va="center", fontsize=T, color=INK)

# (b) signatures
ax.text(37, 932, "(b)", ha="left", va="center", fontsize=9, weight="bold", color=INK)
ROWS = [1060, 1131, 1203, 1275, 1347]
LAB = [r"$\Delta$log  $\rho_R$", r"$\Delta$log  $\rho_L$", r"$\Delta$log  $\rho_C$", r"$\Delta$log  $\rho_R/\rho_L$",
       r"$\Delta$log  $\rho_C/\rho_L$"]
PANELS = [("Feature acts on shared stages", 438, 493, BLUE, [1, 1, 1, 0, 0]),
          ("Feature acts on sharing motive", 1277, 1328, ORANGE, [1, 0, 0, 1, 0]),
          ("Feature acts on evaluation", 2076, 2163, GREEN, [-0.5, -0.5, 0.5, 0, 1])]
for title, xt, x0, color, vals in PANELS:
    ax.text(xt, 995, title, ha="center", va="center", fontsize=T, color=INK)
    ax.plot([x0, x0], [1022, 1383], color=AXIS, lw=1.0)
    for y, lab, v in zip(ROWS, LAB, vals):
        ax.text(x0 - 323, y, lab, ha="right", va="center", fontsize=C, color=AXIS)
        if v == 0:
            ax.text(x0 + 15, y - 4, "0", ha="left", va="center", fontsize=C, color=AXIS)
        else:
            x1 = x0 + 252 * v
            ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                (min(x0, x1), y - 21), abs(x1 - x0), 42, fc=color, ec="none", zorder=2))
save(fig, "sokol2_model")
