"""Fig. 1. Five-module system; module 4.4 (this paper) highlighted. Schematic, no data."""
from common_style import AXIS, BOXFILL, GREY, HIFILL, INK, ORANGE, save, setup
from schematic import arrow, box, canvas

setup(7)
fig, ax = canvas(2576, 976, 3353, 1271)
BOX = 7.0

# module 4: dashed frame
box(ax, 1308, 18, 2200, 950, fc="none", ec=GREY, lw=1.0, ls=(0, (3, 2)), r=30, zorder=0)
ax.text(2183, 915, "Module 4: content triggers", ha="right", va="center", fontsize=6.5, color=AXIS)

blocks = [  # number, x0, y0, text, fill (and optional width; default 327)
    ("1.1", 10, 402, "VK API\ngroup filtering", BOXFILL),
    ("1.2", 426, 402, "Target-group\nclassification", "white"),
    ("1.3", 426, 727, "Target-user\nretrieval", BOXFILL),
    ("2", 843, 95, "Effective posts\n(regression)", "white"),
    ("3", 843, 727, "User posts and\nprofiles", "white"),
    ("4.1", 1349, 95, "Clustering of\ndomain posts", "white"),
    ("4.2", 1349, 727, "User\nclustering", "white"),
    ("4.3", 1785, 402, "Content–audience\nmatching", "white", 390),  # wider: the label must fit inside
    ("5", 2238, 402, "Post generation\nper cluster", "white"),
]
for num, x0, y0, text, fc, *w in blocks:
    box(ax, x0, y0, x0 + (w[0] if w else 327), y0 + 163, text, fc=fc, lw=0.9, r=22, size=BOX)
    ax.text(x0 + 10, y0 - 44, num, ha="left", va="center", fontsize=7.8, weight="bold", color=INK)
# highlighted module 4.4
box(ax, 1349, 402, 1676, 565, "Viral content;\nstage attribution\nof triggers", fc=HIFILL, ec=ORANGE, lw=2.0, r=22,
    size=BOX)
ax.text(1359, 358, "4.4  this paper", ha="left", va="center", fontsize=7.8, weight="bold", color=ORANGE)

arrow(ax, [(337, 483), (422, 483)])
arrow(ax, [(753, 455), (795, 455), (795, 177), (838, 177)])
arrow(ax, [(590, 565), (590, 722)])
arrow(ax, [(753, 808), (838, 808)])
arrow(ax, [(1170, 177), (1345, 177)])
arrow(ax, [(1170, 214), (1297, 214), (1297, 483), (1345, 483)], color=ORANGE, lw=1.2)
arrow(ax, [(1170, 808), (1345, 808)])
arrow(ax, [(1676, 177), (1735, 177), (1735, 435), (1781, 435)])
arrow(ax, [(1676, 483), (1781, 483)], color=ORANGE, lw=1.2)
arrow(ax, [(1676, 808), (1735, 808), (1735, 531), (1781, 531)])
arrow(ax, [(2175, 483), (2234, 483)])
save(fig, "sokol1_system")
