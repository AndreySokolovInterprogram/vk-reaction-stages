"""Fig. 3. Data and methods pipeline (sections of the paper). Schematic, no data; the numbers in the labels come
from the panel export (manifest.json: 310 communities, 166,523 posts), the media slice (9,394 posts in 30 cells)
and the description of the methods."""
from common_style import AXIS, BOXFILL, INK, save, setup
from schematic import arrow, box, canvas

setup(7)
fig, ax = canvas(2576, 817, 3353, 1063)
B = 6.8

COLS = [(10, 585, "Data  (Sec. IV)", [(240, "VK panel: 310 groups,"), (292, "166,523 posts"),
                                      (346, "Media slice: 9,394 posts"), (403, "in 30 cells")]),
        (669, 1245, "Features  (Sec. V-B)", [(240, "VLM: 13 visual concepts"), (297, "Rules: 3 text cues"),
                                            (346, "Blinded audit: 120 posts")]),
        (1330, 1906, "Effects  (Sec. V-C)", [(241, "Mixed models,"), (294, "5 outcomes, 55 BH tests"),
                                            (350, "NB2 / Poisson checks")]),
        (1991, 2566, "Invariance  (Sec. V-D)", [(240, "10 per-stratum fits"), (297, "DerSimonian–Laird τ"),
                                               (350, "Leave-one-stratum-out")])]
for x0, x1, title, lines in COLS:
    box(ax, x0, 88, x1, 456, lw=1.0, r=20)
    ax.text(x0 + 30, 147, title, ha="left", va="center", fontsize=7.3, weight="bold", color=INK)
    for y, s in lines:
        ax.text(x0 + 30, y, s, ha="left", va="center", fontsize=B, color=INK)
for xa, xb in ((585, 665), (1245, 1326), (1906, 1987)):
    arrow(ax, [(xa, 271), (xb, 271)], lw=1.0)

box(ax, 10, 617, 584, 791, fc=BOXFILL, ec="none", lw=0, r=20)
ax.text(40, 640, "YouTube (Sec. VI-F):\n180 channels,\n12,418 videos", ha="left", va="top", fontsize=B, color=INK,
        linespacing=1.15, transform=ax.transData)
box(ax, 669, 617, 2566, 791, fc=BOXFILL, ec="none", lw=0, r=20)
ax.text(701, 701, "Synthetic identifiability (Sec. V-E), N = 10,000 posts:\nT and O merged; detection floor for "
        "τ ≈ 0.1 logit", ha="left", va="center", fontsize=B, color=INK, linespacing=1.15)
arrow(ax, [(584, 704), (627, 704), (627, 568), (1407, 568), (1407, 462)], dashed=True, lw=1.0)
arrow(ax, [(1618, 617), (1618, 462)], dashed=True, lw=1.0)
arrow(ax, [(2279, 617), (2279, 462)], dashed=True, lw=1.0)
save(fig, "sokol3_pipeline")
