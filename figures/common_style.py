"""Shared style of the figures (IEEE Access): font, palette, sizes, saving.

Colour-blind-safe palette (blue / orange; green only together with a distinct line style); significance is encoded
by marker fill (filled - significant, open - not), so the figures read in grayscale.
IEEE column width: single column 3.5", double column 7.16". Output: PDF (vector, TrueType fonts) + PNG 600 dpi.

Paths: FIGDATA_DIR - aggregated data (default data/ of the repository),
       FIG_OUT     - output directory (default figures/out).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("FIGDATA_DIR", REPO / "data"))
OUT = Path(os.environ.get("FIG_OUT", REPO / "figures" / "out"))

COL1, COL2 = 3.5, 7.16  # inches
DPI = 600

# palette
BLUE = "#2a78d6"      # shared stages / reposts, likes
ORANGE = "#eb6834"    # sharing motive
GREEN = "#1baf7a"     # evaluation / comments
GREY = "#9b9a95"      # not significant, views, auxiliary lines
AXIS = "#52514e"      # axes, tick labels, schematic frames
INK = "#0b0b0b"       # main text
GRID = "#e7e6e2"      # grid
LIGHT = "#cfcdc8"     # highlighted vertical line (24 h)
BOXFILL = "#f2f1ee"   # fill of source boxes in schematics
HIFILL = "#fdece4"    # fill of the highlighted box (module 4.4)
SHADE = "#e6f0fb"     # background of the shared stages T/O

FONT = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]


def _register_fonts() -> None:
    """Register Liberation Sans (metric-compatible with Arial) explicitly: the matplotlib font cache may be stale."""
    from matplotlib import font_manager as fm
    for path in fm.findSystemFonts():
        if Path(path).name.startswith("LiberationSans-"):
            fm.fontManager.addfont(path)


def setup(size: float = 8.0) -> None:
    _register_fonts()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT,
        "font.size": size,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT[0],
        "mathtext.it": FONT[0] + ":italic",
        "mathtext.bf": FONT[0] + ":bold",
        "mathtext.sf": FONT[0],
        "mathtext.cal": FONT[0],
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": AXIS,
        "ytick.labelcolor": AXIS,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": DPI,
    })


def figure_px(w_px: int, h_px: int):
    """Figure of a given size in pixels at 600 dpi."""
    return plt.figure(figsize=(w_px / DPI, h_px / DPI))


def axes_px(fig, left: float, top: float, right: float, bottom: float, w_px: int, h_px: int, **kw):
    """Axes from frame coordinates in pixels (origin at the top left corner)."""
    return fig.add_axes((left / w_px, 1 - bottom / h_px, (right - left) / w_px, (bottom - top) / h_px), **kw)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"Creator": None, "Producer": None, "CreationDate": None}
    fig.savefig(OUT / f"{name}.pdf", metadata=meta)
    fig.savefig(OUT / f"{name}.png", dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    print("→", OUT / f"{name}.pdf", "+ .png")
