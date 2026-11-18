"""Primitives for the schematics (Figs. 1-3): canvas in pixel coordinates, rounded boxes, polyline arrows."""
from __future__ import annotations

from matplotlib.patches import FancyBboxPatch

from common_style import AXIS, DPI, INK, plt


def canvas(w_disp: float, h_disp: float, w_px: int, h_px: int):
    """Figure w_px x h_px (600 dpi); frameless axes with coordinates 0..w_disp x 0..h_disp (y downwards)."""
    fig = plt.figure(figsize=(w_px / DPI, h_px / DPI))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w_disp)
    ax.set_ylim(h_disp, 0)
    ax.axis("off")
    return fig, ax


def box(ax, x0, y0, x1, y1, text="", *, fc="white", ec=AXIS, lw=1.0, r=18, size=7, color=INK, weight="normal",
        ls="-", ha="center", tx=None, ty=None, linespacing=1.15, zorder=2, va="center"):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                       lw=lw, ls=ls, zorder=zorder, mutation_aspect=1)
    ax.add_patch(p)
    if text:
        x = (x0 + x1) / 2 if tx is None else tx
        y = (y0 + y1) / 2 if ty is None else ty
        ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color, weight=weight, linespacing=linespacing,
                zorder=zorder + 1)
    return p


def arrow(ax, pts, *, color=AXIS, lw=1.0, dashed=False, head=True, zorder=1, hl=4.2, hw=2.2):
    """Polyline through pts; solid filled arrowhead on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=(0, (3.5, 2.2)) if dashed else "-", solid_capstyle="butt",
            dash_capstyle="butt", zorder=zorder)
    if head:  # arrowhead on a short end segment so that the arrow body does not cover a dashed line
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        d = max(((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5, 1e-9)
        tail = (xb - (xb - xa) / d, yb - (yb - ya) / d)
        ax.annotate("", xy=pts[-1], xytext=tail, zorder=zorder + 1,
                    arrowprops=dict(arrowstyle=f"-|>,head_length={hl / 10:.2f},head_width={hw / 10:.2f}",
                                    color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=10,
                                    connectionstyle="arc3"))
