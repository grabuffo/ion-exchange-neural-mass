"""Small plotting helpers shared by the figure scripts."""
from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
                            "xtick.labelsize": 6, "ytick.labelsize": 6,
                            "legend.fontsize": 6, "pdf.fonttype": 42, "svg.fonttype": "none"})

V_COLOR = "#0032cb"     # membrane potential (blue) in Figs. 2-4
K_COLOR = "#65cc00"     # extracellular potassium (green)
POP_COLOR = "#2f4e9e"   # population (Fig. 3b)
MF_COLOR = "#1f9e89"    # mean field (Fig. 3b)


def strip(ax, bottom=True, left=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
    if left:
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])


def scalebar(ax, x0, length, label, y=None, lw=1.5):
    """Horizontal scale bar in data units, drawn near the bottom of ``ax``."""
    y0, y1 = ax.get_ylim()
    y = y0 + 0.02 * (y1 - y0) if y is None else y
    ax.plot([x0, x0 + length], [y, y], "k-", lw=lw, clip_on=False)
    ax.text(x0 + length / 2, y, label, ha="center", va="top", fontsize=6)


def save(fig, outdir, name, formats=("pdf", "png"), dpi=300):
    os.makedirs(outdir, exist_ok=True)
    for fmt in formats:
        fig.savefig(os.path.join(outdir, f"{name}.{fmt}"), dpi=dpi,
                    bbox_inches="tight", transparent=(fmt == "pdf"))
    print("saved", os.path.join(outdir, name), "+".join(formats))
