#!/usr/bin/env python3
"""Small, self-contained Matplotlib style helper for COVE figures."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


_WIDTHS = {
    "ieee": 3.487,
    "ieee_double": 7.0,
    "custom": 3.5,
}


def setup_style(
    venue="ieee",
    width_inches=None,
    height_ratio=0.618,
    use_latex=False,
    **_,
):
    """Configure portable, vector-friendly defaults at final paper size."""
    width = width_inches or _WIDTHS.get(venue, _WIDTHS["custom"])
    rc = {
        "figure.figsize": (width, width * height_ratio),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": bool(use_latex),
        "axes.prop_cycle": mpl.cycler(
            color=["#2f6fbb", "#538000", "#8c1515", "#59636e"]
        ),
    }
    plt.rcParams.update(rc)
    return rc


def save_fig(fig, output, formats=("pdf", "png"), dpi=300):
    """Save a figure beside its requested stem and return written paths."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix:
        paths = [target]
    else:
        paths = [target.with_suffix(f".{fmt}") for fmt in formats]
    for path in paths:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if path.suffix.lower() in {".png", ".tif", ".tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
    plt.close(fig)
    return paths
