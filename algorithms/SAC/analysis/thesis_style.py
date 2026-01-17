from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt

def apply_thesis_style() -> None:
    # Modern, clean, thesis-friendly. Exports well to PDF.
    mpl.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "legend.frameon": False,
        "lines.linewidth": 2.2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

def thesis_palette():
    # Explicit modern palette (you asked for modern color design)
    return {
        "stable": "#2EC4B6",   # teal
        "trans":  "#FF9F1C",   # amber
        "crisis": "#E71D36",   # red
        "agent":  "#1F77B4",   # blue
        "eqw":    "#6C757D",   # gray
        "spx":    "#111827",   # near-black
        "shade":  "#111827",   # for subtle shading
    }
