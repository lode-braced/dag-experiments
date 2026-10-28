import matplotlib.pyplot as plt
from matplotlib.axes import Axes

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_MUTED,
            "ytick.labelcolor": INK_MUTED,
            "legend.frameon": False,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def set_title(ax: Axes, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=22)
    ax.annotate(
        subtitle, (0, 1), (0, 8), xycoords="axes fraction", textcoords="offset points",
        fontsize=9, color=INK_SECONDARY, va="bottom",
    )
