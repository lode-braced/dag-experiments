"""Charts of the evaluation: a compiled generational plan versus evaluating at runtime."""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from dag_experiments.evaluation.simulation import Makespans, gamma_runtimes, worker_sweep
from dag_experiments.synthetic.generator import BASELINE, generate
from dag_experiments.visualization.style import (
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SURFACE,
    apply_style,
    set_title,
)

SPREAD_MEAN = 10.0
SPREAD_SDS: list[float] = [0, 1, 3, 10, 30]
# Ordinal steps of one blue, light to dark: the wider the spread, the darker the line.
SPREAD_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
SMALLER_SIZES = [400_000, 40_000]
# Shared by every runtime spread chart, so charts of different graph sizes compare directly
WORKER_RANGE = (8, 300_000)
RATIO_RANGE = (0.97, 14)


def runtime_spread(sweeps: dict[float, list[Makespans]], num_nodes: int, mean: float) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    _draw_spread(ax, sweeps, _worker_ticks(10, 100_000))
    ax.set_ylabel("generational ÷ as soon as possible (log scale)")
    # Plain lines in the legend: with the markers' rings they would look dashed.
    ax.legend(
        handles=[Line2D([], [], color=color, label=_spread_label(sd)) for color, sd in zip(SPREAD_COLORS, sweeps)],
        title="runtime standard deviation",
        loc="upper left",
    )
    set_title(
        ax,
        "The wider the runtime spread, the sooner and the further the generational plan falls behind",
        f"Generational run time ÷ as soon as possible, simulated on a {num_nodes:,}-node graph "
        f"with gamma runtimes, mean {mean:g} s",
    )
    return fig


def smaller_graphs(sweeps_by_size: dict[int, dict[float, list[Makespans]]]) -> Figure:
    fig, axes = plt.subplots(1, len(sweeps_by_size), figsize=(10, 3.8), layout="constrained", sharey=True)
    for ax, (num_nodes, sweeps) in zip(axes, sweeps_by_size.items()):
        # Only powers of ten fit under the narrower panels.
        _draw_spread(ax, sweeps, [10, 100, 1000, 10_000, 100_000])
        ax.set_title(f"{num_nodes:,} nodes", loc="left", fontsize=10, fontweight="bold", color=INK)
    axes[0].set_ylabel("generational ÷ as soon as possible")
    fig.suptitle(
        "On smaller graphs, the same curves sit at fewer workers",
        x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK,
    )
    return fig


def _draw_spread(ax: Axes, sweeps: dict[float, list[Makespans]], worker_ticks: list[int]) -> None:
    """One line per runtime spread: the generational plan's run time ÷ as soon as possible's."""
    for color, (sd, sweep) in zip(SPREAD_COLORS, sweeps.items()):
        workers = [point.workers for point in sweep]
        ratios = [point.generational / point.asap for point in sweep]
        ax.plot(workers, ratios, color=color, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5)
        ax.annotate(_spread_label(sd), (workers[-1], ratios[-1]), (8, 0), textcoords="offset points",
                    va="center", color=INK_SECONDARY)

    ax.axhline(1.05, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    # Labeled at the right end, past where the sweeps stop
    ax.annotate("+5%", (WORKER_RANGE[1], 1.05), (-4, 4), textcoords="offset points", ha="right",
                va="bottom", color=INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xlim(*WORKER_RANGE)
    ax.xaxis.set_major_locator(FixedLocator(worker_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _compact(value)))
    ax.xaxis.set_minor_locator(NullLocator())
    # A log scale shows the small gaps where the lines start to part as clearly as the large ones.
    ax.set_yscale("log")
    ax.set_ylim(*RATIO_RANGE)
    ax.yaxis.set_major_locator(FixedLocator([1, 1.1, 1.2, 1.5, 2, 3, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("workers (log scale)")


def _spread_label(sd: float) -> str:
    return "constant" if sd == 0 else f"± {sd:g} s"


def _worker_ticks(first: int, last: int) -> list[int]:
    ticks = []
    scale = 10
    while scale <= last:
        ticks += [tick for tick in (scale, 2 * scale, 5 * scale) if first <= tick <= last]
        scale *= 10
    return ticks


def _compact(count: float) -> str:
    return f"{count / 1000:g}k" if count >= 1000 else f"{count:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the charts of the evaluation")
    parser.add_argument("--output", type=Path, default=Path("docs/images"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", choices=["runtime_spread", "smaller_graphs"])
    args = parser.parse_args()

    apply_style()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "runtime_spread"):
        sweeps = _spread_sweeps(BASELINE.nodes, args.seed)
        _save(runtime_spread(sweeps, BASELINE.nodes, SPREAD_MEAN), args.output / "runtime_spread.png")
    if args.only in (None, "smaller_graphs"):
        sweeps_by_size = {size: _spread_sweeps(size, args.seed) for size in SMALLER_SIZES}
        _save(smaller_graphs(sweeps_by_size), args.output / "runtime_spread_smaller_graphs.png")


def _spread_sweeps(num_nodes: int, seed: int) -> dict[float, list[Makespans]]:
    """A worker sweep per runtime spread, on a graph of `num_nodes` nodes."""
    dag = generate(replace(BASELINE, nodes=num_nodes), seed)
    edges = list(zip(dag.parents.tolist(), dag.children.tolist()))
    return {
        sd: worker_sweep(edges, num_nodes, gamma_runtimes(num_nodes, SPREAD_MEAN, sd, seed))[0]
        for sd in SPREAD_SDS
    }


def _save(figure: Figure, path: Path) -> None:
    figure.savefig(path, dpi=200)
    plt.close(figure)
    print(path)

if __name__ == "__main__":
    main()
