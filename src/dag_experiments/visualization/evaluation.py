"""Charts of the evaluation: a compiled generational plan versus evaluating at runtime."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from dag_experiments.evaluation.simulation import Makespans, gamma_runtimes, worker_sweep
from dag_experiments.synthetic.generator import BASELINE, generate
from dag_experiments.visualization.style import (
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


def runtime_spread(sweeps: dict[float, list[Makespans]], num_nodes: int, mean: float) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    labels = {sd: "constant" if sd == 0 else f"± {sd:g} s" for sd in sweeps}
    for color, (sd, sweep) in zip(SPREAD_COLORS, sweeps.items()):
        workers = [point.workers for point in sweep]
        ratios = [point.generational / point.asap for point in sweep]
        ax.plot(workers, ratios, color=color, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5)
        ax.annotate(labels[sd], (workers[-1], ratios[-1]), (8, 0), textcoords="offset points",
                    va="center", color=INK_SECONDARY)

    first_workers = min(sweep[0].workers for sweep in sweeps.values())
    last_workers = max(sweep[-1].workers for sweep in sweeps.values())
    ax.axhline(1.05, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("+5%", (first_workers, 1.05), (0, 4), textcoords="offset points",
                va="bottom", color=INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xlim(first_workers * 0.8, last_workers * 3)
    ax.xaxis.set_major_locator(FixedLocator(_worker_ticks(first_workers, last_workers)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _compact(value)))
    ax.xaxis.set_minor_locator(NullLocator())
    # A log scale shows the small gaps where the lines start to part as clearly as the large ones.
    top = max(point.generational / point.asap for sweep in sweeps.values() for point in sweep) * 1.15
    ax.set_yscale("log")
    ax.set_ylim(0.97, top)
    ax.yaxis.set_major_locator(FixedLocator([t for t in (1, 1.1, 1.2, 1.5, 2, 3, 5, 10, 20) if t <= top]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("workers (log scale)")
    ax.set_ylabel("generational ÷ as soon as possible (log scale)")
    # Plain lines in the legend: with the markers' rings they would look dashed.
    ax.legend(
        handles=[Line2D([], [], color=color, label=labels[sd]) for color, sd in zip(SPREAD_COLORS, sweeps)],
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
    args = parser.parse_args()

    apply_style()
    # The baseline graph at several runtime spreads, against the number of workers
    dag = generate(BASELINE, args.seed)
    edges = list(zip(dag.parents.tolist(), dag.children.tolist()))
    sweeps = {
        sd: worker_sweep(edges, dag.num_nodes, gamma_runtimes(dag.num_nodes, SPREAD_MEAN, sd, args.seed))[0]
        for sd in SPREAD_SDS
    }
    figure = runtime_spread(sweeps, dag.num_nodes, SPREAD_MEAN)
    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output / "runtime_spread.png", dpi=200)
    print(args.output / "runtime_spread.png")


if __name__ == "__main__":
    main()
