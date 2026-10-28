"""Charts of the baseline graph: its pipeline parameters and the resulting shape."""

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    MaxNLocator,
    MultipleLocator,
    NullLocator,
    PercentFormatter,
)

from dag_experiments.synthetic.generator import (
    BASELINE,
    SyntheticDag,
    generate,
    level_shares,
    typical_depth,
)
from dag_experiments.visualization.pipeline_examples import example_pipelines
from dag_experiments.visualization.style import (
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SERIES,
    SURFACE,
    apply_style,
    set_title,
)

compact = FuncFormatter(lambda value, _: f"{value / 1000:g}K" if value >= 1000 else f"{value:g}")


def _log_size_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 50, 150, 500, 1000, 5000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.xaxis.set_minor_locator(NullLocator())


def pipeline_sizes(dag: SyntheticDag) -> Figure:
    sizes = np.array([pipeline.size for pipeline in dag.pipelines])
    median, max_size = BASELINE.scaled_pipeline_size_median, BASELINE.scaled_pipeline_size_max
    bins = np.geomspace(BASELINE.pipeline_size_min, max_size, 31)
    # The last pipeline takes the remaining nodes and can end up a few nodes over the cap.
    clipped = np.minimum(sizes, max_size)
    large = sizes >= 1000

    fig, (by_count, by_nodes) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
    for ax, weights in ((by_count, None), (by_nodes, sizes)):
        heights, _ = np.histogram(clipped, bins, weights=weights)
        ax.bar(bins[:-1], heights, np.diff(bins), align="edge", color=SERIES[0], edgecolor=SURFACE)
        ax.set_ylim(0, heights.max() * 1.15)
        ax.axvline(median, color=INK_SECONDARY, linewidth=1)
        ax.text(median * 1.07, heights.max() * 1.13, f"median {median:,.0f}", color=INK_SECONDARY, va="top")
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(compact)
        _log_size_axis(ax)
    set_title(
        by_count,
        "Pipelines by size",
        f"Log-normal with median {median:,.0f} and σ {BASELINE.pipeline_size_sigma}, clipped to "
        f"{BASELINE.pipeline_size_min}–{max_size:,}: {len(sizes):,} pipelines for "
        f"{BASELINE.nodes / 1e6:g}M nodes",
    )
    set_title(
        by_nodes,
        "Nodes by the size of their pipeline",
        f"Pipelines of 1,000+ nodes are {large.mean():.0%} of pipelines "
        f"but hold {sizes[large].sum() / sizes.sum():.0%} of the nodes",
    )
    by_count.set_ylabel("pipelines")
    by_nodes.set_ylabel("nodes")
    by_nodes.set_xlabel("pipeline size (nodes, log scale)")
    return fig


def pipeline_depth(dag: SyntheticDag) -> Figure:
    sizes = np.array([pipeline.size for pipeline in dag.pipelines])
    depths = np.array([pipeline.depth for pipeline in dag.pipelines])
    jitter = np.random.default_rng(0).uniform(-0.3, 0.3, len(depths))

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.scatter(
        sizes, depths + jitter, s=6, color=SERIES[0], alpha=0.35, linewidths=0,
        label="pipeline (depth jittered to show overlap)",
    )
    curve = np.geomspace(BASELINE.pipeline_size_min, BASELINE.scaled_pipeline_size_max, 200)
    ax.plot(curve, [typical_depth(size) for size in curve], color=INK, label="1.5 · log2(size)")
    for size in (150, 5000):
        depth = typical_depth(size)
        ax.plot(size, depth, "o", markersize=8, color=INK, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(
            f"{size:,} nodes: {depth:.0f} levels", (size, depth), (-10, 12),
            textcoords="offset points", ha="right", color=INK,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5},
        )
    _log_size_axis(ax)
    ax.set_xlabel("pipeline size (nodes, log scale)")
    ax.set_ylabel("depth (levels)")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper left")
    set_title(
        ax,
        "Pipeline depth grows slowly with size",
        "round(1.5 · log2(size) + Normal(0, 1)), clipped to 1–(size − 1)",
    )
    return fig


def nodes_per_level(dag: SyntheticDag) -> Figure:
    fig, (by_size, source) = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for color, size in zip(SERIES, (10, 150, 5000)):
        depth = round(typical_depth(size))
        shares = [(1 + (size - depth) * share) / size for share in level_shares(depth, False, BASELINE)]
        by_size.plot(shares, color=color, label=f"{size:,} nodes, {depth} levels")
    set_title(by_size, "Wide near the inputs, narrow at the outputs", "(ℓ + 1) · e^(−ℓ/θ), θ = max(1, depth/3)")

    size, depth = 150, round(typical_depth(150))
    for color, is_source, label in ((SERIES[0], False, "other pipelines"), (SERIES[1], True, "source pipelines")):
        shares = [(1 + (size - depth) * share) / size for share in level_shares(depth, is_source, BASELINE)]
        source.plot(shares, color=color, label=label)
    set_title(source, "Source pipelines read many raw tables", "Level 0 gets 3× the weight: 150 nodes, 11 levels")

    for ax in (by_size, source):
        ax.set_xlabel("local level ℓ")
        ax.set_ylabel("expected share of the pipeline's nodes")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")
    return fig


def upstreams(dag: SyntheticDag) -> Figure:
    consumers = [pipeline for pipeline in dag.pipelines if not pipeline.is_source]
    counts = np.bincount([len(pipeline.upstreams) for pipeline in consumers])

    fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    ax.bar(np.arange(1, len(counts)), counts[1:], width=0.5, color=SERIES[0])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("upstream pipelines consumed")
    ax.set_ylabel("pipelines")
    ax.yaxis.set_major_formatter(compact)
    set_title(
        ax,
        "Most pipelines consume one to three earlier pipelines",
        f"1 + Poisson(1.5) distinct pipelines, weighted by √size · "
        f"{len(dag.pipelines) - len(consumers):,} of {len(dag.pipelines):,} pipelines "
        f"are source pipelines with no upstream",
    )
    return fig


def stacking(dag: SyntheticDag) -> Figure:
    pipelines = dag.pipelines
    deepest = max(range(len(pipelines)), key=lambda index: pipelines[index].start + pipelines[index].depth)
    lineage = [deepest]
    while not pipelines[lineage[-1]].is_source:
        lineage.append(pipelines[lineage[-1]].upstreams[0])
    lineage = [pipelines[index] for index in reversed(lineage)]

    fig, ax = plt.subplots(figsize=(9, 0.3 * len(lineage) + 1.6), layout="constrained")
    for row, pipeline in enumerate(lineage):
        ax.barh(row, pipeline.depth, left=pipeline.start, height=0.55, color=SERIES[0])
        label = f"{pipeline.size:,} nodes, {pipeline.depth} levels"
        ax.text(pipeline.start + pipeline.depth + 1, row, label + (" (source)" if row == 0 else ""),
                va="center", color=INK_SECONDARY)
        if row:
            ax.plot([pipeline.start, pipeline.start], [row - 1, row], color=INK_MUTED, linewidth=1)
    max_generations = BASELINE.max_generations
    ax.axvline(max_generations, color=INK_SECONDARY, linewidth=1)
    ax.text(max_generations - 1, -0.5, "maximum depth", ha="right", va="center", color=INK_SECONDARY)
    ax.set_xlim(0, max_generations + 2)
    ax.set_ylim(len(lineage) - 0.4, -0.6)
    ax.yaxis.set_major_locator(NullLocator())
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("generation")
    set_title(
        ax,
        "Pipelines stack on pipelines",
        "The deepest pipeline and its chain of main upstreams: each starts after a random level of the one above",
    )
    return fig


def nodes_per_generation(dag: SyntheticDag) -> Figure:
    counts = np.bincount(dag.generation)

    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    ax.bar(np.arange(len(counts)), counts, width=0.7, color=SERIES[0])
    max_generations = BASELINE.max_generations
    ax.axvline(max_generations, color=INK_SECONDARY, linewidth=1)
    ax.text(max_generations - 1, counts[0], "maximum depth", ha="right", va="top", color=INK_SECONDARY)
    ax.annotate(f"{counts[0]:,} roots", (0, counts[0]), (8, -2), textcoords="offset points",
                va="top", color=INK)
    last = len(counts) - 1
    ax.annotate(f"generation {last}: {counts[-1]} nodes", (last, counts[-1]), (0, 30),
                textcoords="offset points", ha="center", color=INK,
                arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 1})
    ax.set_xlim(-1, max_generations + 2)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(compact)
    ax.set_xlabel("generation")
    ax.set_ylabel("nodes")
    set_title(
        ax,
        "Nodes per generation",
        "Depth comes out of the stacking; the cap only trims and is not reached in the baseline",
    )
    return fig


def degree_distributions(dag: SyntheticDag) -> Figure:
    dependencies = np.bincount(dag.children, minlength=dag.num_nodes)
    dependents = np.bincount(dag.parents, minlength=dag.num_nodes)
    # The max label goes where its curve leaves room: dependencies end in a steep drop,
    # so beside the end point; dependents end in a flat run, so above it.
    series = [
        (dependencies[dependencies > 0], "dependencies, as a share of non-roots", (8, 0), "left"),
        (dependents, "dependents, as a share of all nodes", (0, 10), "center"),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for color, (counts, label, offset, align) in zip(SERIES, series):
        k = np.arange(1, counts.max() + 1)
        at_least_k = 1 - np.searchsorted(np.sort(counts), k, side="left") / len(counts)
        ax.plot(k, at_least_k, color=color, drawstyle="steps-post", label=label)
        ax.plot(k[-1], at_least_k[-1], "o", markersize=8, color=color, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"max {k[-1]:,}", (k[-1], at_least_k[-1]), offset, textcoords="offset points",
                    ha=align, va="bottom" if align == "center" else "center", color=INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.4g}%"))
    ax.set_xlabel("at least k (log scale)")
    ax.set_ylabel("share of nodes (log scale)")
    ax.legend(loc="upper right")
    set_title(
        ax,
        "Most nodes have one to three dependencies; dependents have a long tail",
        f"Share of nodes with at least k · {(dependents == 0).mean():.0%} of nodes are sinks "
        f"and {(dependencies == 0).mean():.0%} are roots",
    )
    return fig


FIGURES: dict[str, Callable[[SyntheticDag], Figure]] = {
    "pipeline_sizes": pipeline_sizes,
    "pipeline_depth": pipeline_depth,
    "nodes_per_level": nodes_per_level,
    "upstreams": upstreams,
    "stacking": stacking,
    "nodes_per_generation": nodes_per_generation,
    "degree_distributions": degree_distributions,
    "example_pipelines": example_pipelines,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the charts of the baseline graph")
    parser.add_argument("--output", type=Path, default=Path("docs/images"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    apply_style()
    dag = generate(BASELINE, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, draw in FIGURES.items():
        figure = draw(dag)
        figure.savefig(args.output / f"{name}.png", dpi=200)
        plt.close(figure)
        print(args.output / f"{name}.png")


if __name__ == "__main__":
    main()
