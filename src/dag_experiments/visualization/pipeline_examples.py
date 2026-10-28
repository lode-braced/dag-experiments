"""Drawings of a few small pipelines from the baseline graph, node by node."""

from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from dag_experiments.synthetic.generator import IntArray, SyntheticDag
from dag_experiments.visualization.style import (
    AXIS,
    INK_MUTED,
    INK_SECONDARY,
    SERIES,
    SURFACE,
    set_title,
)


def example_pipelines(dag: SyntheticDag) -> Figure:
    pipelines = dag.pipelines
    examples = [
        next(i for i, p in enumerate(pipelines) if p.is_source and 12 <= p.size <= 20),
        next(i for i, p in enumerate(pipelines) if not p.is_source and 20 <= p.size <= 35 and p.extra_roots),
        next(i for i, p in enumerate(pipelines) if not p.is_source and 50 <= p.size <= 80),
    ]
    drawings = [_PipelineDrawing.layout(dag, index) for index in examples]
    fig, axes = plt.subplots(
        len(drawings), 1, figsize=(9, 0.28 * sum(d.height for d in drawings) + 3),
        layout="constrained", gridspec_kw={"height_ratios": [d.height for d in drawings]},
    )
    for ax, drawing in zip(axes, drawings):
        drawing.draw(ax, dag)
    fig.legend(
        handles=[
            _legend_dot(SERIES[1], "root (generation 0)"),
            _legend_dot(SERIES[0], "node of this pipeline"),
            _legend_dot(AXIS, "parent in another pipeline or a global hub"),
        ],
        loc="outside upper left", ncols=3, handletextpad=0.3,
    )
    return fig


@dataclass(frozen=True)
class _PipelineDrawing:
    index: int
    parents: IntArray  # edges into the pipeline's nodes
    children: IntArray
    external: IntArray  # parents outside the pipeline
    outgoing: int  # edges from the pipeline to later pipelines
    x: dict[int, float]  # per generation
    y: dict[int, float]  # per node

    @property
    def height(self) -> float:
        return max(self.y.values()) - min(self.y.values()) + 2

    @classmethod
    def layout(cls, dag: SyntheticDag, index: int) -> "_PipelineDrawing":
        nodes = np.flatnonzero(dag.pipeline == index)
        incoming = dag.pipeline[dag.children] == index
        parents, children = dag.parents[incoming], dag.children[incoming]
        external = np.unique(parents[dag.pipeline[parents] != index])
        outgoing = int(((dag.pipeline[dag.parents] == index) & ~incoming).sum())

        y = _layered_positions(dag, nodes, parents, children)
        # Parents from elsewhere go in a band below the pipeline, stacked per generation.
        band = max(y.values()) + 1.5
        stacked = defaultdict(int)
        for node in external.tolist():
            generation = int(dag.generation[node])
            y[node] = band + stacked[generation]
            stacked[generation] += 1
        x = _generation_columns(dag.generation[np.concatenate([nodes, external])])
        return cls(index, parents, children, external, outgoing, x, y)

    def draw(self, ax: Axes, dag: SyntheticDag) -> None:
        for parent, child in zip(self.parents.tolist(), self.children.tolist()):
            own_edge = dag.pipeline[parent] == self.index
            ax.plot(
                [self.x[dag.generation[parent]], self.x[dag.generation[child]]],
                [self.y[parent], self.y[child]],
                color=INK_SECONDARY if own_edge else INK_MUTED,
                alpha=0.55 if own_edge else 0.35, linewidth=1, zorder=1,
            )
        for node, y in self.y.items():
            if dag.pipeline[node] != self.index:
                color = AXIS
            else:
                color = SERIES[1] if dag.generation[node] == 0 else SERIES[0]
            ax.scatter(
                self.x[dag.generation[node]], y, s=60, color=color, edgecolors=SURFACE,
                linewidths=1.5, zorder=2,
            )

        pipeline = dag.pipelines[self.index]
        kind = "Source pipeline" if pipeline.is_source else "Pipeline"
        extra_roots = f", {_count(pipeline.extra_roots, 'extra root')}" if pipeline.extra_roots else ""
        set_title(
            ax,
            f"{kind}: {pipeline.size} nodes in {pipeline.depth} levels{extra_roots}",
            f"{_count(len(self.external), 'parent')} in other pipelines or hubs · "
            f"{_count(self.outgoing, 'edge')} out to later pipelines (not drawn)",
        )
        ax.set_xticks(list(self.x.values()), [str(generation) for generation in self.x])
        ax.set_xlabel("generation (edges point left to right)")
        ax.set_yticks([])
        ax.set_ylim(max(self.y.values()) + 0.8, min(self.y.values()) - 0.8)
        ax.grid(visible=False)
        ax.spines["left"].set_visible(False)


def _generation_columns(generations: npt.NDArray[np.integer]) -> dict[int, float]:
    """x position per generation; a skipped range of generations leaves a wider gap."""
    columns: dict[int, float] = {}
    for generation in sorted(set(generations.tolist())):
        if not columns:
            columns[generation] = 0.0
        else:
            previous = max(columns)
            columns[generation] = columns[previous] + (1 if generation == previous + 1 else 1.8)
    return columns


def _layered_positions(
    dag: SyntheticDag, nodes: IntArray, parents: IntArray, children: IntArray
) -> dict[int, float]:
    """Stack each generation's nodes vertically, ordered by their parents' average height."""
    own = set(nodes.tolist())
    own_parents = defaultdict(list)
    for parent, child in zip(parents.tolist(), children.tolist()):
        if parent in own:
            own_parents[child].append(parent)

    by_generation = defaultdict(list)
    for node in nodes.tolist():
        by_generation[int(dag.generation[node])].append(node)

    position: dict[int, float] = {}
    for generation in sorted(by_generation):
        column = by_generation[generation]
        column.sort(key=lambda node: np.mean([position[p] for p in own_parents[node]] or [0.0]))
        for row, node in enumerate(column):
            position[node] = row - (len(column) - 1) / 2
    return position


def _count(n: int, noun: str) -> str:
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun}s"


def _legend_dot(color: str, label: str) -> Line2D:
    return Line2D(
        [], [], marker="o", linestyle="none", markersize=8, markerfacecolor=color,
        markeredgecolor=SURFACE, label=label,
    )
