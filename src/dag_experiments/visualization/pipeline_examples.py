"""Drawings of a few small pipelines from the baseline graph, node by node."""

import math
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
    source, small, large = (_PipelineDrawing.layout(dag, index) for index in examples)
    row_heights = [max(source.height, small.height), large.height]
    fig, axes = plt.subplot_mosaic(
        [["source", "small"], ["large", "large"]], figsize=(10, 0.25 * sum(row_heights) + 3),
        layout="constrained", height_ratios=row_heights,
    )
    for name, drawing in (("source", source), ("small", small), ("large", large)):
        drawing.draw(axes[name], dag)
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
    columns: dict[int, float]  # x per generation
    position: dict[int, tuple[float, float]]  # x and y per node

    @property
    def height(self) -> float:
        ys = [y for _, y in self.position.values()]
        return max(ys) - min(ys) + 2

    @classmethod
    def layout(cls, dag: SyntheticDag, index: int) -> "_PipelineDrawing":
        nodes = np.flatnonzero(dag.pipeline == index)
        incoming = dag.pipeline[dag.children] == index
        parents, children = dag.parents[incoming], dag.children[incoming]
        external = np.unique(parents[dag.pipeline[parents] != index])
        outgoing = int(((dag.pipeline[dag.parents] == index) & ~incoming).sum())

        columns = _generation_columns(dag.generation[np.concatenate([nodes, external])])
        position = {
            node: (columns[int(dag.generation[node])], y)
            for node, y in _layered_positions(dag, nodes, parents, children).items()
        }
        # Parents from elsewhere go in a band below the pipeline, stacked per generation in
        # up to three narrow sub-columns so a crowded generation stays short.
        band = max(y for _, y in position.values()) + 1.5
        external_by_generation = defaultdict(list)
        for node in external.tolist():
            external_by_generation[int(dag.generation[node])].append(node)
        for generation, column in external_by_generation.items():
            stacks = min(3, math.ceil(len(column) / 6))
            for i, node in enumerate(column):
                offset = (i % stacks - (stacks - 1) / 2) * 0.25
                position[node] = (columns[generation] + offset, band + i // stacks)
        return cls(index, parents, children, external, outgoing, columns, position)

    def draw(self, ax: Axes, dag: SyntheticDag) -> None:
        for parent, child in zip(self.parents.tolist(), self.children.tolist()):
            own_edge = dag.pipeline[parent] == self.index
            (x_parent, y_parent), (x_child, y_child) = self.position[parent], self.position[child]
            ax.plot(
                [x_parent, x_child], [y_parent, y_child],
                color=INK_SECONDARY if own_edge else INK_MUTED,
                alpha=0.55 if own_edge else 0.35, linewidth=1, zorder=1,
            )
        for node, (x, y) in self.position.items():
            if dag.pipeline[node] != self.index:
                color = AXIS
            else:
                color = SERIES[1] if dag.generation[node] == 0 else SERIES[0]
            ax.scatter(x, y, s=50, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=2)

        pipeline = dag.pipelines[self.index]
        kind = "Source pipeline" if pipeline.is_source else "Pipeline"
        extra_roots = f", {_count(pipeline.extra_roots, 'extra root')}" if pipeline.extra_roots else ""
        set_title(
            ax,
            f"{kind}: {pipeline.size} nodes in {pipeline.depth} levels{extra_roots}",
            f"{_count(len(self.external), 'parent')} outside the pipeline · "
            f"{_count(self.outgoing, 'edge')} out (not drawn)",
        )
        ys = [y for _, y in self.position.values()]
        ax.set_xticks(list(self.columns.values()), [str(generation) for generation in self.columns])
        ax.set_xlabel("generation (edges point left to right)")
        ax.set_yticks([])
        ax.set_ylim(max(ys) + 0.8, min(ys) - 0.8)
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
    # Only edges inside the pipeline decide the order; parents from elsewhere are drawn in
    # a separate band below the pipeline.
    own = set(nodes.tolist())
    own_parents = defaultdict(list)
    for parent, child in zip(parents.tolist(), children.tolist()):
        if parent in own:
            own_parents[child].append(parent)

    by_generation = defaultdict(list)
    for node in nodes.tolist():
        by_generation[int(dag.generation[node])].append(node)

    # Place generation by generation, so a node's parents already have a height. Putting
    # each node near its parents' average height keeps edges short and reduces crossings;
    # each column is centered around zero.
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
