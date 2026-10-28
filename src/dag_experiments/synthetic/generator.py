"""Synthetic data-flow DAGs: pipelines stacked on pipelines.

The generator first plans every node's generation, then adds edges under two rules
that make the planned generation the real one:

- every edge goes from a lower to a higher planned generation (so no cycles), and
- every non-root gets exactly one parent one generation lower (its level-fixing parent).

The readme explains the reasoning behind each parameter.
"""

import itertools
import math
import random
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

type IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class GraphParams:
    nodes: int = 1_200_000

    # Pipelines. Median and max size are given for `reference_nodes` and scale with
    # √(nodes / reference_nodes), so bigger graphs get both more and bigger pipelines.
    reference_nodes: int = 1_200_000
    pipeline_size_median: float = 150
    pipeline_size_sigma: float = 1.2
    pipeline_size_min: int = 5
    pipeline_size_max: int = 5_000
    source_pipeline_fraction: float = 0.3
    extra_upstreams_mean: float = 1.5
    max_generations: int = 150

    # Nodes per level
    source_level0_boost: float = 3.0
    extra_root_fraction: float = 0.02

    # Edges
    extra_parents_mean: float = 1.2
    wide_join_fraction: float = 0.01
    wide_join_min_parents: int = 5
    wide_join_max_parents: int = 200
    same_pipeline_share: float = 0.75
    upstream_share: float = 0.15
    hub_share: float = 0.10
    span_p: float = 0.6
    hub_generation_decay: float = 2.0
    popularity_alpha: float = 1.3
    side_output_fraction: float = 0.5

    @property
    def scaled_pipeline_size_median(self) -> float:
        return self.pipeline_size_median * self._size_scale

    @property
    def scaled_pipeline_size_max(self) -> int:
        return max(self.pipeline_size_min, round(self.pipeline_size_max * self._size_scale))

    @property
    def _size_scale(self) -> float:
        return math.sqrt(self.nodes / self.reference_nodes)


BASELINE = GraphParams()


@dataclass(frozen=True)
class Pipeline:
    start: int  # generation of local level 0
    level_sizes: list[int]  # nodes per local level
    extra_roots: int  # nodes placed at generation 0 (seed, lookup or static data)
    upstreams: list[int]  # main upstream first; empty for source pipelines

    @property
    def is_source(self) -> bool:
        return not self.upstreams

    @property
    def depth(self) -> int:
        return len(self.level_sizes)

    @property
    def size(self) -> int:
        return sum(self.level_sizes) + self.extra_roots


@dataclass(frozen=True)
class SyntheticDag:
    """Edge `i` points from `parents[i]` to `children[i]`.

    `generation` is the planned generation of each node, which equals the generation
    Kahn's algorithm computes: use it as the expected answer when testing sorts.
    `pipeline` holds each node's index into `pipelines`, the plan the graph was built from.
    """

    parents: npt.NDArray[np.int32]
    children: npt.NDArray[np.int32]
    generation: npt.NDArray[np.int32]
    pipeline: npt.NDArray[np.int32]
    pipelines: list[Pipeline]

    @property
    def num_nodes(self) -> int:
        return len(self.generation)

    @property
    def num_edges(self) -> int:
        return len(self.parents)


def generate(params: GraphParams = BASELINE, seed: int = 0) -> SyntheticDag:
    rng = random.Random(seed)
    # Plan at the pipeline level first (sizes, levels, start generations, upstreams), then
    # create the nodes that plan asks for, grouped the ways the edge code draws parents.
    pipelines = _plan_pipelines(params, rng)
    layout = _NodeLayout.build(pipelines, params, rng)

    # Every non-root gets one parent exactly one generation lower, which pins its
    # generation, plus a random number of extra parents from lower generations. Separate
    # draws can pick the same parent for a node more than once; the set keeps one edge.
    edges: set[tuple[int, int]] = set()
    for child in layout.nodes:
        if child.generation == 0:
            continue
        parents = [_level_fixing_parent(child, layout, rng)]
        for _ in range(_extra_parent_count(params, rng)):
            parents.append(_extra_parent(child, layout, params, rng))
        edges.update((parent.id, child.id) for parent in parents)
    # Children pick their parents, so nothing so far makes sure every node is read. Give
    # inputs and most intermediate nodes that no child picked a consumer.
    edges.update(_consumer_edges(edges, layout, params, rng))
    _check_guarantees(edges, layout.nodes)

    # Node ids are already in random order. Put the edges in random order too, so neither
    # reveals the topological order. Sorting first makes the order depend on the seed
    # alone, not on how the set happens to store its edges.
    edge_list = sorted(edges)
    rng.shuffle(edge_list)
    return SyntheticDag(
        parents=np.array([parent for parent, _ in edge_list], dtype=np.int32),
        children=np.array([child for _, child in edge_list], dtype=np.int32),
        generation=np.array([node.generation for node in layout.nodes], dtype=np.int32),
        pipeline=np.array([node.pipeline for node in layout.nodes], dtype=np.int32),
        pipelines=pipelines,
    )


# --- Planning pipelines and generations ---


def _plan_pipelines(params: GraphParams, rng: random.Random) -> list[Pipeline]:
    pipelines: list[Pipeline] = []
    # Earlier pipelines that later ones can consume, with their weight as an upstream:
    # √size, because popular pipelines get consumed more.
    consumable: dict[int, float] = {}

    for index, size in enumerate(_pipeline_sizes(params, rng)):
        depth = min(max(round(typical_depth(size) + rng.gauss()), 1), size - 1)
        is_source = index == 0 or rng.random() < params.source_pipeline_fraction
        if is_source:
            # Source pipelines read raw inputs: no upstreams, starting at generation 0.
            # Their level 0 nodes are already roots, so they get no extra roots.
            start, upstreams, extra_roots = 0, [], 0
        else:
            # Pick the upstreams, then start one generation after a random level of the main
            # (first) upstream, as if consuming the output of that step.
            upstreams = _pick_upstreams(consumable, params, rng)
            main = pipelines[upstreams[0]]
            start = main.start + 1 + rng.randrange(main.depth)
            if start + depth > params.max_generations:
                # Past the generation cap: start after an earlier level of the main upstream,
                # and if even its first level is too late, trim this pipeline's depth so its
                # last level stays under the cap.
                start = max(main.start + 1, params.max_generations - depth)
                depth = min(depth, params.max_generations - start)
            # Non-source pipelines can have roots too: external inputs (seed, lookup or static
            # data) pulled in by this pipeline alone.
            extra_roots = min(rng.binomialvariate(size, params.extra_root_fraction), size - depth)
        level_sizes = _level_sizes(size - extra_roots, depth, is_source, params, rng)
        pipelines.append(Pipeline(start, level_sizes, extra_roots, upstreams))

        # A pipeline starting at the last allowed generation leaves no room to consume it.
        if start < params.max_generations - 1:
            consumable[index] = math.sqrt(size)

    return pipelines


def _pipeline_sizes(params: GraphParams, rng: random.Random) -> list[int]:
    # Draw sizes until the node budget is used up. When the leftover would be too small
    # for a pipeline of its own, the current pipeline takes it, so sizes sum to `nodes`.
    sizes = []
    remaining = params.nodes
    while remaining > 0:
        drawn = rng.lognormvariate(
            math.log(params.scaled_pipeline_size_median), params.pipeline_size_sigma
        )
        size = min(max(round(drawn), params.pipeline_size_min), params.scaled_pipeline_size_max)
        if remaining - size < params.pipeline_size_min:
            size = remaining
        sizes.append(size)
        remaining -= size
    return sizes


def _pick_upstreams(
    consumable: dict[int, float], params: GraphParams, rng: random.Random
) -> list[int]:
    """Distinct earlier pipelines, picked by weight; the first pick is the main upstream."""
    candidates = dict(consumable)
    count = min(1 + _poisson(params.extra_upstreams_mean, rng), len(candidates))
    upstreams = []
    for _ in range(count):
        # Each pick leaves the candidates, so the next pick is a different pipeline.
        picked = rng.choices(list(candidates), weights=list(candidates.values()))[0]
        upstreams.append(picked)
        del candidates[picked]
    return upstreams


def typical_depth(size: float) -> float:
    """Number of levels in a pipeline of `size` nodes, before random variation."""
    return 1.5 * math.log2(size)


def level_shares(depth: int, is_source: bool, params: GraphParams) -> list[float]:
    """Share of a pipeline's nodes at each local level, on top of one node per level."""
    # (level + 1) grows and e^(-level / theta) decays, so shares peak at level theta - 1
    # and taper toward the outputs. Theta grows with depth: deeper pipelines stay wide longer.
    theta = max(1.0, depth / 3)
    weights = [(level + 1) * math.exp(-level / theta) for level in range(depth)]
    # Source pipelines start with many raw tables feeding a few transformations.
    if is_source:
        weights[0] *= params.source_level0_boost
    total = sum(weights)
    return [weight / total for weight in weights]


def _level_sizes(
    nodes: int, depth: int, is_source: bool, params: GraphParams, rng: random.Random
) -> list[int]:
    # One node per level is required: it is the level-fixing parent of the next level.
    # Each remaining node goes to a random level, following the shares.
    shares = level_shares(depth, is_source, params)
    levels = rng.choices(range(depth), weights=shares, k=nodes - depth)
    return [1 + levels.count(level) for level in range(depth)]


# --- Nodes ---


@dataclass(frozen=True)
class _Node:
    id: int
    pipeline: int  # index into the list of pipelines
    generation: int
    # How attractive the node is as a parent. Heavy-tailed, so a few nodes become hubs.
    popularity: float


@dataclass(frozen=True)
class _Pool:
    """Nodes to draw a parent from, each with a chance proportional to its popularity."""

    nodes: list[_Node]
    # Running total of popularity. With it, random.choices finds the drawn node with a
    # binary search instead of adding up all weights again on every draw, which would be
    # slow for the thousands of nodes at generation 0.
    cumulative_popularity: list[float]

    @classmethod
    def of(cls, nodes: list[_Node]) -> "_Pool":
        return cls(nodes, list(itertools.accumulate(node.popularity for node in nodes)))

    @property
    def total_popularity(self) -> float:
        return self.cumulative_popularity[-1]

    def draw(self, rng: random.Random) -> _Node:
        return rng.choices(self.nodes, cum_weights=self.cumulative_popularity)[0]


@dataclass(frozen=True)
class _NodeLayout:
    """The planned nodes, grouped the two ways the edge code draws a parent:

    1. From a cell, the nodes of one pipeline at one generation: the main upstream one
       generation lower, the same pipeline `span` generations back, or the pipeline's
       roots at generation 0.
    2. As a global hub, from all nodes below the child's generation.
    """

    pipelines: list[Pipeline]
    nodes: list[_Node]  # node i has id i
    cells: dict[tuple[int, int], _Pool]  # by (pipeline, generation)
    generations: dict[int, _Pool]

    @classmethod
    def build(
        cls, pipelines: list[Pipeline], params: GraphParams, rng: random.Random
    ) -> "_NodeLayout":
        # The planned (pipeline, generation) of every node. Generation 0 holds a pipeline's
        # roots: level 0 of a source pipeline, or the extra roots of any other pipeline. Those
        # start at generation 1 or later, so their extra roots never share a cell with a level.
        planned: list[tuple[int, int]] = []
        for index, pipeline in enumerate(pipelines):
            planned += [(index, 0)] * pipeline.extra_roots
            for level, size in enumerate(pipeline.level_sizes):
                planned += [(index, pipeline.start + level)] * size

        # A node's id is its position in a random order, so ids say nothing about the
        # topological order. Otherwise sorts that happen to follow id order look faster.
        rng.shuffle(planned)
        nodes = [
            _Node(position, pipeline, generation, rng.paretovariate(params.popularity_alpha))
            for position, (pipeline, generation) in enumerate(planned)
        ]

        # Group the nodes per cell and per generation, the two ways parents are drawn.
        cells: defaultdict[tuple[int, int], list[_Node]] = defaultdict(list)
        generations: defaultdict[int, list[_Node]] = defaultdict(list)
        for node in nodes:
            cells[node.pipeline, node.generation].append(node)
            generations[node.generation].append(node)
        return cls(
            pipelines=pipelines,
            nodes=nodes,
            cells={key: _Pool.of(members) for key, members in cells.items()},
            generations={key: _Pool.of(members) for key, members in generations.items()},
        )


# --- Edges ---


def _level_fixing_parent(child: _Node, layout: _NodeLayout, rng: random.Random) -> _Node:
    """A parent exactly one generation lower, picked by popularity."""
    pipeline = layout.pipelines[child.pipeline]
    # Nodes on a pipeline's first level take the parent from the main upstream, all others
    # from their own pipeline's previous level.
    if child.generation == pipeline.start:
        host = pipeline.upstreams[0]
    else:
        host = child.pipeline
    return layout.cells[host, child.generation - 1].draw(rng)


def _extra_parent_count(params: GraphParams, rng: random.Random) -> int:
    # Wide joins and unions get 5-200 extra parents (log-uniform). Every other node gets
    # a geometric number with mean 1.2, so 45% get none.
    if rng.random() < params.wide_join_fraction:
        log_parents = rng.uniform(
            math.log(params.wide_join_min_parents), math.log(params.wide_join_max_parents)
        )
        return round(math.exp(log_parents))
    return _geometric(1 / (1 + params.extra_parents_mean), rng) - 1


def _extra_parent(
    child: _Node, layout: _NodeLayout, params: GraphParams, rng: random.Random
) -> _Node:
    # Where the parent comes from: mostly the child's own pipeline, sometimes an upstream
    # pipeline or a global hub.
    source = rng.choices(
        ["same pipeline", "upstream", "hub"],
        weights=[params.same_pipeline_share, params.upstream_share, params.hub_share],
    )[0]
    if source == "same pipeline":
        return _same_pipeline_parent(child, layout, params, rng)
    if source == "upstream":
        return _upstream_parent(child, layout, rng)
    return _hub_parent(child, layout, params, rng)


def _same_pipeline_parent(
    child: _Node, layout: _NodeLayout, params: GraphParams, rng: random.Random
) -> _Node:
    """A parent `span` generations back in the child's pipeline: mostly one, sometimes more."""
    pipeline = layout.pipelines[child.pipeline]
    target = child.generation - _geometric(params.span_p, rng)
    if target >= pipeline.start:
        return layout.cells[child.pipeline, target].draw(rng)

    # Reaching below the pipeline's first level: source pipelines read their raw inputs
    # (level 0); others use their extra roots half the time, else an upstream. Either way
    # the roots are the pipeline's nodes at generation 0.
    if pipeline.is_source or (pipeline.extra_roots > 0 and rng.random() < 0.5):
        return layout.cells[child.pipeline, 0].draw(rng)
    return _upstream_parent(child, layout, rng)


def _upstream_parent(child: _Node, layout: _NodeLayout, rng: random.Random) -> _Node:
    """A random upstream of the child's pipeline, at a uniformly random level below the child."""
    pipeline = layout.pipelines[child.pipeline]
    # Source pipelines have no upstream pipelines: they read their raw inputs instead.
    if pipeline.is_source:
        return layout.cells[child.pipeline, 0].draw(rng)

    # A non-main upstream may start at or after the child's generation; the main upstream
    # never does.
    upstream = rng.choice(pipeline.upstreams)
    if layout.pipelines[upstream].start >= child.generation:
        upstream = pipeline.upstreams[0]

    # The upstream's levels that sit below the child
    lowest = layout.pipelines[upstream].start
    highest = min(lowest + layout.pipelines[upstream].depth - 1, child.generation - 1)
    return layout.cells[upstream, rng.randint(lowest, highest)].draw(rng)


def _hub_parent(
    child: _Node, layout: _NodeLayout, params: GraphParams, rng: random.Random
) -> _Node:
    """Any node below the child, weighted by popularity × e^(−generation / decay)."""
    # The decay keeps hubs near the bottom of the graph: dimension tables, calendars, config.
    # All nodes of a generation share its decay factor, so drawing a generation by the total
    # weight of its nodes, then a node within it by popularity, gives each node that weight.
    # No generation below the child is empty: each holds one of its chain of level-fixing parents.
    below = range(child.generation)
    weights = [
        math.exp(-generation / params.hub_generation_decay)
        * layout.generations[generation].total_popularity
        for generation in below
    ]
    generation = rng.choices(below, weights=weights)[0]
    return layout.generations[generation].draw(rng)


def _consumer_edges(
    edges: Collection[tuple[int, int]],
    layout: _NodeLayout,
    params: GraphParams,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """An edge to a consumer from each node no child picked as a parent that should be read."""
    picked = {parent for parent, _ in edges}
    consumer_edges = []
    for node in layout.nodes:
        pipeline = layout.pipelines[node.pipeline]
        # A pipeline's last level holds its outputs: they need no consumer in the pipeline.
        if node.id in picked or node.generation == pipeline.start + pipeline.depth - 1:
            continue
        # Inputs (generation 0) are loaded to be used, so they always get one. An intermediate
        # node stays unread part of the time, as a side output such as a report or an export.
        if node.generation > 0 and rng.random() < params.side_output_fraction:
            continue
        # The consumer is a random node one level up in the same pipeline. Extra roots sit
        # below the pipeline's first level, so theirs is a node of that first level.
        generation = max(node.generation + 1, pipeline.start)
        consumer = rng.choice(layout.cells[node.pipeline, generation].nodes)
        consumer_edges.append((node.id, consumer.id))
    return consumer_edges


# --- Distributions the random module lacks ---


def _poisson(mean: float, rng: random.Random) -> int:
    """Number of events in one unit of time, when events happen independently at rate `mean`."""
    # The gaps between such events are exponential: count how many gaps fit in one unit.
    events = 0
    elapsed = rng.expovariate(mean)
    while elapsed < 1:
        events += 1
        elapsed += rng.expovariate(mean)
    return events


def _geometric(p: float, rng: random.Random) -> int:
    """Number of tries up to and including the first success, each succeeding with chance `p`."""
    tries = 1
    while rng.random() >= p:
        tries += 1
    return tries


# --- Finishing ---


def _check_guarantees(edges: Collection[tuple[int, int]], nodes: list[_Node]) -> None:
    # Every edge goes up at least one generation, so no path can loop back: no cycles.
    assert all(
        nodes[parent].generation < nodes[child].generation for parent, child in edges
    ), "edge does not go up a generation"

    # Every non-root has a parent exactly one generation lower. With the check above, the
    # planned generation is then the longest path from a root, as Kahn's algorithm computes.
    has_level_fixing_parent = {
        child for parent, child in edges if nodes[parent].generation == nodes[child].generation - 1
    }
    assert all(
        node.id in has_level_fixing_parent for node in nodes if node.generation > 0
    ), "non-root without level-fixing parent"
