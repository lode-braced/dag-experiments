"""Synthetic data-flow DAGs: pipelines stacked on pipelines.

The generator first plans every node's generation, then adds edges under two rules
that make the planned generation the real one:

- every edge goes from a lower to a higher planned generation (so no cycles), and
- every non-root gets exactly one parent one generation lower (its level-fixing parent).

The readme explains the reasoning behind each parameter.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

type IntArray = npt.NDArray[np.int64]
type FloatArray = npt.NDArray[np.float64]


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
    level_sizes: IntArray  # nodes per local level
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
        return int(self.level_sizes.sum()) + self.extra_roots


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
    rng = np.random.default_rng(seed)
    # Plan at the pipeline level first (sizes, levels, start generations, upstreams), then
    # number the nodes so the nodes of one pipeline at one generation form a range of ids:
    # the edge code draws most parents from such a range.
    pipelines = _plan_pipelines(params, rng)
    layout = _NodeLayout.build(pipelines, params.max_generations)
    # How attractive each node is as a parent. Heavy-tailed, so a few nodes become hubs.
    popularity = rng.pareto(params.popularity_alpha, layout.num_nodes) + 1

    # Every non-root gets one parent exactly one generation lower, which pins its
    # generation, plus a random number of extra parents from lower generations.
    children = np.flatnonzero(layout.generation > 0)
    fixing_parents = _level_fixing_parents(children, layout, popularity, rng)
    extra_children, extra_parents = _extra_parents(children, layout, popularity, params, rng)

    parents = np.concatenate([fixing_parents, extra_parents])
    children = np.concatenate([children, extra_children])
    # Separate draws can pick the same parent for a node more than once; keep one edge.
    parents, children = _deduplicate(parents, children, layout.num_nodes)

    # Hide the construction order in the ids, then verify the generation guarantees.
    dag = _shuffle(parents, children, layout, pipelines, rng)
    _check_guarantees(dag)
    return dag


# --- Planning pipelines and generations ---


def _plan_pipelines(params: GraphParams, rng: np.random.Generator) -> list[Pipeline]:
    sizes = _pipeline_sizes(params, rng)
    # Weight of each pipeline when later pipelines pick their upstreams.
    upstream_weight = np.zeros(len(sizes))
    pipelines: list[Pipeline] = []

    for index, size in enumerate(sizes):
        depth = int(np.clip(round(typical_depth(size) + rng.normal()), 1, size - 1))
        is_source = index == 0 or rng.random() < params.source_pipeline_fraction
        if is_source:
            # Source pipelines read raw inputs: no upstreams, starting at generation 0.
            # Their level 0 nodes are already roots, so they get no extra roots.
            start, upstreams, extra_roots = 0, [], 0
        else:
            # Pick the upstreams, then start one generation after a random level of the main
            # (first) upstream, as if consuming the output of that step.
            upstreams = _pick_upstreams(upstream_weight[:index], params, rng)
            main = pipelines[upstreams[0]]
            start = main.start + 1 + int(rng.integers(main.depth))
            if start + depth > params.max_generations:
                # Past the generation cap: start after an earlier level of the main upstream,
                # and if even its first level is too late, trim this pipeline's depth so its
                # last level stays under the cap.
                start = max(main.start + 1, params.max_generations - depth)
                depth = min(depth, params.max_generations - start)
            # Non-source pipelines can have roots too: external inputs (seed, lookup or static
            # data) pulled in by this pipeline alone.
            extra_roots = min(rng.binomial(size, params.extra_root_fraction), size - depth)
        level_sizes = _level_sizes(size - extra_roots, depth, is_source, params, rng)
        pipelines.append(Pipeline(start, level_sizes, extra_roots, upstreams))

        # A pipeline starting at the last allowed generation leaves no room to consume it.
        if start < params.max_generations - 1:
            upstream_weight[index] = np.sqrt(size)

    return pipelines


def _pipeline_sizes(params: GraphParams, rng: np.random.Generator) -> list[int]:
    # Draw sizes until the node budget is used up. When the leftover would be too small
    # for a pipeline of its own, the current pipeline takes it, so sizes sum to `nodes`.
    sizes = []
    remaining = params.nodes
    while remaining > 0:
        drawn = rng.lognormal(np.log(params.scaled_pipeline_size_median), params.pipeline_size_sigma)
        size = int(np.clip(round(drawn), params.pipeline_size_min, params.scaled_pipeline_size_max))
        if remaining - size < params.pipeline_size_min:
            size = remaining
        sizes.append(size)
        remaining -= size
    return sizes


def _pick_upstreams(weight: FloatArray, params: GraphParams, rng: np.random.Generator) -> list[int]:
    """Distinct earlier pipelines, picked by weight; the first pick is the main upstream."""
    count = min(1 + rng.poisson(params.extra_upstreams_mean), np.count_nonzero(weight))
    return rng.choice(len(weight), size=count, replace=False, p=weight / weight.sum()).tolist()


def typical_depth(size: float) -> float:
    """Number of levels in a pipeline of `size` nodes, before random variation."""
    return 1.5 * math.log2(size)


def level_shares(depth: int, is_source: bool, params: GraphParams) -> FloatArray:
    """Share of a pipeline's nodes at each local level, on top of one node per level."""
    level = np.arange(depth)
    # (level + 1) grows and e^(-level / theta) decays, so shares peak at level theta - 1
    # and taper toward the outputs. Theta grows with depth: deeper pipelines stay wide longer.
    theta = max(1.0, depth / 3)
    weights = (level + 1) * np.exp(-level / theta)
    # Source pipelines start with many raw tables feeding a few transformations.
    if is_source:
        weights[0] *= params.source_level0_boost
    return weights / weights.sum()


def _level_sizes(
    nodes: int, depth: int, is_source: bool, params: GraphParams, rng: np.random.Generator
) -> IntArray:
    # One node per level is required: it is the level-fixing parent of the next level.
    # The remaining nodes are spread over the levels at random, following the shares.
    return 1 + rng.multinomial(nodes - depth, level_shares(depth, is_source, params))


@dataclass(frozen=True)
class _NodeLayout:
    """The planned nodes, numbered so the edge code can quickly find candidate parents.

    A parent is drawn in one of two ways:

    1. From the nodes of one pipeline at one generation, a cell: the main upstream one
       generation lower, the same pipeline `span` generations back, or the pipeline's
       roots at generation 0.
    2. As a global hub, from all nodes below the child's generation.

    Numbering the nodes in (pipeline, generation) order gives each cell consecutive ids,
    `count[p, g]` of them starting at `first[p, g]`. Drawing from a cell is then picking
    an id from a range, which numpy does for all children in one call. Hubs don't use the
    cells, only `generation` per node. The per-pipeline arrays hold the plan in the same
    array form, so the edge code can look them up for all children at once too.
    """

    generation: IntArray  # per node
    pipeline: IntArray
    first: IntArray  # [pipeline, generation]: id of the cell's first node
    count: IntArray  # [pipeline, generation]: number of nodes in the cell, 0 if none
    start: IntArray  # per pipeline from here on
    depth: IntArray
    is_source: npt.NDArray[np.bool_]
    upstreams: IntArray  # [pipeline, i]: its i-th upstream, main first; -1 pads shorter lists
    upstream_count: IntArray

    @property
    def num_nodes(self) -> int:
        return len(self.generation)

    @property
    def main_upstream(self) -> IntArray:
        """Per pipeline; -1 for source pipelines."""
        return self.upstreams[:, 0]

    @classmethod
    def build(cls, pipelines: list[Pipeline], max_generations: int) -> "_NodeLayout":
        # Count the nodes of each pipeline at each generation. Generation 0 holds a pipeline's
        # roots: level 0 of a source pipeline, or the extra roots of any other pipeline. Those
        # start at generation 1 or later, so their extra roots never share a cell with a level.
        count = np.zeros((len(pipelines), max_generations), dtype=np.int64)
        for index, pipeline in enumerate(pipelines):
            count[index, pipeline.start : pipeline.start + pipeline.depth] = pipeline.level_sizes
            count[index, 0] += pipeline.extra_roots

        # Number the nodes cell by cell, row by row: a cell's first id is the number of nodes
        # in all cells before it. Each node's pipeline and generation are its cell's row and
        # column.
        first = (np.cumsum(count) - count.ravel()).reshape(count.shape)
        cell = np.repeat(np.arange(count.size), count.ravel())

        # Upstream lists differ in length, so pad them with -1 into one table. It keeps at
        # least one column, so the main upstream column exists even without upstreams.
        upstream_count = np.array([len(p.upstreams) for p in pipelines])
        upstreams = np.full((len(pipelines), max(1, upstream_count.max())), -1)
        for index, pipeline in enumerate(pipelines):
            upstreams[index, : len(pipeline.upstreams)] = pipeline.upstreams

        return cls(
            generation=cell % max_generations,
            pipeline=cell // max_generations,
            first=first,
            count=count,
            start=np.array([p.start for p in pipelines]),
            depth=np.array([p.depth for p in pipelines]),
            is_source=np.array([p.is_source for p in pipelines]),
            upstreams=upstreams,
            upstream_count=upstream_count,
        )


# --- Edges ---

# Where an extra parent comes from
_SAME_PIPELINE, _UPSTREAM, _HUB = 0, 1, 2


def _level_fixing_parents(
    children: IntArray, layout: _NodeLayout, popularity: FloatArray, rng: np.random.Generator
) -> IntArray:
    """One parent exactly one generation lower per child, picked by popularity."""
    pipeline = layout.pipeline[children]
    generation = layout.generation[children]
    # Nodes on a pipeline's first level take the parent from the main upstream.
    on_first_level = generation == layout.start[pipeline]
    host = np.where(on_first_level, layout.main_upstream[pipeline], pipeline)
    return _pick_in_cells(host, generation - 1, layout, popularity, rng)


def _extra_parents(
    children: IntArray,
    layout: _NodeLayout,
    popularity: FloatArray,
    params: GraphParams,
    rng: np.random.Generator,
) -> tuple[IntArray, IntArray]:
    # How many extra parents each child gets: geometric with mean 1.2 (45% get none),
    # except wide joins and unions, which get 5-200 (log-uniform).
    count = rng.geometric(1 / (1 + params.extra_parents_mean), len(children)) - 1
    wide_join = rng.random(len(children)) < params.wide_join_fraction
    log_parents = rng.uniform(
        np.log(params.wide_join_min_parents),
        np.log(params.wide_join_max_parents),
        wide_join.sum(),
    )
    count[wide_join] = np.round(np.exp(log_parents))

    # From here on, one entry per extra parent to draw. `kind` says where it comes from;
    # `from_pipeline` and `from_generation` collect the cell to pick it from.
    children = np.repeat(children, count)
    n = len(children)
    pipeline = layout.pipeline[children]
    generation = layout.generation[children]
    shares = [params.same_pipeline_share, params.upstream_share, params.hub_share]
    kind = rng.choice(3, size=n, p=shares)
    from_pipeline = pipeline.copy()
    from_generation = np.full(n, -1)

    # Same pipeline: reach back `span` generations, mostly one, sometimes more.
    span = rng.geometric(params.span_p, n)
    target = generation - span
    in_pipeline = (kind == _SAME_PIPELINE) & (target >= layout.start[pipeline])
    from_generation[in_pipeline] = target[in_pipeline]

    # Reaching below the pipeline's first level: source pipelines read their raw inputs
    # (level 0); others use their extra roots half the time, else an upstream. Either way
    # the roots are the pipeline's nodes at generation 0.
    below = (kind == _SAME_PIPELINE) & ~in_pipeline
    has_roots = layout.count[pipeline, 0] > 0
    use_roots = below & (layout.is_source[pipeline] | (has_roots & (rng.random(n) < 0.5)))
    from_generation[use_roots] = 0
    kind[below & ~use_roots] = _UPSTREAM

    # Source pipelines have no upstream pipelines: they read their raw inputs instead.
    from_raw = (kind == _UPSTREAM) & layout.is_source[pipeline]
    from_generation[from_raw] = 0
    from_upstream = (kind == _UPSTREAM) & ~layout.is_source[pipeline]
    from_pipeline[from_upstream], from_generation[from_upstream] = _upstream_levels(
        pipeline[from_upstream], generation[from_upstream], layout, rng
    )

    # Global hubs come from the whole graph; all other parents from their chosen cell.
    hub = kind == _HUB
    parents = np.empty(n, dtype=np.int64)
    parents[~hub] = _pick_in_cells(
        from_pipeline[~hub], from_generation[~hub], layout, popularity, rng
    )
    parents[hub] = _pick_hubs(generation[hub], layout, popularity, params, rng)
    return children, parents


def _upstream_levels(
    pipeline: IntArray, generation: IntArray, layout: _NodeLayout, rng: np.random.Generator
) -> tuple[IntArray, IntArray]:
    """A random upstream of each pipeline, at a uniformly random level below `generation`."""
    # A random position within each pipeline's list of upstreams
    position = (rng.random(len(pipeline)) * layout.upstream_count[pipeline]).astype(np.int64)
    upstream = layout.upstreams[pipeline, position]
    # A non-main upstream may start at or after `generation`; the main upstream never does.
    upstream = np.where(
        layout.start[upstream] < generation, upstream, layout.main_upstream[pipeline]
    )
    # The upstream's levels that sit below `generation`
    lowest = layout.start[upstream]
    highest = np.minimum(lowest + layout.depth[upstream] - 1, generation - 1)
    level_generation = lowest + (rng.random(len(pipeline)) * (highest - lowest + 1)).astype(np.int64)
    return upstream, level_generation


def _pick_in_cells(
    pipeline: IntArray,
    generation: IntArray,
    layout: _NodeLayout,
    popularity: FloatArray,
    rng: np.random.Generator,
) -> IntArray:
    """One node of each `pipeline` at `generation`, chosen proportionally to popularity."""
    # A cell's nodes have consecutive ids, so picking within a cell is picking within
    # a range of ids.
    low = layout.first[pipeline, generation]
    high = low + layout.count[pipeline, generation]
    assert (generation >= 0).all() and (high > low).all(), "no nodes to pick from"
    cumulative = np.concatenate([[0.0], np.cumsum(popularity)])
    return _pick_weighted(cumulative, low, high, rng)


def _pick_hubs(
    generation: IntArray,
    layout: _NodeLayout,
    popularity: FloatArray,
    params: GraphParams,
    rng: np.random.Generator,
) -> IntArray:
    """Any node below `generation`, weighted by popularity × e^(−generation / decay)."""
    # Sorted by generation, "all nodes below generation L" is a prefix of the order.
    by_generation = np.argsort(layout.generation, kind="stable")
    sorted_generation = layout.generation[by_generation]
    # The decay keeps hubs near the bottom of the graph: dimension tables, calendars, config.
    weight = popularity[by_generation] * np.exp(-sorted_generation / params.hub_generation_decay)
    cumulative = np.concatenate([[0.0], np.cumsum(weight)])
    below = np.searchsorted(sorted_generation, generation, side="left")
    position = _pick_weighted(cumulative, np.zeros_like(below), below, rng)
    return by_generation[position]


def _pick_weighted(
    cumulative: FloatArray, low: IntArray, high: IntArray, rng: np.random.Generator
) -> IntArray:
    """For each range [low, high), an index drawn proportionally to its weight.

    `cumulative[i]` is the total weight of all indices before `i`.
    """
    # Each index owns a stretch of the number line as long as its weight. Draw a point
    # within the range's stretches and find whose stretch it falls in.
    targets = cumulative[low] + rng.random(len(low)) * (cumulative[high] - cumulative[low])
    picked = np.searchsorted(cumulative, targets, side="right") - 1
    # Float rounding can land a target on a range's edge.
    return np.clip(picked, low, high - 1)


# --- Finishing ---


def _deduplicate(parents: IntArray, children: IntArray, num_nodes: int) -> tuple[IntArray, IntArray]:
    # Encode each edge as one number, so np.unique can drop duplicates, then decode.
    edges = np.unique(parents * num_nodes + children)
    return edges // num_nodes, edges % num_nodes


def _shuffle(
    parents: IntArray,
    children: IntArray,
    layout: _NodeLayout,
    pipelines: list[Pipeline],
    rng: np.random.Generator,
) -> SyntheticDag:
    """Random node ids and edge order, so neither reveals the topological order."""
    # new_id[old id] is the node's new id; node attributes move along with the node.
    new_id = rng.permutation(layout.num_nodes)
    edge_order = rng.permutation(len(parents))
    generation = np.empty(layout.num_nodes, dtype=np.int32)
    generation[new_id] = layout.generation
    pipeline = np.empty(layout.num_nodes, dtype=np.int32)
    pipeline[new_id] = layout.pipeline
    return SyntheticDag(
        parents=new_id[parents[edge_order]].astype(np.int32),
        children=new_id[children[edge_order]].astype(np.int32),
        generation=generation,
        pipeline=pipeline,
        pipelines=pipelines,
    )


def _check_guarantees(dag: SyntheticDag) -> None:
    # Every edge goes up at least one generation, so no path can loop back: no cycles.
    parent_generation = dag.generation[dag.parents]
    child_generation = dag.generation[dag.children]
    assert (parent_generation < child_generation).all(), "edge does not go up a generation"

    # Every non-root has a parent exactly one generation lower. With the check above, the
    # planned generation is then the longest path from a root, as Kahn's algorithm computes.
    has_level_fixing_parent = np.zeros(dag.num_nodes, dtype=bool)
    has_level_fixing_parent[dag.children[parent_generation == child_generation - 1]] = True
    assert has_level_fixing_parent[dag.generation > 0].all(), "non-root without level-fixing parent"
