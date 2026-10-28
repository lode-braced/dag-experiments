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
    pipelines = _plan_pipelines(params, rng)
    layout = _NodeLayout.build(pipelines)
    popularity = rng.pareto(params.popularity_alpha, layout.num_nodes) + 1

    children = np.flatnonzero(layout.generation > 0)
    fixing_parents = _level_fixing_parents(children, layout, popularity, rng)
    extra_children, extra_parents = _extra_parents(children, layout, popularity, params, rng)

    parents = np.concatenate([fixing_parents, extra_parents])
    children = np.concatenate([children, extra_children])
    parents, children = _deduplicate(parents, children, layout.num_nodes)

    dag = _shuffle(parents, children, layout, pipelines, rng)
    _check_guarantees(dag)
    return dag


# --- Planning pipelines and generations ---


def _plan_pipelines(params: GraphParams, rng: np.random.Generator) -> list[Pipeline]:
    sizes = _pipeline_sizes(params, rng)
    # Cumulative sqrt(size) weights of earlier pipelines, so a pipeline picks its
    # upstreams among the pipelines created before it with one searchsorted.
    cumulative_weight = np.zeros(len(sizes) + 1)
    pipelines: list[Pipeline] = []

    for index, size in enumerate(sizes):
        depth = int(np.clip(round(typical_depth(size) + rng.normal()), 1, size - 1))
        is_source = index == 0 or rng.random() < params.source_pipeline_fraction
        if is_source:
            start, upstreams, extra_roots = 0, [], 0
            depth = min(depth, params.max_generations)
        else:
            upstreams = _pick_upstreams(cumulative_weight[: index + 1], params, rng)
            main = pipelines[upstreams[0]]
            start = main.start + 1 + int(rng.integers(main.depth))
            if start + depth > params.max_generations:
                start = max(main.start + 1, params.max_generations - depth)
                depth = min(depth, params.max_generations - start)
            extra_roots = min(rng.binomial(size, params.extra_root_fraction), size - depth)
        level_sizes = _level_sizes(size - extra_roots, depth, is_source, params, rng)
        pipelines.append(Pipeline(start, level_sizes, extra_roots, upstreams))

        # A pipeline starting at the last allowed generation leaves no room to consume it.
        can_be_consumed = start < params.max_generations - 1
        cumulative_weight[index + 1] = cumulative_weight[index] + can_be_consumed * np.sqrt(size)

    return pipelines


def _pipeline_sizes(params: GraphParams, rng: np.random.Generator) -> list[int]:
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


def _pick_upstreams(
    cumulative_weight: FloatArray, params: GraphParams, rng: np.random.Generator
) -> list[int]:
    count = 1 + rng.poisson(params.extra_upstreams_mean)
    targets = rng.random(count) * cumulative_weight[-1]
    picked = np.searchsorted(cumulative_weight, targets, side="right") - 1
    # Duplicate draws are dropped rather than redrawn; the first draw is the main upstream.
    return list(dict.fromkeys(picked.tolist()))


def typical_depth(size: float) -> float:
    """Number of levels in a pipeline of `size` nodes, before random variation."""
    return 1.5 * math.log2(size)


def level_shares(depth: int, is_source: bool, params: GraphParams) -> FloatArray:
    """Share of a pipeline's nodes at each local level, on top of one node per level."""
    level = np.arange(depth)
    theta = max(1.0, depth / 3)
    weights = (level + 1) * np.exp(-level / theta)
    if is_source:
        weights[0] *= params.source_level0_boost
    return weights / weights.sum()


def _level_sizes(
    nodes: int, depth: int, is_source: bool, params: GraphParams, rng: np.random.Generator
) -> IntArray:
    # One node per level is required: it is the level-fixing parent of the next level.
    return 1 + rng.multinomial(nodes - depth, level_shares(depth, is_source, params))


@dataclass(frozen=True)
class _NodeLayout:
    """Nodes numbered so each (pipeline, generation) group is a contiguous range.

    A pipeline's level groups are consecutive, so the group of pipeline `p` at
    generation `g` is `level_group[p] + g - start[p]`.
    """

    generation: IntArray  # per node
    pipeline: IntArray  # per node
    group_start: IntArray  # first node of each group; one extra entry marks the end
    start: IntArray  # per pipeline from here on
    depth: IntArray
    is_source: npt.NDArray[np.bool_]
    main_upstream: IntArray  # -1 for source pipelines
    upstream_offsets: IntArray  # upstreams of p: upstreams[upstream_offsets[p]:upstream_offsets[p+1]]
    upstreams: IntArray
    level_group: IntArray
    roots_group: IntArray  # extra roots, or level 0 for source pipelines; -1 if none

    @property
    def num_nodes(self) -> int:
        return len(self.generation)

    @classmethod
    def build(cls, pipelines: list[Pipeline]) -> "_NodeLayout":
        group_sizes, group_generation, group_pipeline = [], [], []
        level_group, roots_group = [], []
        for index, pipeline in enumerate(pipelines):
            if pipeline.extra_roots:
                roots_group.append(len(group_sizes))
                group_sizes.append(pipeline.extra_roots)
                group_generation.append(0)
                group_pipeline.append(index)
            else:
                roots_group.append(len(group_sizes) if pipeline.is_source else -1)
            level_group.append(len(group_sizes))
            group_sizes.extend(pipeline.level_sizes.tolist())
            group_generation.extend(range(pipeline.start, pipeline.start + pipeline.depth))
            group_pipeline.extend([index] * pipeline.depth)

        group_sizes = np.array(group_sizes)
        return cls(
            generation=np.repeat(group_generation, group_sizes),
            pipeline=np.repeat(group_pipeline, group_sizes),
            group_start=np.concatenate([[0], np.cumsum(group_sizes)]),
            start=np.array([p.start for p in pipelines]),
            depth=np.array([p.depth for p in pipelines]),
            is_source=np.array([p.is_source for p in pipelines]),
            main_upstream=np.array([p.upstreams[0] if p.upstreams else -1 for p in pipelines]),
            upstream_offsets=np.concatenate([[0], np.cumsum([len(p.upstreams) for p in pipelines])]),
            upstreams=np.array([u for p in pipelines for u in p.upstreams], dtype=np.int64),
            level_group=np.array(level_group),
            roots_group=np.array(roots_group),
        )

    def group_at(self, pipeline: IntArray, generation: IntArray) -> IntArray:
        return self.level_group[pipeline] + generation - self.start[pipeline]


# --- Edges ---

_SAME_PIPELINE, _UPSTREAM, _HUB = 0, 1, 2


def _level_fixing_parents(
    children: IntArray, layout: _NodeLayout, popularity: FloatArray, rng: np.random.Generator
) -> IntArray:
    pipeline = layout.pipeline[children]
    generation = layout.generation[children]
    # Nodes on a pipeline's first level take the parent from the main upstream.
    on_first_level = generation == layout.start[pipeline]
    host = np.where(on_first_level, layout.main_upstream[pipeline], pipeline)
    return _pick_in_groups(layout.group_at(host, generation - 1), layout, popularity, rng)


def _extra_parents(
    children: IntArray,
    layout: _NodeLayout,
    popularity: FloatArray,
    params: GraphParams,
    rng: np.random.Generator,
) -> tuple[IntArray, IntArray]:
    count = rng.geometric(1 / (1 + params.extra_parents_mean), len(children)) - 1
    wide_join = rng.random(len(children)) < params.wide_join_fraction
    log_parents = rng.uniform(
        np.log(params.wide_join_min_parents),
        np.log(params.wide_join_max_parents),
        wide_join.sum(),
    )
    count[wide_join] = np.round(np.exp(log_parents))

    children = np.repeat(children, count)
    n = len(children)
    pipeline = layout.pipeline[children]
    generation = layout.generation[children]
    shares = [params.same_pipeline_share, params.upstream_share, params.hub_share]
    kind = rng.choice(3, size=n, p=shares)
    group = np.full(n, -1)

    span = rng.geometric(params.span_p, n)
    target = generation - span
    in_pipeline = (kind == _SAME_PIPELINE) & (target >= layout.start[pipeline])
    group[in_pipeline] = layout.group_at(pipeline[in_pipeline], target[in_pipeline])

    # Reaching below the pipeline's first level: source pipelines read their raw
    # inputs (level 0); others use their extra roots half the time, else an upstream.
    below = (kind == _SAME_PIPELINE) & ~in_pipeline
    has_roots = layout.roots_group[pipeline] >= 0
    use_roots = below & (layout.is_source[pipeline] | (has_roots & (rng.random(n) < 0.5)))
    group[use_roots] = layout.roots_group[pipeline[use_roots]]
    kind[below & ~use_roots] = _UPSTREAM

    # Source pipelines have no upstream pipelines: they read their raw inputs instead.
    from_raw = (kind == _UPSTREAM) & layout.is_source[pipeline]
    group[from_raw] = layout.roots_group[pipeline[from_raw]]
    from_upstream = (kind == _UPSTREAM) & ~layout.is_source[pipeline]
    group[from_upstream] = _upstream_groups(
        pipeline[from_upstream], generation[from_upstream], layout, rng
    )

    hub = kind == _HUB
    parents = np.empty(n, dtype=np.int64)
    parents[~hub] = _pick_in_groups(group[~hub], layout, popularity, rng)
    parents[hub] = _pick_hubs(generation[hub], layout, popularity, params, rng)
    return children, parents


def _upstream_groups(
    pipeline: IntArray, generation: IntArray, layout: _NodeLayout, rng: np.random.Generator
) -> IntArray:
    """A random upstream of each pipeline, at a uniformly random level below `generation`."""
    first = layout.upstream_offsets[pipeline]
    count = layout.upstream_offsets[pipeline + 1] - first
    upstream = layout.upstreams[first + (rng.random(len(pipeline)) * count).astype(np.int64)]
    # A non-main upstream may start at or after `generation`; the main upstream never does.
    upstream = np.where(
        layout.start[upstream] < generation, upstream, layout.main_upstream[pipeline]
    )
    lowest = layout.start[upstream]
    highest = np.minimum(lowest + layout.depth[upstream] - 1, generation - 1)
    level_generation = lowest + (rng.random(len(pipeline)) * (highest - lowest + 1)).astype(np.int64)
    return layout.group_at(upstream, level_generation)


def _pick_in_groups(
    group: IntArray, layout: _NodeLayout, popularity: FloatArray, rng: np.random.Generator
) -> IntArray:
    """One node per entry of `group`, chosen with probability proportional to popularity."""
    assert (group >= 0).all()
    cumulative = np.concatenate([[0.0], np.cumsum(popularity)])
    return _pick_weighted(
        cumulative, layout.group_start[group], layout.group_start[group + 1], rng
    )


def _pick_hubs(
    generation: IntArray,
    layout: _NodeLayout,
    popularity: FloatArray,
    params: GraphParams,
    rng: np.random.Generator,
) -> IntArray:
    """Any node below `generation`, weighted by popularity × e^(−generation / decay)."""
    by_generation = np.argsort(layout.generation, kind="stable")
    sorted_generation = layout.generation[by_generation]
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
    targets = cumulative[low] + rng.random(len(low)) * (cumulative[high] - cumulative[low])
    picked = np.searchsorted(cumulative, targets, side="right") - 1
    # Float rounding can land a target on a range's edge.
    return np.clip(picked, low, high - 1)


# --- Finishing ---


def _deduplicate(parents: IntArray, children: IntArray, num_nodes: int) -> tuple[IntArray, IntArray]:
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
    parent_generation = dag.generation[dag.parents]
    child_generation = dag.generation[dag.children]
    assert (parent_generation < child_generation).all(), "edge does not go up a generation"

    has_level_fixing_parent = np.zeros(dag.num_nodes, dtype=bool)
    has_level_fixing_parent[dag.children[parent_generation == child_generation - 1]] = True
    assert has_level_fixing_parent[dag.generation > 0].all(), "non-root without level-fixing parent"
