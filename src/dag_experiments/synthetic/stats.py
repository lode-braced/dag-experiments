"""Statistics of a synthetic DAG, to check it against the shape the generator aims for."""

from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import breadth_first_order, connected_components

from dag_experiments.synthetic.generator import IntArray, SyntheticDag


def dependents_graph(dag: SyntheticDag) -> csr_array:
    """Sparse adjacency matrix with an entry at [parent, child] for every edge."""
    weights = np.ones(dag.num_edges)
    return csr_array((weights, (dag.parents, dag.children)), shape=(dag.num_nodes, dag.num_nodes))


def descendant_counts(graph: csr_array, nodes: Sequence[int] | IntArray) -> IntArray:
    """Number of nodes downstream of each node: what a change to it can affect."""
    return np.array(
        [len(breadth_first_order(graph, node, return_predecessors=False)) - 1 for node in nodes]
    )


def redundant_edge_fraction(dag: SyntheticDag, edges: Sequence[int] | IntArray) -> float:
    """Share of `edges` (edge indices) implied by a longer path, like A→C next to A→B→C."""
    graph = dependents_graph(dag)
    parents_graph = graph.T.tocsr()
    redundant = 0
    for edge in edges:
        parent, child = dag.parents[edge], dag.children[edge]
        # Every edge climbs at least one generation, so a longer path climbs at least two.
        if dag.generation[child] - dag.generation[parent] < 2:
            continue
        other_parents = parents_graph.indices[
            parents_graph.indptr[child] : parents_graph.indptr[child + 1]
        ]
        other_parents = other_parents[other_parents != parent]
        downstream = breadth_first_order(graph, parent, return_predecessors=False)
        redundant += bool(np.isin(other_parents, downstream).any())
    return redundant / len(edges)


def report(
    dag: SyntheticDag, seed: int = 0, spread_samples: int = 1_000, redundancy_samples: int = 2_000
) -> str:
    rng = np.random.default_rng(seed)
    graph = dependents_graph(dag)
    dependencies = np.bincount(dag.children, minlength=dag.num_nodes)
    dependents = np.bincount(dag.parents, minlength=dag.num_nodes)
    per_generation = np.bincount(dag.generation)
    span = dag.generation[dag.children] - dag.generation[dag.parents]
    _, component = connected_components(graph, directed=True, connection="weak")

    spread = descendant_counts(graph, rng.choice(dag.num_nodes, spread_samples, replace=False))
    top_hubs = np.argsort(dependents)[-5:][::-1]
    hub_reach = descendant_counts(graph, top_hubs) / dag.num_nodes
    sampled_edges = rng.choice(dag.num_edges, min(redundancy_samples, dag.num_edges), replace=False)

    non_root_dependencies = dependencies[dependencies > 0]
    p = np.percentile
    lines = {
        "Size": f"{dag.num_nodes:,} nodes, {dag.num_edges:,} edges "
        f"({dag.num_edges / dag.num_nodes:.2f} per node)",
        "Dependencies": f"per non-root: mean {non_root_dependencies.mean():.2f}, "
        f"p50 {p(non_root_dependencies, 50):.0f}, p99 {p(non_root_dependencies, 99):.0f}, "
        f"max {non_root_dependencies.max():,}",
        "Dependents": f"per node: p50 {p(dependents, 50):.0f}, p99 {p(dependents, 99):.0f}, "
        f"max {dependents.max():,}",
        "Roots and sinks": f"roots {(dependencies == 0).mean():.1%}, "
        f"sinks {(dependents == 0).mean():.1%}",
        "Generations": f"{len(per_generation)}; generation 0 has {per_generation[0]:,} nodes, "
        f"the median generation {np.median(per_generation):,.0f}, "
        f"the last three {', '.join(str(n) for n in per_generation[-3:])}",
        "Edge reach": f"{(span == 1).mean():.0%} to the previous generation, "
        f"{(span > 5).mean():.0%} skip more than 5",
        "Locality": f"{(dag.pipeline[dag.parents] == dag.pipeline[dag.children]).mean():.0%} "
        f"of edges inside a pipeline; the largest weakly connected component holds "
        f"{np.bincount(component).max() / dag.num_nodes:.1%} of nodes",
        "Downstream spread": f"random node p50 {p(spread, 50):,.0f}, p90 {p(spread, 90):,.0f}, "
        f"p99 {p(spread, 99):,.0f}; top 5 hubs reach "
        f"{', '.join(f'{reach:.0%}' for reach in hub_reach)} of the graph",
        "Redundant edges": f"{redundant_edge_fraction(dag, sampled_edges):.0%} "
        f"(of {len(sampled_edges):,} sampled edges)",
    }
    width = max(len(name) for name in lines)
    return "\n".join(f"{name:<{width}}  {line}" for name, line in lines.items())
