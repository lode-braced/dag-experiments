from dataclasses import replace

import numpy as np
import pytest

from dag_experiments.synthetic.generator import BASELINE, SyntheticDag, generate


def kahn_generations(dag: SyntheticDag) -> list[int]:
    dependents: list[list[int]] = [[] for _ in range(dag.num_nodes)]
    unmet = [0] * dag.num_nodes
    for parent, child in zip(dag.parents.tolist(), dag.children.tolist()):
        dependents[parent].append(child)
        unmet[child] += 1

    generation = [0] * dag.num_nodes
    frontier = [node for node in range(dag.num_nodes) if unmet[node] == 0]
    visited = 0
    current = 0
    while frontier:
        next_frontier = []
        for node in frontier:
            generation[node] = current
            for child in dependents[node]:
                unmet[child] -= 1
                if unmet[child] == 0:
                    next_frontier.append(child)
        visited += len(frontier)
        frontier = next_frontier
        current += 1
    assert visited == dag.num_nodes, "graph has a cycle"
    return generation


@pytest.mark.parametrize("nodes", [10_000, 50_000])
def test_planned_generation_matches_kahn(nodes: int) -> None:
    dag = generate(replace(BASELINE, nodes=nodes), seed=1)

    assert dag.generation.tolist() == kahn_generations(dag)


def test_same_seed_gives_same_graph() -> None:
    params = replace(BASELINE, nodes=10_000)
    first, second = generate(params, seed=7), generate(params, seed=7)

    assert np.array_equal(first.parents, second.parents)
    assert np.array_equal(first.children, second.children)
    assert np.array_equal(first.generation, second.generation)
    assert not np.array_equal(first.parents, generate(params, seed=8).parents)


def test_edges_are_unique_and_ids_and_edge_order_are_shuffled() -> None:
    dag = generate(replace(BASELINE, nodes=50_000))

    edges = dag.parents.astype(np.int64) * dag.num_nodes + dag.children
    assert len(np.unique(edges)) == dag.num_edges
    # Unshuffled, most parents have lower ids than their children and edges come sorted.
    assert 0.48 < (dag.parents < dag.children).mean() < 0.52
    assert 0.48 < (np.diff(dag.parents) > 0).mean() < 0.52


def test_every_input_is_read_by_its_own_pipeline() -> None:
    dag = generate(replace(BASELINE, nodes=10_000), seed=1)

    own_edge = dag.pipeline[dag.parents] == dag.pipeline[dag.children]
    readers_in_own_pipeline = np.bincount(dag.parents[own_edge], minlength=dag.num_nodes)
    # In a single-level source pipeline, the inputs are also the pipeline's outputs.
    single_level = np.array([pipeline.depth == 1 for pipeline in dag.pipelines])
    inputs = (dag.generation == 0) & ~single_level[dag.pipeline]
    assert (readers_in_own_pipeline[inputs] > 0).all()


def test_pipeline_sizes_scale_with_the_square_root_of_the_node_count() -> None:
    dag = generate(replace(BASELINE, nodes=BASELINE.reference_nodes // 4))

    sizes = [pipeline.size for pipeline in dag.pipelines]
    # Half the reference median of 150 and max of 5,000
    assert 65 < np.median(sizes) < 85
    # The last pipeline can take a few leftover nodes beyond the max.
    assert max(sizes) < 2_500 + BASELINE.pipeline_size_min
