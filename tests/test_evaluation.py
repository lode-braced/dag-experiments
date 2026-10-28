from dataclasses import replace

import pytest

from dag_experiments.evaluation.plans import complete, dependency_dict, generational_plan
from dag_experiments.evaluation.simulation import (
    poisson_runtimes,
    run_asap,
    run_generational,
    worker_sweep,
)
from dag_experiments.synthetic.generator import BASELINE, generate

# 0 and 1 are roots, 2 depends on 0, and 3 depends on 1 and 2.
EDGES = [(0, 2), (1, 3), (2, 3)]
RUNTIMES = [1, 5, 1, 1]


def test_generational_plan_matches_the_planned_generations() -> None:
    dag = generate(replace(BASELINE, nodes=10_000), seed=2)

    plan = generational_plan(list(zip(dag.parents.tolist(), dag.children.tolist())), dag.num_nodes)

    planned = dag.generation.tolist()
    assert len(plan) == max(planned) + 1
    for generation, nodes in enumerate(plan):
        assert sorted(nodes) == [node for node in range(dag.num_nodes) if planned[node] == generation]


def test_a_dependent_is_released_by_its_last_dependency() -> None:
    links = dependency_dict(EDGES, 4)

    assert complete(links, 1) == []
    assert complete(links, 0) == [2]
    assert complete(links, 2) == [3]
    with pytest.raises(KeyError):
        complete(links, 2)


def test_one_worker_runs_everything_in_sequence() -> None:
    plan = generational_plan(EDGES, 4)

    assert run_generational(plan, RUNTIMES, workers=1) == sum(RUNTIMES)
    assert run_asap(EDGES, 4, RUNTIMES, workers=1) == sum(RUNTIMES)


def test_each_node_goes_to_the_first_free_worker() -> None:
    # The first worker is busy with node 0 while the second runs 1 and 2, then takes 3 too.
    runtimes = [5, 1, 1, 5]

    assert run_generational([[0, 1, 2, 3]], runtimes, workers=2) == 7
    assert run_asap([], 4, runtimes, workers=2) == 7


def test_asap_runs_ready_nodes_in_the_order_they_became_ready() -> None:
    # Root 2 was waiting before completing 0 made node 3 ready, so 2 goes first and the
    # chain 0 -> 3 -> 4 is held up behind it.
    edges = [(0, 3), (3, 4)]
    runtimes = [1, 5, 5, 1, 5]

    assert run_asap(edges, 5, runtimes, workers=2) == 11


def test_unlimited_workers_wait_only_for_dependencies_or_for_the_generation() -> None:
    plan = generational_plan(EDGES, 4)

    # Node 2 waits for the slow root 1 to end its generation, which asap skips.
    assert run_generational(plan, RUNTIMES, workers=4) == 5 + 1 + 1
    assert run_asap(EDGES, 4, RUNTIMES, workers=4) == 5 + 1


def test_unlimited_asap_takes_the_longest_path() -> None:
    dag = generate(replace(BASELINE, nodes=10_000), seed=2)
    edges = list(zip(dag.parents.tolist(), dag.children.tolist()))
    runtimes = poisson_runtimes(dag.num_nodes, mean=10, seed=2)

    # The longest path, by runtime, ending at each node, filled in generation order.
    parents: list[list[int]] = [[] for _ in range(dag.num_nodes)]
    for parent, child in edges:
        parents[child].append(parent)
    finish = [0] * dag.num_nodes
    plan = generational_plan(edges, dag.num_nodes)
    for nodes in plan:
        for node in nodes:
            finish[node] = runtimes[node] + max((finish[parent] for parent in parents[node]), default=0)

    assert run_asap(edges, dag.num_nodes, runtimes, workers=dag.num_nodes) == max(finish)
    assert run_generational(plan, runtimes, workers=dag.num_nodes) == sum(
        max(runtimes[node] for node in nodes) for nodes in plan
    )


def test_the_worker_sweep_stops_once_asap_saturates() -> None:
    dag = generate(replace(BASELINE, nodes=10_000), seed=2)
    edges = list(zip(dag.parents.tolist(), dag.children.tolist()))
    runtimes = poisson_runtimes(dag.num_nodes, mean=10, seed=2)

    sweep, unlimited = worker_sweep(edges, dag.num_nodes, runtimes)

    assert sweep[0].workers == 10
    assert sweep[-1].asap <= unlimited.asap * 1.01
    assert all(point.asap > unlimited.asap * 1.01 for point in sweep[:-1])


def test_the_worker_sweep_reaches_saturation_on_a_small_graph() -> None:
    # 10 workers leave one of 11 independent nodes for a second round; 11 workers don't.
    sweep, unlimited = worker_sweep([], 11, [1] * 11)

    assert [point.workers for point in sweep] == [10, 11]
    assert sweep[-1].asap == unlimited.asap == 1
