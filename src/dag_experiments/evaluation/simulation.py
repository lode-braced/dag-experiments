"""Simulated execution of a DAG by a fixed number of workers.

Every node gets a runtime up front. A worker runs one node at a time, and handing out work
costs nothing, so the simulation measures only how well each approach keeps the workers busy.
It returns the makespan: the simulated time until the last node finishes.
"""

import heapq
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dag_experiments.evaluation.plans import Edge, complete, dependency_dict, generational_plan


def poisson_runtimes(num_nodes: int, mean: float, seed: int) -> list[int]:
    """A runtime per node in seconds, Poisson distributed around `mean`."""
    return np.random.default_rng(seed).poisson(mean, num_nodes).tolist()


def gamma_runtimes(num_nodes: int, mean: float, sd: float, seed: int) -> list[float]:
    """A runtime per node in seconds, gamma distributed with the given standard deviation.

    Unlike Poisson, the spread is set apart from the mean. A standard deviation of 0 gives
    every node exactly the mean.
    """
    if sd == 0:
        return [mean] * num_nodes
    # A gamma distribution with shape k and scale θ has mean kθ and variance kθ².
    shape, scale = (mean / sd) ** 2, sd**2 / mean
    return np.random.default_rng(seed).gamma(shape, scale, num_nodes).tolist()


def run_generational(plan: list[list[int]], runtimes: Sequence[float], workers: int) -> float:
    """Run the plan batch by batch: a generation starts once the previous one has finished."""
    time = 0.0
    for generation in plan:
        # Each node goes to the worker that is free first. `free_at` holds, per worker, when
        # it finishes the nodes it took so far.
        free_at = [time] * min(workers, len(generation))
        for node in generation:
            start = heapq.heappop(free_at)
            heapq.heappush(free_at, start + runtimes[node])
        # The next generation waits for the last node of this one.
        time = max(free_at)
    return time


def run_asap(edges: list[Edge], num_nodes: int, runtimes: Sequence[float], workers: int) -> float:
    """Run every node as soon as its dependencies have completed and a worker is free."""
    links = dependency_dict(edges, num_nodes)
    ready = deque(node for node in range(num_nodes) if not links[node].dependencies)
    running: list[tuple[float, int]] = []  # (finish time, node), the earliest first
    time = 0.0
    while ready or running:
        # Free workers take the ready nodes that have waited longest.
        while ready and len(running) < workers:
            node = ready.popleft()
            heapq.heappush(running, (time + runtimes[node], node))
        # Jump to the next node to finish; of nodes finishing at the same time, the lowest id
        # goes first. Its completion can make dependents ready, and frees a worker for the
        # next pass of the loop.
        time, node = heapq.heappop(running)
        ready.extend(complete(links, node))
    return time


@dataclass(frozen=True)
class Makespans:
    workers: int
    generational: float
    asap: float


def worker_sweep(
    edges: list[Edge], num_nodes: int, runtimes: Sequence[float], saturation: float = 0.01
) -> tuple[list[Makespans], Makespans]:
    """Makespans for 10, 20, 50, 100, ... workers, up to where as soon as possible saturates.

    Also returns the makespans with a worker for every node, the limit of each approach.
    Saturated means within `saturation` of that limit: more workers barely help.
    """
    plan = generational_plan(edges, num_nodes)
    unlimited = Makespans(
        num_nodes,
        run_generational(plan, runtimes, num_nodes),
        run_asap(edges, num_nodes, runtimes, num_nodes),
    )
    sweep = []
    for workers in _worker_counts(num_nodes):
        point = Makespans(
            workers,
            run_generational(plan, runtimes, workers),
            run_asap(edges, num_nodes, runtimes, workers),
        )
        sweep.append(point)
        if point.asap <= unlimited.asap * (1 + saturation):
            break
    return sweep, unlimited


def _worker_counts(num_nodes: int) -> list[int]:
    """10, 20, 50, 100, 200, 500, ... and finally one worker per node."""
    counts = []
    scale = 10
    while scale < num_nodes:
        counts += [count for count in (scale, 2 * scale, 5 * scale) if count < num_nodes]
        scale *= 10
    # Ending at one worker per node guarantees the sweep reaches saturation.
    return [*counts, num_nodes]
