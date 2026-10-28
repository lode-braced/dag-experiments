"""Time and memory of preparing a DAG for execution, both ways."""

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass

from dag_experiments.evaluation.plans import Edge, complete, dependency_dict


@dataclass(frozen=True)
class Cost:
    seconds: float
    peak_bytes: int  # the most memory the step had allocated at once, on top of its input
    kept_bytes: int  # what its result holds once the input is gone


def measure(make_edges: Callable[[], list[Edge]], step: Callable[[list[Edge]], object]) -> Cost:
    # Time the step on its own. Holding on to the result keeps freeing it out of the time.
    edges = make_edges()
    started = time.perf_counter()
    result = step(edges)
    seconds = time.perf_counter() - started
    del result, edges

    # Tracing allocations slows Python down, so memory comes from a second run. The input
    # is traced too: once it is dropped, what remains is everything the result holds,
    # including the node ids it shares with the input.
    tracemalloc.start()
    edges = make_edges()
    input_bytes, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    result = step(edges)
    peak_bytes = tracemalloc.get_traced_memory()[1] - input_bytes
    del edges
    kept_bytes, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return Cost(seconds, peak_bytes, kept_bytes)


def bookkeeping_seconds(edges: list[Edge], num_nodes: int, plan: list[list[int]]) -> float:
    """Time spent popping completed nodes from the dependency dict over a whole run.

    A generational plan has done this work ahead; evaluating at runtime does it as nodes
    complete. Completing the nodes generation by generation is one valid order.
    """
    links = dependency_dict(edges, num_nodes)
    started = time.perf_counter()
    for generation in plan:
        for node in generation:
            complete(links, node)
    return time.perf_counter() - started
