"""Two ways to prepare a DAG for execution.

- Compiled ahead: a generational plan. Kahn's algorithm splits the nodes into generations,
  batches that can each run once every earlier batch has finished.
- Evaluated at runtime: a dependency dict. Every node keeps its dependents and the
  dependencies that have not completed yet. When a node completes, it is popped from its
  dependents' dependencies, and a dependent whose last dependency was popped can run.
"""

from dataclasses import dataclass, field

type Edge = tuple[int, int]  # (parent, child): the child depends on the parent


def generational_plan(edges: list[Edge], num_nodes: int) -> list[list[int]]:
    """The nodes per generation: the length of their longest path from a root."""
    # List every node's dependents and count its dependencies.
    dependents: list[list[int]] = [[] for _ in range(num_nodes)]
    unmet = [0] * num_nodes
    for parent, child in edges:
        dependents[parent].append(child)
        unmet[child] += 1

    # The roots form the first generation. Each next generation holds the nodes whose last
    # unmet dependency was in the generation before it.
    plan = []
    generation = [node for node in range(num_nodes) if unmet[node] == 0]
    while generation:
        plan.append(generation)
        next_generation = []
        for node in generation:
            for child in dependents[node]:
                unmet[child] -= 1
                if unmet[child] == 0:
                    next_generation.append(child)
        generation = next_generation
    return plan


@dataclass(slots=True)
class Links:
    dependents: list[int] = field(default_factory=list)
    dependencies: set[int] = field(default_factory=set)  # the ones not completed yet


def dependency_dict(edges: list[Edge], num_nodes: int) -> dict[int, Links]:
    """Every node's dependents and dependencies, for evaluating the DAG as it runs."""
    links = {node: Links() for node in range(num_nodes)}
    for parent, child in edges:
        links[parent].dependents.append(child)
        links[child].dependencies.add(parent)
    return links


def complete(links: dict[int, Links], node: int) -> list[int]:
    """Mark `node` as completed and return the dependents that can run now."""
    # Pop the node from each dependent's dependencies. A set makes popping the same
    # dependency twice fail loudly instead of releasing a dependent too early.
    ready = []
    for dependent in links[node].dependents:
        dependencies = links[dependent].dependencies
        dependencies.remove(node)
        if not dependencies:
            ready.append(dependent)
    return ready
