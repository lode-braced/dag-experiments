import numpy as np

from dag_experiments.synthetic.generator import SyntheticDag
from dag_experiments.synthetic.stats import (
    descendant_counts,
    dependents_graph,
    redundant_edge_fraction,
)

# 0 → 1 → 2 → 3, plus the shortcut 0 → 2 and a separate branch 0 → 4
DAG = SyntheticDag(
    parents=np.array([0, 1, 2, 0, 0], dtype=np.int32),
    children=np.array([1, 2, 3, 2, 4], dtype=np.int32),
    generation=np.array([0, 1, 2, 3, 1], dtype=np.int32),
    pipeline=np.zeros(5, dtype=np.int32),
    pipelines=[],
)


def test_descendant_counts() -> None:
    assert descendant_counts(dependents_graph(DAG), [0, 1, 2, 3, 4]).tolist() == [4, 2, 1, 0, 0]


def test_only_the_shortcut_edge_is_redundant() -> None:
    assert redundant_edge_fraction(DAG, range(DAG.num_edges)) == 1 / 5
    assert redundant_edge_fraction(DAG, [3]) == 1
