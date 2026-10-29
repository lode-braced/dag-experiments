import os
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from pymongo import MongoClient
from pymongo.errors import AutoReconnect, ServerSelectionTimeoutError
from pytest_mock import MockerFixture

from dag_experiments.distributed import store, worker
from dag_experiments.distributed.store import Claim, Nodes
from dag_experiments.distributed.worker import work
from dag_experiments.synthetic.generator import BASELINE, generate

# 0 and 1 are roots, 2 depends on 0, and 3 depends on 1 and 2.
EDGES = [(0, 2), (1, 3), (2, 3)]


@pytest.fixture
def nodes() -> Iterator[Nodes]:
    client = MongoClient(os.environ.get("MONGO_URL", "mongodb://localhost:27017"), serverSelectionTimeoutMS=1000)
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError:
        pytest.skip("needs a MongoDB: docker compose up -d mongo")
    database = client[f"test_{uuid.uuid4().hex}"]
    yield database.nodes
    client.drop_database(database.name)
    client.close()


def states(nodes: Nodes) -> dict[int, str]:
    return {node["_id"]: node["state"] for node in nodes.find()}


def claimed(nodes: Nodes, worker: str, lease_seconds: float = 60) -> Claim:
    claim = store.claim(nodes, worker, lease_seconds)
    assert claim is not None
    return claim


def test_a_dependent_becomes_ready_with_its_last_dependency(nodes: Nodes) -> None:
    store.load(nodes, EDGES, 4, [0] * 4)
    zero, one = sorted([claimed(nodes, "a"), claimed(nodes, "b")], key=lambda claim: claim.node)
    assert store.claim(nodes, "c", 60) is None

    store.complete(nodes, one)
    assert states(nodes) == {0: "running", 1: "done", 2: "waiting", 3: "waiting"}
    store.complete(nodes, zero)
    assert states(nodes) == {0: "done", 1: "done", 2: "ready", 3: "waiting"}
    two = claimed(nodes, "a")
    assert two.node == 2
    store.complete(nodes, two)
    assert states(nodes)[3] == "ready"
    assert not store.finished(nodes)


def test_completing_a_node_again_changes_nothing(nodes: Nodes) -> None:
    # A worker whose lease ran out may complete a node that another worker completed too.
    store.load(nodes, [(0, 1), (0, 2), (3, 2)], 4, [0] * 4)
    zero = Claim(0, 0, [1, 2])
    store.complete(nodes, zero)
    before = list(nodes.find())

    store.complete(nodes, zero)

    assert list(nodes.find()) == before
    assert states(nodes)[2] == "waiting"


def test_only_the_claiming_worker_renews_a_lease(nodes: Nodes) -> None:
    store.load(nodes, [], 2, [0] * 2)
    first, second = claimed(nodes, "a", lease_seconds=0), claimed(nodes, "b", lease_seconds=0)
    store.renew(nodes, first.node, "a", 60)
    store.renew(nodes, second.node, "a", 60)
    time.sleep(0.01)

    assert store.release_expired(nodes) == 1
    assert states(nodes) == {first.node: "running", second.node: "ready"}


def test_another_worker_takes_over_a_node_whose_lease_ran_out(nodes: Nodes) -> None:
    store.load(nodes, [(0, 1)], 2, [0] * 2)
    stalled = claimed(nodes, "a", lease_seconds=0)
    time.sleep(0.01)

    assert store.release_expired(nodes) == 1
    assert claimed(nodes, "b") == stalled
    # The stalled worker may still finish; its completion and the second one count once.
    store.complete(nodes, stalled)
    store.complete(nodes, stalled)
    assert states(nodes) == {0: "done", 1: "ready"}
    assert nodes.find_one({"_id": 0}, {"attempts": True}) == {"_id": 0, "attempts": 2}


def test_workers_run_every_node_once_after_its_dependencies(nodes: Nodes) -> None:
    dag = generate(replace(BASELINE, nodes=2_000), seed=2)
    edges = list(zip(dag.parents.tolist(), dag.children.tolist()))
    store.load(nodes, edges, dag.num_nodes, [0.001] * dag.num_nodes)
    runs: dict[int, list[tuple[float, float]]] = {node: [] for node in range(dag.num_nodes)}
    lock = threading.Lock()

    def record(claim: Claim) -> None:
        started = time.perf_counter()
        time.sleep(claim.runtime)
        with lock:
            runs[claim.node].append((started, time.perf_counter()))

    with ThreadPoolExecutor(8) as pool:
        stats = list(pool.map(lambda worker: work(nodes, str(worker), record), range(8)))

    assert store.finished(nodes)
    assert all(len(node_runs) == 1 for node_runs in runs.values())
    assert sum(worker.completed for worker in stats) == dag.num_nodes
    assert all(runs[parent][0][1] <= runs[child][0][0] for parent, child in edges)


def test_a_worker_carries_on_after_a_dropped_connection(nodes: Nodes, mocker: MockerFixture) -> None:
    store.load(nodes, EDGES, 4, [0] * 4)
    mocker.patch.object(worker, "RETRY_WAIT", 0)
    complete = store.complete
    dropped = []

    def drop_the_first_call(nodes: Nodes, claim: Claim) -> None:
        if not dropped:
            dropped.append(claim.node)
            raise AutoReconnect("connection reset by peer")
        complete(nodes, claim)

    mocker.patch.object(store, "complete", side_effect=drop_the_first_call)

    assert work(nodes, "a").completed == 4
    assert store.finished(nodes)
    assert len(dropped) == 1


@pytest.mark.parametrize("draw", [0.0, 1.0])
def test_a_claim_finds_the_only_ready_node_whatever_key_it_draws(
    nodes: Nodes, mocker: MockerFixture, draw: float
) -> None:
    store.load(nodes, [], 1, [0])
    mocker.patch.object(store.random, "random", return_value=draw)

    assert claimed(nodes, "a").node == 0
