"""A worker: claims ready nodes, runs them and completes them, until every node is done.

Workers share the MongoDB collection and nothing else, so they can run as threads, processes
or containers on as many machines as needed. Like the workers in `evaluation/simulation.py`,
a worker runs one node at a time. It also keeps track of the time it spends on MongoDB calls,
the coordination cost the simulation leaves out.
"""

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from pymongo.errors import ConnectionFailure

from dag_experiments.distributed import store
from dag_experiments.distributed.store import Claim, Nodes

LEASE_SECONDS = 60.0
# While no node is ready, a worker asks again after a wait that doubles from the first to the
# last: idle workers pick up new work quickly at first, and don't flood MongoDB after that.
FIRST_IDLE_WAIT = 0.01
LAST_IDLE_WAIT = 1.0
# However many workers there are, while idle they ask for work about this many times a second
# together, so that thousands of them waiting in the tail of a graph don't keep MongoDB busy.
# With thousands asking, a node that becomes ready still gets picked up within milliseconds.
IDLE_ASKS_PER_SECOND = 1_000
# A call that fails on the network is tried this many times, this many seconds apart.
ATTEMPTS = 5
RETRY_WAIT = 1.0


@dataclass
class WorkerStats:
    worker: str
    completed: int = 0
    claim_seconds: float = 0.0  # in the claims that got a node
    complete_seconds: float = 0.0
    idle_seconds: float = 0.0  # asking and waiting while no node was ready


def simulate(claim: Claim) -> None:
    """Run a node by waiting out its runtime. A real worker would run the node's task."""
    time.sleep(claim.runtime)


def work(
    nodes: Nodes,
    worker: str,
    run: Callable[[Claim], None] = simulate,
    lease_seconds: float = LEASE_SECONDS,
    workers: int = 1,
) -> WorkerStats:
    """Run nodes until every node is done; `workers` counts every worker sharing the DAG."""
    stats = WorkerStats(worker)
    lease = _Lease(nodes, worker, lease_seconds)
    # The more workers, the longer each idle one waits between asks.
    first_idle_wait = max(FIRST_IDLE_WAIT, workers / IDLE_ASKS_PER_SECOND)
    last_idle_wait = max(LAST_IDLE_WAIT, first_idle_wait)
    idle_wait = first_idle_wait
    expired_checked = time.monotonic()
    try:
        while True:
            started = time.perf_counter()
            claim = _retried(lambda: store.claim(nodes, worker, lease_seconds))
            if claim is None:
                # Nothing is ready: either every node is done, or the next ones wait for
                # dependencies running elsewhere. Once per lease, also hand back the nodes
                # whose worker stopped renewing their lease.
                if _retried(lambda: store.finished(nodes)):
                    return stats
                if time.monotonic() - expired_checked > lease_seconds:
                    _retried(lambda: store.release_expired(nodes))
                    expired_checked = time.monotonic()
                # The jitter keeps idle workers from asking all at the same moment.
                time.sleep(idle_wait * random.uniform(0.5, 1.5))
                idle_wait = min(2 * idle_wait, last_idle_wait)
                stats.idle_seconds += time.perf_counter() - started
                continue
            stats.claim_seconds += time.perf_counter() - started
            idle_wait = first_idle_wait

            lease.node = claim.node
            run(claim)
            lease.node = None

            started = time.perf_counter()
            _retried(lambda: store.complete(nodes, claim))
            stats.complete_seconds += time.perf_counter() - started
            stats.completed += 1
    finally:
        lease.stop()


def _retried[T](call: Callable[[], T]) -> T:
    # Every call in `store` is safe to repeat. A claim whose reply got lost leaves its node
    # running with no worker on it, until the lease runs out and hands it back.
    for _ in range(ATTEMPTS - 1):
        try:
            return call()
        except ConnectionFailure:
            time.sleep(RETRY_WAIT)
    return call()


class _Lease:
    """Renews the lease on the node a worker is running, from a background thread.

    A node can run for longer than the lease, so renewing between nodes is not enough.
    """

    def __init__(self, nodes: Nodes, worker: str, seconds: float) -> None:
        self.node: int | None = None
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._renew, args=(nodes, worker, seconds), daemon=True)
        self._thread.start()

    def _renew(self, nodes: Nodes, worker: str, seconds: float) -> None:
        # Three renewals per lease leave room for one that is slow or fails.
        while not self._stopped.wait(seconds / 3):
            if (node := self.node) is not None:
                # A renewal that fails on the network gets another go at the next one.
                try:
                    store.renew(nodes, node, worker, seconds)
                except ConnectionFailure:
                    pass

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join()
