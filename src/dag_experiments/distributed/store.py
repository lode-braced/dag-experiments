"""The MongoDB calls through which workers evaluate a DAG at runtime, with no scheduler.

The `nodes` collection is the dependency dict of `evaluation/plans.py`, shared by every
worker: a document per node with its dependents and the dependencies that have not completed
yet. A node goes through four states:

    waiting --its last dependency completes--> ready --a worker claims it--> running --> done

Each step is one atomic update, of a single document or of a node's dependents, so workers
need neither locks nor transactions. Times come from the server's clock ($$NOW), so workers
whose clocks disagree still agree on when a lease runs out.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pymongo import ASCENDING, DESCENDING
from pymongo.collection import Collection

from dag_experiments.evaluation.plans import Edge, dependency_dict

type Nodes = Collection[dict[str, Any]]

WAITING, READY, RUNNING, DONE = "waiting", "ready", "running", "done"
INSERT_BATCH = 10_000
# How many ready nodes a claim tries before it reports that none is left to take.
CLAIM_ATTEMPTS = 3


@dataclass(frozen=True)
class Claim:
    node: int
    runtime: float  # seconds; in a real DAG, the task to run
    dependents: list[int]


def load(nodes: Nodes, edges: list[Edge], num_nodes: int, runtimes: Sequence[float]) -> None:
    """Replace the collection's contents with a fresh DAG, its roots ready to run."""
    links = dependency_dict(edges, num_nodes)
    nodes.drop()
    for start in range(0, num_nodes, INSERT_BATCH):
        nodes.insert_many(
            [
                {
                    "_id": node,
                    "state": WAITING if links[node].dependencies else READY,
                    "dependencies": sorted(links[node].dependencies),
                    "dependents": links[node].dependents,
                    "runtime": runtimes[node],
                    "attempts": 0,
                    "random_key": random.random(),
                }
                for node in range(start, min(start + INSERT_BATCH, num_nodes))
            ],
            ordered=False,
        )
    # Claims find a random ready node through this index. Building it once the documents are
    # in is faster than keeping it up to date during the inserts.
    nodes.create_index([("state", ASCENDING), ("random_key", ASCENDING)])
    # Finding the leases that ran out goes through this one. It holds only the running nodes,
    # so the lookup reads just the expired ones instead of every running node.
    nodes.create_index([("lease_until", ASCENDING)], partialFilterExpression={"state": RUNNING})


def claim(nodes: Nodes, worker: str, lease_seconds: float) -> Claim | None:
    """Take a ready node picked at random, or None if no node is ready."""
    # Every claim going for the same node, such as the one that has waited longest, makes
    # thousands of workers race for one document. MongoDB retries each worker that loses on the
    # next node, so the CPU it spends grows with the number of workers waiting. Instead, a
    # worker looks up a random ready node, then claims that one by its _id. Random picks rarely
    # collide, and a worker that loses finds the node no longer ready and tries another.
    for _ in range(CLAIM_ATTEMPTS):
        node = _random_ready(nodes)
        if node is None:
            return None
        # find_one_and_update checks and changes the document in one atomic step, so of the
        # workers that picked the same node, only one gets it. The lease lets other workers take
        # the node over if this worker stops renewing it.
        document = nodes.find_one_and_update(
            {"_id": node, "state": READY},
            [{"$set": {
                "state": RUNNING,
                "worker": worker,
                "started_at": "$$NOW",
                "lease_until": _from_now(lease_seconds),
                "attempts": {"$add": ["$attempts", 1]},
            }}],
            projection={"runtime": True, "dependents": True},
        )
        if document is not None:
            return Claim(document["_id"], document["runtime"], document["dependents"])
    return None


def renew(nodes: Nodes, node: int, worker: str, lease_seconds: float) -> None:
    """Push back the end of the lease on a node this worker is running."""
    # Matching on the worker leaves the node alone if another worker has taken it over.
    nodes.update_one(
        {"_id": node, "state": RUNNING, "worker": worker},
        [{"$set": {"lease_until": _from_now(lease_seconds)}}],
    )


def complete(nodes: Nodes, claim: Claim) -> None:
    """Release the dependents of a node that has run, then mark it done."""
    # Pull the node from its dependents' dependencies, in one call for all of them. In the
    # same atomic update, a dependent left without dependencies becomes ready, so no worker
    # can find it in between. Matching only dependents that still list the node makes
    # repeating the call change nothing.
    if claim.dependents:
        released = {"$eq": ["$dependencies", []]}
        nodes.update_many(
            {"_id": {"$in": claim.dependents}, "dependencies": claim.node},
            [
                {"$set": {"dependencies": {"$setDifference": ["$dependencies", [claim.node]]}}},
                {"$set": {"state": {"$cond": [released, READY, WAITING]}}},
            ],
        )
    # Mark the node done only once its dependents are released. A worker that crashes in
    # between leaves the node running; when the lease runs out, another worker runs it again
    # and repeats the release, which changes nothing for the dependents already released. The
    # other order could leave a node done whose dependents never become ready.
    nodes.update_one(
        {"_id": claim.node, "state": {"$ne": DONE}},
        [{"$set": {"state": DONE, "finished_at": "$$NOW"}}],
    )


def release_expired(nodes: Nodes) -> int:
    """Make the running nodes whose lease ran out ready again, and return how many."""
    # Their worker stopped renewing the lease: it crashed, lost its connection or stalled. The
    # server's clock decides, as it does for the leases themselves. It is asked for first because a comparison
    # with $$NOW can't use an index: every idle worker checks, so it has to be cheap.
    now = nodes.database.command("hello")["localTime"]
    result = nodes.update_many({"state": RUNNING, "lease_until": {"$lt": now}}, {"$set": {"state": READY}})
    return result.modified_count


def finished(nodes: Nodes) -> bool:
    """Whether every node is done. One index lookup, cheap enough for idle workers to ask."""
    return nodes.find_one({"state": {"$ne": DONE}}, projection={"_id": True}) is None


def _random_ready(nodes: Nodes) -> int | None:
    # The first ready node at or after a random key, or else the last one before it.
    key = random.random()
    for bound, order in (({"$gte": key}, ASCENDING), ({"$lt": key}, DESCENDING)):
        document = nodes.find_one(
            {"state": READY, "random_key": bound}, projection={"_id": True}, sort=[("random_key", order)]
        )
        if document is not None:
            return document["_id"]
    return None


def _from_now(seconds: float) -> dict[str, Any]:
    # Adding a number to a date adds milliseconds.
    return {"$add": ["$$NOW", round(seconds * 1000)]}
