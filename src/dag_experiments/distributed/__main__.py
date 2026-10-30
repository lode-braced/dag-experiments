import argparse
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from statistics import mean
from typing import Any

from pymongo import MongoClient
from pymongo.database import Database

from dag_experiments.distributed import store
from dag_experiments.distributed.worker import LEASE_SECONDS, work
from dag_experiments.evaluation.simulation import gamma_runtimes, run_asap
from dag_experiments.synthetic.generator import BASELINE, generate

DATABASE = "dag_experiments"
# How long the processes of a Cloud Run job wait for each other before starting anyway.
START_TIMEOUT_SECONDS = 300


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m dag_experiments.distributed",
        description="Evaluate a DAG at runtime with workers that coordinate through MongoDB",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    load_parser = commands.add_parser("load", help="generate a DAG and load it, replacing the last run")
    load_parser.add_argument("--nodes", type=int, default=100_000)
    load_parser.add_argument("--runtime-mean", type=float, default=0.1, help="seconds")
    load_parser.add_argument("--runtime-sd", type=float, default=0.3, help="seconds")
    load_parser.add_argument("--seed", type=int, default=0)
    work_parser = commands.add_parser("work", help="run workers until every node is done")
    work_parser.add_argument("--threads", type=int, default=10, help="workers in this process")
    work_parser.add_argument("--lease", type=float, default=LEASE_SECONDS, help="seconds")
    monitor_parser = commands.add_parser("monitor", help="sample the MongoDB server until every node is done")
    monitor_parser.add_argument("--interval", type=float, default=10.0, help="seconds")
    report_parser = commands.add_parser("report", help="compare the last run with the simulation")
    for command in (load_parser, work_parser, monitor_parser, report_parser):
        command.add_argument("--mongo-url", default=os.environ.get("MONGO_URL", "mongodb://localhost:27117"))
    args = parser.parse_args()

    # Each worker thread and the thread renewing its lease can each be in a call at once.
    pool_size = 2 * args.threads if args.command == "work" else 10
    database = MongoClient(args.mongo_url, maxPoolSize=pool_size)[DATABASE]
    if args.command == "load":
        _load(database, args.nodes, args.runtime_mean, args.runtime_sd, args.seed)
    elif args.command == "work":
        _work(database, args.threads, args.lease)
    elif args.command == "monitor":
        _monitor(database, args.interval)
    else:
        _report(database)


def _load(database: Database[dict[str, Any]], nodes: int, mean: float, sd: float, seed: int) -> None:
    dag = generate(replace(BASELINE, nodes=nodes), seed)
    edges = list(zip(dag.parents.tolist(), dag.children.tolist()))
    started = time.perf_counter()
    store.load(database.nodes, edges, dag.num_nodes, gamma_runtimes(dag.num_nodes, mean, sd, seed))
    database.workers.drop()
    database.processes.drop()
    database.server_samples.drop()
    print(f"Loaded {dag.num_nodes:,} nodes and {len(edges):,} edges in {time.perf_counter() - started:.1f} s")


def _work(database: Database[dict[str, Any]], threads: int, lease_seconds: float) -> None:
    # In a docker container the host name is the container's id, so names are unique across
    # containers. Cloud Run doesn't promise unique host names, but it does name every task.
    if "CLOUD_RUN_EXECUTION" in os.environ:
        task = ("CLOUD_RUN_EXECUTION", "CLOUD_RUN_TASK_INDEX", "CLOUD_RUN_TASK_ATTEMPT")
        process = "-".join(os.environ[variable] for variable in task)
    else:
        process = f"{socket.gethostname()}-{os.getpid()}"
    names = [f"{process}-{thread}" for thread in range(threads)]

    # Cloud Run starts the tasks of a job over a minute or two. Waiting until they all run
    # starts the run with every worker, as in the simulation, instead of counting the wait as
    # coordination.
    database.processes.insert_one({"_id": process})
    tasks = int(os.environ.get("CLOUD_RUN_TASK_COUNT", 1))
    workers = tasks * threads
    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    while database.processes.count_documents({}) < tasks and time.monotonic() < deadline:
        time.sleep(1)
    with ThreadPoolExecutor(threads) as pool:
        stats = list(pool.map(
            lambda name: work(database.nodes, name, lease_seconds=lease_seconds, workers=workers), names
        ))
    # The report counts the workers of a run, and their time in MongoDB calls, from these.
    database.workers.insert_many([asdict(worker) for worker in stats])
    print(f"{threads} workers completed {sum(worker.completed for worker in stats):,} nodes")


def _monitor(database: Database[dict[str, Any]], interval: float) -> None:
    # The server's own counters show what the workers cost MongoDB, which the workers can't
    # see from their side: its CPU time, the calls it serves and the connections it holds.
    while True:
        status = database.client.admin.command("serverStatus")
        database.server_samples.insert_one({
            "at": status["localTime"],
            "cpu_seconds": (status["extra_info"]["user_time_us"] + status["extra_info"]["system_time_us"]) / 1e6,
            "calls": sum(status["opcounters"].values()),
            "write_conflicts": status["metrics"]["operation"]["writeConflicts"],
            "connections": status["connections"]["current"],
        })
        if store.finished(database.nodes):
            return
        time.sleep(interval)


def _report(database: Database[dict[str, Any]]) -> None:
    nodes = list(database.nodes.find(
        {}, {"dependents": True, "runtime": True, "started_at": True, "finished_at": True, "attempts": True}
    ))
    workers = list(database.workers.find())
    if not workers or any("finished_at" not in node for node in nodes):
        sys.exit("No finished run: load a DAG, then run workers until they stop")

    # Replay the same DAG and runtimes in the simulation, where handing out work costs nothing.
    num_nodes = len(nodes)
    edges = [(node["_id"], dependent) for node in nodes for dependent in node["dependents"]]
    runtimes = [0.0] * num_nodes
    for node in nodes:
        runtimes[node["_id"]] = node["runtime"]
    simulated = run_asap(edges, num_nodes, runtimes, len(workers))
    measured = (max(node["finished_at"] for node in nodes) - min(node["started_at"] for node in nodes)).total_seconds()

    completed = sum(worker["completed"] for worker in workers)
    claim_ms = 1000 * sum(worker["claim_seconds"] for worker in workers) / completed
    complete_ms = 1000 * sum(worker["complete_seconds"] for worker in workers) / completed
    busy = sum(runtimes) / len(workers)
    print(f"{num_nodes:,} nodes, {len(edges):,} edges, {len(workers)} workers")
    print(f"{'':12} {'seconds':>8} {'nodes/s':>8} {'busy':>5}")
    # Nodes that take no time leave the simulation nothing to compare.
    for name, seconds in [("run", measured), ("simulation", simulated)][: 2 if simulated else 1]:
        print(f"{name:12} {seconds:>8.1f} {num_nodes / seconds:>8,.0f} {busy / seconds:>5.0%}")
    print(f"MongoDB calls per node: claim {claim_ms:.2f} ms, complete {complete_ms:.2f} ms")
    print(f"Nodes run again after their lease ran out: {sum(node['attempts'] for node in nodes) - num_nodes}")

    # The server's load between consecutive samples, if the monitor ran.
    samples = list(database.server_samples.find().sort("at"))
    if len(samples) > 1:
        intervals = list(zip(samples, samples[1:]))
        cores = [_rate(earlier, later, "cpu_seconds") for earlier, later in intervals]
        calls = [_rate(earlier, later, "calls") for earlier, later in intervals]
        connections = [sample["connections"] for sample in samples]
        conflicts = samples[-1]["write_conflicts"] - samples[0]["write_conflicts"]
        print(
            f"MongoDB server, mean and peak: {mean(cores):.1f} and {max(cores):.1f} cores, "
            f"{mean(calls):,.0f} and {max(calls):,.0f} calls/s, "
            f"{mean(connections):,.0f} and {max(connections):,} connections"
        )
        print(f"Write conflicts per node: {conflicts / num_nodes:.1f}")


def _rate(earlier: dict[str, Any], later: dict[str, Any], counter: str) -> float:
    return (later[counter] - earlier[counter]) / (later["at"] - earlier["at"]).total_seconds()


if __name__ == "__main__":
    main()
