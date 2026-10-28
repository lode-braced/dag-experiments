import argparse
from dataclasses import replace

from dag_experiments.evaluation.plans import dependency_dict, generational_plan
from dag_experiments.evaluation.profiling import bookkeeping_seconds, measure
from dag_experiments.evaluation.simulation import gamma_runtimes, poisson_runtimes, worker_sweep
from dag_experiments.synthetic.generator import BASELINE, SyntheticDag, generate

COMPILE_SIZES = [10_000, 100_000, 1_200_000, 5_000_000]
MB = 1_000_000
HOUR = 3600


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m dag_experiments.evaluation",
        description="Compare a compiled generational plan with evaluating the DAG at runtime",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser("compile", help="time and memory of preparing a DAG")
    compile_parser.add_argument("--sizes", type=int, nargs="+", default=COMPILE_SIZES)
    simulate_parser = commands.add_parser("simulate", help="makespan against the number of workers")
    simulate_parser.add_argument("--nodes", type=int, default=BASELINE.nodes)
    simulate_parser.add_argument("--runtime-mean", type=float, default=10.0)
    simulate_parser.add_argument(
        "--runtime-sd", type=float, nargs="+",
        help="gamma runtimes with these standard deviations instead of Poisson ones",
    )
    for command in (compile_parser, simulate_parser):
        command.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.command == "compile":
        print(
            f"{'nodes':>10} {'edges':>11} | {'plan s':>6} {'peak MB':>8} {'kept MB':>8} | "
            f"{'dict s':>6} {'peak MB':>8} {'kept MB':>8} | {'bookkeeping s':>13}"
        )
        for size in args.sizes:
            dag = generate(replace(BASELINE, nodes=size), args.seed)
            plan = measure(lambda: _edges(dag), lambda edges: generational_plan(edges, size))
            links = measure(lambda: _edges(dag), lambda edges: dependency_dict(edges, size))
            edges = _edges(dag)
            bookkeeping = bookkeeping_seconds(edges, size, generational_plan(edges, size))
            print(
                f"{size:>10,} {len(edges):>11,} | {plan.seconds:>6.2f} {plan.peak_bytes / MB:>8,.0f} "
                f"{plan.kept_bytes / MB:>8,.0f} | {links.seconds:>6.2f} {links.peak_bytes / MB:>8,.0f} "
                f"{links.kept_bytes / MB:>8,.0f} | {bookkeeping:>13.2f}"
            )
    else:
        dag = generate(replace(BASELINE, nodes=args.nodes), args.seed)
        edges = _edges(dag)
        mean = args.runtime_mean
        if args.runtime_sd:
            distributions = {
                f"gamma, mean {mean:g} s, standard deviation {sd:g} s":
                    gamma_runtimes(dag.num_nodes, mean, sd, args.seed)
                for sd in args.runtime_sd
            }
        else:
            distributions = {f"Poisson, mean {mean:g} s": poisson_runtimes(dag.num_nodes, mean, args.seed)}
        for name, runtimes in distributions.items():
            sweep, unlimited = worker_sweep(edges, dag.num_nodes, runtimes)
            print(f"\nRuntimes: {name}")
            print(f"{'workers':>9} {'generational h':>14} {'asap h':>8} {'ratio':>6}")
            for point in [*sweep, unlimited]:
                workers = "unlimited" if point is unlimited else f"{point.workers:,}"
                print(
                    f"{workers:>9} {point.generational / HOUR:>14,.2f} {point.asap / HOUR:>8,.2f} "
                    f"{point.generational / point.asap:>6.2f}"
                )


def _edges(dag: SyntheticDag) -> list[tuple[int, int]]:
    return list(zip(dag.parents.tolist(), dag.children.tolist()))


if __name__ == "__main__":
    main()
