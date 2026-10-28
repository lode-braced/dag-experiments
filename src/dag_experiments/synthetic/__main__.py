import argparse
import time
from dataclasses import replace

from dag_experiments.synthetic.generator import BASELINE, generate
from dag_experiments.synthetic.stats import report

SWEEP_SIZES = [10_000, 100_000, 1_200_000, 5_000_000]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m dag_experiments.synthetic", description="Generate synthetic data-flow DAGs"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    report_parser = commands.add_parser("report", help="generate one graph and print its statistics")
    report_parser.add_argument("--nodes", type=int, default=BASELINE.nodes)
    sweep_parser = commands.add_parser("sweep", help="generate increasing sizes to check scaling")
    sweep_parser.add_argument("--sizes", type=int, nargs="+", default=SWEEP_SIZES)
    for command in (report_parser, sweep_parser):
        command.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.command == "report":
        started = time.perf_counter()
        dag = generate(replace(BASELINE, nodes=args.nodes), args.seed)
        print(f"Generated in {time.perf_counter() - started:.1f} s\n")
        print(report(dag, args.seed))
    else:
        print(
            f"{'nodes':>12} {'pipelines':>9} {'edges':>12} {'edges/node':>10} "
            f"{'generations':>11} {'seconds':>8}"
        )
        for size in args.sizes:
            started = time.perf_counter()
            dag = generate(replace(BASELINE, nodes=size), args.seed)
            seconds = time.perf_counter() - started
            print(
                f"{size:>12,} {len(dag.pipelines):>9,} {dag.num_edges:>12,} "
                f"{dag.num_edges / size:>10.2f} {dag.generation.max() + 1:>11} {seconds:>8.1f}"
            )


if __name__ == "__main__":
    main()
