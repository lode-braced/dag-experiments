# Performance experiments for dag workflows

Performance experiments following a discussion
on whether to use generational topological searches
to compile an execution plan vs evaluating the dag
representation in runtime, specifically for large
(500k-5M nodes) flows

This split up in a few parts:

1. Generating synthetic graphs that simulate the shapes of actual data pipeline dags
2. Using the synthetic graphs, evaluating the memory and computational complexity of runtime dag evaluation vs compiling
   an execution plan
3. A solution for coordinating runtime dag evaluation in a distributed system of workers, with MongoDB as the
   central data store.

# Generating synthetic graphs representing actual data pipeline dags

The experiments run on synthetic graphs shaped like data-flow dependency graphs (ETL, lineage,
build-style task execution). A graph is made of thousands of pipelines of different sizes, each
internally layered, and most consume outputs of earlier pipelines. A few heavily shared nodes act
as global hubs. The graph generation logic is built to make graphs that look like real flows, so
the results of the performance experiments reflect what you'd see with large data-flow DAGs.

A node's generation is its longest path from a root: the batch a generational sort puts it in. In
the baseline graph of 1.2M nodes and 2.9M edges:

- **Many small pipelines, a few large ones**: half of the 4,000 pipelines have fewer than 150
  nodes, and the largest have 5,000.
- **Many inputs**: generation 0 is the widest, with 5.5% of the nodes. Most are raw tables at the
  start of a source pipeline, read by a few transformations.
- **Wide early, with a long thin tail**: pipelines build on the outputs of earlier ones, so 87% of
  the nodes sit in the first 40 of 94 generations.
- **Mostly local edges**: 83% of the edges stay inside a pipeline, and two thirds go to the
  previous generation.
- **Few dependencies, a long tail of dependents**: 82% of the nodes with dependencies have one to
  three, while the top hubs (simulating dimension tables, calendars, config) have up to about
  3,400 dependents.
- **Sinks are outputs**: the 17% of nodes that nothing reads are pipeline outputs and side outputs
  such as reports and exports.

![Three small pipelines from the baseline graph, node by node](docs/images/example_pipelines.png)

![Nodes per generation](docs/images/nodes_per_generation.png)

The model, its parameters and why, and the full statistics are in
[docs/synthetic_graphs.md](docs/synthetic_graphs.md).

# Evaluating the memory and computational complexity

The two ways to run a DAG, both in plain Python in `src/dag_experiments/evaluation/`:

- **Compiled generational plan**: before the run, Kahn's algorithm sorts the nodes into
  generations. Workers then run the plan generation by generation: a generation starts once the
  previous one has finished.
- **As soon as possible**: Convert the graph into a representation that can be used easily at runtime: a dict holds every node's dependents and its unfinished dependencies.
  When a node completes, it is popped from its dependents' dependencies, and a dependent whose
  last dependency was popped can run right away.

## Preparing the graph

Both take time and memory linear in the nodes plus edges, O(V + E): compiling the plan visits
every edge once, and so does building the dict. Measured on an Apple M5 Pro with 24 GB, Python
3.13. The peak is memory on top of the input edge list; what is kept is what the result holds
once that list is gone, including the node ids it shares with it.

| Nodes | Compile the plan | Peak while compiling | Plan kept | Build the dict | Dict kept | Pop completed nodes, whole run |
|-------|------------------|----------------------|-----------|----------------|-----------|--------------------------------|
| 10k   | < 0.01 s         | 1 MB                 | 1 MB      | < 0.01 s       | 6 MB      | < 0.01 s                       |
| 100k  | 0.06 s           | 12 MB                | 4 MB      | 0.10 s         | 64 MB     | 0.11 s                         |
| 1.2M  | 1.5 s            | 142 MB               | 49 MB     | 2.0 s          | 769 MB    | 2.0 s                          |
| 5M    | 8.5 s            | 594 MB               | 203 MB    | 10.0 s         | 3,250 MB  | 10.3 s                         |

- **Both take seconds to prepare**, in practice growing a little faster than the node count:
  8.5 s for the plan and 10 s for the dict at 5M nodes.
- **Memory is where they differ.** The plan needs its working memory only while compiling; the
  workers then need only the batches, about 40 bytes per node: 49 MB at 1.2M nodes. The dict has
  to stay in memory for the whole run, about 650 bytes per node: 769 MB at 1.2M nodes and 3.3 GB
  at 5M.
- **Evaluating at runtime does the plan's work during the run.** Popping every completed node
  from the dict takes about as long as compiling the plan, around 2 µs per node. A single process
  could keep up with about 500,000 completions a second this way, far more than the at most 1,300
  a second of the simulated runs below.

## Running the graph

A simulation runs the 1.2M-node baseline graph with a fixed number of workers. Every node gets a
runtime from a gamma distribution with a mean of 10 s, and the chart varies its standard
deviation: from 0, where every node takes exactly 10 s, to ±30 s, where 64% of the nodes take
under a second and 5% over a minute. Both approaches get the same runtimes. A worker runs one node at a time, and
handing out work costs nothing. Within a generation, each node goes to the first free worker; as
soon as possible hands ready nodes to free workers in the order they became ready. The worker
counts go 10, 20, 50, ... and stop at the first one where as soon as possible is within 1% of its
run time with a worker for every node.

![The generational plan's run time divided by as soon as possible's, against workers, per runtime spread](docs/images/runtime_spread.png)

> [!NOTE]
> The worker counts are large because the graph is: the point where the plan starts to fall behind
> grows with the graph. That is also why the chart shows a ratio rather than absolute run times.
> On smaller graphs the curves keep their shape at fewer workers. At ±3 s, the plan falls 5%
> behind at about 500 workers for 1.2M nodes, between 200 and 500 for 400k nodes, and between
> 20 and 50 for 40k nodes.

![The same chart for graphs of 400,000 and 40,000 nodes](docs/images/runtime_spread_smaller_graphs.png)

- **With few workers, the plan costs little.** The workers are the bottleneck: 10 workers take
  about 333 hours either way. The plan stays within about 5% of as soon as possible up to 500
  workers for spreads up to ±3 s, up to 100 workers at ±10 s, and up to 20 workers at ±30 s.
- **The wider the spread, the sooner and the further the plan falls behind.** Each generation
  waits for its slowest node, and a wider spread makes that node slower compared with the
  others. At ±30 s the plan is 8% behind with 50 workers and ends up 11× slower: 12 hours,
  against 67 minutes for the longest path through the graph.
- **Even with no spread, limited workers cost the plan up to 12%.** Between 2,000 and 20,000
  workers, the narrow generations late in the graph leave part of the pool idle, while as soon as
  possible fills it with work from later generations. With a worker for every node, both take the
  same 16 minutes: each of the 94 generations adds exactly 10 s.

> [!NOTE]
> The simulation uses one graph and one seed, and leaves out the cost of coordinating workers,
> which the next section covers.

# a solution for coordinating runtime dag evaluation on distribute

## In a fully local setup

## Deployed as a google cloud run project

# Running the experiments yourself

Install [uv](https://docs.astral.sh/uv/), then from the repository root:

```shell
uv sync
uv run pytest
```

## Synthetic graphs

```shell
# Generate one graph and print the statistics from docs/synthetic_graphs.md
uv run python -m dag_experiments.synthetic report
uv run python -m dag_experiments.synthetic report --nodes 5000000 --seed 1

# Generate 10k, 100k, 1.2M and 5M nodes to check scaling
uv run python -m dag_experiments.synthetic sweep
```

## Evaluation

```shell
# Time and memory of preparing a graph both ways, at 10k, 100k, 1.2M and 5M nodes
uv run python -m dag_experiments.evaluation compile

# Simulated run time against the number of workers, with Poisson runtimes on the baseline graph
uv run python -m dag_experiments.evaluation simulate

# The same with gamma runtimes and several standard deviations, as in the chart
uv run python -m dag_experiments.evaluation simulate --runtime-sd 0 1 3 10 30
```

## Visualizations

```shell
# Redraw the charts in docs/images from the baseline graph
uv run python -m dag_experiments.visualization.charts

# Redraw the runtime spread charts, for the baseline and two smaller graphs (runs the
# simulations, several minutes)
uv run python -m dag_experiments.visualization.evaluation
```

## Distributed workers

### On a development system: docker

### On a google cloud run project
