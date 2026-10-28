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

## Visualizations

```shell
# Redraw the charts in docs/images from the baseline graph
uv run python -m dag_experiments.visualization.charts
```

## Distributed workers

### On a development system: docker

### On a google cloud run project
