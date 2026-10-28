# Performance experiments for dag workflows

Performance experiments following a discussion
on whether to use generational topological searches
to compile an execution plan vs evaluating the dag
representation in runtime, specifically for large
(500k-20M nodes) flows

This split up in a few parts:

1. Generating synthetic graphs that simulate the shapes of actual data pipeline dags
2. Using the synthetic graphs, evaluating the memory and computational complexity of runtime dag evaluation vs compiling
   an execution plan
3. A solution for coordinating runtime dag evaluation in a distributed system of workers, with MongoDB as the
   central data store.

# Generating synthetic graphs representing actual data pipeline dags

# Evaluating the memory and computational complexity

# a solution for coordinating runtime dag evaluation on distribute

## In a fully local setup

## Deployed as a google cloud run project

# Running the experiments yourself

## Distributed workers

### On a development system: docker

### On a google cloud run project