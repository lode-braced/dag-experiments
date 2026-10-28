This project is a minimal comparison of two concrete
options for large (many nodes) dag execution.

Goal being the readme provides the information learned/built from the
experiments, with a reader able to understand the
concepts and cases from the readme, and the python code behind it allowing
them to run the experiments in the readme themselves,
e.g., to compare the results on different hardware.

The experiments have two separations of concerns: running the actual experiments
and producing visualizations to help understand the experiments, e.g.,
distribution of the amount of edges per vertex. Keep those layers separated: experiment code can
depend on other experiments, but not on visualization code.

The experiments are meant to help chart/quantify what to do in production code,
but are not themselves production code. As a result, the speed and ease
at which the reader can read and understand the underlying concepts and results
is the main factor of concern for this codebase.

Comments deviate from the usual "only explain non-obvious choices" rule: a reader
should be able to follow each algorithm from its comments alone. Give each step of an
algorithm (usually a few lines) a short comment on what it does in terms of the modeled
system and why: what a branch represents, which constraint a correction protects, and
what real-world effect a rule mimics. Skip comments on single lines whose code already
reads plainly.
