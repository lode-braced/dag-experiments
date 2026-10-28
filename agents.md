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

