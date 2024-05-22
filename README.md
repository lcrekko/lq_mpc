# Performance Analysis of Certainty-equivalent Linear Model Predictive Control

This Python project contains the code for analyzing the worst-case
**infinite-horizon performance** of certainty-equivalent model
predictive control for **linear** systems under **quadratic** cost
functions. The background information and theoretical analysis can be
found in the paper
> C. Liu, S. Shi, and B. De Schutter, "Stability and performance analysis
> of model predictive control of uncertain linear systems."

## Required packages:

1. Numerical Computation & Plotting
    1. `math`
    2. `numpy`
    3. `scipy`
    4. `matplotlib`
2. Control & Optimization
    1. [`cvxpy`](https://www.cvxpy.org/)
    2. [`control`](https://python-control.readthedocs.io/en/latest/)
    3. [`gurobipy`](https://pypi.org/project/gurobipy/) (to use GUROBI, a valid license is required)

## Brief Usage

- To obtain the results in the paper, you may simply run the file
  `working_example_multiple.py`.
- Important user-defined classes and functions are
  stored, respectively, in `utils_class.py` and `utils.py`.
- To enable fast simulation, some preload data has been saved as `.npy`
  and `.npz` files
- The file `main.py` is left as empty such that potential interested readers can do
  other experiments based on the provided examples files and/or test files.

If you have any questions, please contact Ir. Changrui Liu at
E-mail: [[C.Liu-14@tudelft.nl](mailto:C.Liu-14@tudelft.nl)].
