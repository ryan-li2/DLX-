# Code Appendix

This appendix indexes the source code required to conduct and analyse the
experiments reported in the ORBIT+/DLX+ paper. It should be read together with
[DATA_APPENDIX.md](DATA_APPENDIX.md), which documents the 240 processed
experimental instances.

## Reproducibility Scope

The repository includes code for:

- constructing the CC-MWIS/DLX+ matrix from the processed instances;
- standalone parallel DLX+;
- greedy lower-bound construction, bounding, pruning, and truncated
  backtracking;
- LightGBM feature extraction and ML-guided depth selection;
- LP-guided problem reduction and residual DLX+ search;
- Gurobi and OR-Tools CP-SAT baselines;
- matched-runtime Gurobi and CP-SAT experiments;
- thread-count experiments;
- calculation of optimality gaps, speedups, and paper tables.

## Paper-to-Code Map

| Paper component | Primary implementation |
|---|---|
| Model 2 and DLX+ data structure | `dlx_experiment.cpp`: `build_input` and `DLX` |
| Algorithm 1: DLX+ Main Loop | `DLX::search_range` |
| Algorithm 2: Search | `DLX::search` |
| Algorithm 3: Backtracking | `DLX::backtrack_next_node` |
| Bounding Strategies | `DLX::upper_bound` |
| Backtracking Strategies | `build_input`, `run_worker`, and `DLX::search_range` |
| Parallelisation Strategies | `run_worker` and asynchronous launches in `main` |
| Machine Learning Backtracking | `extract_features_FBO`, `predicted_backtracking_depth`, and `solve_ml_depth` |
| LP-Guided Problem Reduction | `gurobi_LP`, `reduce_after_fixed_rows`, and `solve_dlx_lp` |
| Gurobi baseline | `gurobi_spk.cpp`: `gurobi_SPK` |
| CP-SAT baseline | `cpsat_spk.cpp`: `cpsat_SPK` |
| Time-Constrained Gurobi Runs | `run_exact_solvers_with_dlx_limits` |
| Impact of Parallelisation | `run_all` and Cells 21--23 of `C++.ipynb` |

The C++ source comments reference the corresponding paper headings and
Algorithms 1--3.

## Source Files

### Proposed Methods

- `dlx_experiment.cpp`: main C++17 implementation of DLX+, DLX+:ML search,
  feature extraction, parallel workers, LP-guided reduction, timing, and result
  serialisation.
- `dlx_experiment.py`: Python data-loading and experiment interface.
- `gurobi_spk.cpp` and `gurobi_spk.hpp`: Gurobi integer model and LP
  relaxation used by DLX+:LP Boost.
- `cpsat_spk.cpp` and `cpsat_spk.hpp`: OR-Tools CP-SAT baseline.

### Build and Environment

- `Makefile`: build targets for DLX+, Gurobi, and CP-SAT.
- `CMakeLists.txt`: CMake configuration required by OR-Tools.
- `requirements.txt`: Python dependencies used for loading, ML, validation,
  and analysis.

### Validation and Reference Code

- `functions.py`: original Python implementation and validation utilities.
- `optimized_dlx.py`: earlier Python DLX implementation retained for
  implementation comparison.

## Experiment Notebook

`C++.ipynb` is the executable code appendix for the experiment and analysis
workflow:

| Cells | Purpose | Main output |
|---:|---|---|
| 0 | Full Gurobi runs | `Gurobi_results.csv` |
| 1 | Full CP-SAT runs | `CP_SAT_results.csv` |
| 2 | C++ ML feature extraction | `DLX_ML_features.csv` |
| 3--4 | Leave-one-instance-out LightGBM and DLX+:ML | `DLX_ML_results.csv` |
| 5 | DLX+:LP Boost | `DLX_LP_results.csv` |
| 6--17 | Python/C++ validation and intermediate checks | Validation outputs |
| 18--20 | Exact solvers with DLX+ runtime limits | Matched-time CSV files |
| 21--23 | Parallelisation and Gurobi speedup tables | `parallelisation_results/` |
| 24 | CP-SAT speedup analysis for 10-thread DLX+ | `DLX_speedup_over_CPSAT_10_threads.csv` |
| 25 | Gurobi objective relative to DLX+ | `Gurobi_objective_relative_to_DLX.csv` |

The notebook uses absolute paths from the original machine. To reproduce it
elsewhere, replace:

```text
/Users/ryanli/Documents/GitHub/DLX_plus
```

with the local repository path.

## Reproduction Order

### 1. Install Dependencies

```sh
python -m pip install -r requirements.txt
```

### 2. Build the Solvers

```sh
make dlx

export GUROBI_HOME=/path/to/gurobi
make gurobi

export ORTOOLS_HOME=/path/to/or-tools
make cpsat
```

### 3. Run Standalone DLX+

```python
from dlx_experiment import run_all

dlx_results = run_all(
    data_dir="Data",
    output_dir=".",
    instances=(4, 8, 12),
    users=(500, 1000, 2500, 5000),
    samples=range(20),
    threads=10,
)
```

This produces `DLX_results.csv`.

### 4. Run the Baselines

Use Cells 0--1 of `C++.ipynb` to produce the full Gurobi and CP-SAT results.

### 5. Run DLX+:ML

Use Cells 2--4 of `C++.ipynb`. For each held-out problem, LightGBM is trained
on the other 239 instances, predicts one of seven depth classes, and invokes
`solve_ml_depth`.

The reported Python timing begins before prediction and ends after the C++
search returns. Offline model-training time is excluded.

### 6. Run DLX+:LP Boost

Use Cell 5 of `C++.ipynb`, which calls `solve_dlx_lp` for all 240 problems.

### 7. Run Matched-Time Baselines

```python
from dlx_experiment import run_exact_solvers_with_dlx_limits

gurobi_results, cpsat_results = run_exact_solvers_with_dlx_limits(
    data_dir="Data",
    dlx_results="DLX_results.csv",
    output_dir=".",
)
```

Each baseline receives the computation time of the matching DLX+ problem,
joined by `(Instance, User, Simulation)`.

### 8. Reproduce the Parallelisation Tables

Use Cells 21--23 of `C++.ipynb`. DLX+ is evaluated with `1`, `5`, `10`, and
`20` threads, and the 20 sample-level results are averaged for each class.

## Result Files

| File | Description |
|---|---|
| `DLX_results.csv` | Standalone 10-thread DLX+ |
| `DLX_ML_results.csv` | ML-guided DLX+ |
| `DLX_LP_results.csv` | LP-guided DLX+ |
| `Gurobi_results.csv` | Full Gurobi baseline |
| `CP_SAT_results.csv` | Full CP-SAT baseline |
| `Gurobi_results_timelimit.csv` | Gurobi with matched DLX+ runtime |
| `CP_SAT_results_timelimit.csv` | CP-SAT with matched DLX+ runtime |
| `parallelisation_results/` | Thread-count results and LaTeX tables |

All result comparisons use `(Instance, User, Simulation)` as the join key.

## Timing Conventions

- **DLX+:** greedy incumbent construction plus parallel DLX+ search.
- **DLX+:ML:** Python prediction plus C++ greedy and predicted-depth search;
  offline training is excluded.
- **DLX+:LP Boost:** LP relaxation, reduction, and residual DLX+ search.
- **Gurobi and CP-SAT:** native solver runtime returned by each interface.

These conventions match the experimental descriptions in the paper.
