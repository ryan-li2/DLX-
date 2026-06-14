# DLX+

DLX+ is a parallel bounded-backtracking solver for the Cardinality- and
Colour-Constrained Maximum Weighted Independent Set (CC-MWIS) problem.
The repository also contains C++ interfaces for Gurobi, Google OR-Tools
CP-SAT, an LP-guided DLX+ variant, and Python-based LightGBM experiments.

## Requirements

- Apple Clang or another C++17 compiler
- Python 3.9 or later
- Gurobi 13.0.2 for the Gurobi and LP-guided variants
- OR-Tools 9.12 C++ SDK for the CP-SAT variant

Install the Python packages:

```sh
python -m pip install -r requirements.txt
```

The expected input layout is:

```text
Data/
  User Data/
    Instance_4.xlsx
    Instance_8.xlsx
    Instance_12.xlsx
  Beam Candidate/
    Beams_<instance>_<users>_60_4_sample<sample>.csv
    Allocate_<instance>_<users>_60_4_sample<sample>.txt
    Pairs_<instance>_<users>_sample<sample>.json
```

See [DATA_APPENDIX.md](DATA_APPENDIX.md) for the complete experimental design,
file schemas, and problem-identification conventions.

See [CODE_APPENDIX.md](CODE_APPENDIX.md) for the paper-to-code map, complete
experiment workflow, notebook cell index, outputs, and timing conventions.

## Build

Build the standalone DLX+ solver:

```sh
make dlx
```

Build the Gurobi solver and LP-guided variant:

```sh
export GUROBI_HOME=/Library/gurobi1302/macos_universal2
make gurobi
```

Build the CP-SAT solver:

```sh
export ORTOOLS_HOME="$HOME/Downloads/or-tools_arm64_macOS-15.3.1_cpp_v9.12.4544"
make cpsat
```

The CP-SAT target uses the CMake package bundled with the OR-Tools archive so
that its protobuf and supporting libraries are configured correctly. The SDK
paths can be overridden when they are installed elsewhere.

## Run DLX+ from Python

Run one problem:

```python
from dlx_experiment import solve

result = solve(
    data_dir="Data",
    instance=4,
    users=500,
    sample=0,
    threads=10,
)

print(result["best_value"])
print(result["best_time"])
print(result["best_integer"])
```

Run all 240 experimental problems:

```python
from dlx_experiment import run_all

results = run_all(
    data_dir="Data",
    output_dir="results",
    instances=(4, 8, 12),
    users=(500, 1000, 2500, 5000),
    samples=range(20),
    threads=10,
)
```

This writes `results/DLX_results.csv`.

## Exact Solver Interfaces

```python
from dlx_experiment import solve_cpsat, solve_gurobi

gurobi_integer, gurobi_value, gurobi_time = solve_gurobi(
    data_dir="Data",
    instance=4,
    users=500,
    sample=0,
    time_limit=3600,
)

cpsat_integer, cpsat_value, cpsat_time = solve_cpsat(
    data_dir="Data",
    instance=4,
    users=500,
    sample=0,
    time_limit=3600,
)
```

Run both exact solvers with the computation time from each matching DLX+
result:

```python
from dlx_experiment import run_exact_solvers_with_dlx_limits

gurobi_results, cpsat_results = run_exact_solvers_with_dlx_limits(
    data_dir="Data",
    dlx_results="DLX_results.csv",
    output_dir="results",
)
```

## DLX+:ML

LightGBM training and prediction remain in Python. C++ performs feature
construction, the greedy solution, and the parallel DLX+ search at the
predicted depth.

```python
from dlx_experiment import extract_features_cpp, solve_ml_depth

features = extract_features_cpp(
    data_dir="Data",
    instance=4,
    users=500,
    sample=0,
)

result = solve_ml_depth(
    data_dir="Data",
    predicted_percentile_index=3,
    instance=4,
    users=500,
    sample=0,
    threads=10,
)
```

## DLX+:LP Boost

```python
from dlx_experiment import solve_dlx_lp

result = solve_dlx_lp(
    data_dir="Data",
    instance=4,
    users=500,
    sample=0,
    threads=10,
)
```

## Output

The main result files identify the instance, user count, simulation, solver,
computation time, and total allocated demand. Generated executables, Python
caches, notebook checkpoints, and temporary run directories are excluded by
`.gitignore`.
