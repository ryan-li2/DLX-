"""Python data-loading interface for the compiled DLX+ C++ solver."""

from ast import literal_eval
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import json
import subprocess
import time

import pandas as pd


_MODULE_DIR = Path(__file__).resolve().parent
_EXECUTABLE = _MODULE_DIR / "dlx_experiment"
_LGBM_EXECUTABLE = _MODULE_DIR / "dlx_experiment_lgbm"
_GUROBI_EXECUTABLE = _MODULE_DIR / "gurobi_experiment"
_CPSAT_EXECUTABLE = _MODULE_DIR / "cpsat_experiment"


def run_exact_solvers_with_dlx_limits(
    data_dir,
    dlx_results,
    output_dir=".",
):
    """Run Gurobi and CP-SAT using each matching DLX+ computation time."""
    if isinstance(dlx_results, (str, Path)):
        dlx_results = pd.read_csv(dlx_results)
    else:
        dlx_results = dlx_results.copy()

    required = {
        "Instance",
        "User",
        "Simulation",
        "Computation Time(s)",
    }
    missing = required.difference(dlx_results.columns)
    if missing:
        raise ValueError(
            f"DLX+ results are missing columns: {sorted(missing)}"
        )

    gurobi_records = []
    cpsat_records = []
    for _, row in dlx_results.iterrows():
        instance = int(row["Instance"])
        users = int(row["User"])
        simulation = int(row["Simulation"])
        time_limit = float(row["Computation Time(s)"])

        gurobi_integer, gurobi_value, gurobi_time = solve_gurobi(
            data_dir=data_dir,
            instance=instance,
            users=users,
            sample=simulation,
            time_limit=time_limit,
        )
        cpsat_integer, cpsat_value, cpsat_time = solve_cpsat(
            data_dir=data_dir,
            instance=instance,
            users=users,
            sample=simulation,
            time_limit=time_limit,
        )

        gurobi_records.append({
            "Instance": instance,
            "User": users,
            "Simulation": simulation,
            "Solver": "Gurobi",
            "Computation Time(s)": gurobi_time,
            "Total Allocated Demand": gurobi_value,
        })
        cpsat_records.append({
            "Instance": instance,
            "User": users,
            "Simulation": simulation,
            "Solver": "CP-SAT",
            "Computation Time(s)": cpsat_time,
            "Total Allocated Demand": cpsat_value,
        })

        print(
            f"Completed: z={instance}, INS={users}, "
            f"simulation={simulation}, limit={time_limit:.9f}s"
        )

    columns = [
        "Instance",
        "User",
        "Simulation",
        "Solver",
        "Computation Time(s)",
        "Total Allocated Demand",
    ]
    gurobi_results = pd.DataFrame(gurobi_records, columns=columns)
    cpsat_results = pd.DataFrame(cpsat_records, columns=columns)

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gurobi_results.to_csv(
        output_dir / "Gurobi_results_timelimit.csv", index=False
    )
    cpsat_results.to_csv(
        output_dir / "CP_SAT_results_timelimit.csv", index=False
    )
    return gurobi_results, cpsat_results


def extract_features_cpp(
    data_dir,
    instance=4,
    users=500,
    sample=0,
):
    """Return the C++ equivalent of extract_features_FBO()."""
    if not _EXECUTABLE.exists():
        raise FileNotFoundError(
            f"DLX executable not found at {_EXECUTABLE}."
        )

    data_dir = Path(data_dir).expanduser().resolve()
    with TemporaryDirectory(prefix="dlx_features_") as temporary:
        result_dir = Path(temporary)
        subprocess.run(
            [
                str(_EXECUTABLE),
                str(data_dir),
                str(result_dir),
                str(instance),
                str(users),
                str(sample),
                "1",
                "1",
                "features",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with (result_dir / "DLX_features.json").open() as source:
            return json.load(source)


def solve_ml_depth(
    data_dir,
    predicted_percentile_index,
    instance=4,
    users=500,
    sample=0,
    threads=10,
):
    """Run DLX+ at one ML-predicted backtracking percentile."""
    if predicted_percentile_index not in range(7):
        raise ValueError("predicted_percentile_index must be from 0 to 6")

    data_dir = Path(data_dir).expanduser().resolve()
    with TemporaryDirectory(prefix="dlx_ml_") as temporary:
        result_dir = Path(temporary)
        subprocess.run(
            [
                str(_EXECUTABLE),
                str(data_dir),
                str(result_dir),
                str(instance),
                str(users),
                str(sample),
                "1",
                str(threads),
                "ml",
                str(predicted_percentile_index),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with (result_dir / "DLX_ML_result.json").open() as source:
            return json.load(source)


def solve_ml_model(
    data_dir,
    model,
    instance=4,
    users=500,
    sample=0,
    threads=10,
):
    """Predict in C++ and run DLX+ at the predicted depth."""
    if not _LGBM_EXECUTABLE.exists():
        raise FileNotFoundError(
            f"LightGBM DLX executable not found at {_LGBM_EXECUTABLE}."
        )
    with TemporaryDirectory(prefix="dlx_lgbm_") as temporary:
        temporary = Path(temporary)
        model_path = temporary / "model.txt"
        result_dir = temporary / "results"
        result_dir.mkdir()

        booster = getattr(model, "booster_", model)
        booster.save_model(str(model_path))

        subprocess.run(
            [
                str(_LGBM_EXECUTABLE),
                str(Path(data_dir).expanduser().resolve()),
                str(result_dir),
                str(instance),
                str(users),
                str(sample),
                "1",
                str(threads),
                "ml_model",
                str(model_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with (result_dir / "DLX_ML_result.json").open() as source:
            return json.load(source)


def solve_cpsat(
    data_dir,
    instance=4,
    users=500,
    sample=0,
    time_limit=3600,
):
    """Run the compiled C++ CP-SAT model and return its solution."""
    if not _CPSAT_EXECUTABLE.exists():
        raise FileNotFoundError(
            f"CP-SAT executable not found at {_CPSAT_EXECUTABLE}. "
            "Install the OR-Tools C++ SDK and build it using README.md."
        )
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")

    data_dir = Path(data_dir).expanduser().resolve()
    with TemporaryDirectory(prefix="cpsat_cpp_") as temporary:
        result_dir = Path(temporary)
        subprocess.run(
            [
                str(_CPSAT_EXECUTABLE),
                str(data_dir),
                str(result_dir),
                str(instance),
                str(users),
                str(sample),
                "1",
                "1",
                "cpsat",
                str(float(time_limit)),
            ],
            check=True,
        )
        with (result_dir / "CP_SAT_results.csv").open(
            newline=""
        ) as source:
            row = next(csv.DictReader(source))

    integer_text = row["Integer Solution"].strip()
    integer = (
        [int(value) for value in integer_text.split()]
        if integer_text
        else []
    )
    return (
        integer,
        (
            float(row["Total Allocated Demand"])
            if row["Total Allocated Demand"].strip()
            else None
        ),
        float(row["Computation Time(s)"]),
    )


def solve_gurobi(
    data_dir,
    instance=4,
    users=500,
    sample=0,
    n_colour=4,
    time_limit=3600,
):
    """Run the compiled C++ Gurobi model and return its solution."""
    if not _GUROBI_EXECUTABLE.exists():
        raise FileNotFoundError(
            f"Gurobi executable not found at {_GUROBI_EXECUTABLE}. "
            "Build it using the command in README.md."
        )
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")

    data_dir = Path(data_dir).expanduser().resolve()
    candidate_dir = data_dir / "Beam Candidate"
    prefix = f"{instance}_{users}_60_{n_colour}_sample{sample}"

    required_files = (
        candidate_dir / f"Beams_{prefix}.csv",
        candidate_dir / f"Allocate_{prefix}.txt",
        candidate_dir / f"Pairs_{instance}_{users}_sample{sample}.json",
    )
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    with TemporaryDirectory(prefix="gurobi_cpp_") as temporary:
        result_dir = Path(temporary)
        subprocess.run(
            [
                str(_GUROBI_EXECUTABLE),
                str(data_dir),
                str(result_dir),
                str(instance),
                str(users),
                str(sample),
                "1",
                "1",
                "gurobi",
                str(float(time_limit)),
            ],
            check=True,
        )

        with (result_dir / "Gurobi_results.csv").open(
            newline=""
        ) as source:
            row = next(csv.DictReader(source))

    integer_text = row["Integer Solution"].strip()
    integer = (
        [int(value) for value in integer_text.split()]
        if integer_text
        else []
    )
    return (
        integer,
        (
            float(row["Total Allocated Demand"])
            if row["Total Allocated Demand"].strip()
            else None
        ),
        float(row["Computation Time(s)"]),
    )


def solve_dlx_lp(
    data_dir,
    instance=4,
    users=500,
    sample=0,
    threads=10,
):
    """Run the Gurobi LP-guided DLX+ solver in C++."""
    if not _GUROBI_EXECUTABLE.exists():
        raise FileNotFoundError(
            f"Gurobi executable not found at {_GUROBI_EXECUTABLE}."
        )

    with TemporaryDirectory(prefix="dlx_lp_") as temporary:
        result_dir = Path(temporary)
        subprocess.run(
            [
                str(_GUROBI_EXECUTABLE),
                str(Path(data_dir).expanduser().resolve()),
                str(result_dir),
                str(instance),
                str(users),
                str(sample),
                "1",
                str(threads),
                "lp",
            ],
            check=True,
        )
        with (result_dir / "DLX_LP_result.json").open() as source:
            return json.load(source)


def solve(
    data_dir,
    instance=4,
    users=500,
    sample=0,
    R=None,
    n_colour=4,
    threads=10,
):
    """Load one problem and return C++ input structures and worker results."""
    data_dir = Path(data_dir).expanduser().resolve()
    colours = list(range(n_colour) if R is None else R)
    _validate_colours(colours, n_colour)

    user_data = pd.read_excel(
        data_dir / "User Data" / f"Instance_{instance}.xlsx"
    )
    data = user_data.iloc[:users].copy()
    beams, allocate, pairs = _load_problem(
        data_dir, instance, users, sample, n_colour
    )
    result = _solve_loaded(
        beams=beams,
        allocate=allocate,
        data=data,
        pairs=pairs,
        instance=instance,
        users=users,
        sample=sample,
        n_colour=n_colour,
        threads=threads,
    )
    return result


def run_all(
    data_dir,
    output_dir=".",
    instances=(4, 8, 12),
    users=(500, 1000, 2500, 5000),
    samples=range(20),
    R=None,
    n_colour=4,
    threads=10,
):
    """Run all requested problems and return a pandas result table."""
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    colours = list(range(n_colour) if R is None else R)
    _validate_colours(colours, n_colour)

    results = []
    for instance in instances:
        user_data = pd.read_excel(
            data_dir / "User Data" / f"Instance_{instance}.xlsx"
        )
        for user_count in users:
            data = user_data.iloc[:user_count].copy()
            for sample in samples:
                beams, allocate, pairs = _load_problem(
                    data_dir,
                    instance,
                    user_count,
                    sample,
                    n_colour,
                )
                result = _solve_loaded(
                    beams=beams,
                    allocate=allocate,
                    data=data,
                    pairs=pairs,
                    instance=instance,
                    users=user_count,
                    sample=sample,
                    n_colour=n_colour,
                    threads=threads,
                )
                results.append(result)
                print(
                    f"Completed: z={instance}, INS={user_count}, "
                    f"simulation={sample}"
                )

    summary_rows = []
    for result in results:
        best_worker = max(result["results"], key=lambda item: item[0])
        summary_rows.append(
            {
                "Instance": result["Instance"],
                "User": result["User"],
                "Simulation": result["Simulation"],
                "Backtracking percentiles idx": result[
                    "best_percentile_index"
                ],
                "Total Allocated Demand": result["best_value"],
                "Computation Time(s)": result["best_time"],
            }
        )
    frame = pd.DataFrame(summary_rows)
    frame.to_csv(output_dir / "DLX_results.csv", index=False)
    return frame


def run(
    data_dir,
    output_dir=".",
    instance=4,
    users=500,
    first_sample=0,
    sample_count=20,
    R=None,
    n_colour=4,
    threads=10,
):
    """Compatibility helper for one instance/user batch."""
    return run_all(
        data_dir=data_dir,
        output_dir=output_dir,
        instances=(instance,),
        users=(users,),
        samples=range(first_sample, first_sample + sample_count),
        R=R,
        n_colour=n_colour,
        threads=threads,
    )


def _load_problem(data_dir, instance, users, sample, n_colour):
    candidate_dir = data_dir / "Beam Candidate"
    prefix = f"{instance}_{users}_60_{n_colour}_sample{sample}"

    beams = pd.read_csv(
        candidate_dir / f"Beams_{prefix}.csv",
        index_col=0,
    )
    with (candidate_dir / f"Allocate_{prefix}.txt").open() as source:
        allocate = literal_eval(source.read())
    with (
        candidate_dir / f"Pairs_{instance}_{users}_sample{sample}.json"
    ).open() as source:
        pairs = json.load(source)
    return beams, allocate, pairs


def _solve_loaded(
    beams,
    allocate,
    data,
    pairs,
    instance,
    users,
    sample,
    n_colour,
    threads,
):
    if not _EXECUTABLE.exists():
        raise FileNotFoundError(
            f"Compiled solver not found at {_EXECUTABLE}. "
            "Build it using the command in README.md."
        )

    with TemporaryDirectory(prefix="dlx_python_") as temporary:
        root = Path(temporary)
        candidate_dir = root / "Data" / "Beam Candidate"
        result_dir = root / "results"
        candidate_dir.mkdir(parents=True)
        result_dir.mkdir()

        staged_prefix = f"{instance}_{users}_60_{n_colour}_sample0"
        beams.to_csv(candidate_dir / f"Beams_{staged_prefix}.csv")
        with (
            candidate_dir / f"Allocate_{staged_prefix}.txt"
        ).open("w") as destination:
            destination.write(repr(allocate))
        with (
            candidate_dir / f"Pairs_{instance}_{users}_sample0.json"
        ).open("w") as destination:
            json.dump(pairs, destination)

        start = time.perf_counter()
        subprocess.run(
            [
                str(_EXECUTABLE),
                str(root / "Data"),
                str(result_dir),
                str(instance),
                str(users),
                "0",
                "1",
                str(threads),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - start

        with (result_dir / "DLXv1_results1.csv").open(newline="") as source:
            row = next(csv.DictReader(source))
        with (result_dir / "DLX_details.json").open() as source:
            details = json.load(source)

    integer_text = row["Integer Solution"].strip()
    integer = [int(value) for value in integer_text.split()] if integer_text else []
    details["pos"] = {
        int(beam): position for beam, position in details["pos"].items()
    }
    details["adj"] = {
        int(beam): set(neighbours)
        for beam, neighbours in details["adj"].items()
    }
    raw_worker_results = details.pop("results")
    best_result = details["best_result"]
    worker_results = [
        (
            worker["LB"],
            worker["LB_integer"],
            worker["backtrack_depth"],
        )
        for worker in raw_worker_results
    ]
    return {
        "Instance": instance,
        "User": users,
        "Simulation": sample,
        "R": colours_as_text(range(n_colour)),
        "n_colour": n_colour,
        "integer": integer,
        "objective": float(row["Total Allocated Demand"]),
        "best_value": best_result["value"],
        "best_time": best_result["time"],
        "best_percentile_index": best_result["percentile_index"],
        "best_backtrack_depth": best_result["backtracking_depth"],
        "best_integer": best_result["integer"],
        "time": details["time"],
        "greedy_time": details["greedy_time"],
        "time_by_backtrack_depth": details["time_by_backtrack_depth"],
        "worker_times": [worker["times"] for worker in raw_worker_results],
        "interface_time": elapsed,
        "rows_color": details["rows_color"],
        "orig_row": [tuple(row) for row in details["orig_row"]],
        "pos": details["pos"],
        "D_color": details["D_color"],
        "u": details["u"],
        "N_max": details["N_max"],
        "adj": details["adj"],
        "back": details["back"],
        "results": worker_results,
    }


def colours_as_text(colours):
    return " ".join(str(colour) for colour in colours)


def _validate_colours(colours, n_colour):
    if n_colour != 4 or colours != list(range(4)):
        raise ValueError(
            "The current C++ solver supports R=range(4) and n_colour=4."
        )
