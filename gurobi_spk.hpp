#pragma once

#include <vector>

struct GurobiResult {
    std::vector<int> integer;
    double objective = 0.0;
    double runtime = 0.0;
    bool has_solution = false;
};

struct GurobiLpResult {
    std::vector<int> integer;
    std::vector<int> fraction;
    double integer_demand = 0.0;
    double objective = 0.0;
    double runtime = 0.0;
};

// Paper reference: transpose the DLX+ matrix for the set-packing model.
std::vector<std::vector<int>> incidence_transpose_fast(
    const std::vector<std::vector<int>>& rows_color);

// Paper reference: Model 2 and "Time-Constrained Gurobi Runs".
GurobiResult gurobi_SPK(
    const std::vector<std::vector<int>>& rows_color,
    const std::vector<double>& D_color,
    int N,
    const std::vector<std::vector<int>>& Cons,
    double time_limit_seconds = 3600.0);

// Paper reference: LP relaxation used by "LP-Guided Problem Reduction".
GurobiLpResult gurobi_LP(
    const std::vector<std::vector<int>>& rows_color,
    const std::vector<double>& D_color,
    int N,
    const std::vector<std::vector<int>>& Cons,
    double time_limit_seconds = 3600.0);
