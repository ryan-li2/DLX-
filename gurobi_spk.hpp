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

std::vector<std::vector<int>> incidence_transpose_fast(
    const std::vector<std::vector<int>>& rows_color);

GurobiResult gurobi_SPK(
    const std::vector<std::vector<int>>& rows_color,
    const std::vector<double>& D_color,
    int N,
    const std::vector<std::vector<int>>& Cons,
    double time_limit_seconds = 3600.0);

GurobiLpResult gurobi_LP(
    const std::vector<std::vector<int>>& rows_color,
    const std::vector<double>& D_color,
    int N,
    const std::vector<std::vector<int>>& Cons,
    double time_limit_seconds = 3600.0);
