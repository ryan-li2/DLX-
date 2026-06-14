#pragma once

#include <vector>

struct CpSatResult {
    std::vector<int> integer;
    double objective = 0.0;
    double runtime = 0.0;
    bool has_solution = false;
};

// Paper reference: Experimental Setup - Baselines (OR-Tools CP-SAT).
CpSatResult cpsat_SPK(
    const std::vector<std::vector<int>>& rows_color,
    const std::vector<double>& D_color,
    int N,
    const std::vector<std::vector<int>>& Cons,
    bool log = false,
    double time_limit_seconds = 3600.0,
    int search_workers = 0);
