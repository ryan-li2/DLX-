#include "cpsat_spk.hpp"

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/sat_parameters.pb.h"

/*
Paper reference: Experimental Setup - Baselines and "Weighted Max-Sat
Formulation". This file implements the OR-Tools CP-SAT comparison model.
*/

CpSatResult cpsat_SPK(
    const std::vector<std::vector<int>>& rows_color,
    const std::vector<double>& D_color,
    int N,
    const std::vector<std::vector<int>>& Cons,
    bool log,
    double time_limit_seconds,
    int search_workers) {
    using operations_research::sat::BoolVar;
    using operations_research::sat::CpModelBuilder;
    using operations_research::sat::CpSolverResponse;
    using operations_research::sat::LinearExpr;
    using operations_research::sat::Model;
    using operations_research::sat::NewSatParameters;
    using operations_research::sat::SatParameters;
    using operations_research::sat::SolutionBooleanValue;
    using operations_research::sat::SolveCpModel;

    if (rows_color.size() != D_color.size()) {
        throw std::invalid_argument(
            "rows_color and D_color must have the same size");
    }
    if (N < 0) {
        throw std::invalid_argument("N must be non-negative");
    }

    const int variable_count = static_cast<int>(rows_color.size());
    CpModelBuilder builder;
    std::vector<BoolVar> x;
    x.reserve(variable_count);
    for (int row = 0; row < variable_count; ++row) {
        x.push_back(builder.NewBoolVar().WithName(
            "x[" + std::to_string(row) + "]"));
    }

    // Each DLX coverage/conflict column becomes an at-most-one constraint.
    for (const auto& constraint : Cons) {
        LinearExpr expression;
        for (int row : constraint) {
            if (row < 0 || row >= variable_count) {
                throw std::out_of_range(
                    "Cons contains a row index outside rows_color");
            }
            expression += x[row];
        }
        builder.AddLessOrEqual(expression, 1);
    }

    LinearExpr cardinality;
    LinearExpr objective;
    for (int row = 0; row < variable_count; ++row) {
        cardinality += x[row];
        objective +=
            static_cast<std::int64_t>(D_color[row]) * x[row];
    }
    // Paper model: select at most N rows and maximise allocated demand.
    builder.AddLessOrEqual(cardinality, N);
    builder.Maximize(objective);

    SatParameters parameters;
    parameters.set_max_time_in_seconds(time_limit_seconds);
    parameters.set_log_search_progress(log);
    if (search_workers > 0) {
        parameters.set_num_search_workers(search_workers);
    }

    Model model;
    model.Add(NewSatParameters(parameters));

    // Measure the native CP-SAT solve used in the reported solver runtimes.
    const auto start = std::chrono::steady_clock::now();
    const CpSolverResponse response =
        SolveCpModel(builder.Build(), &model);
    const double runtime = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();

    CpSatResult result;
    result.runtime = runtime;
    if (response.status() !=
            operations_research::sat::CpSolverStatus::OPTIMAL &&
        response.status() !=
            operations_research::sat::CpSolverStatus::FEASIBLE) {
        return result;
    }
    result.has_solution = true;
    result.objective = response.objective_value();
    for (int row = 0; row < variable_count; ++row) {
        if (SolutionBooleanValue(response, x[row])) {
            result.integer.push_back(row);
        }
    }
    return result;
}
