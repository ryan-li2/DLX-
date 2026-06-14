#include "gurobi_spk.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#include "gurobi_c++.h"

std::vector<std::vector<int>> incidence_transpose_fast(
    const std::vector<std::vector<int>>& rows_color) {
    int maximum_column = -1;
    for (const auto& row : rows_color) {
        for (int column : row) {
            if (column < 0) {
                throw std::invalid_argument(
                    "rows_color contains a negative column index");
            }
            maximum_column = std::max(maximum_column, column);
        }
    }

    std::vector<std::vector<int>> constraints(maximum_column + 1);
    for (int row = 0; row < static_cast<int>(rows_color.size()); ++row) {
        for (int column : rows_color[row]) {
            constraints[column].push_back(row);
        }
    }
    return constraints;
}

GurobiResult gurobi_SPK(
    const std::vector<std::vector<int>>& rows_color,
    const std::vector<double>& D_color,
    int N,
    const std::vector<std::vector<int>>& Cons,
    double time_limit_seconds) {
    if (rows_color.size() != D_color.size()) {
        throw std::invalid_argument(
            "rows_color and D_color must have the same size");
    }
    if (N < 0) {
        throw std::invalid_argument("N must be non-negative");
    }

    GRBEnv environment(true);
    environment.set(GRB_IntParam_OutputFlag, 1);
    environment.start();

    GRBModel model(environment);
    model.set(GRB_StringAttr_ModelName, "SPK");
    model.set(GRB_DoubleParam_TimeLimit, time_limit_seconds);

    const int variable_count = static_cast<int>(rows_color.size());
    std::vector<GRBVar> x;
    x.reserve(variable_count);
    for (int row = 0; row < variable_count; ++row) {
        x.push_back(model.addVar(
            0.0,
            1.0,
            0.0,
            GRB_BINARY,
            "x[" + std::to_string(row) + "]"));
    }

    for (int constraint = 0;
         constraint < static_cast<int>(Cons.size());
         ++constraint) {
        GRBLinExpr expression = 0.0;
        for (int row : Cons[constraint]) {
            if (row < 0 || row >= variable_count) {
                throw std::out_of_range(
                    "Cons contains a row index outside rows_color");
            }
            expression += x[row];
        }
        model.addConstr(
            expression <= 1.0,
            "spk[" + std::to_string(constraint) + "]");
    }

    GRBLinExpr cardinality = 0.0;
    GRBLinExpr objective = 0.0;
    for (int row = 0; row < variable_count; ++row) {
        cardinality += x[row];
        objective += D_color[row] * x[row];
    }
    model.addConstr(cardinality <= N, "cardinality");
    model.setObjective(objective, GRB_MAXIMIZE);
    model.optimize();

    GurobiResult result;
    result.runtime = model.get(GRB_DoubleAttr_Runtime);

    const int solution_count = model.get(GRB_IntAttr_SolCount);
    if (solution_count == 0) {
        return result;
    }

    result.has_solution = true;
    result.objective = model.get(GRB_DoubleAttr_ObjVal);
    for (int row = 0; row < variable_count; ++row) {
        if (x[row].get(GRB_DoubleAttr_X) > 0.5) {
            result.integer.push_back(row);
        }
    }
    return result;
}

GurobiLpResult gurobi_LP(
    const std::vector<std::vector<int>>& rows_color,
    const std::vector<double>& D_color,
    int N,
    const std::vector<std::vector<int>>& Cons,
    double time_limit_seconds) {
    const auto start = std::chrono::steady_clock::now();
    if (rows_color.size() != D_color.size()) {
        throw std::invalid_argument(
            "rows_color and D_color must have the same size");
    }

    GRBEnv environment(true);
    environment.set(GRB_IntParam_OutputFlag, 0);
    environment.start();
    GRBModel model(environment);
    model.set(GRB_StringAttr_ModelName, "SPK_LP");
    model.set(GRB_DoubleParam_TimeLimit, time_limit_seconds);

    const int variable_count = static_cast<int>(rows_color.size());
    std::vector<GRBVar> x;
    x.reserve(variable_count);
    for (int row = 0; row < variable_count; ++row) {
        x.push_back(model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS,
                                 "x[" + std::to_string(row) + "]"));
    }

    for (const auto& constraint : Cons) {
        GRBLinExpr expression = 0.0;
        for (int row : constraint) {
            expression += x[row];
        }
        model.addConstr(expression <= 1.0);
    }

    GRBLinExpr cardinality = 0.0;
    GRBLinExpr objective = 0.0;
    for (int row = 0; row < variable_count; ++row) {
        cardinality += x[row];
        objective += D_color[row] * x[row];
    }
    model.addConstr(cardinality <= N);
    model.setObjective(objective, GRB_MAXIMIZE);
    model.optimize();

    GurobiLpResult result;
    if (model.get(GRB_IntAttr_SolCount) > 0) {
        result.objective = model.get(GRB_DoubleAttr_ObjVal);
        for (int row = 0; row < variable_count; ++row) {
            const double value =
                std::round(x[row].get(GRB_DoubleAttr_X) * 100.0) / 100.0;
            if (value > 0.99) {
                result.integer.push_back(row);
                result.integer_demand += D_color[row];
            } else if (value > 1e-6) {
                result.fraction.push_back(row);
            }
        }
    }
    result.runtime = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}
