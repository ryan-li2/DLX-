#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef USE_GUROBI
#include "gurobi_spk.hpp"
#endif
#ifdef USE_CPSAT
#include "cpsat_spk.hpp"
#endif
#ifdef USE_LIGHTGBM
#include "lightgbm_predict.hpp"
#endif

/*
Implementation map to the manuscript:
  - "DLX+: A DLX Extension for CC-MWIS": matrix construction and the search
    procedures specified by Algorithms 1-3.
  - "Bounding Strategies" and "Backtracking Strategies": pruning and truncated
    search-depth control.
  - "Machine Learning Backtracking" and "Feature Design": feature extraction
    and conversion of a predicted percentile class into one search depth.
  - "LP-Guided Problem Reduction": variable fixing and residual DLX+ search.
  - "Parallelisation Strategies": decomposition across starting vertex rows.

Subsection names are used instead of section numbers so these references remain
valid if the manuscript is reordered.
*/

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr int kNMax = 60;
constexpr int kColours = 4;
constexpr std::array<double, 7> kPercentiles{
    0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99};

struct InputData {
    std::vector<std::vector<int>> rows;
    std::vector<std::vector<int>> conflicts;
    std::vector<std::vector<int>> colour_cliques;
    std::vector<std::pair<int, int>> original_rows;
    std::vector<double> demands;
    std::unordered_map<int, int> sorted_position;
    std::unordered_map<int, std::unordered_set<int>> adjacency;
    std::vector<int> backtracking_depths;
    int pair_count = 0;
    int beam_count = 0;
    int column_count = 0;
};

struct Features {
    double B = 0.0;
    double U = 0.0;
    double num_rows = 0.0;
    double density = 0.0;
    double avg_degree = 0.0;
    double max_degree = 0.0;
    double degree_std = 0.0;
    double D_mean = 0.0;
    double D_std = 0.0;
    double D_cv = 0.0;
    double avg_row_len = 0.0;
    double max_row_len = 0.0;
    double row_density = 0.0;
    double conflict_max = 0.0;
    double conflict_avg = 0.0;
    double conflict_std = 0.0;
    double conflict_pressure = 0.0;
    double color_clique_max = 0.0;
    double color_clique_avg = 0.0;
    double color_clique_std = 0.0;
    double color_pressure = 0.0;
};

struct WorkerResult {
    std::array<double, kPercentiles.size()> times{};
    std::array<double, kPercentiles.size()> values{};
    std::array<std::vector<int>, kPercentiles.size()> integers;
    double best_value = 0.0;
    std::vector<int> best_integer;
    int backtrack_depth = -1;
};

struct Summary {
    int percentile_index = 0;
    double time = 0.0;
    double value = 0.0;
    std::vector<int> integers;
};

static void write_int_vector_json(std::ostream& output,
                                  const std::vector<int>& values);

static std::string read_file(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open " + path.string());
    }
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

static std::vector<double> read_demands(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open " + path.string());
    }

    std::string line;
    std::getline(input, line);
    std::vector<std::string> headers;
    std::stringstream header_stream(line);
    while (std::getline(header_stream, line, ',')) {
        headers.push_back(line);
    }

    auto demand_it = std::find(headers.begin(), headers.end(), "Demand");
    if (demand_it == headers.end()) {
        throw std::runtime_error("Demand column missing from " + path.string());
    }
    const std::size_t demand_column =
        static_cast<std::size_t>(std::distance(headers.begin(), demand_it));

    std::vector<double> demands;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream row_stream(line);
        std::string cell;
        std::size_t column = 0;
        while (std::getline(row_stream, cell, ',')) {
            if (column++ == demand_column) {
                demands.push_back(std::stod(cell));
                break;
            }
        }
    }
    return demands;
}

static std::unordered_map<int, std::vector<int>> read_allocations(
    const fs::path& path) {
    const std::string text = read_file(path);
    const std::regex entry_pattern(R"((\d+)\s*:\s*\[([^\]]*)\])");
    const std::regex integer_pattern(R"(\d+)");
    std::unordered_map<int, std::vector<int>> allocations;

    for (auto it = std::sregex_iterator(text.begin(), text.end(), entry_pattern);
         it != std::sregex_iterator(); ++it) {
        const int beam = std::stoi((*it)[1].str());
        const std::string users_text = (*it)[2].str();
        auto& users = allocations[beam];
        for (auto number = std::sregex_iterator(
                 users_text.begin(), users_text.end(), integer_pattern);
             number != std::sregex_iterator(); ++number) {
            users.push_back(std::stoi(number->str()));
        }
    }
    return allocations;
}

static std::vector<std::pair<int, int>> read_pairs(const fs::path& path) {
    const std::string text = read_file(path);
    const std::regex pair_pattern(R"(\[\s*(\d+)\s*,\s*(\d+)\s*\])");
    std::vector<std::pair<int, int>> pairs;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), pair_pattern);
         it != std::sregex_iterator(); ++it) {
        pairs.emplace_back(std::stoi((*it)[1].str()),
                           std::stoi((*it)[2].str()));
    }
    return pairs;
}

static std::vector<std::vector<int>> remove_sublists(
    const std::vector<std::vector<int>>& lists) {
    std::vector<std::set<int>> unique_sets;
    std::set<std::set<int>> seen;
    for (const auto& list : lists) {
        std::set<int> values(list.begin(), list.end());
        if (seen.insert(values).second) {
            unique_sets.push_back(std::move(values));
        }
    }
    std::stable_sort(unique_sets.begin(), unique_sets.end(),
                     [](const auto& left, const auto& right) {
                         return left.size() > right.size();
                     });

    std::vector<std::set<int>> retained;
    for (const auto& candidate : unique_sets) {
        const bool subset = std::any_of(
            retained.begin(), retained.end(), [&](const auto& existing) {
                return std::includes(existing.begin(), existing.end(),
                                     candidate.begin(), candidate.end());
            });
        if (!subset) {
            retained.push_back(candidate);
        }
    }

    std::vector<std::vector<int>> result;
    for (const auto& values : retained) {
        result.emplace_back(values.begin(), values.end());
    }
    return result;
}

static std::vector<std::vector<int>> incidence_transpose(
    const std::vector<std::vector<int>>& rows, int beam_count) {
    std::vector<std::vector<int>> result(beam_count);
    for (int constraint = 0; constraint < static_cast<int>(rows.size());
         ++constraint) {
        for (int beam : rows[constraint]) {
            if (beam >= 0 && beam < beam_count) {
                result[beam].push_back(constraint);
            }
        }
    }
    return result;
}

static std::vector<int> intersect_sorted(const std::vector<int>& left,
                                         const std::vector<int>& right) {
    std::vector<int> result;
    std::set_intersection(left.begin(), left.end(), right.begin(), right.end(),
                          std::back_inserter(result));
    return result;
}

static std::vector<int> difference_sorted(const std::vector<int>& left,
                                          const std::vector<int>& right) {
    std::vector<int> result;
    std::set_difference(left.begin(), left.end(), right.begin(), right.end(),
                        std::back_inserter(result));
    return result;
}

static void maximal_cliques_recursive(
    std::vector<int> current, std::vector<int> candidates,
    std::vector<int> excluded, const std::vector<std::vector<int>>& neighbours,
    std::vector<std::vector<int>>& cliques) {
    if (candidates.empty() && excluded.empty()) {
        cliques.push_back(std::move(current));
        return;
    }

    int pivot = -1;
    std::size_t best_intersection = 0;
    std::vector<int> pivot_pool = candidates;
    pivot_pool.insert(pivot_pool.end(), excluded.begin(), excluded.end());
    for (int vertex : pivot_pool) {
        const auto overlap = intersect_sorted(candidates, neighbours[vertex]);
        if (pivot == -1 || overlap.size() > best_intersection) {
            pivot = vertex;
            best_intersection = overlap.size();
        }
    }

    const std::vector<int> branch_vertices =
        pivot == -1 ? candidates
                    : difference_sorted(candidates, neighbours[pivot]);
    for (int vertex : branch_vertices) {
        auto next_current = current;
        next_current.push_back(vertex);
        maximal_cliques_recursive(
            std::move(next_current),
            intersect_sorted(candidates, neighbours[vertex]),
            intersect_sorted(excluded, neighbours[vertex]), neighbours,
            cliques);
        candidates.erase(
            std::lower_bound(candidates.begin(), candidates.end(), vertex));
        excluded.insert(
            std::lower_bound(excluded.begin(), excluded.end(), vertex),
            vertex);
    }
}

static std::vector<std::vector<int>> maximal_cliques(
    int beam_count, const std::vector<std::pair<int, int>>& pairs) {
    std::vector<std::vector<int>> neighbours(beam_count);
    std::set<int> graph_vertices;
    for (const auto& [left, right] : pairs) {
        if (left < 0 || right < 0 || left >= beam_count || right >= beam_count) {
            continue;
        }
        neighbours[left].push_back(right);
        neighbours[right].push_back(left);
        graph_vertices.insert(left);
        graph_vertices.insert(right);
    }
    for (auto& list : neighbours) {
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
    }
    std::vector<int> candidates(graph_vertices.begin(), graph_vertices.end());
    std::vector<std::vector<int>> result;
    maximal_cliques_recursive({}, std::move(candidates), {}, neighbours, result);
    return result;
}

static InputData build_input(const std::vector<double>& beam_demands,
                             const std::unordered_map<int, std::vector<int>>& allocations,
                             const std::vector<std::pair<int, int>>& pairs,
                             int user_count) {
    // Paper reference: "DLX+: A DLX Extension for CC-MWIS" - Data Structure
    // and Model 2. Convert coverage and colouring conflicts into DLX+ rows.
    const int beam_count = static_cast<int>(beam_demands.size());
    std::vector<std::vector<int>> beams_by_user(user_count);
    for (const auto& [beam, users] : allocations) {
        for (int user : users) {
            if (user >= 0 && user < user_count) {
                beams_by_user[user].push_back(beam);
            }
        }
    }

    const auto constraints = remove_sublists(beams_by_user);
    const auto rows_by_beam = incidence_transpose(constraints, beam_count);

    InputData result;
    result.conflicts = constraints;
    result.pair_count = static_cast<int>(pairs.size());
    result.beam_count = beam_count;
    for (const auto& [left, right] : pairs) {
        result.adjacency[left].insert(right);
        result.adjacency[right].insert(left);
    }

    // Paper reference: "DLX+: A DLX Extension for CC-MWIS" - Data Structure.
    // Sort rows by non-increasing weight to obtain strong incumbents early.
    std::vector<int> order(beam_count);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
        return beam_demands[left] > beam_demands[right];
    });
    for (int index = 0; index < beam_count; ++index) {
        result.sorted_position[order[index]] = index;
    }

    // Paper reference: "Backtracking Strategies" - Backtracking. The seven
    // depth limits are obtained from cumulative weight percentiles.
    std::vector<double> sorted_demands;
    sorted_demands.reserve(beam_count);
    for (int beam : order) {
        sorted_demands.push_back(beam_demands[beam]);
    }
    std::vector<double> cumulative(sorted_demands.size());
    std::partial_sum(sorted_demands.begin(), sorted_demands.end(),
                     cumulative.begin());
    const double total = cumulative.empty() ? 0.0 : cumulative.back();
    for (double percentile : kPercentiles) {
        const double target = percentile * total;
        auto nearest = std::min_element(
            cumulative.begin(), cumulative.end(), [&](double left, double right) {
                return std::abs(left - target) < std::abs(right - target);
            });
        const int index =
            nearest == cumulative.end()
                ? 0
                : static_cast<int>(std::distance(cumulative.begin(), nearest));
        result.backtracking_depths.push_back(index * kColours + 1);
    }

    // One row represents selecting one beam with one colour. Coverage columns
    // prevent incompatible selections; colour-clique columns added below
    // prevent adjacent selected beams from receiving the same colour.
    result.rows.reserve(beam_count * kColours);
    result.original_rows.reserve(beam_count * kColours);
    result.demands.reserve(beam_count * kColours);
    for (int sorted_index = 0; sorted_index < beam_count; ++sorted_index) {
        const int beam = order[sorted_index];
        for (int colour = 0; colour < kColours; ++colour) {
            result.rows.push_back(rows_by_beam[beam]);
            result.original_rows.emplace_back(sorted_index, colour);
            result.demands.push_back(beam_demands[beam]);
        }
    }

    result.colour_cliques = maximal_cliques(beam_count, pairs);
    int next_column = user_count;
    for (const auto& clique : result.colour_cliques) {
        for (int colour = 0; colour < kColours; ++colour) {
            for (int beam : clique) {
                auto position = result.sorted_position.find(beam);
                if (position != result.sorted_position.end()) {
                    result.rows[position->second * kColours + colour].push_back(
                        next_column);
                }
            }
            ++next_column;
        }
    }
    result.column_count = next_column + 1;
    return result;
}

static std::pair<double, double> mean_and_population_std(
    const std::vector<double>& values) {
    if (values.empty()) {
        return {0.0, 0.0};
    }
    const double mean =
        std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double squared_difference_sum = 0.0;
    for (double value : values) {
        const double difference = value - mean;
        squared_difference_sum += difference * difference;
    }
    return {mean, std::sqrt(squared_difference_sum / values.size())};
}

static Features extract_features_FBO(const InputData& input) {
    // Paper reference: "Machine Learning Backtracking" - Feature Design. These
    // graph, demand, and DLX-matrix statistics form the LightGBM input.
    Features features;
    features.B = input.beam_count;
    features.U = input.column_count - 1;
    features.num_rows = input.rows.size();

    std::vector<double> degrees(input.beam_count, 0.0);
    for (const auto& [beam, neighbours] : input.adjacency) {
        if (beam >= 0 && beam < input.beam_count) {
            degrees[beam] = neighbours.size();
        }
    }
    const auto [degree_mean, degree_std] =
        mean_and_population_std(degrees);
    features.avg_degree = degree_mean;
    features.degree_std = degree_std;
    features.max_degree = degrees.empty()
                              ? 0.0
                              : *std::max_element(degrees.begin(),
                                                  degrees.end());
    if (input.beam_count > 1) {
        features.density =
            2.0 * input.pair_count /
            (input.beam_count * (input.beam_count - 1.0));
    }

    const auto [demand_mean, demand_std] =
        mean_and_population_std(input.demands);
    features.D_mean = demand_mean;
    features.D_std = demand_std;
    features.D_cv = demand_mean > 0.0 ? demand_std / demand_mean : 0.0;

    std::vector<double> row_lengths;
    row_lengths.reserve(input.rows.size());
    for (const auto& row : input.rows) {
        row_lengths.push_back(row.size());
    }
    const auto [row_mean, unused_row_std] =
        mean_and_population_std(row_lengths);
    (void)unused_row_std;
    features.avg_row_len = row_mean;
    features.max_row_len =
        row_lengths.empty()
            ? 0.0
            : *std::max_element(row_lengths.begin(), row_lengths.end());
    features.row_density =
        features.U > 0.0 ? row_mean / features.U : 0.0;

    std::vector<double> conflict_sizes;
    conflict_sizes.reserve(input.conflicts.size());
    for (const auto& conflict : input.conflicts) {
        conflict_sizes.push_back(conflict.size());
    }
    const auto [conflict_mean, conflict_std] =
        mean_and_population_std(conflict_sizes);
    features.conflict_avg = conflict_mean;
    features.conflict_std = conflict_std;
    features.conflict_max =
        conflict_sizes.empty()
            ? 0.0
            : *std::max_element(conflict_sizes.begin(),
                                conflict_sizes.end());
    features.conflict_pressure =
        kNMax > 0 ? features.conflict_max / kNMax : 0.0;

    std::vector<double> clique_sizes;
    clique_sizes.reserve(input.colour_cliques.size());
    for (const auto& clique : input.colour_cliques) {
        clique_sizes.push_back(clique.size());
    }
    const auto [clique_mean, clique_std] =
        mean_and_population_std(clique_sizes);
    features.color_clique_avg = clique_mean;
    features.color_clique_std = clique_std;
    features.color_clique_max =
        clique_sizes.empty()
            ? 0.0
            : *std::max_element(clique_sizes.begin(), clique_sizes.end());
    features.color_pressure = features.color_clique_max / kColours;
    return features;
}

static std::vector<double> ml_feature_vector(const Features& features) {
    return {
        features.B,
        features.U,
        features.num_rows,
        features.density,
        features.avg_degree,
        features.max_degree,
        features.degree_std,
        features.D_mean,
        features.D_std,
        features.D_cv,
        features.avg_row_len,
        features.max_row_len,
        features.row_density,
    };
}

static void write_features_json(const fs::path& path,
                                const Features& features) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot create " + path.string());
    }
    output << std::setprecision(15)
           << "{\"B\":" << features.B
           << ",\"U\":" << features.U
           << ",\"num_rows\":" << features.num_rows
           << ",\"density\":" << features.density
           << ",\"avg_degree\":" << features.avg_degree
           << ",\"max_degree\":" << features.max_degree
           << ",\"degree_std\":" << features.degree_std
           << ",\"D_mean\":" << features.D_mean
           << ",\"D_std\":" << features.D_std
           << ",\"D_cv\":" << features.D_cv
           << ",\"avg_row_len\":" << features.avg_row_len
           << ",\"max_row_len\":" << features.max_row_len
           << ",\"row_density\":" << features.row_density
           << ",\"conflict_max\":" << features.conflict_max
           << ",\"conflict_avg\":" << features.conflict_avg
           << ",\"conflict_std\":" << features.conflict_std
           << ",\"conflict_pressure\":" << features.conflict_pressure
           << ",\"color_clique_max\":" << features.color_clique_max
           << ",\"color_clique_avg\":" << features.color_clique_avg
           << ",\"color_clique_std\":" << features.color_clique_std
           << ",\"color_pressure\":" << features.color_pressure << '}';
}

class DLX {
  private:
    struct Node {
        Node* left = this;
        Node* right = this;
        Node* up = this;
        Node* down = this;
        Node* column = nullptr;
        Node* row_header = nullptr;
        int row_id = -1;
        int name = -1;
        int colour = -1;
        bool is_row_header = false;
        bool is_column_header = false;
    };

    Node root_;
    std::vector<Node> columns_;
    std::vector<Node> row_headers_;
    std::vector<std::unique_ptr<Node>> nodes_;
    const std::vector<double>& demands_;
    double lower_bound_;
    int max_size_;
    int backtrack_nodes_;
    int solution_size_ = 0;
    double current_ = 0.0;
    std::vector<int> solution_;
    std::vector<int> best_solution_;

    void cover(Node* column) {
        // Paper reference: Algorithm 1 (DLX+: Main Loop), CoverConstraints.
        // Remove rows conflicting with the selected row from the active matrix.
        for (Node* row = column->down; row != column; row = row->down) {
            for (Node* node = row->right; node != row; node = node->right) {
                node->down->up = node->up;
                node->up->down = node->down;
            }
        }
    }

    void uncover(Node* column) {
        // Restore links in reverse traversal order during backtracking.
        for (Node* row = column->up; row != column; row = row->up) {
            for (Node* node = row->left; node != row; node = node->left) {
                node->down->up = node;
                node->up->down = node;
            }
        }
    }

    double upper_bound(Node* node) const {
        // Paper reference: "Bounding Strategies" and Algorithm 2, line 12.
        // Add weights of distinct remaining vertices up to the size limit.
        double bound = demands_[node->row_id];
        int colour = node->row_header->colour;
        Node* row = node->row_header->down;
        const int remaining = max_size_ - solution_size_;
        int count = 1;
        while (row != &root_ && count < remaining) {
            if (row->colour > colour) {
                bound += demands_[row->name];
                colour = row->colour;
                ++count;
            }
            row = row->down;
        }
        return bound;
    }

    void select_row(Node* node) {
        cover(node->column);
        solution_.push_back(node->row_header->name);
        ++solution_size_;
        current_ += demands_[node->row_header->name];
        for (Node* next = node->right; next != node; next = next->right) {
            if (!next->is_row_header) {
                cover(next->column);
            }
        }
    }

    void restore_row(Node* node) {
        for (Node* previous = node->left; previous != node;
             previous = previous->left) {
            if (!previous->is_row_header) {
                uncover(previous->column);
            }
        }
        current_ -= demands_[solution_.back()];
        solution_.pop_back();
        --solution_size_;
    }

    void search() {
        // Paper reference: Algorithm 2 (Search). Continue only when the upper
        // bound can improve the incumbent.
        Node* row = root_.down;
        if (row == &root_ || solution_size_ >= max_size_) {
            if (current_ > lower_bound_) {
                lower_bound_ = current_;
                best_solution_ = solution_;
            }
            return;
        }

        const int last_name = solution_.back();
        while (row != &root_ && row->name < last_name) {
            row = row->down;
        }
        if (row == &root_) {
            return;
        }

        Node* best = row->right;
        if (upper_bound(best) + current_ > lower_bound_) {
            select_row(best);
            search();
            restore_row(best);
            uncover(best->column);
        }
    }

    void backtrack_next_node(Node* node) {
        // Paper reference: Algorithm 3 (Backtracking). Examine the next row for
        // a different vertex and prune it using the upper bound.
        const int colour = node->row_header->colour;
        Node* alternative = node->down;
        while (!alternative->is_column_header) {
            if (alternative->row_header->colour != colour &&
                upper_bound(alternative) + current_ > lower_bound_) {
                select_row(alternative);
                search();
                restore_row(alternative);
                uncover(alternative->column);
                break;
            }
            alternative = alternative->down;
        }
        uncover(node->column);
    }

    void search_range() {
        // Paper reference: Algorithm 1 and "Backtracking Strategies". Stop at
        // backtrack_nodes_ to control the runtime-quality trade-off.
        Node* row = root_.down;
        if (row == &root_) {
            return;
        }
        if (solution_size_ < backtrack_nodes_ && solution_size_ < max_size_) {
            Node* best = row->right;
            select_row(best);
            search_range();
            restore_row(best);
            backtrack_next_node(best);
        }
    }

  public:
    DLX(int column_count, const std::vector<std::vector<int>>& rows,
        const std::vector<std::pair<int, int>>& row_indices,
        const std::vector<double>& demands, double lower_bound, int max_size,
        int backtrack_nodes)
        : columns_(column_count),
          row_headers_(rows.size()),
          demands_(demands),
          lower_bound_(lower_bound),
          max_size_(max_size),
          backtrack_nodes_(backtrack_nodes) {
        root_.is_column_header = true;
        root_.name = std::numeric_limits<int>::max();

        Node* last_column = &root_;
        for (int index = 0; index < column_count; ++index) {
            Node* column = &columns_[index];
            column->is_column_header = true;
            column->name = index;
            column->left = last_column;
            column->right = last_column->right;
            last_column->right->left = column;
            last_column->right = column;
            last_column = column;
        }

        Node* last_row = &root_;
        for (int row_index = 0; row_index < static_cast<int>(rows.size());
             ++row_index) {
            Node* row_header = &row_headers_[row_index];
            row_header->is_row_header = true;
            row_header->name = row_index;
            row_header->colour = row_indices[row_index].first;
            row_header->column = row_header;
            row_header->up = last_row;
            row_header->down = last_row->down;
            last_row->down->up = row_header;
            last_row->down = row_header;
            last_row = row_header;

            Node* first = nullptr;
            for (int column_index : rows[row_index]) {
                if (column_index < 0 || column_index >= column_count) {
                    throw std::runtime_error("Constraint column out of range");
                }
                auto owned_node = std::make_unique<Node>();
                Node* node = owned_node.get();
                nodes_.push_back(std::move(owned_node));
                Node* column = &columns_[column_index];
                node->column = column;
                node->row_id = row_index;
                node->row_header = row_header;
                node->up = column->up;
                node->down = column;
                column->up->down = node;
                column->up = node;

                if (first == nullptr) {
                    first = node;
                } else {
                    node->left = first->left;
                    node->right = first;
                    first->left->right = node;
                    first->left = node;
                }
            }

            if (first != nullptr) {
                row_header->right = first;
                row_header->left = first->left;
                first->left->right = row_header;
                first->left = row_header;
            }
        }
        root_.up = last_row;
        last_row->down = &root_;
    }

    std::pair<double, std::vector<int>> solve() {
        search_range();
        return {lower_bound_, best_solution_};
    }
};

static std::pair<double, std::vector<int>> greedy_solution(
    const InputData& input) {
    // Paper reference: "DLX+: A DLX Extension for CC-MWIS" - Data Structure.
    // Build the greedy lower bound enabled by non-increasing row weights.
    std::unordered_set<int> used_constraints;
    std::unordered_set<int> selected_beams;
    std::unordered_map<int, int> beam_colours;
    double lower_bound = 0.0;
    int selected_count = 0;
    std::vector<int> integer;

    for (int row_index = 0; row_index < static_cast<int>(input.rows.size());
         ++row_index) {
        const auto [beam_index, colour] = input.original_rows[row_index];
        if (selected_beams.count(beam_index)) {
            continue;
        }
        bool overlaps = false;
        for (int constraint : input.rows[row_index]) {
            if (used_constraints.count(constraint)) {
                overlaps = true;
                break;
            }
        }
        if (overlaps) {
            continue;
        }

        bool colour_conflict = false;
        int original_beam = -1;
        for (const auto& [beam, position] : input.sorted_position) {
            if (position == beam_index) {
                original_beam = beam;
                break;
            }
        }
        auto adjacent = input.adjacency.find(original_beam);
        if (adjacent != input.adjacency.end()) {
            for (int neighbour : adjacent->second) {
                auto position = input.sorted_position.find(neighbour);
                if (position != input.sorted_position.end() &&
                    selected_beams.count(position->second) &&
                    beam_colours[position->second] == colour) {
                    colour_conflict = true;
                    break;
                }
            }
        }
        if (colour_conflict) {
            continue;
        }

        used_constraints.insert(input.rows[row_index].begin(),
                                input.rows[row_index].end());
        selected_beams.insert(beam_index);
        beam_colours[beam_index] = colour;
        lower_bound += input.demands[row_index];
        integer.push_back(row_index);
        if (++selected_count == kNMax) {
            break;
        }
    }
    return {lower_bound, integer};
}

static WorkerResult run_worker(int worker, const InputData& input,
                               double lower_bound,
                               const std::vector<int>& greedy_integer,
                               int max_size = kNMax) {
    // Paper reference: "Parallelisation Strategies". Thread i starts from the
    // first colour row of vertex i and excludes all preceding rows.
    const int begin = worker * kColours;
    WorkerResult result;
    double best = lower_bound;
    std::vector<int> best_integer = greedy_integer;
    if (begin >= static_cast<int>(input.rows.size())) {
        result.values.fill(best);
        result.integers.fill(best_integer);
        result.best_value = best;
        result.best_integer = best_integer;
        return result;
    }

    std::vector<std::vector<int>> rows(input.rows.begin() + begin,
                                       input.rows.end());
    std::vector<std::pair<int, int>> original_rows(
        input.original_rows.begin() + begin, input.original_rows.end());
    std::vector<double> demands(input.demands.begin() + begin,
                                input.demands.end());

    // Evaluate seven depths sequentially per worker; workers run concurrently.
    for (std::size_t percentile = 0; percentile < kPercentiles.size();
         ++percentile) {
        const int depth = input.backtracking_depths[percentile] - begin;
        const auto start = Clock::now();
        if (depth > 0) {
            DLX solver(input.column_count, rows, original_rows, demands,
                       best, max_size, depth);
            auto [value, integer] = solver.solve();
            if (value > best) {
                best = value;
                best_integer = std::move(integer);
                for (int& row : best_integer) {
                    row += begin;
                }
                result.backtrack_depth =
                    static_cast<int>(percentile);
            }
        }
        result.times[percentile] =
            std::chrono::duration<double>(Clock::now() - start).count();
        result.values[percentile] = best;
        result.integers[percentile] = best_integer;
    }
    result.best_value = best;
    result.best_integer = best_integer;
    return result;
}

static std::pair<double, std::vector<int>> run_single_depth_worker(
    int worker, const InputData& input, int backtrack_depth,
    double lower_bound, const std::vector<int>& greedy_integer) {
    // Paper reference: "Machine Learning Backtracking" - Learning Model.
    // DLX+:ML evaluates only the depth selected by the seven-class classifier.
    const int begin = worker * kColours;
    if (begin >= static_cast<int>(input.rows.size())) {
        return {lower_bound, greedy_integer};
    }

    const int depth = backtrack_depth - begin;
    if (depth <= 0) {
        return {lower_bound, greedy_integer};
    }

    std::vector<std::vector<int>> rows(input.rows.begin() + begin,
                                       input.rows.end());
    std::vector<std::pair<int, int>> original_rows(
        input.original_rows.begin() + begin, input.original_rows.end());
    std::vector<double> demands(input.demands.begin() + begin,
                                input.demands.end());

    DLX solver(input.column_count, rows, original_rows, demands,
               lower_bound, kNMax, depth);
    auto [value, integer] = solver.solve();
    if (value <= lower_bound) {
        return {lower_bound, greedy_integer};
    }
    for (int& row : integer) {
        row += begin;
    }
    return {value, std::move(integer)};
}

static int predicted_backtracking_depth(const InputData& input,
                                        int percentile_index,
                                        double lower_bound) {
    // Paper reference: "Machine Learning Backtracking" - Learning Model.
    // Convert the predicted percentile class to a cumulative-demand depth.
    if (percentile_index < 0 ||
        percentile_index >= static_cast<int>(kPercentiles.size())) {
        throw std::out_of_range(
            "Predicted percentile index must be between 0 and 6");
    }

    std::vector<double> beam_demands;
    beam_demands.reserve(input.beam_count);
    for (int row = 0; row < static_cast<int>(input.demands.size());
         row += kColours) {
        beam_demands.push_back(input.demands[row]);
    }
    std::vector<double> cumulative(beam_demands.size());
    std::partial_sum(beam_demands.begin(), beam_demands.end(),
                     cumulative.begin());

    const double target =
        kPercentiles[percentile_index] * lower_bound;
    const auto nearest = std::min_element(
        cumulative.begin(), cumulative.end(),
        [&](double left, double right) {
            return std::abs(left - target) < std::abs(right - target);
        });
    return nearest == cumulative.end()
               ? 1
               : static_cast<int>(
                     std::distance(cumulative.begin(), nearest)) +
                     1;
}

static void write_ml_result_json(
    const fs::path& path, double value, double computation_time,
    int percentile_index, int backtracking_depth,
    const std::vector<int>& integer) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot create " + path.string());
    }
    output << std::setprecision(15)
           << "{\"value\":" << value
           << ",\"time\":" << computation_time
           << ",\"percentile_index\":" << percentile_index
           << ",\"backtracking_depth\":" << backtracking_depth
           << ",\"integer\":";
    write_int_vector_json(output, integer);
    output << '}';
}

#ifdef USE_GUROBI
static InputData reduce_after_fixed_rows(
    const InputData& input, const std::vector<int>& fixed_rows) {
    // Paper reference: "LP-Guided Problem Reduction", Step 1. Fix integral LP
    // rows and remove every residual row sharing a covered DLX column. The
    // surviving rows retain the original DLX+ sparse-matrix structure.
    std::unordered_set<int> covered_columns;
    std::unordered_set<int> fixed(fixed_rows.begin(), fixed_rows.end());
    for (int row : fixed_rows) {
        if (row < 0 || row >= static_cast<int>(input.rows.size())) {
            throw std::out_of_range("LP fixed row is outside rows_color");
        }
        covered_columns.insert(input.rows[row].begin(),
                               input.rows[row].end());
    }

    InputData reduced;
    reduced.column_count = input.column_count;
    for (int row = 0; row < static_cast<int>(input.rows.size()); ++row) {
        if (fixed.count(row)) {
            continue;
        }
        bool conflicts = false;
        for (int column : input.rows[row]) {
            if (covered_columns.count(column)) {
                conflicts = true;
                break;
            }
        }
        if (!conflicts) {
            reduced.rows.push_back(input.rows[row]);
            reduced.demands.push_back(input.demands[row]);
            reduced.original_rows.push_back(input.original_rows[row]);
        }
    }

    // Paper reference: "LP-Guided Problem Reduction", Step 2. Preserve the
    // residual order and recompute percentile depths on the reduced matrix.
    std::vector<double> demand_structure;
    std::vector<int> row_start;
    int last_beam = -1;
    for (int row = 0; row < static_cast<int>(reduced.rows.size()); ++row) {
        const int beam = reduced.original_rows[row].first;
        if (beam != last_beam) {
            demand_structure.push_back(reduced.demands[row]);
            row_start.push_back(row);
            last_beam = beam;
        }
    }

    if (demand_structure.empty()) {
        reduced.backtracking_depths.assign(kPercentiles.size(), 0);
        return reduced;
    }

    std::vector<double> cumulative(demand_structure.size());
    std::partial_sum(demand_structure.begin(), demand_structure.end(),
                     cumulative.begin());
    const double total = cumulative.back();
    for (double percentile : kPercentiles) {
        const double target = percentile * total;
        const auto nearest = std::min_element(
            cumulative.begin(), cumulative.end(),
            [&](double left, double right) {
                return std::abs(left - target) <
                       std::abs(right - target);
            });
        const int index = static_cast<int>(
            std::distance(cumulative.begin(), nearest));
        reduced.backtracking_depths.push_back(row_start[index] + 1);
    }
    return reduced;
}

static void write_lp_result_json(
    const fs::path& path, double value, double computation_time,
    int percentile_index, const std::vector<int>& integer) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot create " + path.string());
    }
    output << std::setprecision(15)
           << "{\"value\":" << value
           << ",\"time\":" << computation_time
           << ",\"percentile_index\":";
    if (percentile_index < 0) {
        output << "null";
    } else {
        output << percentile_index;
    }
    output << ",\"integer\":";
    write_int_vector_json(output, integer);
    output << '}';
}
#endif

static Summary cumulative_analysis(
    const std::vector<WorkerResult>& records) {
    // Paper reference: Experimental Setup - Methods. Report the best objective
    // across depths and its corresponding parallel execution time.
    std::array<Summary, kPercentiles.size()> aggregated{};
    for (std::size_t percentile = 0; percentile < kPercentiles.size();
         ++percentile) {
        aggregated[percentile].percentile_index =
            static_cast<int>(percentile);
        for (const auto& record : records) {
            aggregated[percentile].time =
                std::max(aggregated[percentile].time,
                         record.times[percentile]);
            aggregated[percentile].value =
                std::max(aggregated[percentile].value,
                         record.values[percentile]);
            if (record.values[percentile] ==
                aggregated[percentile].value) {
                aggregated[percentile].integers =
                    record.integers[percentile];
            }
        }
    }

    const double best_value =
        std::max_element(aggregated.begin(), aggregated.end(),
                         [](const auto& left, const auto& right) {
                             return left.value < right.value;
                         })
            ->value;
    Summary fastest_of_best;
    bool found_best = false;
    for (const auto& item : aggregated) {
        if (item.value == best_value &&
            (!found_best || item.time < fastest_of_best.time)) {
            fastest_of_best = item;
            found_best = true;
        }
    }
    return fastest_of_best;
}

static void write_header(std::ofstream& output, const std::string& depth_name) {
    output << "Instance,User,Simulation," << depth_name
           << ",Total Allocated Demand,Computation Time(s),Integer Solution\n";
}

static std::string format_integer(const std::vector<int>& integer) {
    std::ostringstream output;
    output << '"';
    for (std::size_t index = 0; index < integer.size(); ++index) {
        if (index != 0) {
            output << ' ';
        }
        output << integer[index];
    }
    output << '"';
    return output.str();
}

static void write_int_vector_json(std::ostream& output,
                                  const std::vector<int>& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << values[index];
    }
    output << ']';
}

static void write_double_vector_json(std::ostream& output,
                                     const std::vector<double>& values) {
    output << '[' << std::setprecision(15);
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << values[index];
    }
    output << ']';
}

static void write_details_json(const fs::path& path, const InputData& input,
                               const std::vector<WorkerResult>& records,
                               double computation_time,
                               double greedy_time,
                               const Summary& best_result) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot create " + path.string());
    }

    output << "{\"rows_color\":[";
    for (std::size_t index = 0; index < input.rows.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        write_int_vector_json(output, input.rows[index]);
    }

    output << "],\"orig_row\":[";
    for (std::size_t index = 0; index < input.original_rows.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << '[' << input.original_rows[index].first << ','
               << input.original_rows[index].second << ']';
    }

    output << "],\"pos\":{";
    bool first = true;
    for (const auto& [beam, position] : input.sorted_position) {
        if (!first) {
            output << ',';
        }
        first = false;
        output << '"' << beam << "\":" << position;
    }

    output << "},\"D_color\":";
    write_double_vector_json(output, input.demands);
    output << ",\"u\":" << input.column_count - 1;
    output << ",\"N_max\":" << kNMax;

    output << ",\"adj\":{";
    first = true;
    for (const auto& [beam, neighbours] : input.adjacency) {
        if (!first) {
            output << ',';
        }
        first = false;
        output << '"' << beam << "\":";
        std::vector<int> sorted_neighbours(neighbours.begin(),
                                           neighbours.end());
        std::sort(sorted_neighbours.begin(), sorted_neighbours.end());
        write_int_vector_json(output, sorted_neighbours);
    }

    output << "},\"back\":";
    write_int_vector_json(output, input.backtracking_depths);
    output << ",\"time\":" << std::setprecision(15) << computation_time;
    output << ",\"greedy_time\":" << greedy_time;
    output << ",\"best_result\":{\"value\":"
           << std::setprecision(15) << best_result.value
           << ",\"time\":" << greedy_time + best_result.time
           << ",\"percentile_index\":" << best_result.percentile_index
           << ",\"backtracking_depth\":"
           << input.backtracking_depths[best_result.percentile_index]
           << ",\"integer\":";
    write_int_vector_json(output, best_result.integers);
    output << '}';
    output << ",\"time_by_backtrack_depth\":[";
    for (std::size_t percentile = 0; percentile < kPercentiles.size();
         ++percentile) {
        if (percentile != 0) {
            output << ',';
        }
        double depth_time = 0.0;
        for (const auto& record : records) {
            depth_time = std::max(depth_time, record.times[percentile]);
        }
        output << greedy_time + depth_time;
    }
    output << ']';
    output << ",\"results\":[";
    for (std::size_t worker = 0; worker < records.size(); ++worker) {
        if (worker != 0) {
            output << ',';
        }
        const auto& record = records[worker];
        output << "{\"i\":" << worker
               << ",\"LB\":" << std::setprecision(15)
               << record.best_value << ",\"LB_integer\":";
        write_int_vector_json(output, record.best_integer);
        output << ",\"times\":[";
        for (std::size_t percentile = 0; percentile < kPercentiles.size();
             ++percentile) {
            if (percentile != 0) {
                output << ',';
            }
            output << record.times[percentile];
        }
        output << ']';
        output << ",\"backtrack_depth\":";
        if (record.backtrack_depth < 0) {
            output << "null";
        } else {
            output << record.backtrack_depth;
        }
        output << '}';
    }
    output << "]}";
}

int main(int argc, char** argv) {
    try {
        const fs::path data_root = argc > 1 ? argv[1] : "Data";
        const fs::path output_root = argc > 2 ? argv[2] : ".";
        const int instance = argc > 3 ? std::stoi(argv[3]) : 4;
        const int users = argc > 4 ? std::stoi(argv[4]) : 500;
        const int first_sample = argc > 5 ? std::stoi(argv[5]) : 0;
        const int sample_count = argc > 6 ? std::stoi(argv[6]) : 20;
        const int thread_count = argc > 7 ? std::stoi(argv[7]) : 10;
        const bool gurobi_mode = argc > 8 && std::string(argv[8]) == "gurobi";
        const bool lp_mode = argc > 8 && std::string(argv[8]) == "lp";
        const bool cpsat_mode = argc > 8 && std::string(argv[8]) == "cpsat";
        const bool features_mode =
            argc > 8 && std::string(argv[8]) == "features";
        const bool ml_mode = argc > 8 && std::string(argv[8]) == "ml";
        const bool ml_model_mode =
            argc > 8 && std::string(argv[8]) == "ml_model";
        const int predicted_percentile =
            ml_mode && argc > 9 ? std::stoi(argv[9]) : -1;
        const std::string ml_model_path =
            ml_model_mode && argc > 9 ? argv[9] : "";
        const double solver_time_limit =
            (gurobi_mode || cpsat_mode) && argc > 9
                ? std::stod(argv[9])
                : 3600.0;
        if (thread_count <= 0) {
            throw std::runtime_error("Thread count must be positive");
        }
        if (solver_time_limit <= 0.0) {
            throw std::runtime_error(
                "Solver time limit must be positive");
        }
        fs::create_directories(output_root);

#ifndef USE_GUROBI
        if (gurobi_mode || lp_mode) {
            throw std::runtime_error(
                "This executable was not compiled with Gurobi support");
        }
#endif
#ifndef USE_CPSAT
        if (cpsat_mode) {
            throw std::runtime_error(
                "This executable was not compiled with CP-SAT support");
        }
#endif
#ifndef USE_LIGHTGBM
        if (ml_model_mode) {
            throw std::runtime_error(
                "This executable was not compiled with LightGBM support");
        }
#endif

        std::ofstream fastest_of_best_file(output_root / "DLXv1_results1.csv");
        if (!fastest_of_best_file) {
            throw std::runtime_error("Cannot create result CSV files");
        }
        write_header(fastest_of_best_file,
                     "Backtracking percentiles idx");
        fastest_of_best_file << std::setprecision(15);

#ifdef USE_GUROBI
        std::ofstream gurobi_file;
        if (gurobi_mode) {
            gurobi_file.open(output_root / "Gurobi_results.csv");
            if (!gurobi_file) {
                throw std::runtime_error("Cannot create Gurobi_results.csv");
            }
            gurobi_file
                << "Instance,User,Simulation,Solver,Computation Time(s),"
                   "Total Allocated Demand,Integer Solution\n"
                << std::setprecision(15);
        }
#endif

#ifdef USE_CPSAT
        std::ofstream cpsat_file;
        if (cpsat_mode) {
            cpsat_file.open(output_root / "CP_SAT_results.csv");
            if (!cpsat_file) {
                throw std::runtime_error("Cannot create CP_SAT_results.csv");
            }
            cpsat_file
                << "Instance,User,Simulation,Solver,Computation Time(s),"
                   "Total Allocated Demand,Integer Solution\n"
                << std::setprecision(15);
        }
#endif

        const fs::path candidate_root = data_root / "Beam Candidate";
        for (int sample = first_sample;
             sample < first_sample + sample_count; ++sample) {
            const std::string prefix =
                std::to_string(instance) + "_" + std::to_string(users);
            const auto demands = read_demands(
                candidate_root /
                ("Beams_" + prefix + "_60_4_sample" +
                 std::to_string(sample) + ".csv"));
            const auto allocations = read_allocations(
                candidate_root /
                ("Allocate_" + prefix + "_60_4_sample" +
                 std::to_string(sample) + ".txt"));
            const auto pairs = read_pairs(
                candidate_root /
                ("Pairs_" + prefix + "_sample" + std::to_string(sample) +
                 ".json"));

            const InputData input =
                build_input(demands, allocations, pairs, users);

#ifdef USE_GUROBI
            if (lp_mode) {
                // Paper reference: "LP-Guided Problem Reduction", Steps 1-3.
                // fix integral rows, search the residual instance, and combine
                // the fixed and residual objective values.
                const auto constraints =
                    incidence_transpose_fast(input.rows);
                const auto start = Clock::now();
                const GurobiLpResult relaxation = gurobi_LP(
                    input.rows, input.demands, kNMax, constraints);

                if (relaxation.fraction.empty()) {
                    const double elapsed =
                        std::chrono::duration<double>(
                            Clock::now() - start).count();
                    write_lp_result_json(
                        output_root / "DLX_LP_result.json",
                        relaxation.integer_demand, elapsed, -1,
                        relaxation.integer);
                    std::cout << "Completed LP DLX+: z=" << instance
                              << ", INS=" << users
                              << ", simulation=" << sample << '\n';
                    continue;
                }

                InputData reduced =
                    reduce_after_fixed_rows(input, relaxation.integer);
                const int capacity =
                    kNMax - static_cast<int>(relaxation.integer.size());
                const double pre_worker_time =
                    std::chrono::duration<double>(
                        Clock::now() - start).count();
                const std::vector<int> empty_integer;
                std::vector<std::future<WorkerResult>> futures;
                for (int worker = 0; worker < thread_count; ++worker) {
                    futures.push_back(std::async(
                        std::launch::async, run_worker, worker,
                        std::cref(reduced), 0.0,
                        std::cref(empty_integer), capacity));
                }

                std::vector<WorkerResult> records;
                for (auto& future : futures) {
                    records.push_back(future.get());
                }
                for (auto& record : records) {
                    for (double& depth_time : record.times) {
                        depth_time += pre_worker_time;
                    }
                }

                const Summary best = cumulative_analysis(records);
                std::vector<int> combined_integer = relaxation.integer;
                combined_integer.insert(combined_integer.end(),
                                        best.integers.begin(),
                                        best.integers.end());
                write_lp_result_json(
                    output_root / "DLX_LP_result.json",
                    best.value + relaxation.integer_demand,
                    best.time, best.percentile_index,
                    combined_integer);
                std::cout << "Completed LP DLX+: z=" << instance
                          << ", INS=" << users
                          << ", simulation=" << sample << '\n';
                continue;
            }
#endif

            if (features_mode) {
                write_features_json(
                    output_root / "DLX_features.json",
                    extract_features_FBO(input));
                std::cout << "Completed features: z=" << instance
                          << ", INS=" << users
                          << ", simulation=" << sample << '\n';
                continue;
            }

            if (ml_mode) {
                // Paper reference: "Machine Learning Backtracking" and
                // Experimental Setup - Settings for Machine Learning. Python
                // supplies the predicted class; C++ times greedy construction
                // and parallel DLX+ search at only the predicted depth.
                const auto start = Clock::now();
                const auto [lower_bound, greedy_integer] =
                    greedy_solution(input);
                const int depth = predicted_backtracking_depth(
                    input, predicted_percentile, lower_bound);

                std::vector<std::future<
                    std::pair<double, std::vector<int>>>> futures;
                for (int worker = 0; worker < thread_count; ++worker) {
                    futures.push_back(std::async(
                        std::launch::async, run_single_depth_worker,
                        worker, std::cref(input), depth, lower_bound,
                        std::cref(greedy_integer)));
                }

                double best_value = lower_bound;
                std::vector<int> best_integer = greedy_integer;
                for (auto& future : futures) {
                    auto [value, integer] = future.get();
                    if (value > best_value) {
                        best_value = value;
                        best_integer = std::move(integer);
                    }
                }
                const double computation_time =
                    std::chrono::duration<double>(Clock::now() - start)
                        .count();
                write_ml_result_json(
                    output_root / "DLX_ML_result.json", best_value,
                    computation_time, predicted_percentile, depth,
                    best_integer);
                std::cout << "Completed ML DLX+: z=" << instance
                          << ", INS=" << users
                          << ", simulation=" << sample << '\n';
                continue;
            }

#ifdef USE_LIGHTGBM
            if (ml_model_mode) {
                const auto start = Clock::now();
                const int prediction = predict_lightgbm_class(
                    ml_model_path,
                    ml_feature_vector(extract_features_FBO(input)));
                const auto [lower_bound, greedy_integer] =
                    greedy_solution(input);
                const int depth = predicted_backtracking_depth(
                    input, prediction, lower_bound);

                std::vector<std::future<
                    std::pair<double, std::vector<int>>>> futures;
                for (int worker = 0; worker < thread_count; ++worker) {
                    futures.push_back(std::async(
                        std::launch::async, run_single_depth_worker,
                        worker, std::cref(input), depth, lower_bound,
                        std::cref(greedy_integer)));
                }

                double best_value = lower_bound;
                std::vector<int> best_integer = greedy_integer;
                for (auto& future : futures) {
                    auto [value, integer] = future.get();
                    if (value > best_value) {
                        best_value = value;
                        best_integer = std::move(integer);
                    }
                }
                const double computation_time =
                    std::chrono::duration<double>(Clock::now() - start)
                        .count();
                write_ml_result_json(
                    output_root / "DLX_ML_result.json", best_value,
                    computation_time, prediction, depth, best_integer);
                std::cout << "Completed LightGBM DLX+: z=" << instance
                          << ", INS=" << users
                          << ", simulation=" << sample << '\n';
                continue;
            }
#endif

#ifdef USE_GUROBI
            if (gurobi_mode) {
                // Paper reference: "Time-Constrained Gurobi Runs". The caller
                // passes the matching DLX+ runtime as solver_time_limit.
                const auto constraints =
                    incidence_transpose_fast(input.rows);
                const GurobiResult gurobi = gurobi_SPK(
                    input.rows, input.demands, kNMax, constraints,
                    solver_time_limit);
                gurobi_file << instance << ',' << users << ',' << sample
                            << ",Gurobi," << gurobi.runtime << ',';
                if (gurobi.has_solution) {
                    gurobi_file << gurobi.objective;
                }
                gurobi_file << ','
                            << format_integer(gurobi.integer) << '\n';
                std::cout << "Completed Gurobi: z=" << instance
                          << ", INS=" << users
                          << ", simulation=" << sample << '\n';
                continue;
            }
#endif

#ifdef USE_CPSAT
            if (cpsat_mode) {
                // CP-SAT baseline uses the same CC-MWIS rows, objective, and
                // cardinality limit as Gurobi and DLX+.
                const auto constraints =
                    incidence_transpose(input.rows, input.column_count);
                const CpSatResult cpsat = cpsat_SPK(
                    input.rows, input.demands, kNMax, constraints,
                    false, solver_time_limit);
                cpsat_file << instance << ',' << users << ',' << sample
                           << ",CP-SAT," << cpsat.runtime << ',';
                if (cpsat.has_solution) {
                    cpsat_file << cpsat.objective;
                }
                cpsat_file << ','
                           << format_integer(cpsat.integer) << '\n';
                std::cout << "Completed CP-SAT: z=" << instance
                          << ", INS=" << users
                          << ", simulation=" << sample << '\n';
                continue;
            }
#endif

            // Paper reference: Experimental Setup - Methods. Timed standalone
            // DLX+ includes the greedy incumbent and all parallel searches.
            const auto results_start = Clock::now();
            const auto greedy_start = Clock::now();
            const auto [lower_bound, greedy_integer] =
                greedy_solution(input);
            const double greedy_time =
                std::chrono::duration<double>(Clock::now() - greedy_start)
                    .count();

            std::vector<std::future<WorkerResult>> futures;
            for (int worker = 0; worker < thread_count; ++worker) {
                futures.push_back(std::async(
                    std::launch::async, run_worker, worker,
                    std::cref(input), lower_bound,
                    std::cref(greedy_integer), kNMax));
            }
            std::vector<WorkerResult> records;
            for (auto& future : futures) {
                records.push_back(future.get());
            }
            const double computation_time =
                std::chrono::duration<double>(Clock::now() - results_start)
                    .count();

            const Summary fastest_of_best = cumulative_analysis(records);
            write_details_json(output_root / "DLX_details.json", input,
                               records, computation_time, greedy_time,
                               fastest_of_best);
            fastest_of_best_file
                << instance << ',' << users << ',' << sample << ','
                << fastest_of_best.percentile_index << ','
                << fastest_of_best.value << ',' << fastest_of_best.time
                << ',' << format_integer(fastest_of_best.integers) << '\n';
            std::cout << "Completed: z=" << instance << ", INS=" << users
                      << ", simulation=" << sample << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
