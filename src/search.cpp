#include "search.hpp"
#include "encoding.hpp"
#include <iostream>
#include <random>
#include <cmath>
#include <cstdio>

namespace causal_chess {

static torch::Device resolve_device(const std::string& device_str) {
    if (device_str == "mps") {
        return torch::Device(torch::kMPS);
    } else if (device_str == "cuda") {
        return torch::Device(torch::kCUDA);
    } else if (device_str == "cpu") {
        return torch::Device(torch::kCPU);
    }

    // Auto-detect: 1. MPS, 2. CUDA, 3. CPU
    #ifdef __APPLE__
    if (torch::mps::is_available()) {
        return torch::Device(torch::kMPS);
    }
    #endif

    if (torch::cuda::is_available()) {
        return torch::Device(torch::kCUDA);
    }

    return torch::Device(torch::kCPU);
}

Engine::Engine(const SearchConfig& config, ValueNetwork model)
    : config(config),
      device(resolve_device(config.device)) {
    if (model.is_empty()) {
        this->model = ValueNetwork();
    } else {
        this->model = model;
    }
    this->model->to(device);
    this->model->eval();

    optimizer = std::make_unique<torch::optim::Adam>(
        this->model->parameters(),
        torch::optim::AdamOptions(config.learning_rate)
    );
}

std::pair<chess::Move, float> Engine::search_position(chess::Board& board, std::optional<double> temperature) {
    auto start = std::chrono::steady_clock::now();
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    if (moves.empty()) {
        throw std::runtime_error("search_position called on a position with no legal moves");
    }

    bool is_white = (board.sideToMove() == chess::Color::WHITE);

    // 1. Score all moves for initial ordering
    auto move_scores = _score_moves(board, moves);

    // 2. Sort moves based on White-relative score
    std::sort(move_scores.begin(), move_scores.end(), [is_white](const auto& a, const auto& b) {
        if (is_white) {
            return a.second > b.second; // Maximize for White
        } else {
            return a.second < b.second; // Minimize for Black
        }
    });

    // 3. Keep top-N moves
    int keep = std::min(static_cast<int>(move_scores.size()), config.top_n);
    std::vector<std::pair<chess::Move, float>> top_moves(move_scores.begin(), move_scores.begin() + keep);

    // 4. Search recursively
    std::vector<std::pair<chess::Move, float>> searched_moves;
    searched_moves.reserve(top_moves.size());

    for (const auto& ms : top_moves) {
        board.makeMove(ms.first);
        float val = _search(board, config.max_depth - 1);
        board.unmakeMove(ms.first);
        searched_moves.emplace_back(ms.first, val);
    }

    double temp = temperature.value_or(config.temperature);

    chess::Move best_move;
    float best_value;

    if (temp > 0.0) {
        // Temperature-based softmax sampling
        std::vector<double> exp_logits;
        exp_logits.reserve(searched_moves.size());

        double max_score = -9999.0;
        for (const auto& sm : searched_moves) {
            double score = is_white ? sm.second : 1.0 - sm.second;
            if (score > max_score) {
                max_score = score;
            }
        }

        for (const auto& sm : searched_moves) {
            double score = is_white ? sm.second : 1.0 - sm.second;
            exp_logits.push_back(std::exp((score - max_score) / temp));
        }

        static std::mt19937 gen(std::random_device{}());
        std::discrete_distribution<> dist(exp_logits.begin(), exp_logits.end());
        int chosen_idx = dist(gen);

        best_move = searched_moves[chosen_idx].first;
        best_value = searched_moves[chosen_idx].second;
    } else {
        // Greedy choice
        best_move = searched_moves[0].first;
        best_value = searched_moves[0].second;

        for (size_t i = 1; i < searched_moves.size(); ++i) {
            float val = searched_moves[i].second;
            if (is_white) {
                if (val > best_value) {
                    best_value = val;
                    best_move = searched_moves[i].first;
                }
            } else {
                if (val < best_value) {
                    best_value = val;
                    best_move = searched_moves[i].first;
                }
            }
        }
    }



    // 5. Root TD update
    _td_update(board, best_value);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    total_search_time_secs += elapsed.count();

    return {best_move, best_value};
}

float Engine::_search(chess::Board& board, int depth) {
    // 1. Terminal detection
    auto term = _terminal_value(board);
    if (term.has_value()) {
        return *term;
    }

    // 2. Leaf evaluation (no gradients)
    if (depth <= 0) {
        return evaluate(board);
    }

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) {
        return 0.5f; // Draw fallback
    }

    bool is_white = (board.sideToMove() == chess::Color::WHITE);

    // 3. Score all legal moves
    auto move_scores = _score_moves(board, moves);

    // 4. Sort and select top-N
    std::sort(move_scores.begin(), move_scores.end(), [is_white](const auto& a, const auto& b) {
        if (is_white) {
            return a.second > b.second;
        } else {
            return a.second < b.second;
        }
    });

    int keep = std::min(static_cast<int>(move_scores.size()), config.top_n);
    std::vector<std::pair<chess::Move, float>> top_moves(move_scores.begin(), move_scores.begin() + keep);

    // 5. Search recursively
    float best_value = is_white ? -1.0f : 2.0f;
    for (const auto& ms : top_moves) {
        board.makeMove(ms.first);
        float val = _search(board, depth - 1);
        board.unmakeMove(ms.first);

        if (is_white) {
            if (val > best_value) {
                best_value = val;
            }
        } else {
            if (val < best_value) {
                best_value = val;
            }
        }
    }

    // 6. Online TD update
    _td_update(board, best_value);

    return best_value;
}

std::vector<std::pair<chess::Move, float>> Engine::_score_moves(chess::Board& board, const chess::Movelist& moves) {
    positions_evaluated += moves.size();
    std::vector<chess::Board> child_boards;
    child_boards.reserve(moves.size());

    for (const auto& move : moves) {
        board.makeMove(move);
        child_boards.push_back(board);
        board.unmakeMove(move);
    }

    auto start = std::chrono::steady_clock::now();
    torch::Tensor batch_tensor = batch_boards_to_tensor(child_boards).to(device);

    torch::Tensor scores_tensor;
    {
        torch::NoGradGuard no_grad;
        scores_tensor = model->forward(batch_tensor).squeeze(-1); // Shape: (num_moves)
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    forward_time_secs += elapsed.count();

    // Move scores from device to host CPU vector
    scores_tensor = scores_tensor.to(torch::kCPU);
    float* scores_data = scores_tensor.data_ptr<float>();

    std::vector<std::pair<chess::Move, float>> scored_moves;
    scored_moves.reserve(moves.size());
    for (size_t i = 0; i < moves.size(); ++i) {
        scored_moves.emplace_back(moves[i], scores_data[i]);
    }

    return scored_moves;
}

void Engine::_td_update(const chess::Board& board, float target_value) {
    auto start_f = std::chrono::steady_clock::now();
    torch::Tensor orig_tensor = board_to_tensor(board);
    // Flip along files dimension (dim 2) for horizontal symmetry
    torch::Tensor mirrored_tensor = torch::flip(orig_tensor, {2}).clone();

    torch::Tensor batch = torch::stack({orig_tensor, mirrored_tensor}).to(device);
    torch::Tensor target = torch::tensor({{target_value}, {target_value}}, torch::dtype(torch::kFloat32).device(device));

    model->train();
    optimizer->zero_grad();

    torch::Tensor prediction = model->forward(batch);
    auto end_f = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_f = end_f - start_f;
    forward_time_secs += elapsed_f.count();

    auto start_b = std::chrono::steady_clock::now();
    torch::Tensor loss = torch::nn::functional::mse_loss(prediction, target);
    loss.backward();

    // Gradient clipping
    torch::nn::utils::clip_grad_norm_(model->parameters(), config.grad_clip);
    optimizer->step();
    model->eval();
    auto end_b = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_b = end_b - start_b;
    backprop_time_secs += elapsed_b.count();

    total_loss += loss.item<double>();
    update_count++;
}

float Engine::evaluate(const chess::Board& board) {
    positions_evaluated += 1;
    auto start = std::chrono::steady_clock::now();
    torch::Tensor tensor = board_to_tensor(board).unsqueeze(0).to(device);
    torch::Tensor out;
    {
        torch::NoGradGuard no_grad;
        out = model->forward(tensor);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    forward_time_secs += elapsed.count();

    return out.item<float>();
}

std::optional<float> Engine::_terminal_value(const chess::Board& board) {
    auto [reason, result] = board.isGameOver();
    if (reason == chess::GameResultReason::NONE) {
        return std::nullopt;
    }

    if (result == chess::GameResult::DRAW) {
        return 0.5f;
    }

    bool is_white = (board.sideToMove() == chess::Color::WHITE);
    if (result == chess::GameResult::LOSE) {
        // Current player lost, so if White -> 0.0, if Black -> 1.0
        return is_white ? 0.0f : 1.0f;
    } else if (result == chess::GameResult::WIN) {
        // Current player won, so if White -> 1.0, if Black -> 0.0
        return is_white ? 1.0f : 0.0f;
    }

    return 0.5f;
}

double Engine::get_avg_loss() const {
    if (update_count == 0) return 0.0;
    return total_loss / update_count;
}

int Engine::get_update_count() const {
    return update_count;
}

int64_t Engine::get_positions_evaluated() const {
    return positions_evaluated;
}

double Engine::get_forward_time_secs() const {
    return forward_time_secs;
}

double Engine::get_backprop_time_secs() const {
    return backprop_time_secs;
}

double Engine::get_total_search_time_secs() const {
    return total_search_time_secs;
}

void Engine::reset_stats() {
    total_loss = 0.0;
    update_count = 0;
    positions_evaluated = 0;
    forward_time_secs = 0.0;
    backprop_time_secs = 0.0;
    total_search_time_secs = 0.0;
}

void Engine::save_checkpoint(const std::string& path) {
    // libtorch serialize uses torch::save
    torch::save(model, path);
}

void Engine::load_checkpoint(const std::string& path) {
    torch::serialize::InputArchive archive;
    archive.load_from(path, device);
    model->load(archive);
    model->eval();
}

} // namespace causal_chess
