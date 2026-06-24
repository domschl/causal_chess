#include "search.hpp"
#include "encoding.hpp"
#include <iostream>
#include <random>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <filesystem>

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
        this->model = ValueNetwork(15);
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
    std::vector<std::pair<chess::Move, float>> selected_moves(move_scores.begin(), move_scores.begin() + keep);

    // Add all checks and captures from the remaining moves
    for (size_t i = keep; i < move_scores.size(); ++i) {
        const auto& ms = move_scores[i];
        if (board.isCapture(ms.first) || board.givesCheck(ms.first) != chess::CheckType::NO_CHECK) {
            selected_moves.push_back(ms);
        }
    }

    // 4. Search recursively
    std::vector<std::pair<chess::Move, float>> searched_moves;
    searched_moves.reserve(selected_moves.size());

    for (const auto& ms : selected_moves) {
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
        float nn_val = evaluate(board);
        if (config.heuristic_weight <= 0.0) {
            return nn_val;
        }
        float h_val = _calculate_heuristic(board);
        // Track divergence only on imbalanced positions (|H(s) - 0.5| > 0.05)
        if (std::abs(h_val - 0.5f) > 0.05f) {
            total_heuristic_nn_diff += std::abs(nn_val - h_val);
            leaf_eval_count++;
        }
        return (1.0f - config.heuristic_weight) * nn_val + config.heuristic_weight * h_val;
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
    std::vector<std::pair<chess::Move, float>> selected_moves(move_scores.begin(), move_scores.begin() + keep);

    // Add all checks and captures from the remaining moves
    for (size_t i = keep; i < move_scores.size(); ++i) {
        const auto& ms = move_scores[i];
        if (board.isCapture(ms.first) || board.givesCheck(ms.first) != chess::CheckType::NO_CHECK) {
            selected_moves.push_back(ms);
        }
    }

    // 5. Search recursively
    float best_value = is_white ? -1.0f : 2.0f;
    for (const auto& ms : selected_moves) {
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

    // Temporarily scale learning rate for online TD updates
    double base_lr = config.learning_rate;
    double live_lr = base_lr * config.live_lr_scale;
    for (auto& group : optimizer->param_groups()) {
        static_cast<torch::optim::AdamOptions&>(group.options()).lr(live_lr);
    }

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

    // Restore base learning rate
    for (auto& group : optimizer->param_groups()) {
        static_cast<torch::optim::AdamOptions&>(group.options()).lr(base_lr);
    }

    auto end_b = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_b = end_b - start_b;
    backprop_time_secs += elapsed_b.count();

    total_loss += loss.item<double>();
    update_count++;
    live_updates_this_game++;
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

double Engine::get_total_post_game_train_time_secs() const {
    return total_post_game_train_time_secs;
}

void Engine::reset_stats() {
    total_loss = 0.0;
    update_count = 0;
    positions_evaluated = 0;
    forward_time_secs = 0.0;
    backprop_time_secs = 0.0;
    total_search_time_secs = 0.0;
    total_post_game_train_time_secs = 0.0;
    total_heuristic_nn_diff = 0.0;
    leaf_eval_count = 0;
    live_updates_this_game = 0;
    post_updates_this_game = 0;
}

double Engine::get_avg_heuristic_nn_divergence() const {
    if (leaf_eval_count < 20) return 0.25; // Default high divergence if not enough imbalanced states were sampled
    return total_heuristic_nn_diff / leaf_eval_count;
}

void Engine::train_on_outcome(const std::vector<chess::Board>& boards, float outcome) {
    if (boards.empty()) return;
    auto train_start = std::chrono::steady_clock::now();

    // 1. Add all positions from the current game to the replay buffer with their discounted targets
    for (size_t t = 0; t < boards.size(); ++t) {
        size_t moves_to_end = boards.size() - 1 - t;
        float target_val = 0.5f + (outcome - 0.5f) * std::pow(config.discount_factor, moves_to_end);
        replay_buffer.push_back({boards[t], target_val});
    }

    // 2. Enforce replay buffer size limit (FIFO)
    while (replay_buffer.size() > static_cast<size_t>(config.replay_buffer_size)) {
        replay_buffer.pop_front();
    }

    // 3. Sample a random batch of samples from the replay buffer
    size_t batch_size = std::min(replay_buffer.size(), static_cast<size_t>(config.replay_batch_size));
    if (batch_size == 0) return;

    std::vector<ReplaySample> sampled_batch;
    sampled_batch.reserve(batch_size);

    static std::mt19937 gen(std::random_device{}());
    std::vector<size_t> indices(replay_buffer.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    for (size_t i = 0; i < batch_size; ++i) {
        sampled_batch.push_back(replay_buffer[indices[i]]);
    }

    // 4. Prepare tensors (including mirrored versions) for training
    std::vector<torch::Tensor> all_tensors;
    std::vector<float> targets_host;
    all_tensors.reserve(batch_size * 2);
    targets_host.reserve(batch_size * 2);

    for (const auto& sample : sampled_batch) {
        torch::Tensor orig = board_to_tensor(sample.board);
        torch::Tensor mirrored = torch::flip(orig, {2}).clone();
        
        all_tensors.push_back(orig);
        all_tensors.push_back(mirrored);
        
        targets_host.push_back(sample.target);
        targets_host.push_back(sample.target);
    }

    // 5. Run gradient steps on the sampled batch in mini-batches of size 16 (32 boards total after symmetry)
    model->train();
    const size_t mini_batch_size = 16;

    for (int epoch = 0; epoch < config.post_game_epochs; ++epoch) {
        for (size_t i = 0; i < all_tensors.size(); i += mini_batch_size * 2) {
            size_t current_batch_size = std::min(mini_batch_size * 2, all_tensors.size() - i);
            
            auto start_f = std::chrono::steady_clock::now();
            std::vector<torch::Tensor> batch_vec(all_tensors.begin() + i, all_tensors.begin() + i + current_batch_size);
            torch::Tensor batch_input = torch::stack(batch_vec).to(device);
            
            std::vector<float> batch_targets_host(targets_host.begin() + i, targets_host.begin() + i + current_batch_size);
            torch::Tensor batch_target = torch::tensor(batch_targets_host, torch::dtype(torch::kFloat32)).unsqueeze(1).to(device);
            
            optimizer->zero_grad();
            torch::Tensor prediction = model->forward(batch_input);
            auto end_f = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_f = end_f - start_f;
            forward_time_secs += elapsed_f.count();

            auto start_b = std::chrono::steady_clock::now();
            torch::Tensor loss = torch::nn::functional::mse_loss(prediction, batch_target);
            loss.backward();
            
            torch::nn::utils::clip_grad_norm_(model->parameters(), config.grad_clip);
            optimizer->step();
            auto end_b = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_b = end_b - start_b;
            backprop_time_secs += elapsed_b.count();
            
            total_loss += loss.item<double>();
            update_count++;
            post_updates_this_game++;
        }
    }
    model->eval();

    auto train_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = train_end - train_start;
    total_post_game_train_time_secs += elapsed.count();

    // Perform hybrid runtime-adaptive scaling (epochs + live_lr_scale) for the next game
    if (live_updates_this_game > 0 && post_updates_this_game > 0) {
        double updates_per_epoch = static_cast<double>(post_updates_this_game) / config.post_game_epochs;
        if (updates_per_epoch <= 0.0) updates_per_epoch = 16.0;

        // 1. Calculate how many epochs would be required to balance the influence at nominal_live_lr_scale
        double required_epochs = (live_updates_this_game * config.nominal_live_lr_scale) / 
                                  (config.adaptive_influence_ratio * updates_per_epoch);
        
        int next_epochs = std::max(1, std::min(config.max_post_game_epochs, 
                          static_cast<int>(std::round(required_epochs))));
        
        // 2. Adjust live_lr_scale to cover any remaining dampening factor
        double target_scale = config.adaptive_influence_ratio * 
                              ((next_epochs * updates_per_epoch) / static_cast<double>(live_updates_this_game));
        config.live_lr_scale = std::max(1e-5, std::min(1.0, target_scale));
        
        // 3. Update the post-game epochs for the next game
        config.post_game_epochs = next_epochs;

        std::cout << "  ⚖️ Hybrid Adaptive Scaling updated for next game:\n";
        std::cout << "     Post-Game Epochs:  " << config.post_game_epochs << " (calculated: " << required_epochs << ")\n";
        std::cout << "     Live LR Scale:     " << config.live_lr_scale << "\n";
        std::cout << "     (Based on " << live_updates_this_game << " live vs " << post_updates_this_game << " post updates in last game)\n";
    }
}

static std::string get_opt_path(const std::string& path) {
    if (path.length() >= 3 && path.substr(path.length() - 3) == ".pt") {
        return path.substr(0, path.length() - 3) + ".opt";
    }
    return path + ".opt";
}

static std::string get_json_path(const std::string& path) {
    if (path.length() >= 3 && path.substr(path.length() - 3) == ".pt") {
        return path.substr(0, path.length() - 3) + ".json";
    }
    return path + ".json";
}

void Engine::set_learning_rate(double lr) {
    config.learning_rate = lr;
    for (auto& group : optimizer->param_groups()) {
        static_cast<torch::optim::AdamOptions&>(group.options()).lr(lr);
    }
}

void Engine::step_scheduler() {
    scheduler_step++;
    if (config.lr_decay_rate < 1.0 && config.lr_decay_steps > 0) {
        if (scheduler_step % config.lr_decay_steps == 0) {
            double new_lr = config.learning_rate * config.lr_decay_rate;
            if (new_lr < config.min_learning_rate) {
                new_lr = config.min_learning_rate;
            }
            if (new_lr != config.learning_rate) {
                set_learning_rate(new_lr);
                std::cout << "  📉 Scheduler stepped: Learning rate decayed to " << new_lr << "\n";
            }
        }
    }
}

void Engine::save_checkpoint(const std::string& path) {
    // 1. Save model weights
    torch::save(model, path);

    // 2. Save optimizer state
    std::string opt_path = get_opt_path(path);
    try {
        torch::save(*optimizer, opt_path);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to save optimizer state to " << opt_path << ": " << e.what() << "\n";
    }

    // 3. Save config & state to JSON
    std::string json_path = get_json_path(path);
    std::ofstream f(json_path);
    if (f.is_open()) {
        f << "{\n";
        f << "  \"max_depth\": " << config.max_depth << ",\n";
        f << "  \"top_n\": " << config.top_n << ",\n";
        f << "  \"learning_rate\": " << config.learning_rate << ",\n";
        f << "  \"grad_clip\": " << config.grad_clip << ",\n";
        f << "  \"temperature\": " << config.temperature << ",\n";
        f << "  \"post_game_epochs\": " << config.post_game_epochs << ",\n";
        f << "  \"discount_factor\": " << config.discount_factor << ",\n";
        f << "  \"replay_buffer_size\": " << config.replay_buffer_size << ",\n";
        f << "  \"replay_batch_size\": " << config.replay_batch_size << ",\n";
        f << "  \"heuristic_weight\": " << config.heuristic_weight << ",\n";
        f << "  \"lr_decay_rate\": " << config.lr_decay_rate << ",\n";
        f << "  \"lr_decay_steps\": " << config.lr_decay_steps << ",\n";
        f << "  \"min_learning_rate\": " << config.min_learning_rate << ",\n";
        f << "  \"live_lr_scale\": " << config.live_lr_scale << ",\n";
        f << "  \"adaptive_influence_ratio\": " << config.adaptive_influence_ratio << ",\n";
        f << "  \"nominal_live_lr_scale\": " << config.nominal_live_lr_scale << ",\n";
        f << "  \"max_post_game_epochs\": " << config.max_post_game_epochs << ",\n";
        f << "  \"scheduler_step\": " << scheduler_step << "\n";
        f << "}\n";
    }
}

void Engine::load_checkpoint(const std::string& path) {
    // 1. Load model weights
    torch::serialize::InputArchive archive;
    archive.load_from(path, device);
    model->load(archive);
    model->eval();

    // 2. Load optimizer state if it exists
    std::string opt_path = get_opt_path(path);
    if (std::filesystem::exists(opt_path)) {
        try {
            torch::serialize::InputArchive opt_archive;
            opt_archive.load_from(opt_path, device);
            optimizer->load(opt_archive);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to load optimizer state from " << opt_path << ": " << e.what() << "\n";
        }
    }

    // 3. Load config from JSON if it exists
    std::string json_path = get_json_path(path);
    if (std::filesystem::exists(json_path)) {
        std::ifstream f(json_path);
        if (f.is_open()) {
            std::string line;
            while (std::getline(f, line)) {
                size_t colon = line.find(':');
                if (colon == std::string::npos) continue;
                
                size_t first_quote = line.find('"');
                size_t second_quote = line.find('"', first_quote + 1);
                if (first_quote == std::string::npos || second_quote == std::string::npos) continue;
                std::string key = line.substr(first_quote + 1, second_quote - first_quote - 1);
                
                std::string val_str = line.substr(colon + 1);
                while (!val_str.empty() && (std::isspace(val_str.front()) || val_str.front() == '"')) {
                    val_str.erase(val_str.begin());
                }
                while (!val_str.empty() && (std::isspace(val_str.back()) || val_str.back() == ',' || val_str.back() == '}' || val_str.back() == '"')) {
                    val_str.pop_back();
                }
                
                if (val_str.empty()) continue;
                
                try {
                    if (key == "max_depth") config.max_depth = std::stoi(val_str);
                    else if (key == "top_n") config.top_n = std::stoi(val_str);
                    else if (key == "learning_rate") {
                        double lr = std::stod(val_str);
                        set_learning_rate(lr);
                    }
                    else if (key == "grad_clip") config.grad_clip = std::stod(val_str);
                    else if (key == "temperature") config.temperature = std::stod(val_str);
                    else if (key == "post_game_epochs") config.post_game_epochs = std::stoi(val_str);
                    else if (key == "discount_factor") config.discount_factor = std::stod(val_str);
                    else if (key == "replay_buffer_size") config.replay_buffer_size = std::stoi(val_str);
                    else if (key == "replay_batch_size") config.replay_batch_size = std::stoi(val_str);
                    else if (key == "heuristic_weight") config.heuristic_weight = std::stod(val_str);
                    else if (key == "lr_decay_rate") config.lr_decay_rate = std::stod(val_str);
                    else if (key == "lr_decay_steps") config.lr_decay_steps = std::stoi(val_str);
                    else if (key == "min_learning_rate") config.min_learning_rate = std::stod(val_str);
                    else if (key == "live_lr_scale") config.live_lr_scale = std::stod(val_str);
                    else if (key == "adaptive_influence_ratio") config.adaptive_influence_ratio = std::stod(val_str);
                    else if (key == "nominal_live_lr_scale") config.nominal_live_lr_scale = std::stod(val_str);
                    else if (key == "max_post_game_epochs") config.max_post_game_epochs = std::stoi(val_str);
                    else if (key == "scheduler_step") scheduler_step = std::stoi(val_str);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Error parsing checkpoint JSON field '" << key << "' with value '" << val_str << "': " << e.what() << "\n";
                }
            }
        }
    }
}

float Engine::_space_control_score(const chess::Board& board) {
    double white_score = 0.0;
    double black_score = 0.0;

    for (int sq_idx = 0; sq_idx < 64; ++sq_idx) {
        chess::Square sq(sq_idx);
        double weight = 0.1; // Default: Rest of the board
        
        int rank = sq_idx >> 3;
        int file = sq_idx & 7;

        // Check if inside d4-e5 (ranks 3-4, files 3-4, 0-indexed)
        if (rank >= 3 && rank <= 4 && file >= 3 && file <= 4) {
            weight = 1.0;
        }
        // Check if inside c3-f6 (ranks 2-5, files 2-5, 0-indexed)
        else if (rank >= 2 && rank <= 5 && file >= 2 && file <= 5) {
            weight = 0.4;
        }

        if (board.isAttacked(sq, chess::Color::WHITE)) {
            white_score += weight;
        }
        if (board.isAttacked(sq, chess::Color::BLACK)) {
            black_score += weight;
        }
    }

    // Phase factor: active non-pawns / 16.0
    int non_pawns = board.pieces(chess::PieceType::KNIGHT).count() +
                    board.pieces(chess::PieceType::BISHOP).count() +
                    board.pieces(chess::PieceType::ROOK).count() +
                    board.pieces(chess::PieceType::QUEEN).count();
    double phase_factor = static_cast<double>(non_pawns) / 16.0;

    return static_cast<float>((white_score - black_score) * phase_factor);
}

float Engine::_static_material_score(const chess::Board& board) {
    auto score_color = [&](chess::Color color) {
        float val = 0.0f;
        val += board.pieces(chess::PieceType::PAWN, color).count() * 1.0f;
        val += board.pieces(chess::PieceType::KNIGHT, color).count() * 3.0f;
        val += board.pieces(chess::PieceType::BISHOP, color).count() * 3.0f;
        val += board.pieces(chess::PieceType::ROOK, color).count() * 5.0f;
        val += board.pieces(chess::PieceType::QUEEN, color).count() * 9.0f;
        return val;
    };
    return score_color(chess::Color::WHITE) - score_color(chess::Color::BLACK);
}

float Engine::_quiescence_search(chess::Board& board, float alpha, float beta, int q_depth) {
    float stand_pat = _static_material_score(board);
    bool is_white = (board.sideToMove() == chess::Color::WHITE);

    if (is_white) {
        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;
    } else {
        if (stand_pat <= alpha) return alpha;
        if (stand_pat < beta) beta = stand_pat;
    }

    if (q_depth >= 4) {
        return stand_pat;
    }

    chess::Movelist moves;
    chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(moves, board);
    if (moves.empty()) {
        return stand_pat;
    }

    auto get_piece_value = [](chess::PieceType pt) {
        switch (pt.internal()) {
            case chess::PieceType::underlying::PAWN: return 100;
            case chess::PieceType::underlying::KNIGHT: return 300;
            case chess::PieceType::underlying::BISHOP: return 300;
            case chess::PieceType::underlying::ROOK: return 500;
            case chess::PieceType::underlying::QUEEN: return 900;
            case chess::PieceType::underlying::KING: return 10000;
            default: return 0;
        }
    };

    // MVV-LVA sorting for captures
    std::vector<std::pair<chess::Move, int>> sorted_moves;
    sorted_moves.reserve(moves.size());
    for (const auto& m : moves) {
        auto victim_type = board.at(m.to()) != chess::Piece() ? board.at(m.to()).type() : chess::PieceType::PAWN;
        auto attacker_type = board.at(m.from()).type();
        int score = get_piece_value(victim_type) * 10 - get_piece_value(attacker_type);
        sorted_moves.push_back({m, score});
    }
    std::sort(sorted_moves.begin(), sorted_moves.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    if (is_white) {
        for (const auto& sm : sorted_moves) {
            board.makeMove(sm.first);
            float score = _quiescence_search(board, alpha, beta, q_depth + 1);
            board.unmakeMove(sm.first);

            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }
        return alpha;
    } else {
        for (const auto& sm : sorted_moves) {
            board.makeMove(sm.first);
            float score = _quiescence_search(board, alpha, beta, q_depth + 1);
            board.unmakeMove(sm.first);

            if (score <= alpha) return alpha;
            if (score < beta) beta = score;
        }
        return beta;
    }
}

float Engine::_calculate_heuristic(const chess::Board& board) {
    chess::Board board_copy = board;
    float material_diff = _quiescence_search(board_copy, -1e9f, 1e9f, 0);
    float space_diff = _space_control_score(board);
    float total_units = material_diff + 0.1f * space_diff;
    return 0.5f + 0.5f * std::tanh(total_units / 8.0f);
}

} // namespace causal_chess
