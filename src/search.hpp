#ifndef CAUSAL_CHESS_SEARCH_HPP
#define CAUSAL_CHESS_SEARCH_HPP

#include <string>
#include <vector>
#include <utility>
#include <optional>
#include <memory>
#include <deque>
#include <sstream>
#include <mutex>
#include <torch/torch.h>
#include "chess.hpp"
#include "model.hpp"

namespace causal_chess {

class WebServer;

bool parse_top_n_vector(const std::string& top_n_str, int depth, int& out_top_n, std::vector<int>& out_top_n_vector, std::string& error_msg);

struct SearchConfig {
    int max_depth = 4;
    int top_n = 5;
    std::vector<int> top_n_vector = {5};
    double learning_rate = 1e-4;
    std::string device = "cpu"; // "cpu", "mps", "cuda", "auto"
    double grad_clip = 1.0;
    double temperature = 0.0;
    int post_game_epochs = 2;
    double discount_factor = 0.97;
    int replay_buffer_size = 5000;
    int replay_batch_size = 128;
    double heuristic_weight = 0.5;
    double adaptive_weight_smoothing = 0.8;
    double lr_decay_rate = 0.998;
    int lr_decay_steps = 10;
    double min_learning_rate = 1e-6;
    double live_lr_scale = 1.0;
    double adaptive_influence_ratio = 0.5;
    double nominal_live_lr_scale = 1.0;
    int max_post_game_epochs = 15;
    bool adaptive_scaling = false;
};

class Engine {
public:
    Engine(const SearchConfig& config = SearchConfig(), ValueNetwork model = nullptr);

    /**
     * @brief Search for the best move in the given position.
     * Performs selective search and updates the value network online via TD learning.
     * @param board The chess board (modified temporarily but restored before returning).
     * @return A pair of (best_move, evaluation).
     */
    std::pair<chess::Move, float> search_position(chess::Board& board, std::optional<double> temperature = std::nullopt, WebServer* web_server = nullptr);

    /**
     * @brief Evaluate a board state statically (inference only, no gradients, no updates).
     * @param board The chess board.
     * @return White-relative score in [0, 1].
     */
    float evaluate(const chess::Board& board);

    // Monitoring stats
    double get_avg_loss() const;
    int get_update_count() const;
    double get_avg_heuristic_nn_divergence() const;
    void reset_stats();

    // Performance metrics
    int64_t get_positions_evaluated() const;
    double get_forward_time_secs() const;
    double get_backprop_time_secs() const;
    double get_total_search_time_secs() const;
    double get_total_post_game_train_time_secs() const;

    // Post-game outcome training pass
    void train_on_outcome(const std::vector<chess::Board>& boards, float outcome);

    // Checkpoint management
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);

    // Accessors and Scheduler
    ValueNetwork get_model() { return model; }
    torch::Device get_device() { return device; }
    
    void set_heuristic_weight(double w) {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        config.heuristic_weight = w;
    }
    double get_heuristic_weight() const {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        return config.heuristic_weight;
    }
    
    void set_adaptive_weight_smoothing(double val) {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        config.adaptive_weight_smoothing = val;
    }
    double get_adaptive_weight_smoothing() const {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        return config.adaptive_weight_smoothing;
    }
    
    void set_learning_rate(double lr);
    double get_learning_rate() const {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        return config.learning_rate;
    }
    
    bool get_adaptive_scaling() const {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        return config.adaptive_scaling;
    }
    void set_adaptive_scaling(bool val) {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        config.adaptive_scaling = val;
    }
    
    double get_live_lr_scale() const {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        return config.live_lr_scale;
    }
    int get_post_game_epochs() const {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        return config.post_game_epochs;
    }
    
    void set_temperature(double temp) {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        config.temperature = temp;
    }
    double get_temperature() const {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        return config.temperature;
    }
    
    void set_max_depth(int depth) {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        config.max_depth = depth;
    }
    int get_max_depth() const {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        return config.max_depth;
    }
    
    void set_top_n(int top_n) {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        config.top_n = top_n;
    }
    int get_top_n() const {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        return config.top_n;
    }

    void set_top_n_vector(const std::vector<int>& vec) {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        config.top_n_vector = vec;
    }
    std::vector<int> get_top_n_vector() const {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        return config.top_n_vector;
    }
    
    void step_scheduler();
    int get_scheduler_step() const { return scheduler_step; }

    // Session Control API
    bool is_paused() const { return paused; }
    void set_paused(bool val) { paused = val; }
    
    bool is_stop_requested() const { return stop_requested; }
    void set_stop_requested(bool val) { stop_requested = val; }

    bool is_reset_requested() const { return reset_requested; }
    void set_reset_requested(bool val) { reset_requested = val; }

    // Human Move Queue API
    void push_human_move(const std::string& move_str) {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        human_moves.push_back(move_str);
    }
    std::optional<std::string> pop_human_move() {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        if (human_moves.empty()) {
            return std::nullopt;
        }
        std::string mv = human_moves.front();
        human_moves.pop_front();
        return mv;
    }
    void clear_human_moves() {
        std::lock_guard<std::recursive_mutex> lock(config_mutex);
        human_moves.clear();
    }

private:
    float _search(chess::Board& board, int depth, std::vector<chess::Move>& pv);
    std::vector<std::pair<chess::Move, float>> _score_moves(chess::Board& board, const chess::Movelist& moves);
    void _td_update(const chess::Board& board, float target_value);
    static std::optional<float> _terminal_value(const chess::Board& board);
    float _space_control_score(const chess::Board& board);
    float _quiescence_search(chess::Board& board, float alpha, float beta, int q_depth);
    float _static_material_score(const chess::Board& board);
    float _calculate_heuristic(const chess::Board& board);

    SearchConfig config;
    torch::Device device;
    ValueNetwork model;
    std::unique_ptr<torch::optim::Adam> optimizer;

    struct ReplaySample {
        chess::Board board;
        float target;
    };
    std::deque<ReplaySample> replay_buffer;

    double total_loss = 0.0;
    int update_count = 0;
    double total_heuristic_nn_diff = 0.0;
    int64_t leaf_eval_count = 0;
    int scheduler_step = 0;
    int live_updates_this_game = 0;
    int post_updates_this_game = 0;

    int64_t positions_evaluated = 0;
    double forward_time_secs = 0.0;
    double backprop_time_secs = 0.0;
    double total_search_time_secs = 0.0;
    double total_post_game_train_time_secs = 0.0;

    std::atomic<bool> paused{false};
    std::atomic<bool> stop_requested{false};
    std::atomic<bool> reset_requested{false};
    std::deque<std::string> human_moves;

    mutable std::recursive_mutex config_mutex;
};

} // namespace causal_chess

#endif // CAUSAL_CHESS_SEARCH_HPP
