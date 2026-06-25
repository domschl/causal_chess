#ifndef CAUSAL_CHESS_SEARCH_HPP
#define CAUSAL_CHESS_SEARCH_HPP

#include <string>
#include <vector>
#include <utility>
#include <optional>
#include <memory>
#include <deque>
#include <sstream>
#include <torch/torch.h>
#include "chess.hpp"
#include "model.hpp"

namespace causal_chess {

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
    std::pair<chess::Move, float> search_position(chess::Board& board, std::optional<double> temperature = std::nullopt);

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
    void set_heuristic_weight(double w) { config.heuristic_weight = w; }
    double get_heuristic_weight() const { return config.heuristic_weight; }
    void set_learning_rate(double lr);
    double get_learning_rate() const { return config.learning_rate; }
    bool get_adaptive_scaling() const { return config.adaptive_scaling; }
    void set_adaptive_scaling(bool val) { config.adaptive_scaling = val; }
    double get_live_lr_scale() const { return config.live_lr_scale; }
    int get_post_game_epochs() const { return config.post_game_epochs; }
    void step_scheduler();
    int get_scheduler_step() const { return scheduler_step; }

private:
    float _search(chess::Board& board, int depth);
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
};

} // namespace causal_chess

#endif // CAUSAL_CHESS_SEARCH_HPP
