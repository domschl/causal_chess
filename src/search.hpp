#ifndef CAUSAL_CHESS_SEARCH_HPP
#define CAUSAL_CHESS_SEARCH_HPP

#include <string>
#include <vector>
#include <utility>
#include <optional>
#include <memory>
#include <torch/torch.h>
#include "chess.hpp"
#include "model.hpp"

namespace causal_chess {

struct SearchConfig {
    int max_depth = 4;
    int top_n = 5;
    double learning_rate = 1e-4;
    std::string device = "cpu"; // "cpu", "mps", "cuda", "auto"
    double grad_clip = 1.0;
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
    std::pair<chess::Move, float> search_position(chess::Board& board);

    /**
     * @brief Evaluate a board state statically (inference only, no gradients, no updates).
     * @param board The chess board.
     * @return White-relative score in [0, 1].
     */
    float evaluate(const chess::Board& board);

    // Monitoring stats
    double get_avg_loss() const;
    int get_update_count() const;
    void reset_stats();

    // Performance metrics
    int64_t get_positions_evaluated() const;
    double get_forward_time_secs() const;
    double get_backprop_time_secs() const;
    double get_total_search_time_secs() const;

    // Checkpoint management
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);

    // Accessors
    ValueNetwork get_model() { return model; }
    torch::Device get_device() { return device; }

private:
    float _search(chess::Board& board, int depth);
    std::vector<std::pair<chess::Move, float>> _score_moves(chess::Board& board, const chess::Movelist& moves);
    void _td_update(const chess::Board& board, float target_value);
    static std::optional<float> _terminal_value(const chess::Board& board);

    SearchConfig config;
    torch::Device device;
    ValueNetwork model;
    std::unique_ptr<torch::optim::Adam> optimizer;

    double total_loss = 0.0;
    int update_count = 0;

    int64_t positions_evaluated = 0;
    double forward_time_secs = 0.0;
    double backprop_time_secs = 0.0;
    double total_search_time_secs = 0.0;
};

} // namespace causal_chess

#endif // CAUSAL_CHESS_SEARCH_HPP
