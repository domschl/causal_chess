#ifndef CAUSAL_CHESS_PLAY_HPP
#define CAUSAL_CHESS_PLAY_HPP

#include <string>
#include <map>
#include "search.hpp"

namespace causal_chess {

struct PlayStats {
    int white_wins = 0;
    int black_wins = 0;
    int draws = 0;
};

/**
 * @brief Run a continuous self-play training loop.
 *
 * @param engine The chess engine containing the value network and optimizer.
 * @param num_games Number of games to play.
 * @param save_dir Directory where model checkpoints will be saved.
 * @param save_interval How often (in games) to save checkpoints.
 * @param max_moves Maximum moves before declaring a draw.
 * @param verbose If true, print detailed summaries and PGN of games.
 * @return PlayStats containing outcomes of the self-play loop.
 */
PlayStats self_play_loop(
    Engine& engine,
    int num_games = 100,
    const std::string& save_dir = "checkpoints",
    int save_interval = 10,
    int max_moves = 200,
    bool verbose = true
);

} // namespace causal_chess

#endif // CAUSAL_CHESS_PLAY_HPP
