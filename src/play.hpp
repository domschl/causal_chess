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

class WebServer;

/**
 * @brief Run a continuous self-play training loop.
 *
 * @param engine The chess engine containing the value network and optimizer.
 * @param num_games Number of games to play.
 * @param save_dir Directory where model checkpoints will be saved.
 * @param save_interval How often (in games) to save checkpoints.
 * @param verbose If true, print detailed summaries and PGN of games.
 * @return PlayStats containing outcomes of the self-play loop.
 */
PlayStats self_play_loop(
    Engine& engine,
    int num_games = 100,
    const std::string& save_dir = "checkpoints",
    int save_interval = 10,
    bool verbose = true,
    bool resume = true,
    WebServer* web_server = nullptr
);

/**
 * @brief Print the board to the terminal using Unicode chess symbols.
 *
 * @param board Current board state.
 * @param perspective The player's color perspective.
 */
void print_board_unicode(const chess::Board& board, chess::Color perspective = chess::Color::WHITE);

/**
 * @brief Run an interactive game loop allowing a human to play against the model.
 *
 * @param engine The chess engine.
 * @param human_color Color of the human player.
 */
void play_human_loop(Engine& engine, chess::Color human_color, WebServer* web_server = nullptr);

} // namespace causal_chess

#endif // CAUSAL_CHESS_PLAY_HPP
