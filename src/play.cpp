#include "play.hpp"
#include <chrono>
#include <sstream>
#include <iostream>
#include <filesystem>

namespace causal_chess {

PlayStats self_play_loop(
    Engine& engine,
    int num_games,
    const std::string& save_dir,
    int save_interval,
    int max_moves,
    bool verbose
) {
    PlayStats stats;
    std::filesystem::create_directories(save_dir);

    for (int game_num = 1; game_num <= num_games; ++game_num) {
        auto game_start = std::chrono::steady_clock::now();
        engine.reset_stats();

        chess::Board board;
        std::vector<std::string> moves_san;
        int move_count = 0;

        // Play game until termination or max moves
        while (board.isGameOver().first == chess::GameResultReason::NONE && move_count < max_moves) {
            auto [best_move, value] = engine.search_position(board);

            // Record SAN string before making move
            std::string san = chess::uci::moveToSan(board, best_move);
            moves_san.push_back(san);

            board.makeMove(best_move);
            move_count++;
        }

        // Determine result
        std::string result = "1/2-1/2";
        auto [reason, res] = board.isGameOver();

        if (reason != chess::GameResultReason::NONE) {
            if (res == chess::GameResult::WIN) {
                // If it's a win, the side whose turn it IS won.
                // Wait! Let's check how we made the last move.
                // When board.isGameOver() was called, the active player is the one WHO HAS NO LEGAL MOVES or is in mate.
                // Let's re-verify: isGameOver() checkmate returns LOSE for the player whose turn it is.
                // So if it returned WIN, it means the player whose turn it is won.
                // But normally checkmate returns LOSE for the side to move.
                // So let's check:
                if (board.sideToMove() == chess::Color::WHITE) {
                    // White is in checkmate/stalemate and lost, so Black wins (0-1)
                    result = "0-1";
                    stats.black_wins++;
                } else {
                    // Black is in checkmate/stalemate and lost, so White wins (1-0)
                    result = "1-0";
                    stats.white_wins++;
                }
            } else if (res == chess::GameResult::LOSE) {
                // Side to move lost, so if White -> Black wins (0-1)
                if (board.sideToMove() == chess::Color::WHITE) {
                    result = "0-1";
                    stats.black_wins++;
                } else {
                    result = "1-0";
                    stats.white_wins++;
                }
            } else {
                stats.draws++;
            }
        } else {
            // Max moves draw
            stats.draws++;
        }

        auto game_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = game_end - game_start;

        if (verbose) {
            int total = stats.white_wins + stats.black_wins + stats.draws;
            std::cout << "\n============================================================\n";
            std::cout << "Game " << game_num << "/" << num_games << ": " << result << "\n";
            std::cout << "  Moves:       " << (move_count + 1) / 2 << "\n";
            std::cout << "  Time:        " << elapsed.count() << "s\n";
            std::cout << "  TD updates:  " << engine.get_update_count() << "\n";
            std::cout << "  Avg TD loss: " << engine.get_avg_loss() << "\n";
            std::cout << "  Record:      W=" << stats.white_wins << "  B=" << stats.black_wins << "  D=" << stats.draws << "  (" << total << " games)\n";
            std::cout << "------------------------------------------------------------\n";

            // Print PGN
            std::cout << "[Event \"Causal Chess Self-Play\"]\n";
            std::cout << "[White \"CausalChess\"]\n";
            std::cout << "[Black \"CausalChess\"]\n";
            std::cout << "[Round \"" << game_num << "\"]\n";
            std::cout << "[Result \"" << result << "\"]\n\n";

            for (size_t i = 0; i < moves_san.size(); ++i) {
                if (i % 2 == 0) {
                    std::cout << (i / 2 + 1) << ". " << moves_san[i] << " ";
                } else {
                    std::cout << moves_san[i] << " ";
                }
            }
            std::cout << result << "\n\n";
        }

        // Save checkpoint
        if (game_num % save_interval == 0) {
            std::stringstream ss;
            ss << save_dir << "/checkpoint_" << std::setw(4) << std::setfill('0') << game_num << ".pt";
            engine.save_checkpoint(ss.str());
            if (verbose) {
                std::cout << "  💾 Checkpoint saved: " << ss.str() << "\n\n";
            }
        }
    }

    // Save final checkpoint
    engine.save_checkpoint(save_dir + "/checkpoint_final.pt");

    if (verbose) {
        std::cout << "============================================================\n";
        std::cout << "Self-play complete!\n";
        std::cout << "  Games:  " << num_games << "\n";
        std::cout << "  White:  " << stats.white_wins << "\n";
        std::cout << "  Black:  " << stats.black_wins << "\n";
        std::cout << "  Draws:  " << stats.draws << "\n";
        std::cout << "============================================================\n";
    }

    return stats;
}

} // namespace causal_chess
