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
            double total_search_time = engine.get_total_search_time_secs();
            double nps = 0.0;
            double pct_forward = 0.0;
            double pct_backprop = 0.0;

            if (total_search_time > 0.0) {
                nps = engine.get_positions_evaluated() / total_search_time;
                pct_forward = (engine.get_forward_time_secs() / total_search_time) * 100.0;
                pct_backprop = (engine.get_backprop_time_secs() / total_search_time) * 100.0;
            }

            std::cout << "Game " << game_num << "/" << num_games << ": " << result << "\n";
            std::cout << "  Moves:       " << (move_count + 1) / 2 << "\n";
            std::cout << "  Time:        " << elapsed.count() << "s (Search: " << total_search_time << "s)\n";
            std::cout << "  Speed:       " << static_cast<int64_t>(nps) << " positions/sec\n";
            std::cout << "  Forward:     " << pct_forward << "%\n";
            std::cout << "  Backprop:    " << pct_backprop << "%\n";
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
            std::string checkpoint_path = save_dir + "/checkpoint.pt";
            engine.save_checkpoint(checkpoint_path);
            if (verbose) {
                std::cout << "  💾 Checkpoint saved: " << checkpoint_path << "\n\n";
            }
        }
    }

    // Save final checkpoint
    engine.save_checkpoint(save_dir + "/checkpoint.pt");

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

static std::string piece_to_unicode(chess::Piece p) {
    if (p == chess::Piece::NONE) {
        return "·"; // U+00B7 middle dot
    }

    if (p.color() == chess::Color::WHITE) {
        switch (p.type().internal()) {
            case chess::PieceType::underlying::PAWN:   return "♙";
            case chess::PieceType::underlying::KNIGHT: return "♘";
            case chess::PieceType::underlying::BISHOP: return "♗";
            case chess::PieceType::underlying::ROOK:   return "♖";
            case chess::PieceType::underlying::QUEEN:  return "♕";
            case chess::PieceType::underlying::KING:   return "♔";
            default: return ".";
        }
    } else {
        switch (p.type().internal()) {
            case chess::PieceType::underlying::PAWN:   return "♟";
            case chess::PieceType::underlying::KNIGHT: return "♞";
            case chess::PieceType::underlying::BISHOP: return "♝";
            case chess::PieceType::underlying::ROOK:   return "♜";
            case chess::PieceType::underlying::QUEEN:  return "♛";
            case chess::PieceType::underlying::KING:   return "♚";
            default: return ".";
        }
    }
}

void print_board_unicode(const chess::Board& board, chess::Color perspective) {
    std::cout << "\n";
    if (perspective == chess::Color::WHITE) {
        std::cout << "  a b c d e f g h\n";
        for (int rank = 7; rank >= 0; --rank) {
            std::cout << (rank + 1) << " ";
            for (int file = 0; file < 8; ++file) {
                chess::Square square(rank * 8 + file);
                std::cout << piece_to_unicode(board.at(square)) << " ";
            }
            std::cout << (rank + 1) << "\n";
        }
        std::cout << "  a b c d e f g h\n";
    } else {
        std::cout << "  h g f e d c b a\n";
        for (int rank = 0; rank < 8; ++rank) {
            std::cout << (rank + 1) << " ";
            for (int file = 7; file >= 0; --file) {
                chess::Square square(rank * 8 + file);
                std::cout << piece_to_unicode(board.at(square)) << " ";
            }
            std::cout << (rank + 1) << "\n";
        }
        std::cout << "  h g f e d c b a\n";
    }
    std::cout << "\n";
}

void play_human_loop(Engine& engine, chess::Color human_color) {
    chess::Board board;

    std::cout << "Game started! You are playing as " 
              << (human_color == chess::Color::WHITE ? "White" : "Black") << ".\n";
    std::cout << "Enter your moves in SAN (e.g. e4, Nf3) or UCI (e.g. e2e4) format.\n";

    while (true) {
        auto [reason, result] = board.isGameOver();
        if (reason != chess::GameResultReason::NONE) {
            print_board_unicode(board, human_color);
            std::cout << "Game Over! ";
            if (result == chess::GameResult::DRAW) {
                std::cout << "It's a draw.\n";
            } else if (result == chess::GameResult::LOSE) {
                std::cout << (board.sideToMove() == chess::Color::WHITE ? "Black" : "White") << " wins by checkmate!\n";
            }
            break;
        }

        bool human_turn = (board.sideToMove() == human_color);

        if (human_turn) {
            print_board_unicode(board, human_color);
            chess::Move move = chess::Move::NO_MOVE;
            while (true) {
                std::cout << "Your move: ";
                std::string input;
                if (!std::getline(std::cin, input)) {
                    std::cout << "\nGame aborted.\n";
                    return;
                }
                // Strip whitespace
                if (!input.empty()) {
                    input.erase(input.find_last_not_of(" \t\r\n") + 1);
                    input.erase(0, input.find_first_not_of(" \t\r\n"));
                }

                if (input.empty()) continue;

                // Try parsing as SAN
                try {
                    move = chess::uci::parseSan(board, input);
                    if (move != chess::Move::NO_MOVE && board.isLegal(move)) {
                        break;
                    }
                } catch (...) {}

                // Try parsing as UCI
                try {
                    move = chess::uci::uciToMove(board, input);
                    if (move != chess::Move::NO_MOVE && board.isLegal(move)) {
                        break;
                    }
                } catch (...) {}

                std::cout << "Invalid or illegal move. Try again.\n";
            }

            std::cout << "You played: " << chess::uci::moveToSan(board, move) << "\n";
            board.makeMove(move);
        } else {
            std::cout << "Model is thinking...\n";
            auto [best_move, value] = engine.search_position(board);
            std::cout << "Model played: " << chess::uci::moveToSan(board, best_move) 
                      << " (eval: " << value << ")\n";
            board.makeMove(best_move);
        }
    }
}

} // namespace causal_chess
