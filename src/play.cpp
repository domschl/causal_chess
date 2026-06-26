#include "play.hpp"
#include <chrono>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <fstream>
#include "web_server.hpp"
#include "nlohmann/json.hpp"

namespace causal_chess {

static std::string safe_move_to_san(const chess::Board& board, chess::Move move) {
    if (move == chess::Move::NO_MOVE) return "";
    try {
        if (board.at(move.to()) != chess::Piece() && board.at(move.to()).type() == chess::PieceType::KING) {
            return chess::uci::moveToUci(move);
        }
        return chess::uci::moveToSan(board, move);
    } catch (...) {
        return chess::uci::moveToUci(move);
    }
}

static int get_last_game_num(const std::string& csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        return 0;
    }
    std::string line;
    std::string last_line;
    // Skip header line
    std::getline(file, line);
    while (std::getline(file, line)) {
        if (!line.empty()) {
            last_line = line;
        }
    }
    if (last_line.empty()) {
        return 0;
    }
    std::stringstream ss(last_line);
    std::string game_num_str;
    if (std::getline(ss, game_num_str, ',')) {
        try {
            return std::stoi(game_num_str);
        } catch (...) {
            return 0;
        }
    }
    return 0;
}

PlayStats self_play_loop(
    Engine& engine,
    int num_games,
    const std::string& save_dir,
    int save_interval,
    bool verbose,
    bool resume,
    WebServer* web_server
) {
    PlayStats stats;
    std::filesystem::create_directories(save_dir);

    std::string csv_path = save_dir + "/stats.csv";
    int start_game_num = 0;
    std::ofstream csv_file;
    double initial_w = engine.get_heuristic_weight();
    if (resume && std::filesystem::exists(csv_path)) {
        start_game_num = get_last_game_num(csv_path);
        csv_file.open(csv_path, std::ios::app);
    } else {
        csv_file.open(csv_path, std::ios::trunc);
        csv_file << "game,loss,moves,time,nps,divergence,heuristic_weight\n";
        csv_file.flush();
    }

    for (int game_num = 1; game_num <= num_games; ++game_num) {
        auto game_start = std::chrono::steady_clock::now();
        engine.reset_stats();

        chess::Board board;
        std::vector<std::string> moves_san;
        std::vector<float> move_evals;
        std::vector<chess::Board> visited_boards;
        int move_count = 0;

        engine.set_active_position(board.getFen(), board.sideToMove() == chess::Color::WHITE ? "w" : "b", start_game_num + game_num, "");
        // Broadcast starting position
        if (web_server) {
            nlohmann::json pos_msg;
            pos_msg["type"] = "position";
            pos_msg["fen"] = board.getFen();
            pos_msg["turn"] = board.sideToMove() == chess::Color::WHITE ? "w" : "b";
            pos_msg["game_index"] = start_game_num + game_num;
            pos_msg["last_move"] = "";
            web_server->broadcast(pos_msg.dump());
        }

        // Play game until termination
        while (board.isGameOver().first == chess::GameResultReason::NONE) {
            while (engine.is_paused() && !engine.is_stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            if (engine.is_stop_requested()) {
                break;
            }

            visited_boards.push_back(board);
            auto [best_move, value] = engine.search_position(board, std::nullopt, web_server);

            if (engine.is_stop_requested()) {
                break;
            }

            // Record SAN string before making move
            std::string san = safe_move_to_san(board, best_move);
            moves_san.push_back(san);
            move_evals.push_back(value);

            board.makeMove(best_move);
            move_count++;

            engine.set_active_position(board.getFen(), board.sideToMove() == chess::Color::WHITE ? "w" : "b", start_game_num + game_num, san);
            // Broadcast move position update
            if (web_server) {
                nlohmann::json pos_msg;
                pos_msg["type"] = "position";
                pos_msg["fen"] = board.getFen();
                pos_msg["turn"] = board.sideToMove() == chess::Color::WHITE ? "w" : "b";
                pos_msg["game_index"] = start_game_num + game_num;
                pos_msg["last_move"] = san;
                web_server->broadcast(pos_msg.dump());
            }
        }

        if (engine.is_stop_requested()) {
            break;
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
        }

        // Post-game training pass: Map result to target value (White-relative)
        float outcome_val = 0.5f;
        if (result == "1-0") {
            outcome_val = 1.0f;
        } else if (result == "0-1") {
            outcome_val = 0.0f;
        }
        engine.train_on_outcome(visited_boards, outcome_val);

        auto game_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = game_end - game_start;

        // Write stats to CSV log
        if (csv_file.is_open()) {
            double total_search_time = engine.get_total_search_time_secs();
            double nps = 0.0;
            if (total_search_time > 0.0) {
                nps = engine.get_positions_evaluated() / total_search_time;
            }
            double avg_div = engine.get_avg_heuristic_nn_divergence();
            double current_w = engine.get_heuristic_weight();
            csv_file << (start_game_num + game_num) << ","
                     << engine.get_avg_loss() << ","
                     << move_count << ","
                     << elapsed.count() << ","
                     << static_cast<int64_t>(nps) << ","
                     << avg_div << ","
                     << current_w << "\n";
            csv_file.flush();
        }

        if (web_server) {
            nlohmann::json stats_msg;
            stats_msg["type"] = "stats";
            stats_msg["game"] = start_game_num + game_num;
            stats_msg["loss"] = engine.get_avg_loss();
            stats_msg["moves"] = move_count;
            stats_msg["duration_secs"] = elapsed.count();
            
            double total_search_time = engine.get_total_search_time_secs();
            double nps = 0.0;
            if (total_search_time > 0.0) {
                nps = engine.get_positions_evaluated() / total_search_time;
            }
            stats_msg["nps"] = static_cast<int64_t>(nps);
            stats_msg["h_nn_div"] = engine.get_avg_heuristic_nn_divergence();
            stats_msg["heuristic_weight"] = engine.get_heuristic_weight();
            stats_msg["record"]["white_wins"] = stats.white_wins;
            stats_msg["record"]["black_wins"] = stats.black_wins;
            stats_msg["record"]["draws"] = stats.draws;
            
            web_server->broadcast(stats_msg.dump());
        }

        if (verbose) {
            int total = stats.white_wins + stats.black_wins + stats.draws;
            std::cout << "\n============================================================\n";
            double total_search_time = engine.get_total_search_time_secs();
            double nps = 0.0;
            double pct_forward = 0.0;
            double pct_backprop = 0.0;

            if (total_search_time > 0.0) {
                nps = engine.get_positions_evaluated() / total_search_time;
            }
            double total_engine_time = total_search_time + engine.get_total_post_game_train_time_secs();
            if (total_engine_time > 0.0) {
                pct_forward = (engine.get_forward_time_secs() / total_engine_time) * 100.0;
                pct_backprop = (engine.get_backprop_time_secs() / total_engine_time) * 100.0;
            }

            std::cout << "Game " << game_num << "/" << num_games << ": " << result << "\n";
            std::cout << "  Moves:       " << (move_count + 1) / 2 << "\n";
            std::cout << "  Time:        " << elapsed.count() << "s (Search: " << total_search_time << "s, Train: " << engine.get_total_post_game_train_time_secs() << "s)\n";
            std::cout << "  Speed:       " << static_cast<int64_t>(nps) << " positions/sec\n";
            std::cout << "  Forward:     " << pct_forward << "%\n";
            std::cout << "  Backprop:    " << pct_backprop << "%\n";
            std::cout << "  TD updates:  " << engine.get_update_count() << "\n";
            std::cout << "  Avg TD loss: " << engine.get_avg_loss() << "\n";
            std::cout << "  Avg H-NN Div:" << engine.get_avg_heuristic_nn_divergence() << "\n";
            std::cout << "  Heur. Weight:" << engine.get_heuristic_weight() << "\n";
            std::cout << "  Record:      W=" << stats.white_wins << "  B=" << stats.black_wins << "  D=" << stats.draws << "  (" << total << " games)\n";
            std::cout << "------------------------------------------------------------\n";

            // Print PGN
            std::cout << "[Event \"Causal Chess Self-Play\"]\n";
            std::cout << "[White \"CausalChess\"]\n";
            std::cout << "[Black \"CausalChess\"]\n";
            std::cout << "[Round \"" << game_num << "\"]\n";
            std::cout << "[Result \"" << result << "\"]\n\n";

            for (size_t i = 0; i < moves_san.size(); ++i) {
                double pseudo_cp = (move_evals[i] - 0.5) * 20.0;
                char cp_buf[32];
                std::snprintf(cp_buf, sizeof(cp_buf), "(%+.2f)", pseudo_cp);

                if (i % 2 == 0) {
                    std::cout << (i / 2 + 1) << ". " << moves_san[i] << " " << cp_buf << " ";
                } else {
                    std::cout << moves_san[i] << " " << cp_buf << " ";
                }
            }
            std::cout << result << "\n\n";
        }

        // Step the learning rate scheduler
        engine.step_scheduler();

        // Save checkpoint
        if (game_num % save_interval == 0) {
            std::string checkpoint_path = save_dir + "/checkpoint.pt";
            engine.save_checkpoint(checkpoint_path);
            if (verbose) {
                std::cout << "  💾 Checkpoint saved: " << checkpoint_path << "\n\n";
            }
        }

        // Scheme A: Adaptive weight controller (Divergence-based Monotonic Annealing)
        if (initial_w > 0.0) {
            double avg_div = engine.get_avg_heuristic_nn_divergence();
            double target_w = initial_w * std::max(0.0, std::min(1.0, avg_div / 0.15));
            double current_w = engine.get_heuristic_weight();
            double new_w = std::min(current_w, target_w);
            double alpha = engine.get_adaptive_weight_smoothing();
            double blended_w = alpha * current_w + (1.0 - alpha) * new_w;
            engine.set_heuristic_weight(blended_w);
        }

        // Broadcast updated configuration
        if (web_server) {
            nlohmann::json msg;
            msg["type"] = "config";
            nlohmann::json cfg;
            cfg["max_depth"] = engine.get_max_depth();
            cfg["top_n"] = engine.get_top_n();
            cfg["top_n_vector"] = engine.get_top_n_vector();
            cfg["heuristic_weight"] = engine.get_heuristic_weight();
            cfg["adaptive_weight_smoothing"] = engine.get_adaptive_weight_smoothing();
            cfg["learning_rate"] = engine.get_learning_rate();
            cfg["temperature"] = engine.get_temperature();
            cfg["adaptive_scaling"] = engine.get_adaptive_scaling();
            msg["config"] = cfg;
            web_server->broadcast(msg.dump());
        }
    }

    // Save final checkpoint
    engine.save_checkpoint(save_dir + "/checkpoint.pt");

    if (csv_file.is_open()) {
        csv_file.close();
    }

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

std::string preprocess_move_input(const std::string& raw_input) {
    std::string input = raw_input;
    // Strip leading and trailing spaces
    if (!input.empty()) {
        input.erase(input.find_last_not_of(" \t\r\n") + 1);
        input.erase(0, input.find_first_not_of(" \t\r\n"));
    }

    if (input.empty()) return input;

    // 1. Map German piece letters at the beginning of the SAN move (must be uppercase)
    if (input.size() > 1 && std::isupper(input[0])) {
        if (input[0] == 'D') input[0] = 'Q';
        else if (input[0] == 'T') input[0] = 'R';
        else if (input[0] == 'L') input[0] = 'B';
        else if (input[0] == 'S') input[0] = 'N';
    }

    // 2. Map promotion characters at the end of the move (e.g. g7h8d, gxh8=D, gxh8(D), g7h8-d)
    int last_letter_idx = -1;
    for (int i = static_cast<int>(input.size()) - 1; i >= 0; --i) {
        if (std::isalpha(input[i])) {
            last_letter_idx = i;
            break;
        }
    }

    if (last_letter_idx != -1) {
        char promo = input[last_letter_idx];
        char mapped = '\0';
        if (promo == 'd' || promo == 'D') mapped = 'q';
        else if (promo == 't' || promo == 'T') mapped = 'r';
        else if (promo == 'l' || promo == 'L') mapped = 'b';
        else if (promo == 's' || promo == 'S') mapped = 'n';

        if (mapped != '\0') {
            if (last_letter_idx >= 3) {
                input[last_letter_idx] = std::isupper(promo) ? std::toupper(mapped) : mapped;
            }
        }
    }

    return input;
}

void play_human_loop(Engine& engine, chess::Color human_color, WebServer* web_server) {
    chess::Board board;

    std::cout << "Game started! You are playing as " 
              << (human_color == chess::Color::WHITE ? "White" : "Black") << ".\n";
    if (!web_server) {
        std::cout << "Enter your moves in SAN (e.g. e4, Nf3) or UCI (e.g. e2e4) format.\n";
    }

    engine.set_active_position(board.getFen(), board.sideToMove() == chess::Color::WHITE ? "w" : "b", 0, "");
    if (web_server) {
        engine.clear_human_moves();
        nlohmann::json pos_msg;
        pos_msg["type"] = "position";
        pos_msg["fen"] = board.getFen();
        pos_msg["turn"] = board.sideToMove() == chess::Color::WHITE ? "w" : "b";
        pos_msg["game_index"] = 0;
        pos_msg["last_move"] = "";
        web_server->broadcast(pos_msg.dump());
    }

    while (true) {
        if (engine.is_stop_requested()) {
            return;
        }

        auto [reason, result] = board.isGameOver();
        if (reason != chess::GameResultReason::NONE) {
            print_board_unicode(board, human_color);
            std::cout << "Game Over! ";
            std::string outcome_str = "Game Over! ";
            if (result == chess::GameResult::DRAW) {
                std::cout << "It's a draw.\n";
                outcome_str += "It's a draw.";
            } else if (result == chess::GameResult::LOSE) {
                std::string winner = (board.sideToMove() == chess::Color::WHITE ? "Black" : "White");
                std::cout << winner << " wins by checkmate!\n";
                outcome_str += winner + " wins by checkmate!";
            }
            if (web_server) {
                nlohmann::json over_msg;
                over_msg["type"] = "game_over";
                over_msg["result"] = outcome_str;
                web_server->broadcast(over_msg.dump());
            }
            break;
        }

        bool human_turn = (board.sideToMove() == human_color);

        if (human_turn) {
            if (!web_server) {
                print_board_unicode(board, human_color);
            }
            chess::Move move = chess::Move::NO_MOVE;
            while (true) {
                if (engine.is_stop_requested()) {
                    return;
                }

                std::string input;
                if (web_server) {
                    auto opt_move = engine.pop_human_move();
                    if (opt_move.has_value()) {
                        input = *opt_move;
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        continue;
                    }
                } else {
                    std::cout << "Your move: ";
                    if (!std::getline(std::cin, input)) {
                        std::cout << "\nGame aborted.\n";
                        return;
                    }
                }

                // Strip whitespace
                if (!input.empty()) {
                    input.erase(input.find_last_not_of(" \t\r\n") + 1);
                    input.erase(0, input.find_first_not_of(" \t\r\n"));
                }
                input = preprocess_move_input(input);
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

                if (web_server) {
                    nlohmann::json err_msg;
                    err_msg["type"] = "move_error";
                    err_msg["error"] = "Invalid or illegal move: " + input;
                    web_server->broadcast(err_msg.dump());
                } else {
                    std::cout << "Invalid or illegal move. Try again.\n";
                }
            }

            std::string san = safe_move_to_san(board, move);
            std::cout << "You played: " << san << "\n";
            board.makeMove(move);

            engine.set_active_position(board.getFen(), board.sideToMove() == chess::Color::WHITE ? "w" : "b", 0, san);
            if (web_server) {
                nlohmann::json pos_msg;
                pos_msg["type"] = "position";
                pos_msg["fen"] = board.getFen();
                pos_msg["turn"] = board.sideToMove() == chess::Color::WHITE ? "w" : "b";
                pos_msg["game_index"] = 0;
                pos_msg["last_move"] = san;
                web_server->broadcast(pos_msg.dump());
            }
        } else {
            std::cout << "Model is thinking...\n";
            auto [best_move, value] = engine.search_position(board, std::nullopt, web_server);
            
            if (engine.is_stop_requested()) {
                return;
            }

            std::string san = safe_move_to_san(board, best_move);
            std::cout << "Model played: " << san << " (eval: " << value << ")\n";
            board.makeMove(best_move);

            engine.set_active_position(board.getFen(), board.sideToMove() == chess::Color::WHITE ? "w" : "b", 0, san);
            if (web_server) {
                nlohmann::json pos_msg;
                pos_msg["type"] = "position";
                pos_msg["fen"] = board.getFen();
                pos_msg["turn"] = board.sideToMove() == chess::Color::WHITE ? "w" : "b";
                pos_msg["game_index"] = 0;
                pos_msg["last_move"] = san;
                web_server->broadcast(pos_msg.dump());
            }
        }
    }
}

} // namespace causal_chess
