#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <nlohmann/json.hpp>
#include "search.hpp"
#include "play.hpp"
#include "web_server.hpp"
#include "chess.hpp"

using namespace causal_chess;

static void broadcast_config(Engine& engine, WebServer& web_server) {
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
    web_server.broadcast(msg.dump());
}

// Auto-detect the latest .pt checkpoint in a directory, prioritizing checkpoint.pt
std::string find_latest_checkpoint(const std::string& directory) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(directory)) {
        return "";
    }

    // Prioritize checkpoint.pt if it exists
    fs::path preferred = fs::path(directory) / "checkpoint.pt";
    if (fs::is_regular_file(preferred)) {
        return preferred.string();
    }

    fs::path latest;
    fs::file_time_type latest_time;
    bool found = false;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".pt") {
            auto time = entry.last_write_time();
            if (!found || time > latest_time) {
                latest_time = time;
                latest = entry.path();
                found = true;
            }
        }
    }
    return found ? latest.string() : "";
}

void print_global_help() {
    std::cout << "Usage: causal-chess-cpp <command> [options]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  play        Start a self-play training loop\n";
    std::cout << "  play-human  Play interactively against the model\n";
    std::cout << "  eval        Evaluate a single FEN position\n";
    std::cout << "  move        Find the best move in a FEN position\n";
    std::cout << "  plot        Plot training loss development using gnuplot\n\n";
    std::cout << "Run 'causal-chess-cpp <command> --help' for details on a specific command.\n";
}

void print_plot_help() {
    std::cout << "Usage: causal-chess-cpp plot [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --save-dir <path>    Checkpoint directory (default: checkpoints)\n";
    std::cout << "  --output <path>      Output plot image file path (default: show in terminal)\n";
    std::cout << "  --watch              Live watch mode: updates every 2 seconds\n";
}

void print_play_human_help() {
    std::cout << "Usage: causal-chess-cpp play-human [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --depth <n>          Search depth (default: 4)\n";
    std::cout << "  --top-n <val>        Moves to expand per node (constant or comma-separated list of size <depth> in non-increasing order, default: 5)\n";
    std::cout << "  --color <str>        Your color: white, black (default: white)\n";
    std::cout << "  --checkpoint <path>  Specific checkpoint file to load\n";
    std::cout << "  --device <str>       Torch device: cpu, mps, cuda, auto (default: cpu)\n";
    std::cout << "  --heuristic-weight <val> Weight of quiescent material/space heuristic in [0, 1] (default: 0.5)\n";
    std::cout << "  --adaptive-weight-smoothing <val> Smoothing factor for adaptive heuristic weight controller (default: 0.8)\n";
}

void print_play_help() {
    std::cout << "Usage: causal-chess-cpp play [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --games <n>            Number of self-play games (default: 100)\n";
    std::cout << "  --depth <n>            Search depth (default: 4)\n";
    std::cout << "  --top-n <val>          Moves to expand per node (constant or comma-separated list of size <depth> in non-increasing order, default: 5)\n";
    std::cout << "  --lr <val>             Learning rate (default: 1e-4)\n";
    std::cout << "  --device <str>         Torch device: cpu, mps, cuda, auto (default: cpu)\n";
    std::cout << "  --save-dir <path>      Checkpoint directory (default: checkpoints)\n";
    std::cout << "  --save-interval <n>    Save checkpoint every n games (default: 10)\n";
    std::cout << "  --temperature <val>    Exploration temperature for self-play (default: 0.0)\n";
    std::cout << "  --post-game-epochs <n> Number of epochs for post-game outcome training (default: 2)\n";
    std::cout << "  --discount-factor <val> Decay factor for post-game outcome training (default: 0.97)\n";
    std::cout << "  --replay-buffer-size <n> Max size of replay buffer (default: 5000)\n";
    std::cout << "  --replay-batch-size <n> Mini-batch size sampled from replay buffer (default: 128)\n";
    std::cout << "  --checkpoint <path>    Specific checkpoint file to load\n";
    std::cout << "  --fresh                Ignore existing checkpoints and start fresh\n";
    std::cout << "  --heuristic-weight <val> Weight of quiescent material/space heuristic in [0, 1] (default: 0.5)\n";
    std::cout << "  --adaptive-weight-smoothing <val> Smoothing factor for adaptive heuristic weight controller (default: 0.8)\n";
    std::cout << "  --lr-decay-rate <val>  Learning rate decay multiplier (default: 0.998)\n";
    std::cout << "  --lr-decay-steps <n>   Interval of games to decay learning rate (default: 10)\n";
    std::cout << "  --min-lr <val>         Minimum learning rate threshold (default: 1e-6)\n";
    std::cout << "  --live-lr-scale <val>  Learning rate scale for online TD updates (default: 1.0)\n";
    std::cout << "  --influence-ratio <val> Target influence ratio for adaptive scaling (default: 0.5)\n";
    std::cout << "  --adaptive-scaling     Enable dynamic/adaptive scaling of live learning rate and epochs (default: disabled)\n";
}

void print_eval_help() {
    std::cout << "Usage: causal-chess-cpp eval <fen> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --checkpoint <path>  Model checkpoint file\n";
    std::cout << "  --device <str>       Torch device (default: cpu)\n";
}

void print_move_help() {
    std::cout << "Usage: causal-chess-cpp move <fen> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --depth <n>          Search depth (default: 4)\n";
    std::cout << "  --top-n <val>        Moves to expand per node (constant or comma-separated list of size <depth> in non-increasing order, default: 5)\n";
    std::cout << "  --lr <val>           Learning rate (default: 1e-4)\n";
    std::cout << "  --checkpoint <path>  Model checkpoint file\n";
    std::cout << "  --device <str>       Torch device (default: cpu)\n";
}

int handle_play(const std::vector<std::string>& args) {
    int games = 100;
    int depth = 4;
    std::string top_n_str = "5";
    double lr = 1e-4;
    std::string device = "cpu";
    std::string save_dir = "checkpoints";
    int save_interval = 10;
    std::string checkpoint_path = "";
    bool fresh = false;
    double temperature = 0.0;
    int post_game_epochs = 2;
    double discount_factor = 0.97;
    int replay_buffer_size = 5000;
    int replay_batch_size = 128;
    double heuristic_weight = 0.5;
    double adaptive_weight_smoothing = 0.8;
    double lr_decay_rate = 0.998;
    int lr_decay_steps = 10;
    double min_lr = 1e-6;
    double live_lr_scale = 1.0;
    double influence_ratio = 0.5;
    bool adaptive_scaling = false;

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--help" || args[i] == "-h") {
            print_play_help();
            return 0;
        } else if (args[i] == "--games" && i + 1 < args.size()) {
            games = std::stoi(args[++i]);
        } else if (args[i] == "--depth" && i + 1 < args.size()) {
            depth = std::stoi(args[++i]);
        } else if (args[i] == "--top-n" && i + 1 < args.size()) {
            top_n_str = args[++i];
        } else if (args[i] == "--lr" && i + 1 < args.size()) {
            lr = std::stod(args[++i]);
        } else if (args[i] == "--device" && i + 1 < args.size()) {
            device = args[++i];
        } else if (args[i] == "--save-dir" && i + 1 < args.size()) {
            save_dir = args[++i];
        } else if (args[i] == "--save-interval" && i + 1 < args.size()) {
            save_interval = std::stoi(args[++i]);
        } else if (args[i] == "--checkpoint" && i + 1 < args.size()) {
            checkpoint_path = args[++i];
        } else if (args[i] == "--fresh") {
            fresh = true;
        } else if (args[i] == "--temperature" && i + 1 < args.size()) {
            temperature = std::stod(args[++i]);
        } else if (args[i] == "--post-game-epochs" && i + 1 < args.size()) {
            post_game_epochs = std::stoi(args[++i]);
        } else if (args[i] == "--discount-factor" && i + 1 < args.size()) {
            discount_factor = std::stod(args[++i]);
        } else if (args[i] == "--replay-buffer-size" && i + 1 < args.size()) {
            replay_buffer_size = std::stoi(args[++i]);
        } else if (args[i] == "--replay-batch-size" && i + 1 < args.size()) {
            replay_batch_size = std::stoi(args[++i]);
        } else if (args[i] == "--heuristic-weight" && i + 1 < args.size()) {
            heuristic_weight = std::stod(args[++i]);
        } else if (args[i] == "--adaptive-weight-smoothing" && i + 1 < args.size()) {
            adaptive_weight_smoothing = std::stod(args[++i]);
        } else if (args[i] == "--lr-decay-rate" && i + 1 < args.size()) {
            lr_decay_rate = std::stod(args[++i]);
        } else if (args[i] == "--lr-decay-steps" && i + 1 < args.size()) {
            lr_decay_steps = std::stoi(args[++i]);
        } else if (args[i] == "--min-lr" && i + 1 < args.size()) {
            min_lr = std::stod(args[++i]);
        } else if (args[i] == "--live-lr-scale" && i + 1 < args.size()) {
            live_lr_scale = std::stod(args[++i]);
        } else if (args[i] == "--influence-ratio" && i + 1 < args.size()) {
            influence_ratio = std::stod(args[++i]);
        } else if (args[i] == "--adaptive-scaling") {
            adaptive_scaling = true;
        } else {
            std::cerr << "Unknown play option: " << args[i] << "\n";
            return 1;
        }
    }

    int top_n = 5;
    std::vector<int> top_n_vector;
    std::string error_msg;
    if (!parse_top_n_vector(top_n_str, depth, top_n, top_n_vector, error_msg)) {
        std::cerr << "Error parsing --top-n: " << error_msg << "\n";
        return 1;
    }

    SearchConfig config;
    config.max_depth = depth;
    config.top_n = top_n;
    config.top_n_vector = top_n_vector;
    config.learning_rate = lr;
    config.device = device;
    config.temperature = temperature;
    config.post_game_epochs = post_game_epochs;
    config.discount_factor = discount_factor;
    config.replay_buffer_size = replay_buffer_size;
    config.replay_batch_size = replay_batch_size;
    config.heuristic_weight = heuristic_weight;
    config.adaptive_weight_smoothing = adaptive_weight_smoothing;
    config.lr_decay_rate = lr_decay_rate;
    config.lr_decay_steps = lr_decay_steps;
    config.min_learning_rate = min_lr;
    config.live_lr_scale = live_lr_scale;
    config.adaptive_influence_ratio = influence_ratio;
    config.adaptive_scaling = adaptive_scaling;

    Engine engine(config);

    // Auto-resume logic
    std::string loaded_checkpoint = "";
    if (!checkpoint_path.empty()) {
        engine.load_checkpoint(checkpoint_path);
        loaded_checkpoint = checkpoint_path;
    } else if (!fresh) {
        std::string latest = find_latest_checkpoint(save_dir);
        if (latest.empty() && save_dir == "checkpoints") {
            latest = find_latest_checkpoint("../checkpoints");
        }
        if (!latest.empty()) {
            engine.load_checkpoint(latest);
            loaded_checkpoint = latest;
        }
    }

    if (!loaded_checkpoint.empty()) {
        std::cout << "Resumed from checkpoint: " << loaded_checkpoint << "\n";
    } else {
        std::cout << "Starting with fresh (random) weights\n";
    }

    std::cout << "============================================================\n";
    std::cout << "Causal Chess (C++) — Self-Play Training\n";
    std::cout << "  Depth:               " << config.max_depth << "\n";
    std::cout << "  Top-N:               " << config.top_n;
    if (!config.top_n_vector.empty()) {
        std::cout << " [";
        for (size_t i = 0; i < config.top_n_vector.size(); ++i) {
            std::cout << config.top_n_vector[i] << (i + 1 < config.top_n_vector.size() ? "," : "");
        }
        std::cout << "]";
    }
    std::cout << "\n";
    std::cout << "  LR:                  " << config.learning_rate << "\n";
    std::cout << "  Device:              " << engine.get_device() << "\n";
    std::cout << "  Games:               " << games << "\n";
    std::cout << "  Temperature:         " << config.temperature << "\n";
    std::cout << "  Post-Game Epochs:    " << config.post_game_epochs << "\n";
    std::cout << "  Discount Factor:     " << config.discount_factor << "\n";
    std::cout << "  Replay Buffer Size:  " << config.replay_buffer_size << "\n";
    std::cout << "  Replay Batch Size:   " << config.replay_batch_size << "\n";
    std::cout << "  Heuristic Weight (w):" << config.heuristic_weight << "\n";
    std::cout << "  Weight Smoothing:    " << config.adaptive_weight_smoothing << "\n";
    std::cout << "  Live LR Scale:       " << config.live_lr_scale << "\n";
    std::cout << "  Adaptive Scaling:    " << (config.adaptive_scaling ? "yes" : "no") << "\n";
    std::cout << "  Params:              " << engine.get_model()->param_count() << "\n";
    std::cout << "============================================================\n";

    std::string web_dir = "web";
    if (!std::filesystem::exists(web_dir) && std::filesystem::exists("../web")) {
        web_dir = "../web";
    }
    web_dir = std::filesystem::weakly_canonical(web_dir).string();
    WebServer web_server(8080, web_dir);
    web_server.start();

    // Game thread state variables
    std::mutex game_thread_mutex;
    std::thread game_thread;
    std::atomic<bool> game_thread_running{false};
    std::string active_mode = "self_play";

    auto stop_active_game = [&]() {
        if (game_thread_running) {
            engine.set_stop_requested(true);
            if (game_thread.joinable()) {
                game_thread.join();
            }
            game_thread_running = false;
        }
    };

    auto start_game_thread = [&](const std::string& mode) {
        stop_active_game();
        engine.set_stop_requested(false);
        engine.set_paused(false);
        active_mode = mode;
        game_thread_running = true;

        game_thread = std::thread([&engine, &web_server, mode, save_dir, save_interval, games, fresh, &game_thread_running]() {
            try {
                if (mode == "self_play") {
                    self_play_loop(engine, games, save_dir, save_interval, true, !fresh, &web_server);
                } else if (mode == "human_white") {
                    play_human_loop(engine, chess::Color::WHITE, &web_server);
                } else if (mode == "human_black") {
                    play_human_loop(engine, chess::Color::BLACK, &web_server);
                }
            } catch (const std::exception& e) {
                std::cerr << "Game loop error: " << e.what() << "\n";
            }
            game_thread_running = false;
        });
    };

    // WebSocket Message Handling
    web_server.on_message([&](const std::string& msg_str) {
        try {
            auto js = nlohmann::json::parse(msg_str);
            if (js.contains("action")) {
                std::string action = js["action"];
                if (action == "pause") {
                    engine.set_paused(true);
                    std::cout << "  ⏸️ Play paused via Web Interface\n";
                } else if (action == "resume") {
                    engine.set_paused(false);
                    std::cout << "  ▶️ Play resumed via Web Interface\n";
                } else if (action == "start") {
                    std::lock_guard<std::mutex> lock(game_thread_mutex);
                    std::string mode = js.value("mode", "self_play");
                    
                    if (js.contains("config")) {
                        auto cfg = js["config"];
                        if (cfg.contains("max_depth")) {
                            engine.set_max_depth(cfg["max_depth"].get<int>());
                        }
                        if (cfg.contains("top_n")) {
                            engine.set_top_n(cfg["top_n"].get<int>());
                        }
                        if (cfg.contains("top_n_vector")) {
                            std::vector<int> vec = cfg["top_n_vector"].get<std::vector<int>>();
                            engine.set_top_n_vector(vec);
                        }
                        if (cfg.contains("heuristic_weight")) {
                            engine.set_heuristic_weight(cfg["heuristic_weight"].get<double>());
                        }
                        if (cfg.contains("adaptive_weight_smoothing")) {
                            engine.set_adaptive_weight_smoothing(cfg["adaptive_weight_smoothing"].get<double>());
                        }
                        if (cfg.contains("learning_rate")) {
                            engine.set_learning_rate(cfg["learning_rate"].get<double>());
                        }
                        if (cfg.contains("temperature")) {
                            engine.set_temperature(cfg["temperature"].get<double>());
                        }
                        if (cfg.contains("adaptive_scaling")) {
                            engine.set_adaptive_scaling(cfg["adaptive_scaling"].get<bool>());
                        }
                    }

                    std::cout << "  🚀 Starting session (" << mode << ") via Web Interface\n";
                    start_game_thread(mode);
                    broadcast_config(engine, web_server);
                } else if (action == "update_config") {
                    if (js.contains("config")) {
                        auto cfg = js["config"];
                        if (cfg.contains("max_depth")) {
                            engine.set_max_depth(cfg["max_depth"].get<int>());
                        }
                        if (cfg.contains("top_n")) {
                            engine.set_top_n(cfg["top_n"].get<int>());
                        }
                        if (cfg.contains("top_n_vector")) {
                            std::vector<int> vec = cfg["top_n_vector"].get<std::vector<int>>();
                            engine.set_top_n_vector(vec);
                        }
                        if (cfg.contains("heuristic_weight")) {
                            engine.set_heuristic_weight(cfg["heuristic_weight"].get<double>());
                        }
                        if (cfg.contains("adaptive_weight_smoothing")) {
                            engine.set_adaptive_weight_smoothing(cfg["adaptive_weight_smoothing"].get<double>());
                        }
                        if (cfg.contains("learning_rate")) {
                            engine.set_learning_rate(cfg["learning_rate"].get<double>());
                        }
                        if (cfg.contains("temperature")) {
                            engine.set_temperature(cfg["temperature"].get<double>());
                        }
                        if (cfg.contains("adaptive_scaling")) {
                            engine.set_adaptive_scaling(cfg["adaptive_scaling"].get<bool>());
                        }
                    }
                    std::cout << "  🔧 Configuration updated via Web Interface\n";
                    broadcast_config(engine, web_server);
                } else if (action == "get_config") {
                    broadcast_config(engine, web_server);
                } else if (action == "human_move") {
                    if (js.contains("move")) {
                        std::string mv = js["move"];
                        engine.push_human_move(mv);
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error handling websocket message: " << e.what() << "\n";
        }
    });

    start_game_thread("self_play");

    // Keep process alive while serving Web UI
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    stop_active_game();
    return 0;
}

int handle_eval(const std::vector<std::string>& args) {
    if (args.empty() || args[0] == "--help" || args[0] == "-h") {
        print_eval_help();
        return args.empty() ? 1 : 0;
    }

    std::string fen = args[0];
    std::string checkpoint_path = "";
    std::string device = "cpu";

    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--checkpoint" && i + 1 < args.size()) {
            checkpoint_path = args[++i];
        } else if (args[i] == "--device" && i + 1 < args.size()) {
            device = args[++i];
        } else {
            std::cerr << "Unknown eval option: " << args[i] << "\n";
            return 1;
        }
    }

    SearchConfig config;
    config.device = device;
    Engine engine(config);

    if (!checkpoint_path.empty()) {
        engine.load_checkpoint(checkpoint_path);
    }

    chess::Board board(fen);
    float value = engine.evaluate(board);

    std::cout << "FEN:   " << fen << "\n";
    std::cout << "Value: " << value << "  (White-relative win probability)\n";
    if (value > 0.55f) {
        std::cout << "       → White is better\n";
    } else if (value < 0.45f) {
        std::cout << "       → Black is better\n";
    } else {
        std::cout << "       → Roughly equal\n";
    }
    return 0;
}

int handle_move(const std::vector<std::string>& args) {
    if (args.empty() || args[0] == "--help" || args[0] == "-h") {
        print_move_help();
        return args.empty() ? 1 : 0;
    }

    std::string fen = args[0];
    int depth = 4;
    std::string top_n_str = "5";
    double lr = 1e-4;
    std::string checkpoint_path = "";
    std::string device = "cpu";

    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--depth" && i + 1 < args.size()) {
            depth = std::stoi(args[++i]);
        } else if (args[i] == "--top-n" && i + 1 < args.size()) {
            top_n_str = args[++i];
        } else if (args[i] == "--lr" && i + 1 < args.size()) {
            lr = std::stod(args[++i]);
        } else if (args[i] == "--checkpoint" && i + 1 < args.size()) {
            checkpoint_path = args[++i];
        } else if (args[i] == "--device" && i + 1 < args.size()) {
            device = args[++i];
        } else {
            std::cerr << "Unknown move option: " << args[i] << "\n";
            return 1;
        }
    }

    int top_n = 5;
    std::vector<int> top_n_vector;
    std::string error_msg;
    if (!parse_top_n_vector(top_n_str, depth, top_n, top_n_vector, error_msg)) {
        std::cerr << "Error parsing --top-n: " << error_msg << "\n";
        return 1;
    }

    SearchConfig config;
    config.max_depth = depth;
    config.top_n = top_n;
    config.top_n_vector = top_n_vector;
    config.learning_rate = lr;
    config.device = device;

    Engine engine(config);

    if (!checkpoint_path.empty()) {
        engine.load_checkpoint(checkpoint_path);
    }

    chess::Board board(fen);
    auto [move, value] = engine.search_position(board);

    double total_search_time = engine.get_total_search_time_secs();
    double nps = 0.0;
    double pct_forward = 0.0;
    double pct_backprop = 0.0;

    if (total_search_time > 0.0) {
        nps = engine.get_positions_evaluated() / total_search_time;
        pct_forward = (engine.get_forward_time_secs() / total_search_time) * 100.0;
        pct_backprop = (engine.get_backprop_time_secs() / total_search_time) * 100.0;
    }

    std::cout << "FEN:   " << fen << "\n";
    std::cout << "Move:  " << chess::uci::moveToSan(board, move) << "  (" << chess::uci::moveToUci(move) << ")\n";
    std::cout << "Value: " << value << "  (White-relative)\n";
    std::cout << "Performance:\n";
    std::cout << "  Search time: " << total_search_time << "s\n";
    std::cout << "  Speed:       " << static_cast<int64_t>(nps) << " positions/sec\n";
    std::cout << "  Forward:     " << pct_forward << "%\n";
    std::cout << "  Backprop:    " << pct_backprop << "%\n";
    std::cout << "  TD updates:  " << engine.get_update_count() << "\n";
    return 0;
}

int handle_play_human(const std::vector<std::string>& args) {
    int depth = 4;
    std::string top_n_str = "5";
    std::string device = "cpu";
    std::string color_str = "white";
    std::string checkpoint_path = "";
    double heuristic_weight = 0.5;
    double adaptive_weight_smoothing = 0.8;

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--help" || args[i] == "-h") {
            print_play_human_help();
            return 0;
        } else if (args[i] == "--depth" && i + 1 < args.size()) {
            depth = std::stoi(args[++i]);
        } else if (args[i] == "--top-n" && i + 1 < args.size()) {
            top_n_str = args[++i];
        } else if (args[i] == "--color" && i + 1 < args.size()) {
            color_str = args[++i];
        } else if (args[i] == "--checkpoint" && i + 1 < args.size()) {
            checkpoint_path = args[++i];
        } else if (args[i] == "--device" && i + 1 < args.size()) {
            device = args[++i];
        } else if (args[i] == "--heuristic-weight" && i + 1 < args.size()) {
            heuristic_weight = std::stod(args[++i]);
        } else if (args[i] == "--adaptive-weight-smoothing" && i + 1 < args.size()) {
            adaptive_weight_smoothing = std::stod(args[++i]);
        } else {
            std::cerr << "Unknown play-human option: " << args[i] << "\n";
            return 1;
        }
    }

    chess::Color human_color = chess::Color::WHITE;
    if (color_str == "black" || color_str == "b") {
        human_color = chess::Color::BLACK;
    } else if (color_str != "white" && color_str != "w") {
        std::cerr << "Warning: unknown color '" << color_str << "', defaulting to White.\n";
    }

    int top_n = 5;
    std::vector<int> top_n_vector;
    std::string error_msg;
    if (!parse_top_n_vector(top_n_str, depth, top_n, top_n_vector, error_msg)) {
        std::cerr << "Error parsing --top-n: " << error_msg << "\n";
        return 1;
    }

    SearchConfig config;
    config.max_depth = depth;
    config.top_n = top_n;
    config.top_n_vector = top_n_vector;
    config.learning_rate = 0.0; // Freeze learning during human play
    config.device = device;
    config.heuristic_weight = heuristic_weight;
    config.adaptive_weight_smoothing = adaptive_weight_smoothing;

    Engine engine(config);

    if (!checkpoint_path.empty()) {
        engine.load_checkpoint(checkpoint_path);
        std::cout << "Loaded checkpoint: " << checkpoint_path << "\n";
    } else {
        std::string latest = find_latest_checkpoint("checkpoints");
        if (latest.empty()) {
            latest = find_latest_checkpoint("../checkpoints");
        }
        if (!latest.empty()) {
            engine.load_checkpoint(latest);
            std::cout << "Auto-resumed from latest checkpoint: " << latest << "\n";
        } else {
            std::cout << "No checkpoints found. Playing with fresh (random) weights.\n";
        }
    }

    std::string web_dir = "web";
    if (!std::filesystem::exists(web_dir) && std::filesystem::exists("../web")) {
        web_dir = "../web";
    }
    web_dir = std::filesystem::weakly_canonical(web_dir).string();
    WebServer web_server(8080, web_dir);
    web_server.start();

    // Game thread state variables
    std::mutex game_thread_mutex;
    std::thread game_thread;
    std::atomic<bool> game_thread_running{false};
    std::string active_mode = (human_color == chess::Color::WHITE ? "human_white" : "human_black");

    auto stop_active_game = [&]() {
        if (game_thread_running) {
            engine.set_stop_requested(true);
            if (game_thread.joinable()) {
                game_thread.join();
            }
            game_thread_running = false;
        }
    };

    auto start_game_thread = [&](const std::string& mode) {
        stop_active_game();
        engine.set_stop_requested(false);
        engine.set_paused(false);
        active_mode = mode;
        game_thread_running = true;

        game_thread = std::thread([&engine, &web_server, mode, &game_thread_running]() {
            try {
                if (mode == "self_play") {
                    self_play_loop(engine, 100, "checkpoints", 10, true, true, &web_server);
                } else if (mode == "human_white") {
                    play_human_loop(engine, chess::Color::WHITE, &web_server);
                } else if (mode == "human_black") {
                    play_human_loop(engine, chess::Color::BLACK, &web_server);
                }
            } catch (const std::exception& e) {
                std::cerr << "Game loop error: " << e.what() << "\n";
            }
            game_thread_running = false;
        });
    };

    // WebSocket Message Handling
    web_server.on_message([&](const std::string& msg_str) {
        try {
            auto js = nlohmann::json::parse(msg_str);
            if (js.contains("action")) {
                std::string action = js["action"];
                if (action == "pause") {
                    engine.set_paused(true);
                    std::cout << "  ⏸️ Play paused via Web Interface\n";
                } else if (action == "resume") {
                    engine.set_paused(false);
                    std::cout << "  ▶️ Play resumed via Web Interface\n";
                } else if (action == "start") {
                    std::lock_guard<std::mutex> lock(game_thread_mutex);
                    std::string mode = js.value("mode", "self_play");
                    
                    if (js.contains("config")) {
                        auto cfg = js["config"];
                        if (cfg.contains("max_depth")) {
                            engine.set_max_depth(cfg["max_depth"].get<int>());
                        }
                        if (cfg.contains("top_n")) {
                            engine.set_top_n(cfg["top_n"].get<int>());
                        }
                        if (cfg.contains("top_n_vector")) {
                            std::vector<int> vec = cfg["top_n_vector"].get<std::vector<int>>();
                            engine.set_top_n_vector(vec);
                        }
                        if (cfg.contains("heuristic_weight")) {
                            engine.set_heuristic_weight(cfg["heuristic_weight"].get<double>());
                        }
                        if (cfg.contains("adaptive_weight_smoothing")) {
                            engine.set_adaptive_weight_smoothing(cfg["adaptive_weight_smoothing"].get<double>());
                        }
                        if (cfg.contains("learning_rate")) {
                            engine.set_learning_rate(cfg["learning_rate"].get<double>());
                        }
                        if (cfg.contains("temperature")) {
                            engine.set_temperature(cfg["temperature"].get<double>());
                        }
                        if (cfg.contains("adaptive_scaling")) {
                            engine.set_adaptive_scaling(cfg["adaptive_scaling"].get<bool>());
                        }
                    }

                    std::cout << "  🚀 Starting session (" << mode << ") via Web Interface\n";
                    start_game_thread(mode);
                    broadcast_config(engine, web_server);
                } else if (action == "update_config") {
                    if (js.contains("config")) {
                        auto cfg = js["config"];
                        if (cfg.contains("max_depth")) {
                            engine.set_max_depth(cfg["max_depth"].get<int>());
                        }
                        if (cfg.contains("top_n")) {
                            engine.set_top_n(cfg["top_n"].get<int>());
                        }
                        if (cfg.contains("top_n_vector")) {
                            std::vector<int> vec = cfg["top_n_vector"].get<std::vector<int>>();
                            engine.set_top_n_vector(vec);
                        }
                        if (cfg.contains("heuristic_weight")) {
                            engine.set_heuristic_weight(cfg["heuristic_weight"].get<double>());
                        }
                        if (cfg.contains("adaptive_weight_smoothing")) {
                            engine.set_adaptive_weight_smoothing(cfg["adaptive_weight_smoothing"].get<double>());
                        }
                        if (cfg.contains("learning_rate")) {
                            engine.set_learning_rate(cfg["learning_rate"].get<double>());
                        }
                        if (cfg.contains("temperature")) {
                            engine.set_temperature(cfg["temperature"].get<double>());
                        }
                        if (cfg.contains("adaptive_scaling")) {
                            engine.set_adaptive_scaling(cfg["adaptive_scaling"].get<bool>());
                        }
                    }
                    std::cout << "  🔧 Configuration updated via Web Interface\n";
                    broadcast_config(engine, web_server);
                } else if (action == "get_config") {
                    broadcast_config(engine, web_server);
                } else if (action == "human_move") {
                    if (js.contains("move")) {
                        std::string mv = js["move"];
                        engine.push_human_move(mv);
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error handling websocket message: " << e.what() << "\n";
        }
    });

    start_game_thread(human_color == chess::Color::WHITE ? "human_white" : "human_black");

    // Keep process alive while serving Web UI
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    stop_active_game();
    return 0;
}

bool has_gnuplot() {
#ifdef _WIN32
    return std::system("where gnuplot > NUL 2>&1") == 0;
#else
    return std::system("which gnuplot > /dev/null 2>&1") == 0;
#endif
}

int handle_plot(const std::vector<std::string>& args) {
    std::string save_dir = "checkpoints";
    std::string output = "";
    bool watch = false;

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--help" || args[i] == "-h") {
            print_plot_help();
            return 0;
        } else if (args[i] == "--save-dir" && i + 1 < args.size()) {
            save_dir = args[++i];
        } else if (args[i] == "--output" && i + 1 < args.size()) {
            output = args[++i];
        } else if (args[i] == "--watch") {
            watch = true;
        } else {
            std::cerr << "Unknown plot option: " << args[i] << "\n";
            return 1;
        }
    }

    std::string csv_path = save_dir + "/stats.csv";
    if (!std::filesystem::exists(csv_path)) {
        std::cerr << "Error: Log file not found at " << csv_path << ". Run training first.\n";
        return 1;
    }

    if (!has_gnuplot()) {
        std::cerr << "Error: 'gnuplot' is not installed or not in PATH.\n";
        std::cerr << "Please install gnuplot (e.g. 'brew install gnuplot') or inspect the log file directly in:\n";
        std::cerr << "  " << csv_path << "\n";
        return 1;
    }

    std::stringstream ss;
    ss << "gnuplot -e \"set datafile separator ','; ";

    if (!output.empty()) {
        std::filesystem::path out_path(output);
        std::string ext = out_path.extension().string();
        for (auto& c : ext) c = std::tolower(c);

        if (ext == ".png") {
            ss << "set terminal png size 800,600; ";
        } else if (ext == ".svg") {
            ss << "set terminal svg size 800,600; ";
        } else if (ext == ".pdf") {
            ss << "set terminal pdf; ";
        } else {
            ss << "set terminal png size 800,600; ";
        }
        ss << "set output '" << output << "'; ";
        ss << "set title 'TD Loss Development'; set xlabel 'Game'; set ylabel 'Average TD Loss'; ";
        ss << "plot '" << csv_path << "' using 1:2 with lines title 'TD Loss'";
    } else {
        // terminal plotting
        ss << "set terminal dumb 80 25; ";
        ss << "set title 'TD Loss Development'; set xlabel 'Game'; set ylabel 'Avg Loss'; ";
        ss << "plot '" << csv_path << "' using 1:2 with lines title 'Loss'";
    }

    if (watch) {
        ss << "; while (1) { pause 2; replot }";
    }

    ss << "\"";

    if (watch && output.empty()) {
        std::cout << "Watching loss live. Press Ctrl+C to exit...\n\n";
    }

    int ret = std::system(ss.str().c_str());
    if (ret != 0) {
        std::cerr << "Gnuplot process exited with code " << ret << "\n";
        return 1;
    }

    if (!output.empty()) {
        std::cout << "Plot saved successfully to: " << output << "\n";
    }

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_global_help();
        return 1;
    }

    std::string command = argv[1];
    std::vector<std::string> args;
    for (int i = 2; i < argc; ++i) {
        args.push_back(argv[i]);
    }

    try {
        if (command == "play") {
            return handle_play(args);
        } else if (command == "play-human") {
            return handle_play_human(args);
        } else if (command == "eval") {
            return handle_eval(args);
        } else if (command == "move") {
            return handle_move(args);
        } else if (command == "plot") {
            return handle_plot(args);
        } else if (command == "--help" || command == "-h") {
            print_global_help();
            return 0;
        } else {
            std::cerr << "Unknown command: " << command << "\n\n";
            print_global_help();
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
