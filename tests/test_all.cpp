#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>
#include "chess.hpp"
#include "encoding.hpp"
#include "model.hpp"
#include "search.hpp"

using namespace causal_chess;

namespace causal_chess {
    std::string preprocess_move_input(const std::string& raw_input);
}

void test_encoding_shape_and_values() {
    std::cout << "Running: test_encoding_shape_and_values... ";

    chess::Board board; // starting position
    torch::Tensor tensor = board_to_tensor(board);

    // Verify shape and type
    assert(tensor.dim() == 3);
    assert(tensor.size(0) == NUM_PLANES);
    assert(tensor.size(1) == 8);
    assert(tensor.size(2) == 8);
    assert(tensor.dtype() == torch::kFloat32);

    float* data = tensor.data_ptr<float>();

    // 1. Plane 0 should contain White Pawns at rank 1 (index 1)
    for (int file = 0; file < 8; ++file) {
        // rank 1, file
        assert(data[0 * 64 + 1 * 8 + file] == 1.0f);
        // other ranks should be empty for White pawns
        for (int rank = 0; rank < 8; ++rank) {
            if (rank != 1) {
                assert(data[0 * 64 + rank * 8 + file] == 0.0f);
            }
        }
    }

    // 2. Plane 6 should contain Black Pawns at rank 6 (index 6)
    for (int file = 0; file < 8; ++file) {
        assert(data[6 * 64 + 6 * 8 + file] == 1.0f);
    }

    // 3. Side to move: starting position is White to move
    for (int i = 0; i < 64; ++i) {
        assert(data[12 * 64 + i] == 1.0f);
    }

    // After push_san (Black's turn) side to move plane should be 0.0
    board.makeMove(chess::Move::make(chess::Square("e2"), chess::Square("e4")));
    tensor = board_to_tensor(board);
    data = tensor.data_ptr<float>();
    for (int i = 0; i < 64; ++i) {
        assert(data[12 * 64 + i] == 0.0f);
    }

    std::cout << "PASSED\n";
}

void test_model_output_range_and_gradients() {
    std::cout << "Running: test_model_output_range_and_gradients... ";

    ValueNetwork model(15);
    torch::Tensor x = torch::randn({4, 15, 8, 8});
    torch::Tensor out = model->forward(x);

    // Verify shape
    assert(out.dim() == 2);
    assert(out.size(0) == 4);
    assert(out.size(1) == 1);

    // Verify sigmoid range
    float* data = out.data_ptr<float>();
    for (int i = 0; i < 4; ++i) {
        assert(data[i] >= 0.0f && data[i] <= 1.0f);
    }

    // Verify gradients
    torch::Tensor loss = out.sum();
    loss.backward();

    bool has_grad = false;
    for (const auto& p : model->parameters()) {
        if (p.grad().defined() && p.grad().abs().sum().item<float>() > 0.0f) {
            has_grad = true;
            break;
        }
    }
    assert(has_grad);

    std::cout << "PASSED\n";
}

void test_search_depth_1() {
    std::cout << "Running: test_search_depth_1... ";

    SearchConfig config;
    config.max_depth = 1;
    config.top_n = 3;
    config.device = "cpu";

    Engine engine(config);
    chess::Board board; // starting pos

    auto [move, val] = engine.search_position(board);

    // Verify returned move is legal
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    bool is_legal = false;
    for (const auto& m : moves) {
        if (m == move) {
            is_legal = true;
            break;
        }
    }
    assert(is_legal);

    // Verify val range
    assert(val >= 0.0f && val <= 1.0f);

    std::cout << "PASSED\n";
}

void test_td_learning_weight_changes() {
    std::cout << "Running: test_td_learning_weight_changes... ";

    SearchConfig config;
    config.max_depth = 2;
    config.top_n = 3;
    config.device = "cpu";

    Engine engine(config);
    ValueNetwork model = engine.get_model();

    // Snapshot parameters
    std::vector<torch::Tensor> before_weights;
    for (const auto& p : model->parameters()) {
        before_weights.push_back(p.clone());
    }

    chess::Board board;
    engine.search_position(board);

    // Verify that weights changed
    bool changed = false;
    auto params = model->parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        if (!torch::allclose(params[i], before_weights[i])) {
            changed = true;
            break;
        }
    }
    assert(changed);
    assert(engine.get_update_count() > 0);

    std::cout << "PASSED\n";
}

void test_checkpoint_roundtrip() {
    std::cout << "Running: test_checkpoint_roundtrip... ";

    SearchConfig config;
    config.device = "cpu";
    config.lr_decay_rate = 0.95;
    config.lr_decay_steps = 1; // Decay every step for testing
    config.min_learning_rate = 4.9e-5; // Learning rate won't decay below this

    Engine engine1(config);
    chess::Board board;

    float val1 = engine1.evaluate(board);

    // Set custom parameter values
    engine1.set_heuristic_weight(0.1234);
    engine1.set_learning_rate(5e-5);
    engine1.step_scheduler(); // scheduler_step becomes 1, decayed lr (4.75e-5) is clamped to min_lr (4.9e-5)

    std::string path = "test_temp_checkpoint.pt";
    engine1.save_checkpoint(path);

    Engine engine2(config);
    engine2.load_checkpoint(path);

    float val2 = engine2.evaluate(board);

    // Clean up
    std::filesystem::remove(path);
    std::filesystem::remove("test_temp_checkpoint.opt");
    std::filesystem::remove("test_temp_checkpoint.json");

    // Verify float equivalence
    assert(std::abs(val1 - val2) < 1e-6f);

    // Verify state and hyperparameter loading
    assert(std::abs(engine2.get_heuristic_weight() - 0.1234) < 1e-6);
    assert(std::abs(engine2.get_learning_rate() - 4.9e-5) < 1e-9);
    assert(engine2.get_scheduler_step() == 1);

    std::cout << "PASSED\n";
}

void test_temperature_exploration() {
    std::cout << "Running: test_temperature_exploration... ";

    SearchConfig config;
    config.max_depth = 1;
    config.top_n = 5;
    config.device = "cpu";
    config.temperature = 10.0;

    Engine engine(config);
    chess::Board board;

    std::vector<chess::Move> moves;
    for (int i = 0; i < 50; ++i) {
        auto [move, val] = engine.search_position(board);
        if (std::find(moves.begin(), moves.end(), move) == moves.end()) {
            moves.push_back(move);
        }
    }

    assert(moves.size() > 1);

    std::cout << "PASSED\n";
}

void test_tensor_symmetry() {
    std::cout << "Running: test_tensor_symmetry... ";

    chess::Board board;
    torch::Tensor orig = board_to_tensor(board);
    torch::Tensor mirrored = torch::flip(orig, {2});

    assert(torch::allclose(orig[0], mirrored[0]));

    // Let's make a move to make the board asymmetric, e.g. e4
    board.makeMove(chess::Move::make(chess::Square("e2"), chess::Square("e4")));
    orig = board_to_tensor(board);
    mirrored = torch::flip(orig, {2});

    float* orig_data = orig.data_ptr<float>();
    float* mirrored_data = mirrored.data_ptr<float>();

    // orig White pawn is at plane 0, rank 3, file 4.
    // Index: plane_idx * 64 + rank * 8 + file = 0 * 64 + 3 * 8 + 4 = 28.
    assert(orig_data[28] == 1.0f);
    assert(orig_data[3 * 8 + 3] == 0.0f);

    // mirrored White pawn should be at file 3, rank 3.
    // Index: 0 * 64 + 3 * 8 + 3 = 27.
    assert(mirrored_data[27] == 1.0f);
    assert(mirrored_data[28] == 0.0f);

    std::cout << "PASSED\n";
}

void test_heuristic_evaluation() {
    std::cout << "Running: test_heuristic_evaluation... ";

    SearchConfig config;
    config.max_depth = 1;
    config.heuristic_weight = 0.5;

    Engine engine(config);
    chess::Board board; // standard starting position

    // Starting position is equal, so evaluation should be around 0.5
    auto [move, val] = engine.search_position(board);
    assert(val >= 0.0f && val <= 1.0f);

    // Let's load a position where White has a huge material advantage (White has a queen, Black only king)
    chess::Board white_up("k1Q5/8/8/8/8/8/8/4K3 w - - 0 1");
    auto [move_wu, val_wu] = engine.search_position(white_up);
    assert(val_wu > 0.6f); // White is heavily favored

    // Position where Black has a queen, White only king
    chess::Board black_up("k1q5/8/8/8/8/8/8/4K3 w - - 0 1");
    auto [move_bu, val_bu] = engine.search_position(black_up);
    assert(val_bu < 0.4f); // Black is heavily favored

    std::cout << "PASSED\n";
}

void test_german_notation_preprocessing() {
    std::cout << "Running: test_german_notation_preprocessing... ";

    // Test German piece abbreviations to English in SAN
    assert(preprocess_move_input("Td1") == "Rd1");
    assert(preprocess_move_input("Sf3") == "Nf3");
    assert(preprocess_move_input("Le4") == "Be4");
    assert(preprocess_move_input("Dd5") == "Qd5");

    // Test lowercase/uppercase promotion mapping in UCI
    assert(preprocess_move_input("g7h8d") == "g7h8q");
    assert(preprocess_move_input("g7h8D") == "g7h8Q");
    assert(preprocess_move_input("g7h8t") == "g7h8r");
    assert(preprocess_move_input("g7h8T") == "g7h8R");
    assert(preprocess_move_input("g7h8l") == "g7h8b");
    assert(preprocess_move_input("g7h8L") == "g7h8B");
    assert(preprocess_move_input("g7h8s") == "g7h8n");
    assert(preprocess_move_input("g7h8S") == "g7h8N");

    // Test promotion with special symbols (= or - or parenthesis)
    assert(preprocess_move_input("g7h8=D") == "g7h8=Q");
    assert(preprocess_move_input("g7h8(D)") == "g7h8(Q)");
    assert(preprocess_move_input("g7h8-d") == "g7h8-q");

    // Test that standard English moves are untouched
    assert(preprocess_move_input("e4") == "e4");
    assert(preprocess_move_input("Nf3") == "Nf3");
    assert(preprocess_move_input("O-O") == "O-O");
    assert(preprocess_move_input("g7h8q") == "g7h8q");

    std::cout << "PASSED\n";
}


int main() {
    std::cout << "============================================================\n";
    std::cout << "Starting C++ Causal Chess Unit Tests\n";
    std::cout << "============================================================\n";

    try {
        test_encoding_shape_and_values();
        test_model_output_range_and_gradients();
        test_search_depth_1();
        test_td_learning_weight_changes();
        test_checkpoint_roundtrip();
        test_temperature_exploration();
        test_tensor_symmetry();
        test_heuristic_evaluation();
        test_german_notation_preprocessing();

        std::cout << "============================================================\n";
        std::cout << "All C++ Unit Tests PASSED successfully!\n";
        std::cout << "============================================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nERROR during testing: " << e.what() << "\n";
        return 1;
    }
}
