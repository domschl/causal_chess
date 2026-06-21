#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>
#include "chess.hpp"
#include "encoding.hpp"
#include "model.hpp"
#include "search.hpp"

using namespace causal_chess;

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

    ValueNetwork model;
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

    Engine engine1(config);
    chess::Board board;

    float val1 = engine1.evaluate(board);

    std::string path = "test_temp_checkpoint.pt";
    engine1.save_checkpoint(path);

    Engine engine2(config);
    engine2.load_checkpoint(path);

    float val2 = engine2.evaluate(board);

    // Clean up
    std::filesystem::remove(path);

    // Verify float equivalence
    assert(std::abs(val1 - val2) < 1e-6f);

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

        std::cout << "============================================================\n";
        std::cout << "All C++ Unit Tests PASSED successfully!\n";
        std::cout << "============================================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nERROR during testing: " << e.what() << "\n";
        return 1;
    }
}
