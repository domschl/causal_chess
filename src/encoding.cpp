#include "encoding.hpp"

namespace causal_chess {

torch::Tensor board_to_tensor(const chess::Board& board) {
    // Create zero-initialized tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor = torch::zeros({NUM_PLANES, 8, 8}, options);
    float* data = tensor.data_ptr<float>();

    // 1. Piece planes (0-11)
    for (int sq = 0; sq < 64; ++sq) {
        chess::Square square(sq);
        chess::Piece piece = board.at(square);
        if (piece != chess::Piece::NONE) {
            int type_idx = static_cast<int>(piece.type());
            int color_idx = static_cast<int>(piece.color());
            int plane_idx = color_idx * 6 + type_idx;

            int rank = square.rank(); // 0..7
            int file = square.file(); // 0..7

            // Index in flat 1D array for (plane, rank, file)
            // Shape is (15, 8, 8), so strides are (64, 8, 1)
            int index = plane_idx * 64 + rank * 8 + file;
            data[index] = 1.0f;
        }
    }

    // 2. Side to move plane (12)
    if (board.sideToMove() == chess::Color::WHITE) {
        for (int i = 0; i < 64; ++i) {
            data[12 * 64 + i] = 1.0f;
        }
    }

    // 3. Castling rights planes (13 & 14)
    auto cr = board.castlingRights();

    // White castling (plane 13)
    if (cr.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE)) {
        // a1 (file 0, rank 0)
        data[13 * 64 + 0 * 8 + 0] = 1.0f;
    }
    if (cr.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE)) {
        // h1 (file 7, rank 0)
        data[13 * 64 + 0 * 8 + 7] = 1.0f;
    }

    // Black castling (plane 14)
    if (cr.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE)) {
        // a8 (file 0, rank 7)
        data[14 * 64 + 7 * 8 + 0] = 1.0f;
    }
    if (cr.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE)) {
        // h8 (file 7, rank 7)
        data[14 * 64 + 7 * 8 + 7] = 1.0f;
    }

    return tensor;
}

torch::Tensor batch_boards_to_tensor(const std::vector<chess::Board>& boards) {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(boards.size());
    for (const auto& board : boards) {
        tensors.push_back(board_to_tensor(board));
    }
    return torch::stack(tensors);
}

} // namespace causal_chess
