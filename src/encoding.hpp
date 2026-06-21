#ifndef CAUSAL_CHESS_ENCODING_HPP
#define CAUSAL_CHESS_ENCODING_HPP

#include <torch/torch.h>
#include "chess.hpp"

namespace causal_chess {

constexpr int NUM_PLANES = 15;

/**
 * @brief Encode a chess::Board as a (15, 8, 8) float32 tensor.
 *
 * Layout (White-relative):
 *   Planes 0-5:   White pieces (P, N, B, R, Q, K)
 *   Planes 6-11:  Black pieces (P, N, B, R, Q, K)
 *   Plane 12:     Side to move (all 1.0 if White, all 0.0 if Black)
 *   Plane 13:     White castling rights (1.0 at a1/h1 if present)
 *   Plane 14:     Black castling rights (1.0 at a8/h8 if present)
 *
 * @param board Position to encode.
 * @return torch::Tensor of shape (15, 8, 8).
 */
torch::Tensor board_to_tensor(const chess::Board& board);

/**
 * @brief Encode a batch of chess boards.
 *
 * @param boards Vector of chess boards.
 * @return torch::Tensor of shape (batch_size, 15, 8, 8).
 */
torch::Tensor batch_boards_to_tensor(const std::vector<chess::Board>& boards);

} // namespace causal_chess

#endif // CAUSAL_CHESS_ENCODING_HPP
