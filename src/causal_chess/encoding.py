"""Board encoding: converts chess.Board to a tensor for the value network.

Encoding uses 15 planes of 8x8, all values White-relative:
  Planes 0-5:   White pieces (P, N, B, R, Q, K) — binary
  Planes 6-11:  Black pieces (P, N, B, R, Q, K) — binary
  Plane 12:     Side to move (all 1s if White to move, all 0s if Black)
  Plane 13:     White castling rights (1s at a1 if queenside, h1 if kingside)
  Plane 14:     Black castling rights (1s at a8 if queenside, h8 if kingside)
"""

import chess
import numpy as np
import torch


# Mapping from (piece_type, color) to plane index
_PIECE_PLANE: dict[tuple[int, bool], int] = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

NUM_PLANES = 15


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Encode a chess.Board as a (15, 8, 8) float32 tensor.

    The encoding is White-relative: plane layout is always from White's
    perspective regardless of whose turn it is.

    Args:
        board: The chess position to encode.

    Returns:
        A float32 tensor of shape (15, 8, 8).
    """
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    # Planes 0-11: piece positions
    for square, piece in board.piece_map().items():
        rank = square // 8  # 0 = rank 1, 7 = rank 8
        file = square % 8   # 0 = a-file, 7 = h-file
        plane_idx = _PIECE_PLANE[(piece.piece_type, piece.color)]
        planes[plane_idx, rank, file] = 1.0

    # Plane 12: side to move
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    # Plane 13: White castling rights
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[13, 0, 0] = 1.0  # a1
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13, 0, 7] = 1.0  # h1

    # Plane 14: Black castling rights
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[14, 7, 0] = 1.0  # a8
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[14, 7, 7] = 1.0  # h8

    return torch.from_numpy(planes)


def batch_boards_to_tensor(boards: list[chess.Board]) -> torch.Tensor:
    """Encode multiple boards into a batched tensor.

    Args:
        boards: List of chess positions to encode.

    Returns:
        A float32 tensor of shape (len(boards), 15, 8, 8).
    """
    return torch.stack([board_to_tensor(b) for b in boards])
