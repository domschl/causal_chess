"""Tests for board encoding."""

import chess
import torch

from causal_chess.encoding import NUM_PLANES, batch_boards_to_tensor, board_to_tensor


class TestBoardToTensor:
    """Test board_to_tensor encoding."""

    def test_output_shape(self) -> None:
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor.shape == (NUM_PLANES, 8, 8)

    def test_output_dtype(self) -> None:
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor.dtype == torch.float32

    def test_starting_position_white_pawns(self) -> None:
        """White pawns should be on rank 1 (index 1) in plane 0."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        # Plane 0 = White pawns
        pawn_plane = tensor[0]
        # Rank 1 (index 1) should have all pawns
        for file in range(8):
            assert pawn_plane[1, file] == 1.0, f"Expected pawn at rank 1, file {file}"
        # Rank 0 and ranks 2-7 should be empty for pawns
        for rank in [0] + list(range(2, 8)):
            for file in range(8):
                assert pawn_plane[rank, file] == 0.0

    def test_starting_position_black_pawns(self) -> None:
        """Black pawns should be on rank 6 (index 6) in plane 6."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        pawn_plane = tensor[6]
        for file in range(8):
            assert pawn_plane[6, file] == 1.0

    def test_starting_position_white_king(self) -> None:
        """White king on e1 = square e1 = file 4, rank 0."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        king_plane = tensor[5]  # Plane 5 = White king
        assert king_plane[0, 4] == 1.0
        # Only one king
        assert king_plane.sum().item() == 1.0

    def test_side_to_move_white(self) -> None:
        """Plane 12 should be all 1s when White to move."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor[12].sum().item() == 64.0

    def test_side_to_move_black(self) -> None:
        """Plane 12 should be all 0s when Black to move."""
        board = chess.Board()
        board.push_san("e4")
        tensor = board_to_tensor(board)
        assert tensor[12].sum().item() == 0.0

    def test_castling_rights_starting_position(self) -> None:
        """All four castling rights should be set in the starting position."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        # Plane 13: White castling — a1 and h1
        assert tensor[13, 0, 0] == 1.0  # White queenside (a1)
        assert tensor[13, 0, 7] == 1.0  # White kingside (h1)
        # Plane 14: Black castling — a8 and h8
        assert tensor[14, 7, 0] == 1.0  # Black queenside (a8)
        assert tensor[14, 7, 7] == 1.0  # Black kingside (h8)

    def test_no_castling_rights(self) -> None:
        """After removing castling rights, planes 13-14 should be empty."""
        board = chess.Board()
        board.set_castling_fen("-")
        tensor = board_to_tensor(board)
        assert tensor[13].sum().item() == 0.0
        assert tensor[14].sum().item() == 0.0

    def test_empty_board(self) -> None:
        """An empty board (just kings) should have minimal non-zero entries."""
        board = chess.Board(fen="4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        tensor = board_to_tensor(board)
        # Only white king (plane 5) and black king (plane 11) should have entries
        assert tensor[5].sum().item() == 1.0  # White king
        assert tensor[11].sum().item() == 1.0  # Black king
        # No pawns
        assert tensor[0].sum().item() == 0.0
        assert tensor[6].sum().item() == 0.0


class TestBatchBoardsToTensor:
    """Test batched encoding."""

    def test_batch_shape(self) -> None:
        boards = [chess.Board() for _ in range(4)]
        tensor = batch_boards_to_tensor(boards)
        assert tensor.shape == (4, NUM_PLANES, 8, 8)

    def test_single_board_batch(self) -> None:
        board = chess.Board()
        single = board_to_tensor(board)
        batch = batch_boards_to_tensor([board])
        assert torch.allclose(single.unsqueeze(0), batch)
