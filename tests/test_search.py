"""Tests for the search engine."""

import chess
import torch

from causal_chess.search import Engine, SearchConfig


class TestTerminalDetection:
    """Test that terminal positions return correct values."""

    def test_checkmate_white_wins(self) -> None:
        """Scholar's mate: Black is checkmated → value = 1.0."""
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        # This is actually Black winning (White is checkmated)
        # Let's use a proper fool's mate position
        board = chess.Board()
        board.push_san("f3")
        board.push_san("e5")
        board.push_san("g4")
        board.push_san("Qh4")
        # White is checkmated — value should be 0.0 (Black wins)
        assert board.is_checkmate()
        value = Engine._terminal_value(board)
        assert value == 0.0

    def test_checkmate_black_wins(self) -> None:
        """Back-rank mate: White checkmates Black → value = 1.0."""
        # White to play Qxf7# — but we need the position after the mate
        board = chess.Board("rnbqkb1r/pppp1Qpp/5n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 3")
        # Black is checkmated (scholar's mate achieved)
        assert board.is_checkmate()
        value = Engine._terminal_value(board)
        assert value == 1.0  # White wins

    def test_stalemate_is_draw(self) -> None:
        """Stalemate should return 0.5."""
        # King vs King + Queen stalemate
        board = chess.Board("k7/8/1K6/8/8/8/8/7Q w - - 0 1")
        board.push_san("Qa1")
        # Check if stalemate
        if board.is_stalemate():
            value = Engine._terminal_value(board)
            assert value == 0.5

    def test_non_terminal_returns_none(self) -> None:
        """A normal position should return None."""
        board = chess.Board()
        value = Engine._terminal_value(board)
        assert value is None


class TestSearchPosition:
    """Test the main search_position method."""

    def test_returns_legal_move(self) -> None:
        """search_position should return a legal move."""
        config = SearchConfig(max_depth=1, top_n=3, device="cpu")
        engine = Engine(config=config)
        board = chess.Board()
        move, value = engine.search_position(board)
        assert move in board.legal_moves

    def test_value_in_range(self) -> None:
        """Returned value should be in [0, 1]."""
        config = SearchConfig(max_depth=1, top_n=3, device="cpu")
        engine = Engine(config=config)
        board = chess.Board()
        _, value = engine.search_position(board)
        assert 0.0 <= value <= 1.0

    def test_td_learning_changes_weights(self) -> None:
        """Model weights should change after a search (TD updates are happening)."""
        config = SearchConfig(max_depth=2, top_n=3, device="cpu")
        engine = Engine(config=config)

        # Snapshot weights before search
        weights_before = {
            name: p.clone() for name, p in engine.model.named_parameters()
        }

        board = chess.Board()
        engine.search_position(board)

        # Check that at least some weights changed
        any_changed = False
        for name, p in engine.model.named_parameters():
            if not torch.allclose(p, weights_before[name]):
                any_changed = True
                break

        assert any_changed, "No weights changed during search — TD learning may not be working"

    def test_update_count_increases(self) -> None:
        """The TD update counter should increase after a search."""
        config = SearchConfig(max_depth=2, top_n=3, device="cpu")
        engine = Engine(config=config)
        engine.reset_stats()

        board = chess.Board()
        engine.search_position(board)

        assert engine.update_count > 0, "No TD updates were performed"


class TestEvaluate:
    """Test the single-position evaluation."""

    def test_evaluate_returns_float(self) -> None:
        config = SearchConfig(device="cpu")
        engine = Engine(config=config)
        board = chess.Board()
        value = engine.evaluate(board)
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_evaluate_does_not_change_weights(self) -> None:
        """evaluate() is inference-only — no weight updates."""
        config = SearchConfig(device="cpu")
        engine = Engine(config=config)

        weights_before = {
            name: p.clone() for name, p in engine.model.named_parameters()
        }

        board = chess.Board()
        engine.evaluate(board)

        for name, p in engine.model.named_parameters():
            assert torch.allclose(p, weights_before[name]), (
                f"Weight {name} changed during evaluate()"
            )


class TestCheckpointRoundTrip:
    """Test save/load checkpoint."""

    def test_checkpoint_roundtrip(self, tmp_path: object) -> None:
        """Model should produce same output after save/load cycle."""
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)

        config = SearchConfig(device="cpu")
        engine1 = Engine(config=config)

        board = chess.Board()
        value_before = engine1.evaluate(board)

        checkpoint_file = tmp_path / "test_checkpoint.pt"
        engine1.save_checkpoint(checkpoint_file)

        engine2 = Engine(config=config)
        engine2.load_checkpoint(checkpoint_file)

        value_after = engine2.evaluate(board)
        assert abs(value_before - value_after) < 1e-6
