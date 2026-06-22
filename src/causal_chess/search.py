"""Selective tree search with online temporal-difference learning.

The engine evaluates positions with a CNN value network and trains
the network *during* search using TD(0) updates:

    Loss = (V(s_t) - V_target)²

where V_target is the minimax-backed value from deeper search
(treated as a fixed target — no gradient flows through it).

Scores are White-relative throughout: V(s) ∈ [0, 1] where
1.0 = White wins, 0.0 = Black wins, 0.5 = draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import random

import chess
import torch
import torch.nn.functional as F

from causal_chess.encoding import board_to_tensor
from causal_chess.model import ValueNetwork


@dataclass
class SearchConfig:
    """Configuration for the search engine."""

    max_depth: int = 4
    """Maximum search depth (number of plies)."""

    top_n: int = 5
    """Number of moves to expand at each node (selective breadth)."""

    learning_rate: float = 1e-4
    """Learning rate for online TD updates."""

    device: str = "cpu"
    """Torch device: 'cpu', 'mps', or 'cuda'."""

    grad_clip: float = 1.0
    """Maximum gradient norm for clipping (stability)."""

    temperature: float = 0.0
    """Exploration temperature for move selection at the root."""


class Engine:
    """Chess engine with selective search and online TD learning.

    The value network is updated during every search, so evaluation
    quality improves as more positions are explored.

    Args:
        config: Search configuration. Uses defaults if ``None``.
        model: Pre-built value network. A fresh one is created if ``None``.
    """

    def __init__(
        self,
        config: SearchConfig | None = None,
        model: ValueNetwork | None = None,
    ) -> None:
        self.config = config or SearchConfig()
        self.device = torch.device(self.config.device)

        self.model = (model or ValueNetwork()).to(self.device)
        self.model.eval()  # default to eval; switched to train for TD updates

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        # Monitoring counters (reset per game / per search_position call)
        self._total_loss: float = 0.0
        self._update_count: int = 0
        self._positions_evaluated: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_position(
        self, board: chess.Board, temperature: float | None = None
    ) -> tuple[chess.Move, float]:
        """Find the best move for the current position.

        Runs a selective tree search of depth ``config.max_depth``,
        updating the value network via TD learning at each internal node.

        Args:
            board: Current board position (not modified).
            temperature: Optional override for exploration temperature.

        Returns:
            A tuple of (best_move, value) where *value* is the
            White-relative evaluation backed up from the search.

        Raises:
            ValueError: If the position has no legal moves.
        """
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("No legal moves in position")

        is_white = board.turn == chess.WHITE

        # Score all legal moves for initial ordering
        move_scores = self._score_moves(board, moves)

        # Select top-n for deeper search
        move_scores.sort(key=lambda ms: ms[1], reverse=is_white)
        top_moves = move_scores[: self.config.top_n]

        # Search each top move
        searched_moves: list[tuple[chess.Move, float]] = []
        for move, _ in top_moves:
            board.push(move)
            v = self._search(board, self.config.max_depth - 1)
            board.pop()
            searched_moves.append((move, v))

        # Use temperature if specified or configured
        temp = temperature if temperature is not None else self.config.temperature

        if temp > 0.0:
            scores = [v if is_white else 1.0 - v for _, v in searched_moves]
            max_score = max(scores)
            logits = [(s - max_score) / temp for s in scores]
            exp_logits = [math.exp(l) for l in logits]
            sum_exp = sum(exp_logits)
            probs = [e / sum_exp for e in exp_logits]
            
            chosen_idx = random.choices(range(len(searched_moves)), weights=probs, k=1)[0]
            best_move, best_value = searched_moves[chosen_idx]
        else:
            best_move, best_value = searched_moves[0]
            for move, v in searched_moves[1:]:
                if is_white and v > best_value:
                    best_value = v
                    best_move = move
                elif not is_white and v < best_value:
                    best_value = v
                    best_move = move

        # TD update at root position
        self._td_update(board, best_value)

        return best_move, best_value

    @property
    def avg_loss(self) -> float:
        """Average TD loss since the last reset."""
        if self._update_count == 0:
            return 0.0
        return self._total_loss / self._update_count

    @property
    def update_count(self) -> int:
        """Number of TD updates since the last reset."""
        return self._update_count

    @property
    def positions_evaluated(self) -> int:
        """Number of positions evaluated since the last reset."""
        return self._positions_evaluated

    def reset_stats(self) -> None:
        """Reset monitoring counters."""
        self._total_loss = 0.0
        self._update_count = 0
        self._positions_evaluated = 0

    def evaluate(self, board: chess.Board) -> float:
        """Evaluate a position without searching (single forward pass).

        Args:
            board: Position to evaluate.

        Returns:
            White-relative value in [0, 1].
        """
        self._positions_evaluated += 1
        tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(tensor).item()

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model as a scripted module for C++ compatibility.

        Args:
            path: File path for the checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        scripted = torch.jit.script(self.model)
        scripted.save(str(path))

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model state from a checkpoint.

        Args:
            path: File path to the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if hasattr(checkpoint, "state_dict"):
            state_dict = checkpoint.state_dict()
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

    # ------------------------------------------------------------------
    # Internal search
    # ------------------------------------------------------------------

    def _search(self, board: chess.Board, depth: int) -> float:
        """Recursive selective search returning a White-relative value.

        At each internal node the value network is updated via TD learning:
        V(s) is trained toward the minimax-backed value from the subtree.

        Args:
            board: Current position (mutated via push/pop).
            depth: Remaining search depth.

        Returns:
            White-relative value in [0, 1].
        """
        # Terminal position — ground-truth value
        terminal = self._terminal_value(board)
        if terminal is not None:
            return terminal

        # Leaf node — neural network evaluation (no TD update)
        if depth == 0:
            self._positions_evaluated += 1
            tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.model(tensor).item()

        moves = list(board.legal_moves)
        if not moves:
            # No legal moves but not detected as game-over — treat as draw
            return 0.5

        is_white = board.turn == chess.WHITE

        # Score all legal moves (batched forward pass, no gradient)
        move_scores = self._score_moves(board, moves)

        # Select top-n moves
        move_scores.sort(key=lambda ms: ms[1], reverse=is_white)
        top_moves = move_scores[: self.config.top_n]

        # Recursively search selected moves
        best_value: float | None = None
        for move, _ in top_moves:
            board.push(move)
            v = self._search(board, depth - 1)
            board.pop()

            if best_value is None:
                best_value = v
            elif is_white and v > best_value:
                best_value = v
            elif not is_white and v < best_value:
                best_value = v

        assert best_value is not None

        # TD update: train V(current position) toward backed-up value
        self._td_update(board, best_value)

        return best_value

    def _score_moves(
        self, board: chess.Board, moves: list[chess.Move]
    ) -> list[tuple[chess.Move, float]]:
        """Evaluate each move by running the value net on the resulting position.

        Uses a single batched forward pass for efficiency.

        Args:
            board: Current position (not modified).
            moves: Legal moves to evaluate.

        Returns:
            List of (move, White-relative score) pairs.
        """
        tensors: list[torch.Tensor] = []
        for move in moves:
            board.push(move)
            tensors.append(board_to_tensor(board))
            board.pop()

        self._positions_evaluated += len(moves)
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            scores = self.model(batch).squeeze(-1)  # (num_moves,)

        return list(zip(moves, scores.tolist()))

    def _td_update(self, board: chess.Board, target_value: float) -> None:
        """Perform a single TD(0) gradient step.

        Trains V(board) and its horizontally mirrored position toward
        ``target_value`` using MSE loss. The target is detached (no gradient
        flows through it).

        Args:
            board: Position whose evaluation is being updated.
            target_value: Minimax-backed White-relative value (fixed target).
        """
        orig_tensor = board_to_tensor(board)
        # Flip along files dimension (dim 2) for horizontal symmetry
        mirrored_tensor = torch.flip(orig_tensor, dims=[2])

        batch = torch.stack([orig_tensor, mirrored_tensor]).to(self.device)
        target = torch.tensor(
            [[target_value], [target_value]], dtype=torch.float32, device=self.device
        )

        self.model.train()
        prediction = self.model(batch)
        loss = F.mse_loss(prediction, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.config.grad_clip
        )
        self.optimizer.step()
        self.model.eval()

        self._total_loss += loss.item()
        self._update_count += 1

    @staticmethod
    def _terminal_value(board: chess.Board) -> float | None:
        """Return ground-truth value for terminal positions, or None.

        Values are White-relative:
            Checkmate with White checkmated → 0.0
            Checkmate with Black checkmated → 1.0
            Any draw (stalemate, repetition, 50-move, insufficient material) → 0.5

        Args:
            board: Position to check.

        Returns:
            White-relative value, or ``None`` if the position is not terminal.
        """
        if not board.is_game_over():
            return None

        outcome = board.outcome()
        if outcome is None:
            return 0.5  # fallback — should not happen
        if outcome.winner is None:
            return 0.5  # draw
        return 1.0 if outcome.winner == chess.WHITE else 0.0
