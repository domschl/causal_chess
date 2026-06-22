"""Self-play loop for continuous learning.

The engine plays complete games against itself.  Because the value
network is updated during search, the evaluation function improves
across (and within) games.  Game results are printed in PGN format
so that learning progress is visible.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import chess
import chess.pgn

from causal_chess.search import Engine, SearchConfig


def _get_last_game_num(csv_path: Path) -> int:
    """Read the last game index from the CSV file."""
    if not csv_path.is_file():
        return 0
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) <= 1:
                return 0
            # Read last non-empty line
            for line in reversed(lines):
                line = line.strip()
                if line:
                    parts = line.split(",")
                    if parts and parts[0].isdigit():
                        return int(parts[0])
    except Exception:
        pass
    return 0


def self_play_loop(
    engine: Engine,
    num_games: int = 100,
    save_dir: str | Path = "checkpoints",
    save_interval: int = 10,
    max_moves: int = 200,
    verbose: bool = True,
    resume: bool = True,
) -> dict[str, int]:
    """Run a continuous self-play training loop.

    The engine plays ``num_games`` complete games against itself.
    After each game the model weights are carried over, enabling
    continuous learning.

    Args:
        engine: The chess engine (with value network and optimizer).
        num_games: Number of games to play.
        save_dir: Directory for model checkpoints.
        save_interval: Save a checkpoint every N games.
        max_moves: Maximum moves per game before declaring a draw.
        verbose: If ``True``, print game summaries and PGN.
        resume: If ``True``, appends to CSV log and continues game numbering.

    Returns:
        A dict with keys ``'white_wins'``, ``'black_wins'``, ``'draws'``.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {"white_wins": 0, "black_wins": 0, "draws": 0}

    csv_path = save_path / "loss.csv"
    if resume and csv_path.is_file():
        start_game_num = _get_last_game_num(csv_path)
        mode = "a"
    else:
        start_game_num = 0
        mode = "w"

    try:
        csv_file = open(csv_path, mode, encoding="utf-8")
        if mode == "w":
            csv_file.write("game,loss,moves,time,nps\n")
            csv_file.flush()
    except Exception as e:
        csv_file = None
        if verbose:
            print(f"  ⚠️ Warning: Could not open {csv_path} for writing: {e}")

    try:
        for game_num in range(1, num_games + 1):
            game_start = time.time()
            engine.reset_stats()

            board = chess.Board()
            game = chess.pgn.Game()
            game.headers["Event"] = "Causal Chess Self-Play"
            game.headers["White"] = "CausalChess"
            game.headers["Black"] = "CausalChess"
            game.headers["Round"] = str(game_num)
            node = game

            move_count = 0

            while not board.is_game_over(claim_draw=True):
                if move_count >= max_moves:
                    break

                best_move, value = engine.search_position(board)
                board.push(best_move)
                node = node.add_variation(best_move)
                # Linearize 0..1 score to pseudo-centipawn score: (value - 0.5) * 20.0
                pseudo_cp = (value - 0.5) * 20.0
                node.comment = f"{pseudo_cp:+.2f}"
                move_count += 1

            # Determine result
            if board.is_game_over(claim_draw=True):
                outcome = board.outcome(claim_draw=True)
                if outcome is not None and outcome.winner == chess.WHITE:
                    result = "1-0"
                    stats["white_wins"] += 1
                elif outcome is not None and outcome.winner == chess.BLACK:
                    result = "0-1"
                    stats["black_wins"] += 1
                else:
                    result = "1/2-1/2"
                    stats["draws"] += 1
            else:
                # Max moves reached — declare draw
                result = "1/2-1/2"
                stats["draws"] += 1

            game.headers["Result"] = result
            elapsed = time.time() - game_start

            # Write game metrics to CSV log
            nps = int(engine.positions_evaluated / elapsed) if elapsed > 0 else 0
            if csv_file is not None:
                try:
                    csv_file.write(f"{start_game_num + game_num},{engine.avg_loss:.6f},{move_count},{elapsed:.1f},{nps}\n")
                    csv_file.flush()
                except Exception:
                    pass

            if verbose:
                _print_game_summary(game_num, num_games, result, board, engine, elapsed, stats)
                # Print PGN
                pgn_str = _game_to_pgn(game)
                print(pgn_str)
                print()

            # Save checkpoint periodically
            if game_num % save_interval == 0:
                checkpoint_path = save_path / "checkpoint.pt"
                engine.save_checkpoint(checkpoint_path)
                if verbose:
                    print(f"  💾 Checkpoint saved: {checkpoint_path}")
                    print()

        # Final checkpoint
        engine.save_checkpoint(save_path / "checkpoint.pt")

        if verbose:
            print("=" * 60)
            print("Self-play complete!")
            print(f"  Games:  {num_games}")
            print(f"  White:  {stats['white_wins']}")
            print(f"  Black:  {stats['black_wins']}")
            print(f"  Draws:  {stats['draws']}")
            print("=" * 60)

        return stats
    finally:
        if csv_file is not None:
            try:
                csv_file.close()
            except Exception:
                pass


def play_single_game(engine: Engine, max_moves: int = 200) -> chess.pgn.Game:
    """Play a single self-play game and return the PGN.

    Args:
        engine: The chess engine.
        max_moves: Maximum number of moves before declaring a draw.

    Returns:
        A ``chess.pgn.Game`` object with the completed game.
    """
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Causal Chess Self-Play"
    game.headers["White"] = "CausalChess"
    game.headers["Black"] = "CausalChess"
    node = game

    move_count = 0
    while not board.is_game_over(claim_draw=True):
        if move_count >= max_moves:
            break
        best_move, value = engine.search_position(board)
        board.push(best_move)
        node = node.add_variation(best_move)
        # Linearize 0..1 score to pseudo-centipawn score: (value - 0.5) * 20.0
        pseudo_cp = (value - 0.5) * 20.0
        node.comment = f"{pseudo_cp:+.2f}"
        move_count += 1

    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None and outcome.winner == chess.WHITE:
            game.headers["Result"] = "1-0"
        elif outcome is not None and outcome.winner == chess.BLACK:
            game.headers["Result"] = "0-1"
        else:
            game.headers["Result"] = "1/2-1/2"
    else:
        game.headers["Result"] = "1/2-1/2"

    return game


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _print_game_summary(
    game_num: int,
    total_games: int,
    result: str,
    board: chess.Board,
    engine: Engine,
    elapsed: float,
    stats: dict[str, int],
) -> None:
    """Print a human-readable summary of a completed game."""
    total = stats["white_wins"] + stats["black_wins"] + stats["draws"]
    print(f"\n{'=' * 60}")
    print(f"Game {game_num}/{total_games}: {result}")
    print(f"  Moves:       {board.fullmove_number}")
    print(f"  Time:        {elapsed:.1f}s")
    print(f"  TD updates:  {engine.update_count}")
    print(f"  Avg TD loss: {engine.avg_loss:.6f}")
    print(f"  Record:      W={stats['white_wins']}  B={stats['black_wins']}  D={stats['draws']}  ({total} games)")
    print(f"{'-' * 60}")


def _game_to_pgn(game: chess.pgn.Game) -> str:
    """Serialise a game to a PGN string."""
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=True)
    pgn_str = game.accept(exporter)
    # Replace PGN brace comments with parentheses: {+0.20} -> (+0.20)
    return pgn_str.replace("{", "(").replace("}", ")")
