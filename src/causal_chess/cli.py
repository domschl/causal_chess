"""Command-line interface for Causal Chess."""

from __future__ import annotations

import argparse
import sys

import chess

from causal_chess.model import ValueNetwork
from causal_chess.play import self_play_loop
from causal_chess.search import Engine, SearchConfig


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``causal-chess`` CLI."""
    parser = argparse.ArgumentParser(
        prog="causal-chess",
        description="Chess engine that learns during search via TD learning",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- play ----
    play_parser = subparsers.add_parser("play", help="Start a self-play training loop")
    play_parser.add_argument(
        "--games", type=int, default=100, help="Number of games (default: 100)"
    )
    play_parser.add_argument(
        "--depth", type=int, default=4, help="Search depth in plies (default: 4)"
    )
    play_parser.add_argument(
        "--top-n", type=int, default=5, help="Moves to expand per node (default: 5)"
    )
    play_parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    play_parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device: cpu, mps, cuda (default: cpu)"
    )
    play_parser.add_argument(
        "--save-dir", type=str, default="checkpoints", help="Checkpoint directory (default: checkpoints)"
    )
    play_parser.add_argument(
        "--save-interval", type=int, default=10, help="Save checkpoint every N games (default: 10)"
    )
    play_parser.add_argument(
        "--max-moves", type=int, default=200, help="Max moves per game (default: 200)"
    )
    play_parser.add_argument(
        "--checkpoint", type=str, default=None, help="Resume from a specific checkpoint file"
    )
    play_parser.add_argument(
        "--fresh", action="store_true", help="Ignore existing checkpoints and start from scratch"
    )

    # ---- eval ----
    eval_parser = subparsers.add_parser("eval", help="Evaluate a FEN position")
    eval_parser.add_argument("fen", type=str, help="FEN string of the position")
    eval_parser.add_argument(
        "--checkpoint", type=str, default=None, help="Model checkpoint file"
    )
    eval_parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device (default: cpu)"
    )

    # ---- move ----
    move_parser = subparsers.add_parser("move", help="Find the best move for a FEN position")
    move_parser.add_argument("fen", type=str, help="FEN string of the position")
    move_parser.add_argument(
        "--depth", type=int, default=4, help="Search depth (default: 4)"
    )
    move_parser.add_argument(
        "--top-n", type=int, default=5, help="Moves to expand per node (default: 5)"
    )
    move_parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    move_parser.add_argument(
        "--checkpoint", type=str, default=None, help="Model checkpoint file"
    )
    move_parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device (default: cpu)"
    )

    args = parser.parse_args(argv)

    if args.command == "play":
        _cmd_play(args)
    elif args.command == "eval":
        _cmd_eval(args)
    elif args.command == "move":
        _cmd_move(args)


# ------------------------------------------------------------------
# Sub-commands
# ------------------------------------------------------------------


def _find_latest_checkpoint(directory: str) -> str | None:
    """Find the most recently modified .pt file in a directory."""
    from pathlib import Path

    save_path = Path(directory)
    if not save_path.is_dir():
        return None
    checkpoints = sorted(save_path.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        return None
    return str(checkpoints[-1])


def _cmd_play(args: argparse.Namespace) -> None:
    config = SearchConfig(
        max_depth=args.depth,
        top_n=args.top_n,
        learning_rate=args.lr,
        device=args.device,
    )
    engine = Engine(config=config)

    # Load checkpoint: explicit path > auto-detect latest > fresh start
    loaded_checkpoint: str | None = None
    if args.checkpoint:
        engine.load_checkpoint(args.checkpoint)
        loaded_checkpoint = args.checkpoint
    elif not args.fresh:
        latest = _find_latest_checkpoint(args.save_dir)
        if latest:
            engine.load_checkpoint(latest)
            loaded_checkpoint = latest

    if loaded_checkpoint:
        print(f"Resumed from checkpoint: {loaded_checkpoint}")
    else:
        print("Starting with fresh (random) weights")

    print("=" * 60)
    print("Causal Chess — Self-Play Training")
    print(f"  Depth:     {config.max_depth}")
    print(f"  Top-N:     {config.top_n}")
    print(f"  LR:        {config.learning_rate}")
    print(f"  Device:    {config.device}")
    print(f"  Games:     {args.games}")
    print(f"  Params:    {engine.model.param_count():,}")
    print("=" * 60)

    self_play_loop(
        engine=engine,
        num_games=args.games,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        max_moves=args.max_moves,
    )


def _cmd_eval(args: argparse.Namespace) -> None:
    config = SearchConfig(device=args.device)
    engine = Engine(config=config)

    if args.checkpoint:
        engine.load_checkpoint(args.checkpoint)

    board = chess.Board(args.fen)
    value = engine.evaluate(board)

    print(f"FEN:   {args.fen}")
    print(f"Value: {value:.4f}  (White-relative win probability)")
    if value > 0.55:
        print("       → White is better")
    elif value < 0.45:
        print("       → Black is better")
    else:
        print("       → Roughly equal")


def _cmd_move(args: argparse.Namespace) -> None:
    config = SearchConfig(
        max_depth=args.depth,
        top_n=args.top_n,
        learning_rate=args.lr,
        device=args.device,
    )
    engine = Engine(config=config)

    if args.checkpoint:
        engine.load_checkpoint(args.checkpoint)

    board = chess.Board(args.fen)
    move, value = engine.search_position(board)

    print(f"FEN:   {args.fen}")
    print(f"Move:  {board.san(move)}  ({move.uci()})")
    print(f"Value: {value:.4f}  (White-relative)")
    print(f"TD updates during search: {engine.update_count}")


if __name__ == "__main__":
    main()
