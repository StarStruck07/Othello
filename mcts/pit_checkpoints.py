"""
pit_checkpoints.py
------------------
Loads all othello_az_iter*.pt checkpoints found in the current directory,
pits each one against the earliest checkpoint, and plots a strength curve.

Usage:
    python pit_checkpoints.py                  # auto-finds all iter checkpoints
    python pit_checkpoints.py --sims 100       # more sims = more accurate, slower
    python pit_checkpoints.py --games 40       # games per matchup (even number)
    python pit_checkpoints.py --baseline othello_supervised.pt  # custom baseline
"""

import argparse
import glob
import os
import random
import sys

import matplotlib.pyplot as plt
import torch

# Make ../common importable; `model` and `mcts` resolve to this folder.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

from model import OthelloPlayer
from mcts import MCTS, PASS_ACTION
from game import Environment


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path, device):
    m = OthelloPlayer().to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    return m


def best_action(mcts_obj, env):
    root = mcts_obj.run(env, add_noise=False)
    board_actions = [a for a in root.children if a != PASS_ACTION]
    if not board_actions:
        return PASS_ACTION
    return max(board_actions, key=lambda a: root.children[a].visit_count)


def pit(model_a, model_b, device, num_games=20, sims=50):
    """
    Returns model_a's score rate (wins + 0.5*draws) / num_games.
    Plays num_games games alternating who starts as Black.
    """
    mcts_a = MCTS(model_a, device, n_simulations=sims)
    mcts_b = MCTS(model_b, device, n_simulations=sims)
    score = 0.0

    for i in range(num_games):
        env = Environment()
        game = env.game
        player_a = 1 if i % 2 == 0 else -1  # alternate colours

        while not game.is_game_over():
            legal = game.get_legal_moves()
            if not legal:
                game.current_turn *= -1
                continue
            mcts = mcts_a if game.current_turn == player_a else mcts_b
            action = best_action(mcts, env)
            if action == PASS_ACTION:
                game.current_turn *= -1
            else:
                game.take_turn(*divmod(action, 8))

        winner = game.get_winner_id()
        if winner == player_a:
            score += 1.0
        elif winner == 0:
            score += 0.5

    return score / num_games


def find_checkpoints(pattern="othello_az_iter*.pt"):
    paths = sorted(glob.glob(pattern),
                   key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p)))))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims",     type=int,   default=50)
    parser.add_argument("--games",    type=int,   default=20)
    parser.add_argument("--baseline", type=str,   default=None,
                        help="Checkpoint to use as the fixed opponent. "
                             "Defaults to the earliest iter checkpoint found.")
    parser.add_argument("--pattern",  type=str,   default="othello_az_iter*.pt")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}  |  sims={args.sims}  |  games={args.games}")

    checkpoints = find_checkpoints(args.pattern)
    if not checkpoints:
        print(f"No checkpoints found matching '{args.pattern}'. "
              "Run from the same directory as your .pt files.")
        return

    print(f"\nFound {len(checkpoints)} checkpoint(s):")
    for p in checkpoints: print(f"  {p}")

    baseline_path = args.baseline if args.baseline else checkpoints[0]
    print(f"\nBaseline opponent: {baseline_path}")
    baseline_model = load_model(baseline_path, device)

    iterations, scores = [], []
    for path in checkpoints:
        iter_num = int("".join(filter(str.isdigit, os.path.basename(path))))
        current_model = load_model(path, device)
        score = pit(current_model, baseline_model, device,
                    num_games=args.games, sims=args.sims)
        iterations.append(iter_num)
        scores.append(score)
        bar = "█" * int(score * 20)
        print(f"  iter {iter_num:>4}  vs  baseline: {score*100:5.1f}%  {bar}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iterations, scores, marker="o", linewidth=2, color="#5c8fe0")
    ax.axhline(0.5,  color="grey",  linestyle="--", linewidth=1, label="50% (equal strength)")
    ax.axhline(1.0,  color="green", linestyle=":",  linewidth=1, label="100% (dominant)")
    ax.fill_between(iterations, 0.5, scores,
                    where=[s >= 0.5 for s in scores], alpha=0.15, color="green")
    ax.fill_between(iterations, scores, 0.5,
                    where=[s <  0.5 for s in scores], alpha=0.15, color="red")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score vs Baseline")
    ax.set_title(f"Strength Progression  (baseline = {os.path.basename(baseline_path)})")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    out = "pit_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
