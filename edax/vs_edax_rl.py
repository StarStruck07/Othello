"""
vs_edax_rl.py
-------------
Benchmark an **RL (Dueling-DQN)** checkpoint against Edax.

The RL model returns Q-values over the 64 squares; we mask illegal moves and
play the argmax — the same greedy policy used during RL evaluation.

Usage:
    python vs_edax_rl.py --binary /path/to/mEdax-native \
        --checkpoint ../rl/othello_dqn_model.pt --levels 1 3 5 --games 20
"""

import argparse
import os
import sys

# Resolve the shared engine (this dir), ../common (game), and ../rl (model.py).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "common"))
sys.path.insert(0, os.path.join(_ROOT, "rl"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from model import OthelloPlayer        # ../rl/model.py
from edax_engine import get_device, evaluate, plot_results


def load_model(path, device):
    m = OthelloPlayer().to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    return m


def make_pick_action(model, device):
    @torch.no_grad()
    def pick_action(env):
        legal = env.game.get_legal_moves()
        if not legal:
            return None
        state = env.get_state().unsqueeze(0).to(device)
        q = model(state).squeeze(0)                      # Dueling-DQN Q-values
        mask = state[0, 2, :, :].reshape(64)             # legal-moves plane
        q[mask == 0] = -torch.inf
        return int(torch.argmax(q).item())
    return pick_action


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary',     required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--levels',     nargs='+', type=int, default=[1, 3, 5, 7])
    p.add_argument('--games',      type=int, default=20)
    p.add_argument('--out',        default='vs_edax_rl.png')
    args = p.parse_args()

    binary = os.path.expanduser(args.binary)
    if not os.path.isfile(binary):
        print(f"Binary not found: {binary}"); return
    if not os.path.isfile(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}"); return

    device = get_device()
    print(f"[RL] Device: {device} | levels: {args.levels} | games/level: {args.games}\n")

    label = os.path.basename(args.checkpoint).replace('.pt', '')
    print(f"Evaluating: {label}\n{'='*50}")
    model   = load_model(args.checkpoint, device)
    results = evaluate(make_pick_action(model, device), binary, args.levels, args.games)
    plot_results({label: results}, args.levels, out=args.out)


if __name__ == '__main__':
    main()
