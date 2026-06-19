"""
debug_edax.py
-------------
Play ONE verbose game against Edax, printing every command sent, every parsed
response, and piece counts each ply — so you can confirm the games are real
before running the full benchmark. Works for either agent type via --variant.

Usage:
    python debug_edax.py --variant mcts --binary /path/to/mEdax-native \
        --checkpoint ../mcts/othello_az_iter100.pt --level 5
    python debug_edax.py --variant rl  --binary ... \
        --checkpoint ../rl/othello_dqn_model.pt --level 5 --white
"""

import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant',    choices=['rl', 'mcts'], required=True)
    p.add_argument('--binary',     required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--level',      type=int, default=5)
    p.add_argument('--white',      action='store_true',
                   help='Agent plays White instead of Black')
    args = p.parse_args()

    # Put ../common, ../<variant> and this dir on sys.path before importing
    # variant-specific `model` and the shared engine.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "common"))
    sys.path.insert(0, os.path.join(root, args.variant))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import torch
    from model import OthelloPlayer
    from edax_engine import get_device, play_one_game

    binary = os.path.expanduser(args.binary)
    if not os.path.isfile(binary):
        print(f"Binary not found: {binary}"); return
    if not os.path.isfile(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}"); return

    device = get_device()
    model = OthelloPlayer().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=True))
    model.eval()

    if args.variant == 'mcts':
        @torch.no_grad()
        def pick_action(env):
            legal = env.game.get_legal_moves()
            if not legal:
                return None
            state = env.get_state().unsqueeze(0).to(device)
            logits, _ = model(state)
            logits = logits.squeeze(0)
            idxs = torch.tensor([8 * r + c for r, c in legal], dtype=torch.long)
            return int(idxs[logits[idxs].argmax()].item())
    else:  # rl
        @torch.no_grad()
        def pick_action(env):
            legal = env.game.get_legal_moves()
            if not legal:
                return None
            state = env.get_state().unsqueeze(0).to(device)
            q = model(state).squeeze(0)
            mask = state[0, 2, :, :].reshape(64)
            q[mask == 0] = -torch.inf
            return int(torch.argmax(q).item())

    model_is_black = not args.white
    print(f"Binary : {binary}")
    print(f"Level  : {args.level}")
    print(f"Device : {device}")
    print(f"Variant: {args.variant}")
    print(f"Agent  : {'Black' if model_is_black else 'White'}  "
          f"(mode 3 — we drive both sides)\n")

    play_one_game(pick_action, binary, args.level, model_is_black, verbose=True)


if __name__ == '__main__':
    main()
