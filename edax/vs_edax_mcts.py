"""
vs_edax_mcts.py
---------------
Benchmark an **MCTS / AlphaZero-style** checkpoint against Edax.

The model returns (policy_logits, value). Move-selection modes:
  --sims N>0   (default 200) : run N MCTS simulations, play most-visited move.
  --sims 0                   : raw policy — argmax over legal squares (fast,
                               but NO lookahead; mostly for debugging).

Test-time strength (no retraining):
  --tta        : average the net's (policy, value) over the 8 board symmetries
                 before search. Pure inference-time augmentation.

IMPORTANT: point --checkpoint at your *final / best* model (e.g.
othello_az_model.pt or your strongest othello_az_iterN.pt), NOT an early one.

Usage:
    python vs_edax_mcts.py --binary /path/to/mEdax-native \
        --checkpoint ../mcts/othello_az_model.pt --levels 1 3 5 --games 20
    python vs_edax_mcts.py --binary ... --checkpoint ... --sims 400 --tta
"""

import argparse
import os
import sys

# Resolve the shared engine (this dir), ../common (game), and ../mcts (model, mcts).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "common"))
sys.path.insert(0, os.path.join(_ROOT, "mcts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from model import OthelloPlayer        # ../mcts/model.py
from mcts import MCTS, PASS_ACTION     # ../mcts/mcts.py
from edax_engine import get_device, evaluate, plot_results


# ── test-time augmentation ──────────────────────────────────────────────────
class SymmetrizedModel(torch.nn.Module):
    """
    Average the net's (policy_logits, value) over the 8 dihedral symmetries at
    inference time. Pure test-time augmentation — no retraining.

    Othello is equivariant under the 8 symmetries, and the 3x8x8 input planes
    and the 64-square policy live on the same 8x8 grid. For each symmetry we
    transform the input, run the net, map the policy back to the original
    frame, and average. Value is symmetry-invariant, so we just mean it.
    Returns the same (logits[B,64], value[B,1]) tuple as the wrapped model, so
    MCTS and the raw-policy path use it transparently.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    @staticmethod
    def _apply(x, rot, flip):
        # Forward transform on a (B, C, 8, 8) grid: rotate, then flip L-R.
        if rot:
            x = torch.rot90(x, rot, dims=(-2, -1))
        if flip:
            x = torch.flip(x, dims=(-1,))
        return x

    @staticmethod
    def _invert_policy(p, rot, flip):
        # p: (B, 64) logits in the transformed frame -> original frame.
        # Inverse of (flip ∘ rot) is (rot^-1 ∘ flip): undo flip, then rotation.
        g = p.view(-1, 8, 8)
        if flip:
            g = torch.flip(g, dims=(-1,))
        if rot:
            g = torch.rot90(g, -rot, dims=(-2, -1))
        return g.reshape(-1, 64)

    @torch.no_grad()
    def forward(self, x):
        logits_sum, value_sum, n = 0.0, 0.0, 0
        for flip in (False, True):
            for rot in range(4):
                logits, value = self.model(self._apply(x, rot, flip))
                logits_sum = logits_sum + self._invert_policy(logits, rot, flip)
                value_sum = value_sum + value
                n += 1
        return logits_sum / n, value_sum / n


def load_model(path, device, tta=False):
    m = OthelloPlayer().to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    if tta:
        m = SymmetrizedModel(m).to(device).eval()
    return m


def make_policy_pick(model, device):
    """Fast: argmax of the raw policy head, restricted to legal squares."""
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
    return pick_action


def make_mcts_pick(model, device, sims):
    """Stronger: run MCTS and play the most-visited child move."""
    searcher = MCTS(model, device, n_simulations=sims)

    def pick_action(env):
        legal = env.game.get_legal_moves()
        if not legal:
            return None
        root = searcher.run(env, add_noise=False)
        action = max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]
        # MCTS may report PASS_ACTION, but pick_action is only ever called when
        # the agent has a legal move, so a concrete square is always available.
        return action if action != PASS_ACTION else (8 * legal[0][0] + legal[0][1])
    return pick_action


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary',     required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--levels',     nargs='+', type=int, default=[1, 3, 5, 7])
    p.add_argument('--games',      type=int, default=20)
    p.add_argument('--sims',       type=int, default=200,
                   help='MCTS simulations per move (0 = raw policy argmax, no search).')
    p.add_argument('--tta',        action='store_true',
                   help='Average over the 8 board symmetries at inference (no retraining).')
    p.add_argument('--out',        default='vs_edax_mcts.png')
    args = p.parse_args()

    binary = os.path.expanduser(args.binary)
    if not os.path.isfile(binary):
        print(f"Binary not found: {binary}"); return
    if not os.path.isfile(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}"); return

    device = get_device()
    mode = f"MCTS x{args.sims}" if args.sims > 0 else "raw policy (no search)"
    tta = " +TTA" if args.tta else ""
    print(f"[MCTS:{mode}{tta}] Device: {device} | levels: {args.levels} | "
          f"games/level: {args.games}\n")

    label = os.path.basename(args.checkpoint).replace('.pt', '') + (tta and "_tta")
    print(f"Evaluating: {label}\n{'='*50}")
    model = load_model(args.checkpoint, device, tta=args.tta)
    pick  = make_mcts_pick(model, device, args.sims) if args.sims > 0 \
            else make_policy_pick(model, device)
    results = evaluate(pick, binary, args.levels, args.games)
    plot_results({label: results}, args.levels, out=args.out)


if __name__ == '__main__':
    main()