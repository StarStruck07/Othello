"""
edax_engine.py
--------------
Model-agnostic machinery for playing Othello agents against the Edax engine.

Everything here is independent of *how* the agent chooses a move: callers pass a
`pick_action(env) -> int | None` callable (None means "no legal move / pass").
The RL and MCTS front-ends (vs_edax_rl.py / vs_edax_mcts.py) just supply that
callable, so the Edax protocol, sync-safety, evaluation loop and plotting live
in exactly one place.

Requires pexpect:  pip install pexpect

ARCHITECTURE (mode 3):
  Edax always runs in mode 3 ("human vs human": Edax auto-plays NEITHER side).
  We drive both sides explicitly through one uniform protocol:
    - Agent's move     : send the coordinate string, e.g. 'f5'
    - Edax's move      : send 'go' -> Edax plays the side to move, printing
                         "Edax plays F6" (or "Edax plays PS" on a forced pass)
    - Agent forced pass: also send 'go' -> Edax executes the forced pass for
                         that side, keeping both states in sync.

SYNC SAFETY:
  Any 'illegal move' / 'unknown command' in Edax's output sets a desync flag;
  that game is reported as unreliable instead of silently counted. A safety
  counter aborts pathological loops.
"""

import os
import re
import sys

try:
    import pexpect
except ImportError:
    print("Install pexpect first:  pip install pexpect")
    sys.exit(1)

import torch


# ── patterns ──────────────────────────────────────────────────────────────────

PROMPT = '>'
# "Edax plays F6" for a normal move, "Edax plays PS" (or PA) for a pass.
EDAX_PLAYS_RE = re.compile(r'(?i)edax plays\s+(ps|pa|[a-h][1-8])')
# Anything indicating Edax rejected our input => the two game states diverged.
DESYNC_RE = re.compile(r'(?i)(illegal move|unknown command)')


# ── helpers ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():         return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


def action_to_edax(action: int) -> str:
    r, c = divmod(action, 8)
    return chr(ord('a') + c) + str(r + 1)


def edax_to_action(move_str: str) -> int:
    s = move_str.strip().lower()
    return 8 * (int(s[1]) - 1) + (ord(s[0]) - ord('a'))


# ── Edax wrapper ──────────────────────────────────────────────────────────────

class EdaxEngine:
    def __init__(self, binary_path: str, level: int):
        self.binary_path = os.path.abspath(os.path.expanduser(binary_path))
        self.bin_dir     = os.path.dirname(self.binary_path)
        self.level       = level
        self.proc        = None

    def start(self, verbose: bool = False):
        # mode 3 = human vs human: Edax auto-plays NEITHER side.
        self.proc = pexpect.spawn(
            self.binary_path,
            args=['-level', str(self.level), '-mode', '3'],
            cwd=self.bin_dir,
            timeout=60,
            encoding='utf-8',
        )
        self.proc.setecho(False)

        # Startup prints one '>' attached to the book line ("...done>") and
        # usually a second '>' as the real game prompt. Consume the first, then
        # *try* to consume a second with a short timeout — tolerant of one or two.
        idx = self.proc.expect([PROMPT, pexpect.EOF], timeout=30)
        first = self.proc.before or ''
        if idx != 0:
            raise RuntimeError(f"Edax exited at startup.\n{first}")

        self.proc.expect([PROMPT, pexpect.EOF, pexpect.TIMEOUT], timeout=5)
        second = self.proc.before or ''

        if verbose:
            print("── Before 1st '>' ──────────────────────────────────────")
            print(repr(first[:200]))
            print("── Before 2nd '>' ──────────────────────────────────────")
            print(repr(second[:200]))
            print("─────────────────────────────────────────────────────────\n")

        if 'cannot open' in (first + second).lower():
            raise RuntimeError(
                "Edax can't find data/eval.dat.\n"
                f"Copy it to: {self.bin_dir}/data/eval.dat"
            )

    def send(self, command: str, timeout: int = 90) -> str:
        """Send a command, wait for the next prompt, return Edax's output."""
        self.proc.sendline(command)
        self.proc.expect([PROMPT, pexpect.EOF, pexpect.TIMEOUT], timeout=timeout)
        return self.proc.before or ''

    def stop(self):
        if self.proc and self.proc.isalive():
            try:
                self.proc.sendline('quit')
                self.proc.close(force=True)
            except Exception:
                pass
        self.proc = None


# ── single game ───────────────────────────────────────────────────────────────

def play_one_game(pick_action, binary_path: str, level: int,
                  model_is_black: bool, verbose: bool = False):
    """
    Play one game: our agent (via `pick_action(env) -> int | None`) versus Edax.

    Returns (result, desync):
      result : +1 agent win, 0 draw, -1 Edax win  (from Python ground truth)
      desync : True if Edax ever rejected input or returned an impossible move —
               the result should then be treated as unreliable.

    `Environment` is imported lazily so this module can be imported before the
    variant has put ../common on sys.path.
    """
    from game import Environment

    env         = Environment()
    game        = env.game
    model_color = 1 if model_is_black else -1
    desync      = False

    edax = EdaxEngine(binary_path, level)
    edax.start(verbose=verbose)

    def log(msg):
        if verbose:
            print(msg)

    try:
        safety = 0
        while not game.is_game_over():
            safety += 1
            if safety > 200:           # no legal game lasts this long
                log("SAFETY LIMIT — aborting")
                desync = True
                break

            legal = game.get_legal_moves()

            if verbose:
                b = int((game.board == 1).sum()); w = int((game.board == -1).sum())
                turn = "BLACK" if game.current_turn == 1 else "WHITE"
                who  = "agent" if game.current_turn == model_color else "edax "
                print(f"Ply {safety:>3} | {turn} to move ({who}) | "
                      f"B:{b} W:{w} | legal:{len(legal)}")

            if game.current_turn == model_color and legal:
                # ── agent's real move ─────────────────────────────────────────
                action = pick_action(env)
                mv = action_to_edax(action)
                log(f"        agent plays {mv.upper()}")
                out = edax.send(mv)
                if DESYNC_RE.search(out):
                    log(f"        *** DESYNC: Edax rejected our move! {repr(out[:150])}")
                    desync = True
                game.take_turn(*divmod(action, 8))

            else:
                # ── Edax decides: its own move, OR the agent's forced pass ─────
                reason = "its own move" if legal else "forced pass for side to move"
                log(f"        sending 'go' ({reason})")
                out = edax.send('go')
                m = EDAX_PLAYS_RE.search(out)

                if m:
                    mv = m.group(1).lower()
                    if mv in ('ps', 'pa'):
                        log("        edax plays PS (pass)")
                        game.current_turn *= -1
                    else:
                        log(f"        edax plays {mv.upper()}")
                        a = edax_to_action(mv)
                        r, c = divmod(a, 8)
                        if game.is_valid_move(r, c):
                            game.take_turn(r, c)
                        else:
                            log(f"        *** DESYNC: {mv} invalid in Python game! ***")
                            desync = True
                            game.current_turn *= -1
                elif re.search(r'(?i)pass', out):
                    log("        edax output mentions pass — flipping turn")
                    game.current_turn *= -1
                else:
                    log(f"        no move parsed; raw: {repr(out[:150])}")
                    game.current_turn *= -1

    finally:
        edax.stop()

    winner = game.get_winner_id()
    if   winner == model_color: result =  1
    elif winner == 0:           result =  0
    else:                       result = -1

    if verbose:
        b = int((game.board == 1).sum()); w = int((game.board == -1).sum())
        names = {1: 'Black', -1: 'White', 0: 'nobody (draw)'}
        mine  = ' (AGENT)' if winner == model_color else (' (EDAX)' if winner == -model_color else '')
        print(f"\nGame over after {safety} plies.  Final: Black {b} – White {w}")
        print(f"Winner: {names[winner]}{mine}")
        print("⚠ DESYNC — result NOT reliable." if desync
              else "✓ No desync detected.")

    return result, desync


# ── evaluation ────────────────────────────────────────────────────────────────

def wilson_ci(p, n, z=1.96):
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = (z * (p * (1 - p) / n + z**2 / (4 * n**2))**0.5) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def evaluate(pick_action, binary_path, levels, num_games):
    results = {}
    for level in levels:
        score, desyncs = 0.0, 0
        for i in range(num_games):
            result, desync = play_one_game(
                pick_action, binary_path, level,
                model_is_black=(i % 2 == 0))
            score += {1: 1.0, 0: 0.5, -1: 0.0}[result]
            desyncs += int(desync)
            tag  = {1: 'WIN ', 0: 'DRAW', -1: 'LOSS'}[result]
            note = '  [DESYNC — unreliable]' if desync else ''
            colr = 'B' if i % 2 == 0 else 'W'
            print(f"  lvl {level:>2} | game {i+1:>2}/{num_games} ({colr}) | "
                  f"{tag} | running {score/(i+1)*100:5.1f}%{note}")

        rate = score / num_games
        lo, hi = wilson_ci(rate, num_games)
        bar = '█' * int(rate * 20)
        warn = f"  ⚠ {desyncs} desynced game(s)!" if desyncs else ""
        print(f"\n  → Level {level}: {rate*100:.1f}%  "
              f"(95% CI {lo*100:.0f}–{hi*100:.0f}%)  {bar}{warn}\n")
        results[level] = rate
    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(all_results, levels, out='vs_edax.png'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap('tab10')
    for i, (label, res) in enumerate(all_results.items()):
        y = [res.get(lv, float('nan')) * 100 for lv in levels]
        ax.plot(levels, y, marker='o', linewidth=2, color=cmap(i), label=label)
    ax.axhline(50, color='grey', linestyle='--', linewidth=1, label='50%')
    ax.set_xlabel('Edax Level')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Model Strength vs Edax')
    ax.set_xticks(levels)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved {out}")
