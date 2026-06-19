import json
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os as _os, sys as _sys
# Make ../common importable; `model` and `mcts` resolve to this folder.
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))

from game import Environment
from model import OthelloPlayer
from mcts import MCTS, PASS_ACTION


# ───────────────────────── D4 symmetry augmentation ──────────────────────────
# Othello is invariant under the 8 dihedral symmetries (4 rotations × 2
# reflections). For each (state, pi, z) we can present the network with any of
# the 8 transformed views, multiplying effective data 8× and regularising the
# policy/value heads — a standard, near-free AlphaZero trick.

def _transform_grid(x: torch.Tensor, t: int) -> torch.Tensor:
    """Apply symmetry `t` (0..7) to the trailing (..., 8, 8) spatial dims."""
    if t >= 4:
        x = torch.flip(x, dims=(-1,))
    k = t % 4
    return torch.rot90(x, k, dims=(-2, -1)) if k else x


# Flat 64-vector permutations matching _transform_grid, precomputed once.
_BASE_GRID = torch.arange(64).reshape(8, 8)
_PI_PERMS = torch.stack([_transform_grid(_BASE_GRID, t).reshape(64) for t in range(8)])


def augment_batch(states: torch.Tensor, pis: torch.Tensor):
    """
    Apply an independent random D4 symmetry to each example in the batch.
    states: (B, C, 8, 8) ; pis: (B, 64). z is symmetry-invariant (unchanged).
    """
    B = states.shape[0]
    ts = torch.randint(0, 8, (B,), device=states.device)
    out_s = states.clone()
    out_p = pis.clone()
    for t in range(8):
        m = ts == t
        if m.any():
            out_s[m] = _transform_grid(states[m], t)
            out_p[m] = pis[m][:, _PI_PERMS[t].to(pis.device)]
    return out_s, out_p


class Agent:
    def __init__(
            self,
            lr=5e-4,
            weight_decay=1e-4,
            buffer_size=200_000,
            batch_size=512,
            n_simulations=200,
            c_puct=1.5,
            temp_threshold=12,
            min_buffer=20_000,        # warm up the buffer before training
            augment=True,             # 8-fold D4 symmetry augmentation
            load_pretrained=True,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.temp_threshold = temp_threshold  # plies after which we play greedily in self-play
        self.min_buffer = min_buffer
        self.augment = augment

        self.model = OthelloPlayer().to(self.device)
        if load_pretrained:
            sd = torch.load("othello_supervised_final.pt", map_location=self.device, weights_only=True)
            # strict=False: trunk + policy(classifier) + value load; the old
            # DQN 'advantage' head (if present) is ignored.
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            print(f"Loaded pretrained weights (missing={len(missing)}, unexpected={len(unexpected)})")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = None  # created in train() once total step budget is known
        self.buffer = deque(maxlen=buffer_size)  # (state_tensor, pi_tensor[64], z float)
        self.mcts = MCTS(self.model, self.device, c_puct=c_puct, n_simulations=n_simulations)

        self.logs = {"against_random_player": [], "loss": [], "policy_loss": [], "value_loss": [], "lr": []}

    # ----------------------------------------------------------- self-play
    @torch.no_grad()
    def self_play_game(self):
        """
        Play one game against itself with MCTS. Returns a list of training
        examples (state, pi, z) and the winner. One canonical orientation is
        stored per position; symmetries are applied at training time.
        """
        self.model.eval()
        env = Environment()
        game = env.game
        records = []  # (state_tensor, player_to_move, pi[64])
        ply = 0

        while not game.is_game_over():
            legal = game.get_legal_moves()
            if len(legal) == 0:
                game.current_turn *= -1  # forced pass; no decision to learn
                continue

            root = self.mcts.run(env, add_noise=True)

            # Visit-count policy target over the 64 board actions.
            visits = np.zeros(64, dtype=np.float32)
            for action, child in root.children.items():
                if action != PASS_ACTION:
                    visits[action] = child.visit_count
            total = visits.sum()
            pi = visits / total if total > 0 else visits

            records.append((env.get_state().clone(), game.current_turn, pi))

            # Pick the move: sample (temperature 1) early, argmax later.
            board_actions = [a for a in root.children if a != PASS_ACTION]
            counts = np.array([root.children[a].visit_count for a in board_actions], dtype=np.float64)
            if ply < self.temp_threshold:
                probs = counts / counts.sum()
                action = int(np.random.choice(board_actions, p=probs))
            else:
                action = int(board_actions[int(np.argmax(counts))])

            r, c = divmod(action, 8)
            game.take_turn(r, c)
            ply += 1

        winner = game.get_winner_id()  # 1, -1, or 0

        examples = []
        for state, player, pi in records:
            if winner == 0:
                z = 0.0
            else:
                z = 1.0 if winner == player else -1.0
            examples.append((state, torch.from_numpy(pi), z))
        return examples, winner

    # ------------------------------------------------------------- training
    def train_step(self):
        if len(self.buffer) < max(self.batch_size, self.min_buffer):
            return None

        self.model.train()
        batch = random.sample(self.buffer, self.batch_size)
        states, pis, zs = zip(*batch)

        states = torch.stack(states).to(self.device)
        target_pi = torch.stack(pis).to(self.device)
        target_z = torch.tensor(zs, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 8-fold D4 augmentation: state planes and policy target transform together.
        if self.augment:
            states, target_pi = augment_batch(states, target_pi)

        policy_logits, value = self.model(states)

        # Policy: cross-entropy against the (augmented) MCTS visit distribution.
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(target_pi * log_probs).sum(dim=1).mean()
        # Value: regress to the game outcome.
        value_loss = F.mse_loss(value, target_z)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item(), policy_loss.item(), value_loss.item()

    # ----------------------------------------------------------- evaluation
    def evaluate_against_random(self, num_games=40, eval_sims=50):
        """Play num_games vs a random opponent (alternating colors). eval_sims=1 ~ raw policy."""
        self.model.eval()
        eval_mcts = MCTS(self.model, self.device, c_puct=self.mcts.c_puct, n_simulations=eval_sims)
        score = 0.0

        for i in range(num_games):
            env = Environment()
            game = env.game
            rl_player = 1 if i % 2 == 0 else -1

            while not game.is_game_over():
                legal = game.get_legal_moves()
                if len(legal) == 0:
                    game.current_turn *= -1
                    continue

                if game.current_turn == rl_player:
                    action = self._mcts_best_action(eval_mcts, env)
                else:
                    r, c = random.choice(legal)
                    action = 8 * r + c

                game.take_turn(*divmod(action, 8))

            winner = game.get_winner_id()
            if winner == rl_player:
                score += 1.0
            elif winner == 0:
                score += 0.5

        win_rate = score / num_games
        self.logs["against_random_player"].append(win_rate)
        print(f"Win rate vs random ({eval_sims} sims): {win_rate * 100:.1f}%")
        return win_rate

    def pit(self, opponent_model, num_games=20, sims=100):
        """Current model vs a frozen opponent_model. Returns current model's score rate."""
        self.model.eval()
        opponent_model.eval()
        cur_mcts = MCTS(self.model, self.device, c_puct=self.mcts.c_puct, n_simulations=sims)
        opp_mcts = MCTS(opponent_model, self.device, c_puct=self.mcts.c_puct, n_simulations=sims)
        score = 0.0

        for i in range(num_games):
            env = Environment()
            game = env.game
            cur_player = 1 if i % 2 == 0 else -1

            while not game.is_game_over():
                legal = game.get_legal_moves()
                if len(legal) == 0:
                    game.current_turn *= -1
                    continue

                mcts = cur_mcts if game.current_turn == cur_player else opp_mcts
                action = self._mcts_best_action(mcts, env)
                game.take_turn(*divmod(action, 8))

            winner = game.get_winner_id()
            if winner == cur_player:
                score += 1.0
            elif winner == 0:
                score += 0.5

        return score / num_games

    @staticmethod
    def _mcts_best_action(mcts, env):
        root = mcts.run(env, add_noise=False)
        board_actions = [a for a in root.children if a != PASS_ACTION]
        if not board_actions:  # only a forced pass is available
            return PASS_ACTION
        return max(board_actions, key=lambda a: root.children[a].visit_count)

    def snapshot(self):
        """Return a frozen copy of the current network (for use as a pit opponent)."""
        clone = OthelloPlayer().to(self.device)
        clone.load_state_dict(self.model.state_dict())
        clone.eval()
        return clone

    # ----------------------------------------------------------- main loop
    def train(
            self,
            iterations=400,
            games_per_iter=50,
            train_steps_per_iter=400,
            eval_every=5,
            checkpoint_every=10,
    ):
        print(f"Training AlphaZero-style Othello on {self.device}")

        # Cosine LR decay over the (approximate) total optimizer-step budget,
        # to counter the slow loss drift seen late in training.
        total_steps = max(1, iterations * train_steps_per_iter)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=5e-6)

        for it in range(1, iterations + 1):
            # --- Self-play: generate fresh games into the buffer ---
            for _ in range(games_per_iter):
                examples, _ = self.self_play_game()
                self.buffer.extend(examples)

            # --- Learn from the buffer (skipped until warmed up) ---
            losses, p_losses, v_losses = [], [], []
            for _ in range(train_steps_per_iter):
                out = self.train_step()
                if out is not None:
                    losses.append(out[0])
                    p_losses.append(out[1])
                    v_losses.append(out[2])

            cur_lr = self.optimizer.param_groups[0]["lr"]
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            self.logs["loss"].append(mean_loss)
            self.logs["policy_loss"].append(float(np.mean(p_losses)) if p_losses else float("nan"))
            self.logs["value_loss"].append(float(np.mean(v_losses)) if v_losses else float("nan"))
            self.logs["lr"].append(cur_lr)
            warm = "" if losses else "  [warming up buffer]"
            print(f"[iter {it}] buffer={len(self.buffer)} lr={cur_lr:.2e} loss={mean_loss:.4f}{warm}")

            # --- Periodic evaluation / checkpoints ---
            if it % eval_every == 0:
                self.evaluate_against_random()

            if it % checkpoint_every == 0:
                torch.save(self.model.state_dict(), f"othello_az_iter{it}.pt")
                with open("training_logs.json", "w") as f:
                    json.dump(self.logs, f, indent=2)

        torch.save(self.model.state_dict(), "othello_az_model.pt")
        with open("training_logs.json", "w") as f:
            json.dump(self.logs, f, indent=2)


if __name__ == "__main__":
    agent = Agent(
        lr=5e-4,
        batch_size=512,
        buffer_size=200_000,
        n_simulations=200,
        min_buffer=20_000,
        augment=True,
        load_pretrained=True,
    )
    try:
        agent.train(
            iterations=400,
            games_per_iter=50,
            train_steps_per_iter=400,
            eval_every=5,
            checkpoint_every=10,
        )
    except KeyboardInterrupt:
        torch.save(agent.model.state_dict(), "othello_az_model_interrupted.pt")
        with open("training_logs.json", "w") as f:
            json.dump(agent.logs, f, indent=2)
        print("\nInterrupted; saved current model and logs.")
