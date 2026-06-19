import math
import os as _os, sys as _sys

import numpy as np
import torch

# Make ../common importable for the shared game engine.
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))

from game import Environment

# Board actions are 0..63 (8*row + col). 64 is reserved for a forced "pass",
# which only ever happens when the side to move has no legal moves but the
# game is not over (the opponent still has a move).
PASS_ACTION = 64


class Node:
    """A single node in the search tree (one game state)."""
    __slots__ = ("prior", "visit_count", "value_sum", "children")

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # action (int) -> Node

    @property
    def value(self) -> float:
        # Mean value from the perspective of the player to move AT THIS node.
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0


def clone_env(env: Environment) -> Environment:
    """
    Cheap, fully-isolated copy of the game state. Faster and safer than
    copy.deepcopy and avoids any shared mutable state between simulations.
    """
    new = Environment()
    new.game.board = env.game.board.copy()
    new.game.current_turn = env.game.current_turn
    return new


class MCTS:
    def __init__(self, model, device, c_puct=1.5, n_simulations=200,
                 dirichlet_alpha=0.3, dirichlet_eps=0.25):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    # ---------------------------------------------------------------- network
    @torch.no_grad()
    def _evaluate(self, env: Environment):
        """Single network pass: returns (policy_logits[64] np.array, value float)."""
        state = env.get_state().unsqueeze(0).to(self.device)
        policy_logits, value = self.model(state)
        return policy_logits.squeeze(0).cpu().numpy(), float(value.item())

    @staticmethod
    def _priors_for_legal(logits: np.ndarray, legal_moves):
        """Softmax restricted to legal squares -> {action_index: prior}."""
        idxs = [8 * r + c for (r, c) in legal_moves]
        z = logits[idxs]
        z = z - z.max()
        e = np.exp(z)
        p = e / e.sum()
        return {idx: float(pi) for idx, pi in zip(idxs, p)}

    # ----------------------------------------------------------------- search
    def _expand(self, node: Node, env: Environment) -> float:
        """
        Run the network on `env`, attach children with priors, and return the
        value estimate (from the perspective of the player to move at `env`).
        Only ever called on non-terminal states.
        """
        legal = env.game.get_legal_moves()
        logits, value = self._evaluate(env)

        if len(legal) == 0:
            # Forced pass: a single deterministic child.
            node.children[PASS_ACTION] = Node(1.0)
            return value

        for idx, prior in self._priors_for_legal(logits, legal).items():
            node.children[idx] = Node(prior)
        return value

    def _select_child(self, node: Node):
        """Pick the child maximizing the PUCT score."""
        total = math.sqrt(sum(c.visit_count for c in node.children.values()))
        best_score, best_action, best_child = -float("inf"), None, None

        # Iterate highest-prior first so that on the very first visits (when all
        # visit counts are 0 and the exploration term is 0) the policy prior
        # still decides the ordering.
        for action, child in sorted(node.children.items(),
                                    key=lambda kv: kv[1].prior, reverse=True):
            # child.value is from the opponent's perspective, so negate it to
            # score the move from THIS node's player's perspective.
            q = -child.value if child.visit_count > 0 else 0.0
            u = self.c_puct * child.prior * total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score, best_action, best_child = score, action, child
        return best_action, best_child

    @staticmethod
    def _apply(env: Environment, action: int):
        if action == PASS_ACTION:
            env.game.current_turn *= -1
        else:
            r, c = divmod(action, 8)
            env.game.take_turn(r, c)

    @staticmethod
    def _terminal_value(env: Environment) -> float:
        """Game result from the perspective of the player to move at `env`."""
        winner = env.game.get_winner_id()  # 1, -1, or 0
        if winner == 0:
            return 0.0
        return 1.0 if winner == env.game.current_turn else -1.0

    def _add_dirichlet_noise(self, root: Node):
        actions = list(root.children.keys())
        if len(actions) == 0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for action, n in zip(actions, noise):
            child = root.children[action]
            child.prior = (1 - self.dirichlet_eps) * child.prior + self.dirichlet_eps * float(n)

    def run(self, root_env: Environment, add_noise: bool = False) -> Node:
        """
        Build a search tree rooted at `root_env` (which must be non-terminal)
        and return the root. `root_env` is NOT mutated.
        """
        root = Node(0.0)
        self._expand(root, root_env)
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.n_simulations):
            env = clone_env(root_env)
            node = root
            path = [root]

            # SELECT down to a leaf.
            while node.is_expanded():
                action, node = self._select_child(node)
                self._apply(env, action)
                path.append(node)

            # EVALUATE the leaf (terminal -> exact result, else -> network).
            if env.game.is_game_over():
                value = self._terminal_value(env)
            else:
                value = self._expand(node, env)

            # BACKUP: flip sign each ply since players alternate.
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                value = -value

        return root
