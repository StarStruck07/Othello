import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)  # keep bias off for GroupNorm (it has its own learned bias)
        self.gn1 = nn.GroupNorm(num_groups, channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += residual
        return F.relu(out)


class OthelloPlayer(nn.Module):
    def __init__(
            self,
            in_channels=3,
            base_channels=64,
            num_blocks=8,
            hidden_dim=512,
    ):
        """
        ResNet trunk with an AlphaZero-style policy head and value head.

        - The POLICY head reuses the `classifier` module so weights from
          supervised pre-training (which predicts expert moves) transfer
          directly as a strong prior for MCTS.
        - The VALUE head outputs a scalar in [-1, 1] (tanh) estimating the
          game result from the perspective of the player to move.

        Args:
            in_channels (int): input planes (own pieces, opp pieces, legal moves)
            base_channels (int): channels in the residual stream
            num_blocks (int): number of residual blocks
            hidden_dim (int): hidden size of the head MLPs
        """
        super().__init__()

        # Initial stem
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)
        self.gn_in = nn.GroupNorm(8, base_channels)

        # Residual tower
        self.res_blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])

        self.flatten = nn.Flatten()

        # Policy head (also used as the supervised classifier head).
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 8 * 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
        )

        # Value head -> scalar in [-1, 1].
        self.value = nn.Sequential(
            nn.Linear(base_channels * 8 * 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _trunk(self, x):
        x = F.relu(self.gn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        return self.flatten(x)

    def forward(self, x, supervised=False):
        """
        x shape: (batch, in_channels, 8, 8)

        - supervised=True  -> returns policy logits only (back-compat with
          superVisionClassifier.py).
        - supervised=False -> returns (policy_logits, value), value in [-1, 1].
        """
        features = self._trunk(x)

        if supervised:
            return self.classifier(features)

        policy_logits = self.classifier(features)
        value = torch.tanh(self.value(features))
        return policy_logits, value


if __name__ == "__main__":
    model = OthelloPlayer()
    total = sum(p.numel() for p in model.parameters())
    print(total)
