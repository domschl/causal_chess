"""Value network: a small CNN that maps board tensors to a scalar in [0, 1].

Architecture:
    Input (15 × 8 × 8)
      → Conv2d(15, 32, 3, padding=1) → ReLU
      → Conv2d(32, 64, 3, padding=1) → ReLU
      → Conv2d(64, 128, 3, padding=1) → ReLU
      → AdaptiveAvgPool2d(1)          → (128, 1, 1)
      → Flatten
      → Linear(128, 64) → ReLU
      → Linear(64, 1)   → Sigmoid

Output is White-relative win probability: 1.0 = White wins, 0.0 = Black wins.

No batch normalisation is used — online TD learning often updates with
batch_size=1, which makes BatchNorm unreliable.  The model is small enough
(~105 K parameters) to train stably without normalisation.
"""

import torch
import torch.nn as nn

from causal_chess.encoding import NUM_PLANES


class ValueNetwork(nn.Module):
    """Convolutional value network for chess position evaluation.

    Args:
        in_channels: Number of input planes (default: 15).
    """

    def __init__(self, in_channels: int = NUM_PLANES) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # (batch, 128, 1, 1)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate a batch of board tensors.

        Args:
            x: Tensor of shape (batch, 15, 8, 8).

        Returns:
            Tensor of shape (batch, 1) with values in [0, 1].
        """
        x = self.conv_layers(x)
        return self.fc_layers(x)

    def param_count(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
