# Python 3.11
"""Lightweight conv-linear-linear stub."""
from __future__ import annotations

import torch
import torch.nn as nn


class MatMul(nn.Module):
    """Module that performs conv -> linear -> linear."""

    def __init__(self):
        super().__init__()
        # Input shape: (batch_size, 3, 32, 32)
        # Output of conv: (batch_size, 16, 16, 16) after stride=2, padding=1
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.linear1 = nn.Linear(16 * 16 * 16, 64)
        self.linear2 = nn.Linear(64, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., 3, 32, 32)
        Returns:
            Tensor of shape (..., 4)
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


# ───────────────────────────── API ─────────────────────────────
def get_model() -> nn.Module:
    """Returns a ready-to-use Conv-Linear-Linear model."""
    return MatMul()


def get_dummy_input() -> tuple[int, int, int, int]:
    """Shape tuple for dummy input (to be random-filled elsewhere)."""
    return (1, 3, 32, 32)

