# Python 3.11
from __future__ import annotations
import torch
import torch.nn as nn

# ───────────────────────────── SIZE ─────────────────────────────
N, C, H, W = 16, 3, 32, 32
OC, IC, KS = 8, 3, 3 # KS: Kernel Size

LIN = OC * (H - KS + 1) * (W - KS + 1)

# ───────────────────────────── MODEL ─────────────────────────────
class ConvLinearReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(IC, OC, kernel_size=int(KS), stride=1, padding=0, bias=True),
            nn.BatchNorm2d(OC),
            nn.Flatten(),
            nn.Linear(LIN, 10, bias=True),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ───────────────────────────── API ─────────────────────────────
def get_model() -> nn.Module:
    return ConvLinearReLU()

def get_dummy_input() -> tuple[int, int, int, int]:
    """Return shape tuple — will be random-filled elsewhere."""
    return (N, C, H, W)

# m = get_model().eval()
# print(m)
# x = torch.randn(get_dummy_input())
# y= m(x)
