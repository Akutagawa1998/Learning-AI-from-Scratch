# src/models/linear.py
from __future__ import annotations

import torch
import torch.nn as nn


class LinearMNIST(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.fc(x)
