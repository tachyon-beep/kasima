from __future__ import annotations

import torch
from torch import nn


class SentinelSeed(nn.Module):
    """Morphogenetic component representing a latent seed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.state = "Dormant"
        # We keep parameters for the potential layer but do not register them
        self.weight = None
        self.bias = None

    def germinate(self) -> None:
        if self.state == "Active":
            return
        self.weight = nn.Parameter(
            torch.randn(self.out_features, self.in_features) * 0.1
        )
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)
        self.state = "Active"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.state == "Dormant":
            return x
        return torch.matmul(x, self.weight.t()) + self.bias

    def get_health_signal(self, loss_history: list[float]) -> float:
        if not loss_history:
            return 0.0
        return loss_history[-1]


class BaseNet(nn.Module):
    """Simple MLP host network with two sentinel seeds."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        hidden = 64
        self.fc1 = nn.Linear(input_dim, hidden)
        self.seed1 = SentinelSeed(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.seed2 = SentinelSeed(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.seed1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.seed2(x))
        return self.fc3(x)

    def freeze_weights(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
