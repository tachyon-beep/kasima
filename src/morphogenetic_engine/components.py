from collections import deque
import torch
from torch import nn

from .core import SeedManager


class SentinelSeed(nn.Module):
    def __init__(self, seed_id: str, dim: int):
        super().__init__()
        self.seed_id = seed_id
        self.child = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        for p in self.child.parameters():
            nn.init.uniform_(p, -1e-3, 1e-3)
            p.requires_grad = False
        self.seed_manager = SeedManager()
        self.seed_manager.register_seed(self, seed_id)

    def forward(self, x):
        info = self.seed_manager.seeds[self.seed_id]
        if info["status"] != "active":
            info["buffer"].append(x.detach())
            return x
        residual = self.child(x)
        return x + residual

    def germinate(self):
        for m in self.child.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for p in self.child.parameters():
            p.requires_grad = True

    def get_health_signal(self) -> float:
        buf = self.seed_manager.seeds[self.seed_id]["buffer"]
        if not buf:
            return float("inf")
        data = torch.cat(list(buf), dim=0)
        return data.var().item()


class BaseNet(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.act1 = nn.ReLU()
        self.seed1 = SentinelSeed("seed1", hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.seed2 = SentinelSeed("seed2", hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.seed1(x)
        x = self.act2(self.fc2(x))
        x = self.seed2(x)
        return self.out(x)

    def _freeze_backbone(self):
        for name, module in self.named_modules():
            if isinstance(module, SentinelSeed):
                continue
            for p in getattr(module, "parameters", lambda recurse=False: [])(recurse=False):
                p.requires_grad = False
