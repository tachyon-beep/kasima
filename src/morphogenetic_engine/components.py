from typing import List, Optional

import torch
from torch import nn

from .core import SeedManager
from .metrics import SeedUtilityMetric, VarianceMetric


class SentinelSeed(nn.Module):
    def __init__(
        self,
        seed_id: str,
        dim: int,
        buffer_size: int = 100,
        metric: Optional[SeedUtilityMetric] = None,
    ) -> None:
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
        self.metric: SeedUtilityMetric = metric or VarianceMetric()
        self.seed_manager = SeedManager()
        self.seed_manager.register_seed(self, seed_id, buffer_size=buffer_size)

    def forward(self, x):
        info = self.seed_manager.seeds[self.seed_id]
        if info["status"] != "active":
            with info["lock"]:
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

    def compute_health_signal(self, buf: List[torch.Tensor]) -> float:
        self.metric.update(buf)
        return self.metric.score()

    def get_health_signal(self) -> float:
        return self.seed_manager.get_health_signal(self.seed_id)


class BaseNet(nn.Module):
    """Seeded ResNet-18 architecture."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        from kasima.models.resnet import SeededResNet18

        self.seed1 = SentinelSeed("seed1", 512)
        self.seed2 = SentinelSeed("seed2", num_classes)
        self.backbone = SeededResNet18(
            num_classes=num_classes,
            seed1=self.seed1,
            seed2=self.seed2,
        )
        self._freeze_backbone()

    def forward(self, x):
        return self.backbone(x)

    def _freeze_backbone(self):
        for name, module in self.named_modules():
            if isinstance(module, SentinelSeed):
                continue
            for p in getattr(module, "parameters", lambda recurse=False: [])(recurse=False):
                p.requires_grad = False
