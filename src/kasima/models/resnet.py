"""ResNet helpers with optional SentinelSeed hooks."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch
from torch import nn
from torchvision.models import resnet18 as tv_resnet18

if TYPE_CHECKING:
    from morphogenetic_engine.components import SentinelSeed


class SeededResNet18(nn.Module):
    """Slim ResNet-18 backbone with SentinelSeed hooks."""

    def __init__(
        self,
        num_classes: int = 10,
        seed1: Optional["SentinelSeed"] = None,
        seed2: Optional["SentinelSeed"] = None,
    ) -> None:
        super().__init__()
        self.model = tv_resnet18(num_classes=num_classes)
        # replace pooling + fc to expose seeds
        self.model.fc = nn.Linear(512, num_classes)
        self.seed1 = seed1
        self.seed2 = seed2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        if self.seed1 is not None:
            x = self.seed1(x)
        x = self.model.fc(x)
        if self.seed2 is not None:
            x = self.seed2(x)
        return x
