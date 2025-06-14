"""CIFAR-10 and CIFAR-100 data utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class CIFARDataModule:
    """Simple data module wrapping TorchVision datasets."""

    _MEAN_STD = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    }

    def __init__(
        self,
        dataset: Literal["cifar10", "cifar100"] = "cifar10",
        root: str | Path = "./data",
        batch_size: int = 128,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.root = Path(root)
        self.batch_size = batch_size
        self.seed = seed
        mean, std = self._MEAN_STD[dataset]
        self.train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.test_tf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        self.train_ds: torch.utils.data.Dataset | None = None
        self.val_ds: torch.utils.data.Dataset | None = None
        self.test_ds: torch.utils.data.Dataset | None = None

    def setup(self) -> None:
        ds_cls = datasets.CIFAR10 if self.dataset == "cifar10" else datasets.CIFAR100
        full_train = ds_cls(self.root, train=True, download=True, transform=self.train_tf)
        self.test_ds = ds_cls(self.root, train=False, download=True, transform=self.test_tf)
        gen = torch.Generator().manual_seed(self.seed)
        train_size = int(0.9 * len(full_train))
        val_size = len(full_train) - train_size
        self.train_ds, self.val_ds = random_split(full_train, [train_size, val_size], generator=gen)

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None
        return DataLoader(self.test_ds, batch_size=self.batch_size)
