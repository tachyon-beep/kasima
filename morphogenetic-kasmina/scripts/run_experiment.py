from __future__ import annotations
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import math
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from morphogenetic_engine import BaseNet, KasminaMicro, SeedManager


def create_spirals(
    n_samples: int = 1000, noise: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = torch.linspace(0.0, 1.0, n_samples // 2)
    theta = n * 4 * math.pi
    r = n
    x1 = r * torch.sin(theta)
    y1 = r * torch.cos(theta)
    x2 = r * torch.sin(theta + math.pi)
    y2 = r * torch.cos(theta + math.pi)
    spiral1 = torch.stack([x1, y1], dim=1)
    spiral2 = torch.stack([x2, y2], dim=1)
    X = torch.cat([spiral1, spiral2], dim=0)
    y = torch.cat([torch.zeros(n_samples // 2), torch.ones(n_samples // 2)], dim=0)
    X += torch.randn_like(X) * noise
    return X, y.long()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for data, target in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(data)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            out = model(data)
            loss = criterion(out, target)
            total_loss += loss.item() * len(data)
            pred = out.argmax(dim=1)
            correct += (pred == target).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def main() -> None:
    X, y = create_spirals(800, noise=0.2)
    dataset = TensorDataset(X, y)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [600, 200], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = BaseNet(2, 2)
    seeds = [model.seed1, model.seed2]
    kasmina = KasminaMicro(seeds, patience=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    val_loss_history = []
    seed_manager = SeedManager()

    for epoch in range(50):
        train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        val_loss_history.append(val_loss)
        print(f"Epoch {epoch:03d}: val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        kasmina.step(val_loss)
        if len(seed_manager.germination_log) > 0 and seed_manager.germination_log[
            -1
        ].endswith(str(model.seed2)):
            pass
        if any(seed.state == "Active" for seed in seeds):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        if val_acc > 0.95:
            break

    print("\nGermination Log:")
    for entry in seed_manager.germination_log:
        print(entry)


if __name__ == "__main__":
    main()
