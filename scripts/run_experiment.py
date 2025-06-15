"""Train a seeded ResNet on CIFAR with Kasmina.

This entrypoint glues together the CIFAR data module, the seeded ResNet
architecture and the Kasmina controller.  Training progress is logged to
ClearML and TensorBoard.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import torch
from clearml import Task
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from kasima.data.cifar import CIFARDataModule
from morphogenetic_engine.components import BaseNet
from morphogenetic_engine.core import KasminaMicro, SeedManager


PROJECT_NAME = os.getenv("CLEARML_PROJECT_NAME", "kasima-cifar")
TASK_NAME = os.getenv("CLEARML_TASK_NAME", "run_experiment")

# Allow tests to patch ``Task.init`` on import.
_task = Task.init(PROJECT_NAME, TASK_NAME)  # pylint: disable=unused-variable


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _env_default(key: str, default: str) -> str:
    return os.getenv(key, default)


def _loader_metrics(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Return mean loss and accuracy for the given loader."""

    model.eval()
    loss_accum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(x_batch)
            loss_accum += criterion(preds, y_batch).item()
            pred_labels = preds.argmax(dim=1)
            correct += (pred_labels == y_batch).sum().item()
            total += y_batch.size(0)

    return loss_accum / max(len(loader), 1), correct / max(total, 1)


def _train_epoch(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
) -> tuple[float, float]:
    """Train for a single epoch and return loss and accuracy."""

    model.train()
    loss_accum = 0.0
    correct = 0
    total = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimiser.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        if any(p.requires_grad for p in model.parameters()):
            loss.backward()
            optimiser.step()
        loss_accum += loss.item()
        pred_labels = preds.argmax(dim=1)
        correct += (pred_labels == y_batch).sum().item()
        total += y_batch.size(0)

    return loss_accum / max(len(loader), 1), correct / max(total, 1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kasmina CIFAR experiment")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default=_env_default("DATASET", "cifar10"))
    parser.add_argument("--batch_size", type=int, default=int(_env_default("BATCH_SIZE", "128")))
    parser.add_argument("--lr", type=float, default=float(_env_default("LR", "0.001")))
    parser.add_argument("--patience", type=int, default=int(_env_default("PATIENCE", "20")))
    parser.add_argument("--delta", type=float, default=float(_env_default("DELTA", "0.001")))
    parser.add_argument("--num_epochs", type=int, default=int(_env_default("NUM_EPOCHS", "10")))
    parser.add_argument("--device", default=_env_default("DEVICE", "cpu"))
    parser.add_argument("--log_dir", type=str, default=_env_default("LOG_DIR", "runs"))
    parser.add_argument("--resume-from", dest="resume_from", type=str, default=None)
    return parser


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    writer = SummaryWriter(args.log_dir)
    Task.current_task().connect(vars(args))

    dm = CIFARDataModule(dataset=args.dataset, batch_size=args.batch_size)
    dm.setup()

    num_classes = 10 if args.dataset == "cifar10" else 100
    model = BaseNet(num_classes=num_classes).to(args.device)
    model._freeze_backbone()

    optimiser = torch.optim.Adam(
        list(model.seed1.parameters()) + list(model.seed2.parameters()),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda _: 1.0)
    criterion = nn.CrossEntropyLoss()
    seed_manager = SeedManager()
    kasmina = KasminaMicro(seed_manager, patience=args.patience, delta=args.delta)

    start_epoch = 0
    best_loss = float("inf")
    ckpt_path = Path(args.log_dir) / "checkpoint.pt"
    if args.resume_from:
        data = torch.load(args.resume_from, map_location=args.device)
        model.load_state_dict(data["model"])
        optimiser.load_state_dict(data["optimiser"])
        scheduler.load_state_dict(data["scheduler"])
        start_epoch = data.get("epoch", 0)
        best_loss = data.get("best_loss", float("inf"))

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_acc = _train_epoch(
            model, dm.train_dataloader(), args.device, criterion, optimiser
        )
        val_loss, val_acc = _loader_metrics(
            model, dm.val_dataloader(), args.device, criterion
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)

        germinated = kasmina.step(val_loss)
        if germinated:
            writer.add_scalar("germination/event", 1, epoch)
            for sid, info in seed_manager.seeds.items():
                if info["status"] == "active":
                    params = torch.cat([
                        p.detach().view(-1) for p in info["module"].parameters()
                    ])
                    writer.add_histogram(f"weights/{sid}", params, epoch)

        scheduler.step()
        if val_loss < best_loss or germinated or epoch == args.num_epochs - 1:
            best_loss = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_loss": best_loss,
                },
                ckpt_path,
            )

    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
