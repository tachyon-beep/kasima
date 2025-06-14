"""Run a morphogenetic architecture experiment on the two spirals dataset."""

import os
from clearml import Task

project_name = os.getenv("CLEARML_PROJECT_NAME", "kasima-cifar")
task_name = os.getenv("CLEARML_TASK_NAME", "run_experiment")

import json
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from morphogenetic_engine.core import SeedManager, KasminaMicro
from morphogenetic_engine.components import BaseNet


def create_spirals(n_samples=2000, noise=0.2, rotations=2):
    n = np.sqrt(np.random.rand(n_samples // 2)) * rotations * 2 * np.pi
    d1x = np.cos(n) * n + np.random.rand(n_samples // 2) * noise
    d1y = np.sin(n) * n + np.random.rand(n_samples // 2) * noise
    X = np.vstack((
        np.hstack((d1x, -d1x)),
        np.hstack((d1y, -d1y)),
    )).T
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))
    return X.astype(np.float32), y.astype(np.int64)


def train_epoch(model, loader, opt, crit, device, use_amp, scaler):
    model.train()
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        opt.zero_grad()
        with autocast(enabled=use_amp):
            preds = model(X)
            loss = crit(preds, y)
        if any(p.requires_grad for p in model.parameters()):
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()


def evaluate(model, loader, crit, device, use_amp):
    model.eval()
    loss_accum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            with autocast(enabled=use_amp):
                preds = model(X)
                loss_accum += crit(preds, y).item()
            pred_labels = preds.argmax(dim=1)
            correct += (pred_labels == y).sum().item()
            total += y.size(0)
    return loss_accum / len(loader), correct / total


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"

    lr = random.choice([1e-2, 5e-3, 1e-3, 5e-4])
    patience = random.randint(10, 50)
    epochs = random.randint(100, 300)
    hidden_dim = random.choice([64, 128, 256])
    config = dict(lr=lr, patience=patience, epochs=epochs, hidden_dim=hidden_dim)
    print('Selected hyperparameters:', config)

    X, y = create_spirals()
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    scaler = GradScaler(enabled=use_amp)

    seed_manager = SeedManager()
    model = BaseNet(hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    kasmina = KasminaMicro(seed_manager, patience=patience, delta=1e-3)

    best_acc = 0.0
    warm_up_epochs = 20
    for epoch in range(warm_up_epochs):
        train_epoch(model, train_loader, optimizer, criterion, device, use_amp, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp)
        best_acc = max(best_acc, val_acc)
        print(f'Warm-up Epoch {epoch+1}/{warm_up_epochs} - loss: {val_loss:.4f}, acc: {val_acc:.4f}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(warm_up_epochs, epochs):
        train_epoch(model, train_loader, optimizer, criterion, device, use_amp, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp)
        best_acc = max(best_acc, val_acc)
        print(f'Epoch {epoch+1}/{epochs} - loss: {val_loss:.4f}, acc: {val_acc:.4f}')
        if kasmina.step(val_loss):
            print('Germination occurred! Reinitializing optimizer.')
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('Best validation accuracy:', best_acc)
    print('Germination log path:', seed_manager.log_file)
    if seed_manager.log_file.exists():
        for line in seed_manager.log_file.read_text().splitlines():
            print(json.loads(line)["event"])


if __name__ == '__main__':
    main()
