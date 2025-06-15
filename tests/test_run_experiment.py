import importlib
import os
import sys

import clearml
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from kasima.data.cifar import CIFARDataModule  # noqa: E402


class TinyDS(torch.utils.data.Dataset):
    def __init__(self, n=4, num_classes=10):
        self.x = torch.randn(n, 3, 32, 32)
        self.y = torch.randint(0, num_classes, (n,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def fake_setup(self):
    ds = TinyDS()
    self.train_ds = ds
    self.val_ds = ds
    self.test_ds = ds


def dataloader(self):
    return torch.utils.data.DataLoader(self.train_ds, batch_size=2)


def test_run_experiment_smoke(monkeypatch, tmp_path):
    monkeypatch.setattr(CIFARDataModule, "setup", fake_setup)
    monkeypatch.setattr(CIFARDataModule, "train_dataloader", dataloader)
    monkeypatch.setattr(CIFARDataModule, "val_dataloader", dataloader)

    called = {}

    def fake_init(project, task):
        called["project"] = project
        called["task"] = task
        class Dummy:
            def connect(self, *_a, **_k):
                pass
        clearml.Task.current_task = staticmethod(lambda: Dummy())
        return Dummy()

    monkeypatch.setattr(clearml.Task, "init", fake_init)

    if "scripts.run_experiment" in sys.modules:
        del sys.modules["scripts.run_experiment"]
    exp = importlib.import_module("scripts.run_experiment")

    ret = exp.main([
        "--dataset",
        "cifar10",
        "--batch_size",
        "2",
        "--lr",
        "0.001",
        "--patience",
        "1",
        "--delta",
        "0.0",
        "--num_epochs",
        "1",
        "--device",
        "cpu",
        "--log_dir",
        str(tmp_path),
    ])
    assert ret == 0
    assert called["project"] == "kasima-cifar"
    assert (tmp_path / "checkpoint.pt").exists()
    assert any(tmp_path.glob("events.out.tfevents.*"))


def test_resume_from_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(CIFARDataModule, "setup", fake_setup)
    monkeypatch.setattr(CIFARDataModule, "train_dataloader", dataloader)
    monkeypatch.setattr(CIFARDataModule, "val_dataloader", dataloader)

    def fake(project, task):
        class D:
            def connect(self, *_a, **_k):
                pass
        clearml.Task.current_task = staticmethod(lambda: D())
        return D()

    monkeypatch.setattr(clearml.Task, "init", fake)

    if "scripts.run_experiment" in sys.modules:
        del sys.modules["scripts.run_experiment"]
    exp = importlib.import_module("scripts.run_experiment")

    exp.main([
        "--dataset",
        "cifar10",
        "--batch_size",
        "2",
        "--lr",
        "0.001",
        "--patience",
        "1",
        "--delta",
        "0.0",
        "--num_epochs",
        "1",
        "--device",
        "cpu",
        "--log_dir",
        str(tmp_path),
    ])

    ckpt = tmp_path / "checkpoint.pt"
    assert ckpt.exists()

    if "scripts.run_experiment" in sys.modules:
        del sys.modules["scripts.run_experiment"]
    exp = importlib.import_module("scripts.run_experiment")

    exp.main([
        "--dataset",
        "cifar10",
        "--batch_size",
        "2",
        "--lr",
        "0.001",
        "--patience",
        "1",
        "--delta",
        "0.0",
        "--num_epochs",
        "2",
        "--device",
        "cpu",
        "--log_dir",
        str(tmp_path),
        "--resume-from",
        str(ckpt),
    ])

    data = torch.load(ckpt)
    assert data["epoch"] == 2
