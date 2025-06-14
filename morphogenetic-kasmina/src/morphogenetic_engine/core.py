from __future__ import annotations

import threading
from typing import List, Optional


class SeedManager:
    _instance: Optional["SeedManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "SeedManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self.seeds = []
        self.germination_log: List[str] = []
        self._seed_lock = threading.Lock()

    def register(self, seed) -> None:
        with self._seed_lock:
            self.seeds.append(seed)

    def request_germination(self, seed) -> None:
        with self._seed_lock:
            if getattr(seed, "state", "Dormant") != "Active":
                seed.germinate()
                self.germination_log.append(f"Germinated {seed}")


class KasminaMicro:
    """Heuristic controller monitoring validation loss."""

    def __init__(self, seeds: List, patience: int = 5, min_delta: float = 0.001):
        self.seeds = seeds
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.wait = 0
        self.seed_manager = SeedManager()
        for seed in seeds:
            self.seed_manager.register(seed)

    def step(self, val_loss: float) -> None:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                for seed in self.seeds:
                    if getattr(seed, "state", "Dormant") == "Dormant":
                        self.seed_manager.request_germination(seed)
                        self.wait = 0
                        break
