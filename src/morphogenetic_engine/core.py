"""Seed management and optimisation utilities.

Kasmina germinates the dormant seed with the lowest health signal to maximise
exploratory capacity.
"""

import logging
import threading
import time
from collections import deque
from typing import Dict


class SeedManager:
    _instance = None
    _singleton_lock = threading.Lock()

    def __new__(cls):
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.seeds: Dict[str, Dict] = {}
                cls._instance.germination_log = []
                cls._instance.lock = threading.Lock()
            return cls._instance

    def register_seed(self, seed_module, seed_id: str, buffer_size: int = 100):
        self.seeds[seed_id] = {
            "module": seed_module,
            "status": "dormant",
            "buffer": deque(maxlen=buffer_size),
            "lock": threading.Lock(),
        }

    def get_health_signal(self, seed_id: str) -> float:
        """Return the variance-based health signal of the given seed.

        Access to the underlying buffer is protected by the seed-specific
        lock to avoid data races when multiple dataloaders operate in
        parallel.
        """
        seed_info = self.seeds.get(seed_id)
        if seed_info is None:
            raise KeyError(seed_id)
        lock = seed_info["lock"]
        with lock:
            buf_copy = list(seed_info["buffer"])
        return seed_info["module"].compute_health_signal(buf_copy)

    def request_germination(self, seed_id: str) -> bool:
        with self.lock:
            seed_info = self.seeds.get(seed_id)
            if not seed_info or seed_info["status"] != "dormant":
                return False
            try:
                seed_info["module"].germinate()
                seed_info["status"] = "active"
                self._log_event(seed_id, True)
                return True
            except Exception as e:
                logging.exception(
                    f"Error during germination of seed '{seed_id}': {e}"
                )
                self._log_event(seed_id, False)
                return False

    def _log_event(self, seed_id: str, success: bool):
        self.germination_log.append(
            {
                "seed_id": seed_id,
                "success": success,
                "timestamp": time.time(),
            }
        )


class KasminaMicro:
    def __init__(self, seed_manager: SeedManager, patience: int, delta: float):
        self.seed_manager = seed_manager
        self.patience = patience
        self.delta = delta
        self.plateau = 0
        self.prev_loss = None

    def step(self, val_loss: float) -> bool:
        germinated = False
        if self.prev_loss is not None:
            if abs(val_loss - self.prev_loss) < self.delta:
                self.plateau += 1
            else:
                self.plateau = 0
            if self.plateau >= self.patience:
                self.plateau = 0
                seed_id = self._select_seed()
                if seed_id:
                    germinated = self.seed_manager.request_germination(seed_id)
        self.prev_loss = val_loss
        return germinated

    def _select_seed(self) -> str | None:
        """Select the dormant seed with the lowest health signal.

        A lower variance indicates a seed whose activations have become
        stagnant, so the controller prefers to germinate these first to
        encourage exploration.
        """
        best_id = None
        best_signal = float("inf")
        with self.seed_manager.lock:
            for sid, info in self.seed_manager.seeds.items():
                if info["status"] == "dormant":
                    signal = info["module"].get_health_signal()
                    if signal < best_signal:
                        best_signal = signal
                        best_id = sid
        return best_id
