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

    def register_seed(self, seed_module, seed_id: str):
        self.seeds[seed_id] = {
            "module": seed_module,
            "status": "dormant",
            "buffer": deque(maxlen=100),
        }

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
