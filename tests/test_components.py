import os
import sys
import threading
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from morphogenetic_engine.components import BaseNet, SentinelSeed
from morphogenetic_engine.core import SeedManager


def test_backbone_frozen_on_init():
    model = BaseNet(hidden_dim=16)
    backbone_params = []
    for name, module in model.named_modules():
        if isinstance(module, SentinelSeed):
            continue
        for p in getattr(module, "parameters", lambda recurse=False: [])(recurse=False):
            backbone_params.append(p)
    assert backbone_params
    assert all(not p.requires_grad for p in backbone_params)


def test_health_signal_thread_safe():
    seed = SentinelSeed("race", dim=2, buffer_size=32)
    manager = SeedManager()
    manager.seeds["race"]["status"] = "active"  # avoid buffering early exit

    def worker():
        for _ in range(3000):
            x = torch.randn(1, 2)
            seed(x)
            seed.get_health_signal()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # call once more to ensure no race after threads complete
    seed.get_health_signal()

