import os
import sys
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from morphogenetic_engine.core import SeedManager, KasminaMicro  # noqa: E402


class DummySeed:
    def __init__(self, seed_id: str, signal: float = 0.0):
        self.seed_id = seed_id
        self.signal = signal
        self.germinated = False

    def germinate(self):
        self.germinated = True

    def get_health_signal(self):
        return self.signal


def test_seed_manager_singleton():
    sm1 = SeedManager()
    sm2 = SeedManager()
    assert sm1 is sm2


def test_register_and_germinate():
    sm = SeedManager()
    # reset state for test isolation
    sm.seeds.clear()
    sm.germination_log.clear()

    seed = DummySeed("s1")
    sm.register_seed(seed, "s1")
    assert sm.seeds["s1"]["status"] == "dormant"

    assert sm.request_germination("s1") is True
    assert sm.seeds["s1"]["status"] == "active"
    assert seed.germinated
    assert sm.germination_log[-1]["seed_id"] == "s1"
    assert sm.germination_log[-1]["success"] is True


def test_register_seed_creates_lock():
    sm = SeedManager()
    sm.seeds.clear()
    dummy = DummySeed("bar")
    sm.register_seed(dummy, "bar")
    assert isinstance(sm.seeds["bar"].get("lock"), type(threading.Lock()))


def test_kasmina_micro_selects_lowest_signal_seed():
    sm = SeedManager()
    sm.seeds.clear()
    sm.germination_log.clear()

    seed_low = DummySeed("low", signal=0.1)
    seed_high = DummySeed("high", signal=0.5)
    sm.register_seed(seed_low, "low")
    sm.register_seed(seed_high, "high")

    kasmina = KasminaMicro(sm, patience=2, delta=0.2)

    # first step sets prev_loss
    kasmina.step(1.0)
    # plateau increment 1
    kasmina.step(0.9)
    # plateau increment 2 -> triggers germination
    kasmina.step(0.85)

    assert seed_low.germinated
    assert not seed_high.germinated
