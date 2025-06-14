import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from kaslog import verify  # noqa: E402
from morphogenetic_engine.core import SeedManager  # noqa: E402
from morphogenetic_engine.components import SentinelSeed  # noqa: E402


def test_verify_detects_corruption(tmp_path):
    sm = SeedManager()
    sm.seeds.clear()
    sm.log_file = tmp_path / "log.jsonl"
    sm.prev_hash = ""
    sm.register_seed(SentinelSeed("foo", dim=2), "foo")
    sm.request_germination("foo")
    log_path = sm.log_file
    lines = log_path.read_text().splitlines()
    assert verify(log_path)
    # corrupt second field
    bad = json.loads(lines[-1])
    bad["event"]["seed_id"] = "bar"
    lines[-1] = json.dumps(bad)
    log_path.write_text("\n".join(lines) + "\n")
    assert not verify(log_path)
