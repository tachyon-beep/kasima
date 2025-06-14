import sys
import os
from unittest import mock

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from kasima.data.cifar import CIFARDataModule  # noqa: E402


class DummyDS:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def test_module_split_lengths(tmp_path):
    with mock.patch(
        "torchvision.datasets.CIFAR10",
        side_effect=[DummyDS(50000), DummyDS(10000)],
    ):
        dm = CIFARDataModule(root=tmp_path)
        dm.setup()
        assert len(dm.train_ds) + len(dm.val_ds) == 50000
        assert len(dm.test_ds) == 10000
