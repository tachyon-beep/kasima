"""Seed utility metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch


class SeedUtilityMetric(ABC):
    """Abstract interface for seed health metrics."""

    @abstractmethod
    def update(self, buf: List[torch.Tensor]) -> None:  # pragma: no cover - interface
        """Update internal state from buffered activations."""

    @abstractmethod
    def score(self) -> float:  # pragma: no cover - interface
        """Return the metric score."""


class VarianceMetric(SeedUtilityMetric):
    """Variance-based health metric."""

    def __init__(self) -> None:
        self._score = float("inf")

    def update(self, buf: List[torch.Tensor]) -> None:
        if not buf:
            self._score = float("inf")
            return
        data = torch.cat(buf, dim=0)
        self._score = data.var().item()

    def score(self) -> float:
        return self._score


class GradientMetric(SeedUtilityMetric):
    """Placeholder for gradient-based metric implementation."""

    def update(self, buf: List[torch.Tensor]) -> None:
        raise NotImplementedError

    def score(self) -> float:
        raise NotImplementedError
