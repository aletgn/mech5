import os
import sys
sys.path.append('../../src/')

from typing import Protocol
from datetime import datetime, timezone

import numpy as np


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class Criterion(Protocol):
    def __init__(self, criterion: callable):
        self.criterion = criterion

    def __call__(self, owner, path: str) -> np.ndarray:
        _, mask = owner.query(path, self.criterion)
        return np.asarray(mask, dtype=bool)


class Mask(Protocol):
    def __init__(self, mask: np.ndarray):
        self.mask = np.asarray(mask, dtype=bool)

    def __call__(self, owner, path: str) -> np.ndarray:
        return self.mask


def test_protocols_instance():
    c = Criterion(lambda x: x > 10)
    m = Mask(np.array([0, 1, 0]))


if __name__ == "__main__":

    test_protocols_instance()