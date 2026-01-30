import os
import sys
sys.path.append('../../src/')

from typing import Protocol, Dict, Any
from datetime import datetime, timezone

import numpy as np


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class CriterionMaskProtocol(Protocol):
    def __call__(self, data: np.ndarray) -> np.ndarray: ...


class Criterion:
    """
    Wrapper for a callable criterion function that produces a boolean mask.

    Parameters
    ----------
    criterion : callable
        Function that takes a NumPy array and optional keyword arguments,
        and returns a boolean mask of the same length as the input.
    **cargs : dict
        Additional keyword arguments passed to the criterion function.

    Methods
    -------
    __call__(data)
        Apply the criterion to the provided data and return a boolean mask.
    """
    def __init__(self, criterion: callable, **cargs: Dict[str, Any]) -> None:
        self.criterion = criterion
        self.cargs = cargs


    def __call__(self, data: np.ndarray) -> np.ndarray:
        mask = self.criterion(data, **self.cargs)
        return np.asarray(mask, dtype=bool)


class Mask:
    """
    Wrapper for a precomputed boolean mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean array representing the mask to apply.

    Methods
    -------
    __call__(data)
        Return the stored mask. The input is ignored but kept for polymorphism.
    """
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = np.asarray(mask, dtype=bool)


    def __call__(self, data: np.ndarray = None) -> np.ndarray:
        return self.mask


class TrueMask:
    """
    A mask that selects all elements (no-op).

    Methods
    -------
    __call__(data)
        Return a boolean array of all True, same length as data.
    """
    def __init__(self) -> None:
        pass

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return np.ones(data.shape[0], dtype=bool)


def test_protocols_instance():
    c = Criterion(lambda x: x > 10)
    m = Mask(np.array([0, 1, 0]))


if __name__ == "__main__":

    test_protocols_instance()