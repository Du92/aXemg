"""
Axion potentials.

The RHS should never hard-code -m^2 a directly. Instead, it should use
-dV/da, allowing different potentials in later phases.
"""

from __future__ import annotations

import numpy as np


class AxionPotential:
    """
    Base class for axion potentials.
    """

    def V(self, a: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def dV_da(self, a: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MassivePotential(AxionPotential):
    """
    Quadratic potential:

        V(a) = 1/2 m^2 a^2

    so that

        dV/da = m^2 a.
    """

    def __init__(self, m: float) -> None:
        self.m = float(m)

    def V(self, a: np.ndarray) -> np.ndarray:
        return 0.5 * self.m**2 * a**2

    def dV_da(self, a: np.ndarray) -> np.ndarray:
        return self.m**2 * a


class ZeroPotential(AxionPotential):
    """
    Massless scalar potential.
    """

    def V(self, a: np.ndarray) -> np.ndarray:
        return np.zeros_like(a)

    def dV_da(self, a: np.ndarray) -> np.ndarray:
        return np.zeros_like(a)
