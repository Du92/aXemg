"""
Base classes for 3+1 geometry.

In Phase 1, we only use flat spacetime. The interface is nevertheless
written in terms of 3+1 quantities to make later extensions natural.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axion_em_gr.core.grid import Grid


@dataclass
class GeometryFields:
    """
    3+1 geometric fields evaluated on the numerical grid.

    Attributes
    ----------
    lapse:
        Lapse N.
    shift:
        Shift beta^i with shape (3, *grid.shape_full).
    gamma_down:
        Spatial metric gamma_ij with shape (3, 3, *grid.shape_full).
    gamma_up:
        Inverse spatial metric gamma^ij with shape (3, 3, *grid.shape_full).
    sqrt_gamma:
        Square root of the determinant of gamma_ij.
    K:
        Trace of extrinsic curvature.
    """

    lapse: np.ndarray
    shift: np.ndarray
    gamma_down: np.ndarray
    gamma_up: np.ndarray
    sqrt_gamma: np.ndarray
    K: np.ndarray


class BaseMetric:
    """
    Abstract base class for 3+1 backgrounds.
    """

    def evaluate(self, t: float, grid: Grid) -> GeometryFields:
        raise NotImplementedError
