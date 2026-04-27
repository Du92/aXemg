"""
Toy diagonal 1D geometry.

This metric is not intended to represent a full physical spacetime solution.
It is a controlled background for testing the geometric terms of the 3+1
evolution system.

We assume all quantities depend only on x.

Spatial metric:

    gamma_ij = diag(gxx(x), gyy(x), gzz(x))

Lapse:

    N = N(x)

Shift:

    beta^i = (beta_x(x), 0, 0)

The determinant is:

    gamma = gxx gyy gzz.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.geometry.base_metric import BaseMetric, GeometryFields


class DiagonalMetric1D(BaseMetric):
    """
    Simple diagonal 1D 3+1 background.

    Parameters
    ----------
    lapse_amplitude:
        Amplitude of the lapse deformation.
    metric_amplitude:
        Amplitude of the spatial metric deformation.
    shift_amplitude:
        Amplitude of beta^x.
    center:
        Center of the Gaussian deformation.
    width:
        Width of the Gaussian deformation.
    K_value:
        Constant trace of extrinsic curvature.
    """

    def __init__(
        self,
        lapse_amplitude: float = 0.1,
        metric_amplitude: float = 0.1,
        shift_amplitude: float = 0.0,
        center: float = 50.0,
        width: float = 10.0,
        K_value: float = 0.0,
    ) -> None:
        self.lapse_amplitude = lapse_amplitude
        self.metric_amplitude = metric_amplitude
        self.shift_amplitude = shift_amplitude
        self.center = center
        self.width = width
        self.K_value = K_value

    def evaluate(self, t: float, grid: Grid) -> GeometryFields:
        if grid.ndim != 1:
            raise NotImplementedError("DiagonalMetric1D supports only 1D grids.")

        x = grid.coordinates_1d()
        shape = grid.shape_full

        profile = np.exp(-0.5 * ((x - self.center) / self.width) ** 2)

        lapse = 1.0 + self.lapse_amplitude * profile

        shift = np.zeros((3, *shape), dtype=float)
        shift[0] = self.shift_amplitude * profile

        gxx = 1.0 + self.metric_amplitude * profile
        gyy = 1.0 - 0.5 * self.metric_amplitude * profile
        gzz = 1.0 - 0.5 * self.metric_amplitude * profile

        gamma_down = np.zeros((3, 3, *shape), dtype=float)
        gamma_up = np.zeros((3, 3, *shape), dtype=float)

        gamma_down[0, 0] = gxx
        gamma_down[1, 1] = gyy
        gamma_down[2, 2] = gzz

        gamma_up[0, 0] = 1.0 / gxx
        gamma_up[1, 1] = 1.0 / gyy
        gamma_up[2, 2] = 1.0 / gzz

        sqrt_gamma = np.sqrt(gxx * gyy * gzz)

        K = self.K_value * np.ones(shape, dtype=float)

        return GeometryFields(
            lapse=lapse,
            shift=shift,
            gamma_down=gamma_down,
            gamma_up=gamma_up,
            sqrt_gamma=sqrt_gamma,
            K=K,
        )
