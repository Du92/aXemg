"""
Weak gravitational-wave background in TT gauge.

Phase 5 implements a 1D TT gravitational wave propagating along the
numerical x direction. The transverse plane is therefore (y, z).

The 3+1 fields are:

    N = 1
    beta^i = 0

and the spatial metric is

    gamma_ij =
        [[1,       0,        0],
         [0, 1 + h_+,  h_x],
         [0,   h_x, 1 - h_+]]

where h_+ = h_plus(t, x), h_x = h_cross(t, x).

This is a prescribed background. The metric is not evolved dynamically.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.geometry.base_metric import BaseMetric, GeometryFields


class GWTTMetric1D(BaseMetric):
    """
    One-dimensional gravitational-wave background in TT gauge.

    Parameters
    ----------
    h_plus_amplitude:
        Amplitude of the plus polarization.
    h_cross_amplitude:
        Amplitude of the cross polarization.
    wavelength:
        GW wavelength in grid units.
    omega:
        GW angular frequency. If None, omega = k is used, corresponding
        to propagation speed c=1.
    phase_plus:
        Phase of the plus polarization.
    phase_cross:
        Phase of the cross polarization.
    packet:
        If True, multiply the sinusoidal wave by a Gaussian envelope.
    packet_center:
        Initial center of the envelope.
    packet_width:
        Width of the Gaussian envelope.
    direction:
        +1 for a wave moving toward +x, -1 for a wave moving toward -x.
    compute_K_exact:
        If True, compute the trace K from K_ij = -1/2 ∂_t gamma_ij.
        At linear order in TT gauge, K is zero, but the exact finite-amplitude
        trace contains O(h ∂_t h) corrections.
    """

    def __init__(
        self,
        h_plus_amplitude: float = 1.0e-3,
        h_cross_amplitude: float = 0.0,
        wavelength: float = 50.0,
        omega: float | None = None,
        phase_plus: float = 0.0,
        phase_cross: float = 0.0,
        packet: bool = False,
        packet_center: float = 50.0,
        packet_width: float = 15.0,
        direction: int = +1,
        compute_K_exact: bool = True,
    ) -> None:
        if direction not in (+1, -1):
            raise ValueError("direction must be +1 or -1.")

        self.h_plus_amplitude = float(h_plus_amplitude)
        self.h_cross_amplitude = float(h_cross_amplitude)
        self.wavelength = float(wavelength)
        self.k = 2.0 * np.pi / self.wavelength
        self.omega = self.k if omega is None else float(omega)
        self.phase_plus = float(phase_plus)
        self.phase_cross = float(phase_cross)
        self.packet = bool(packet)
        self.packet_center = float(packet_center)
        self.packet_width = float(packet_width)
        self.direction = direction
        self.compute_K_exact = bool(compute_K_exact)

    def _phase(self, t: float, x: np.ndarray, phase0: float) -> np.ndarray:
        """
        Phase for a wave travelling in the selected direction.

        direction = +1:
            sin(k x - omega t + phase0)

        direction = -1:
            sin(k x + omega t + phase0)
        """
        return self.k * x - self.direction * self.omega * t + phase0

    def _envelope(self, t: float, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return envelope and its time derivative.

        If packet=False, envelope = 1 and d_envelope/dt = 0.

        For a packet moving with the wave:

            u = x - direction * t - packet_center

            envelope = exp[-u^2 / (2 sigma^2)].
        """
        if not self.packet:
            envelope = np.ones_like(x)
            d_envelope_dt = np.zeros_like(x)
            return envelope, d_envelope_dt

        u = x - self.direction * t - self.packet_center
        sigma = self.packet_width

        envelope = np.exp(-0.5 * (u / sigma) ** 2)

        # d/dt exp[-u^2/(2 sigma^2)]
        # u = x - direction * t - x0
        # du/dt = -direction
        # dE/dt = E * direction * u / sigma^2
        d_envelope_dt = envelope * self.direction * u / sigma**2

        return envelope, d_envelope_dt

    def _polarizations(
        self,
        t: float,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return h_plus, h_cross and their time derivatives.
        """
        envelope, d_envelope_dt = self._envelope(t, x)

        phase_p = self._phase(t, x, self.phase_plus)
        phase_c = self._phase(t, x, self.phase_cross)

        sin_p = np.sin(phase_p)
        cos_p = np.cos(phase_p)

        sin_c = np.sin(phase_c)
        cos_c = np.cos(phase_c)

        h_plus = self.h_plus_amplitude * envelope * sin_p
        h_cross = self.h_cross_amplitude * envelope * sin_c

        # d/dt sin(kx - direction omega t + phase)
        # = -direction omega cos(...)
        dh_plus_dt = self.h_plus_amplitude * (
            d_envelope_dt * sin_p
            - self.direction * self.omega * envelope * cos_p
        )

        dh_cross_dt = self.h_cross_amplitude * (
            d_envelope_dt * sin_c
            - self.direction * self.omega * envelope * cos_c
        )

        return h_plus, h_cross, dh_plus_dt, dh_cross_dt

    def evaluate(self, t: float, grid: Grid) -> GeometryFields:
        if grid.ndim != 1:
            raise NotImplementedError("GWTTMetric1D currently supports only 1D grids.")

        x = grid.coordinates_1d()
        shape = grid.shape_full

        h_plus, h_cross, dh_plus_dt, dh_cross_dt = self._polarizations(t, x)

        lapse = np.ones(shape, dtype=float)
        shift = np.zeros((3, *shape), dtype=float)

        gamma_down = np.zeros((3, 3, *shape), dtype=float)
        gamma_up = np.zeros((3, 3, *shape), dtype=float)

        gamma_down[0, 0] = 1.0
        gamma_down[1, 1] = 1.0 + h_plus
        gamma_down[2, 2] = 1.0 - h_plus
        gamma_down[1, 2] = h_cross
        gamma_down[2, 1] = h_cross

        # Determinant of the transverse yz block:
        #
        # det [[1+h_plus, h_cross],
        #      [h_cross,  1-h_plus]]
        #
        # = 1 - h_plus^2 - h_cross^2.
        det_transverse = 1.0 - h_plus**2 - h_cross**2

        if np.any(det_transverse <= 0.0):
            raise ValueError(
                "GW spatial metric became non-positive. "
                "Reduce h_plus_amplitude or h_cross_amplitude."
            )

        sqrt_gamma = np.sqrt(det_transverse)

        gamma_up[0, 0] = 1.0

        gamma_up[1, 1] = (1.0 - h_plus) / det_transverse
        gamma_up[2, 2] = (1.0 + h_plus) / det_transverse
        gamma_up[1, 2] = -h_cross / det_transverse
        gamma_up[2, 1] = -h_cross / det_transverse

        if self.compute_K_exact:
            # K_ij = -1/2 ∂_t gamma_ij for N=1, beta=0.
            K_down = np.zeros((3, 3, *shape), dtype=float)

            K_down[1, 1] = -0.5 * dh_plus_dt
            K_down[2, 2] = +0.5 * dh_plus_dt
            K_down[1, 2] = -0.5 * dh_cross_dt
            K_down[2, 1] = -0.5 * dh_cross_dt

            K = np.zeros(shape, dtype=float)

            for i in range(3):
                for j in range(3):
                    K += gamma_up[i, j] * K_down[i, j]
        else:
            K = np.zeros(shape, dtype=float)

        return GeometryFields(
            lapse=lapse,
            shift=shift,
            gamma_down=gamma_down,
            gamma_up=gamma_up,
            sqrt_gamma=sqrt_gamma,
            K=K,
        )
