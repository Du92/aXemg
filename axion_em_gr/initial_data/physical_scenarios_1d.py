"""
Physically motivated 1D scenarios.

Phase 16 introduces reduced physical setups such as:

    GW + axion halo + magnetic field + axion gradient.

These are not full 3D astrophysical simulations. They are controlled reduced
models intended to isolate coupling channels.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.core.state import State


def localized_gradient_profile(
    x: np.ndarray,
    center: float,
    width: float,
    amplitude: float,
) -> np.ndarray:
    """
    Localized odd profile with a strong gradient around center.

    This behaves like:

        A (x-x0)/sigma exp[-(x-x0)^2/(2 sigma^2)]

    It is useful because it creates a localized axion gradient without imposing
    a nonzero linear trend at the boundaries.
    """
    u = (x - center) / width

    return amplitude * u * np.exp(-0.5 * u**2)


def gaussian_profile(
    x: np.ndarray,
    center: float,
    width: float,
    amplitude: float,
) -> np.ndarray:
    """
    Gaussian profile.
    """
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def axion_halo_gradient_magnetized_1d(
    grid: Grid,
    axion_amplitude: float = 1.0,
    axion_center: float = 45.0,
    axion_width: float = 10.0,
    axion_background: float = 0.0,
    gradient_amplitude: float = 0.3,
    gradient_width: float = 12.0,
    Pi_amplitude: float = 0.0,
    Pi_width: float | None = None,
    g_agamma: float = 0.03,
    background_Bx: float = 1.0,
    background_By: float = 0.0,
    background_Bz: float = 0.0,
    em_pulse_amplitude: float = 0.15,
    em_pulse_center: float = 40.0,
    em_pulse_width: float = 8.0,
    em_pulse_polarization: str = "y",
    em_pulse_propagation: str = "right",
    constraint_solved_Ex: bool = True,
) -> State:
    """
    1D axion halo with localized gradient, magnetic background and optional
    transverse EM pulse.

    The axion profile is

        a(x) = a_bg
             + A exp[-(x-x0)^2/(2 sigma^2)]
             + A_grad ((x-x0)/sigma_g) exp[-(x-x0)^2/(2 sigma_g^2)].

    The magnetic background can contain B^x, B^y, B^z.

    If constraint_solved_Ex=True and B^x is constant, we set

        E^x = - g B^x a,

    which solves the 1D axion-modified Gauss constraint approximately:

        partial_x E^x + g B^x partial_x a = 0.

    A transverse EM pulse may be added as:

        E^y = f(x), B^z = +/- f(x)

    or

        E^z = f(x), B^y = -/+ f(x)

    using the stable Maxwell sign convention implemented in the code.
    """
    if grid.ndim != 1:
        raise NotImplementedError("axion_halo_gradient_magnetized_1d requires 1D.")

    if em_pulse_polarization not in ("y", "z"):
        raise ValueError("em_pulse_polarization must be 'y' or 'z'.")

    if em_pulse_propagation not in ("right", "left"):
        raise ValueError("em_pulse_propagation must be 'right' or 'left'.")

    x = grid.coordinates_1d()

    halo = gaussian_profile(
        x=x,
        center=axion_center,
        width=axion_width,
        amplitude=axion_amplitude,
    )

    gradient_part = localized_gradient_profile(
        x=x,
        center=axion_center,
        width=gradient_width,
        amplitude=gradient_amplitude,
    )

    a = axion_background + halo + gradient_part

    if Pi_width is None:
        Pi_width = axion_width

    Pi = gaussian_profile(
        x=x,
        center=axion_center,
        width=Pi_width,
        amplitude=Pi_amplitude,
    )

    E = grid.zeros_vector()
    B = grid.zeros_vector()

    B[0] = background_Bx
    B[1] = background_By
    B[2] = background_Bz

    if constraint_solved_Ex:
        E[0] = -g_agamma * background_Bx * a

    em_profile = gaussian_profile(
        x=x,
        center=em_pulse_center,
        width=em_pulse_width,
        amplitude=em_pulse_amplitude,
    )

    sign = 1.0 if em_pulse_propagation == "right" else -1.0

    if em_pulse_polarization == "y":
        E[1] += em_profile
        B[2] += sign * em_profile
    else:
        E[2] += em_profile
        B[1] += -sign * em_profile

    return State(a=a, Pi=Pi, E=E, B=B)