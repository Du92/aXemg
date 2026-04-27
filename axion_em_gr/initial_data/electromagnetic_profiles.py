"""
Initial data for electromagnetic fields.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.core.state import State


def gaussian_em_pulse_1d(
    grid: Grid,
    amplitude: float = 1.0,
    center: float = 50.0,
    width: float = 5.0,
    polarization: str = "y",
    propagation: str = "right",
) -> State:
    """
    Electromagnetic Gaussian pulse in flat 1D.

    The dynamical axion variables are initialized to zero.

    For a right-moving wave in the +x direction, one usual flat-space
    choice is:

        E_y = f(x)
        B_z = -f(x)

    for the sign convention used by the current Maxwell equations:

        ∂_t B = -curl E
        ∂_t E = -curl B

    This choice gives approximately a right-moving profile for the
    implemented convention.

    Parameters
    ----------
    polarization:
        "y" or "z".
    propagation:
        "right" or "left".
    """
    if grid.ndim != 1:
        raise NotImplementedError("Phase 2 supports only 1D EM initial data.")

    if polarization not in ("y", "z"):
        raise ValueError("polarization must be 'y' or 'z'.")

    if propagation not in ("right", "left"):
        raise ValueError("propagation must be 'right' or 'left'.")

    x = grid.coordinates_1d()

    profile = amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)

    a = np.zeros_like(x)
    Pi = np.zeros_like(x)

    E = grid.zeros_vector()
    B = grid.zeros_vector()

    sign = -1.0 if propagation == "right" else 1.0

    if polarization == "y":
        E[1] = profile
        B[2] = sign * profile
    else:
        E[2] = profile
        B[1] = -sign * profile

    return State(a=a, Pi=Pi, E=E, B=B)


def sinusoidal_em_mode_1d(
    grid: Grid,
    amplitude: float = 1.0,
    mode_number: int = 1,
    polarization: str = "y",
    propagation: str = "right",
) -> State:
    """
    Sinusoidal electromagnetic mode in flat 1D.
    """
    if grid.ndim != 1:
        raise NotImplementedError("Phase 2 supports only 1D EM initial data.")

    if polarization not in ("y", "z"):
        raise ValueError("polarization must be 'y' or 'z'.")

    if propagation not in ("right", "left"):
        raise ValueError("propagation must be 'right' or 'left'.")

    x = grid.coordinates_1d()
    xmin, xmax = grid.bounds[0]
    L = xmax - xmin
    k = 2.0 * np.pi * mode_number / L

    profile = amplitude * np.sin(k * (x - xmin))

    a = np.zeros_like(x)
    Pi = np.zeros_like(x)

    E = grid.zeros_vector()
    B = grid.zeros_vector()

    sign = -1.0 if propagation == "right" else 1.0

    if polarization == "y":
        E[1] = profile
        B[2] = sign * profile
    else:
        E[2] = profile
        B[1] = -sign * profile

    return State(a=a, Pi=Pi, E=E, B=B)
