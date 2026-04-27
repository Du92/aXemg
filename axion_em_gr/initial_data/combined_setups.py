"""
Combined initial data setups for axion-electromagnetic evolutions.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.core.state import State


def gaussian_axion_uniform_magnetic_field_1d(
    grid: Grid,
    axion_amplitude: float = 1.0,
    axion_center: float = 50.0,
    axion_width: float = 5.0,
    axion_momentum_amplitude: float = 0.0,
    B0: tuple[float, float, float] = (1.0, 0.0, 0.0),
    E0: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> State:
    """
    Gaussian axion profile plus uniform electromagnetic background.

    This is useful for Phase 3.

    The most direct coupling test is obtained with a background B^x and a
    nontrivial Pi. Then the Maxwell equation contains

        ∂_t E^x = -g_agamma Pi B^x,

    so an electric field parallel to B^x is generated. This then sources the
    axion through

        -g_agamma E_i B^i.
    """
    if grid.ndim != 1:
        raise NotImplementedError("Phase 3 combined setup supports only 1D.")

    x = grid.coordinates_1d()

    a = axion_amplitude * np.exp(
        -0.5 * ((x - axion_center) / axion_width) ** 2
    )

    Pi = axion_momentum_amplitude * np.exp(
        -0.5 * ((x - axion_center) / axion_width) ** 2
    )

    E = grid.zeros_vector()
    B = grid.zeros_vector()

    for i in range(3):
        E[i] = E0[i]
        B[i] = B0[i]

    return State(a=a, Pi=Pi, E=E, B=B)


def gaussian_axion_plus_em_wave_1d(
    grid: Grid,
    axion_amplitude: float = 1.0,
    axion_center: float = 50.0,
    axion_width: float = 8.0,
    axion_momentum_amplitude: float = 0.0,
    em_amplitude: float = 0.1,
    em_center: float = 40.0,
    em_width: float = 6.0,
    background_Bx: float = 1.0,
) -> State:
    """
    Gaussian axion plus a transverse electromagnetic pulse and background B^x.

    This setup gives several coupling channels:

    - background B^x couples to Pi and generates E^x;
    - transverse EM fields can contribute to E_i B^i if nonorthogonal
      components are added later;
    - gradients of a couple to transverse electric fields through
      g epsilon^{ijk} E_k ∂_j a.
    """
    if grid.ndim != 1:
        raise NotImplementedError("Phase 3 combined setup supports only 1D.")

    x = grid.coordinates_1d()

    a = axion_amplitude * np.exp(
        -0.5 * ((x - axion_center) / axion_width) ** 2
    )

    Pi = axion_momentum_amplitude * np.exp(
        -0.5 * ((x - axion_center) / axion_width) ** 2
    )

    em_profile = em_amplitude * np.exp(
        -0.5 * ((x - em_center) / em_width) ** 2
    )

    E = grid.zeros_vector()
    B = grid.zeros_vector()

    B[0] = background_Bx

    # A transverse pulse using the convention of Phase 2.
    E[1] = em_profile
    B[2] = -em_profile

    return State(a=a, Pi=Pi, E=E, B=B)

def gaussian_axion_em_wave_background_Bx_1d(
    grid: Grid,
    axion_amplitude: float = 1.0,
    axion_center: float = 40.0,
    axion_width: float = 7.0,
    axion_momentum_amplitude: float = 0.3,
    em_amplitude: float = 0.2,
    em_center: float = 45.0,
    em_width: float = 8.0,
    background_Bx: float = 1.0,
    propagation: str = "right",
) -> State:
    """
    Axion packet + transverse EM pulse + longitudinal background B^x.

    This is useful for testing a GW background because the transverse EM
    pulse feels the TT metric through gamma_yy, gamma_zz, gamma_yz, while
    the axion coupling still has a clean longitudinal channel through B^x.

    With the stable Maxwell sign convention used in the code, a right-moving
    y-polarized EM pulse is initialized as:

        E^y = f(x)
        B^z = f(x).
    """
    if grid.ndim != 1:
        raise NotImplementedError("This setup supports only 1D.")

    if propagation not in ("right", "left"):
        raise ValueError("propagation must be 'right' or 'left'.")

    x = grid.coordinates_1d()

    a = axion_amplitude * np.exp(
        -0.5 * ((x - axion_center) / axion_width) ** 2
    )

    Pi = axion_momentum_amplitude * np.exp(
        -0.5 * ((x - axion_center) / axion_width) ** 2
    )

    em_profile = em_amplitude * np.exp(
        -0.5 * ((x - em_center) / em_width) ** 2
    )

    sign = 1.0 if propagation == "right" else -1.0

    E = grid.zeros_vector()
    B = grid.zeros_vector()

    B[0] = background_Bx

    E[1] = em_profile
    B[2] = sign * em_profile

    return State(a=a, Pi=Pi, E=E, B=B)

def gaussian_axion_uniform_Bx_constraint_solved_1d(
    grid: Grid,
    axion_amplitude: float = 1.0,
    axion_center: float = 50.0,
    axion_width: float = 5.0,
    axion_momentum_amplitude: float = 0.0,
    g_agamma: float = 0.03,
    Bx: float = 1.0,
    Ex_constant: float = 0.0,
) -> State:
    """
    1D Gaussian axion profile plus uniform B^x with Gauss-solved E^x.

    In 1D, with rho=0 and B^x = constant, the axion-modified Gauss
    constraint is:

        partial_x E^x = - g B^x partial_x a.

    A direct solution is:

        E^x = - g B^x a + const.

    This makes:

        partial_x E^x + g B^x partial_x a = 0

    up to finite-difference and boundary errors.
    """
    if grid.ndim != 1:
        raise NotImplementedError("This setup supports only 1D grids.")

    x = grid.coordinates_1d()

    a = axion_amplitude * np.exp(
        -0.5 * ((x - axion_center) / axion_width) ** 2
    )

    Pi = axion_momentum_amplitude * np.exp(
        -0.5 * ((x - axion_center) / axion_width) ** 2
    )

    E = grid.zeros_vector()
    B = grid.zeros_vector()

    B[0] = Bx
    E[0] = -g_agamma * Bx * a + Ex_constant

    return State(a=a, Pi=Pi, E=E, B=B)
