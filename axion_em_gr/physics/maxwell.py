"""
Maxwell evolution equations in 3+1 form.

The evolved variables are the electric and magnetic fields measured by
Eulerian observers:

    E^i, B^i.

The equations are:

    partial_t B^i =
        L_beta B^i
        + N K B^i
        - epsilon^{ijk} D_j(N E_k),

    partial_t E^i =
        L_beta E^i
        + N K E^i
        - epsilon^{ijk} D_j(N B_k)
        + N j^i
        + g N epsilon^{ijk} E_k D_j a
        - g N Pi B^i.

Phase 20A uses curved curl operators in 1D/2D.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.covariant_derivatives import (
    axion_gradient_cross_E_curved,
    curl_lapse_weighted_covector_curved,
    lie_derivative_vector,
)
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.state import State
from axion_em_gr.core.tensors import lower_vector
from axion_em_gr.geometry.base_metric import GeometryFields
from axion_em_gr.physics.sources import SourceModel


def compute_maxwell_rhs(
    state: State,
    t: float,
    grid: Grid,
    geom: GeometryFields,
    sources: SourceModel,
    numerics: NumericalParameters,
    physical: PhysicalParameters,
    include_axion_coupling: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute RHS for Maxwell fields.

    Returns
    -------
    rhs_E:
        partial_t E^i

    rhs_B:
        partial_t B^i
    """
    if state.E is None or state.B is None:
        raise ValueError("Maxwell RHS requires both E and B fields.")

    order = numerics.derivative_order

    E_down = lower_vector(
        state.E,
        geom.gamma_down,
    )

    B_down = lower_vector(
        state.B,
        geom.gamma_down,
    )

    lie_B = lie_derivative_vector(
        vector_up=state.B,
        beta=geom.shift,
        grid=grid,
        order=order,
    )

    lie_E = lie_derivative_vector(
        vector_up=state.E,
        beta=geom.shift,
        grid=grid,
        order=order,
    )

    curl_NE = curl_lapse_weighted_covector_curved(
        covector_down=E_down,
        grid=grid,
        geom=geom,
        order=order,
    )

    curl_NB = curl_lapse_weighted_covector_curved(
        covector_down=B_down,
        grid=grid,
        geom=geom,
        order=order,
    )

    rhs_B = (
        lie_B
        +
        geom.lapse * geom.K * state.B
        -
        curl_NE
    )

    current = sources.current_density(
        t=t,
        grid=grid,
        state=state,
        geom=geom,
    )

    rhs_E = (
        lie_E
        +
        geom.lapse * geom.K * state.E
        -
        curl_NB
        +
        geom.lapse * current
    )

    if include_axion_coupling:
        axion_cross = axion_gradient_cross_E_curved(
            state=state,
            grid=grid,
            geom=geom,
            order=order,
        )

        rhs_E += (
            physical.g_agamma
            *
            geom.lapse
            *
            axion_cross
        )

        rhs_E -= (
            physical.g_agamma
            *
            geom.lapse
            *
            state.Pi
            *
            state.B
        )

    return rhs_E, rhs_B