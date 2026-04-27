"""
2D combined initial data for axion-electromagnetic evolutions.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.geometry.flat import FlatMetric
#from axion_em_gr.physics.constraint_cleaning import clean_electric_constraint_flat
from axion_em_gr.physics.sources import VacuumSources

from axion_em_gr.core.grid import Grid
from axion_em_gr.core.state import State

from axion_em_gr.physics.constraint_cleaning import (
    clean_electric_constraint_flat,
    clean_electric_constraint_curved,
)


def gaussian_2d(
    X: np.ndarray,
    Y: np.ndarray,
    amplitude: float,
    center: tuple[float, float],
    width: tuple[float, float],
) -> np.ndarray:
    """
    2D Gaussian profile.
    """
    x0, y0 = center
    sx, sy = width

    return amplitude * np.exp(
        -0.5 * (
            ((X - x0) / sx) ** 2
            +
            ((Y - y0) / sy) ** 2
        )
    )


def gaussian_axion_uniform_Bz_2d(
    grid: Grid,
    axion_amplitude: float = 1.0,
    axion_center: tuple[float, float] = (0.0, 0.0),
    axion_width: tuple[float, float] = (5.0, 5.0),
    axion_momentum_amplitude: float = 0.3,
    B0: tuple[float, float, float] = (0.0, 0.0, 1.0),
    E0: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> State:
    """
    2D Gaussian axion profile plus a uniform magnetic background.

    Useful first 2D axion-EM coupling test.
    """
    if grid.ndim != 2:
        raise NotImplementedError("This setup requires a 2D grid.")

    X, Y = grid.coordinates_2d()

    a = gaussian_2d(
        X,
        Y,
        amplitude=axion_amplitude,
        center=axion_center,
        width=axion_width,
    )

    Pi = gaussian_2d(
        X,
        Y,
        amplitude=axion_momentum_amplitude,
        center=axion_center,
        width=axion_width,
    )

    E = grid.zeros_vector()
    B = grid.zeros_vector()

    for i in range(3):
        E[i] = E0[i]
        B[i] = B0[i]

    return State(a=a, Pi=Pi, E=E, B=B)


def gaussian_axion_em_ring_2d(
    grid: Grid,
    axion_amplitude: float = 1.0,
    axion_center: tuple[float, float] = (0.0, 0.0),
    axion_width: tuple[float, float] = (6.0, 6.0),
    axion_momentum_amplitude: float = 0.3,
    em_amplitude: float = 0.2,
    em_center: tuple[float, float] = (0.0, 0.0),
    em_width: tuple[float, float] = (8.0, 8.0),
    background_Bz: float = 1.0,
) -> State:
    """
    2D axion packet plus a transverse electromagnetic perturbation.

    This setup is not a constraint-solved physical EM wave. It is intended
    as a first 2D visualization and coupling test.

    Fields:
        a  = Gaussian
        Pi = Gaussian
        B^z = background_Bz
        E^z = Gaussian EM perturbation

    The E^z perturbation couples to transverse axion gradients through
    the Maxwell axion source:

        C^x ~ E_z ∂_y a
        C^y ~ -E_z ∂_x a.
    """
    if grid.ndim != 2:
        raise NotImplementedError("This setup requires a 2D grid.")

    X, Y = grid.coordinates_2d()

    a = gaussian_2d(
        X,
        Y,
        amplitude=axion_amplitude,
        center=axion_center,
        width=axion_width,
    )

    Pi = gaussian_2d(
        X,
        Y,
        amplitude=axion_momentum_amplitude,
        center=axion_center,
        width=axion_width,
    )

    em_profile = gaussian_2d(
        X,
        Y,
        amplitude=em_amplitude,
        center=em_center,
        width=em_width,
    )

    E = grid.zeros_vector()
    B = grid.zeros_vector()

    B[2] = background_Bz
    E[2] = em_profile

    return State(a=a, Pi=Pi, E=E, B=B)

def gaussian_axion_uniform_Bxy_constraint_cleaned_2d(
    grid: Grid,
    axion_amplitude: float = 1.0,
    axion_center: tuple[float, float] = (0.0, 0.0),
    axion_width: tuple[float, float] = (8.0, 8.0),
    axion_momentum_amplitude: float = 0.3,
    g_agamma: float = 0.03,
    B0: tuple[float, float, float] = (1.0, 0.5, 0.0),
    E0: tuple[float, float, float] = (0.0, 0.0, 0.0),
    dt_for_cleaning: float = 0.01,
    poisson_solver: str = "periodic_fft",
    poisson_boundary: str = "periodic",
    dirichlet_value: float = 0.0,
    max_iterations: int = 50_000,
    tolerance: float = 1.0e-8,
    omega: float | None = None,
    cleaning_geometry: str = "flat",
    metric=None,
) -> tuple[State, object]:
    """
    2D Gaussian axion profile with uniform in-plane magnetic field, followed
    by electric constraint cleaning.

    The initial E field is projected so that:

        div E - rho + g B^i partial_i a ≈ 0.

    Supported cleaning methods:
        poisson_solver = "periodic_fft", "jacobi", "sor"
        poisson_boundary = "periodic", "dirichlet", "neumann", "outflow"

    Returns
    -------
    state:
        Cleaned State.
    report:
        CleaningReport with before/after constraint norms.
    """
    if grid.ndim != 2:
        raise NotImplementedError("This setup requires a 2D grid.")

    X, Y = grid.coordinates_2d()

    a = gaussian_2d(
        X,
        Y,
        amplitude=axion_amplitude,
        center=axion_center,
        width=axion_width,
    )

    Pi = gaussian_2d(
        X,
        Y,
        amplitude=axion_momentum_amplitude,
        center=axion_center,
        width=axion_width,
    )

    E = grid.zeros_vector()
    B = grid.zeros_vector()

    for i in range(3):
        E[i] = E0[i]
        B[i] = B0[i]

    raw_state = State(a=a, Pi=Pi, E=E, B=B)

    physical = PhysicalParameters(
        m_axion=0.0,
        g_agamma=g_agamma,
    )

    numerics = NumericalParameters(
        dt=dt_for_cleaning,
        t_final=dt_for_cleaning,
        output_every=1,
        derivative_order=2,
    )

    metric = FlatMetric()
    geom = metric.evaluate(0.0, grid)

    if cleaning_geometry == "flat":
        metric_for_cleaning = FlatMetric()
        geom = metric_for_cleaning.evaluate(0.0, grid)

        cleaned_state, report = clean_electric_constraint_flat(
            state=raw_state,
            t=0.0,
            grid=grid,
            geom=geom,
            sources=VacuumSources(),
            numerics=numerics,
            physical=physical,
            include_axion_coupling=True,
            poisson_solver=poisson_solver,
            poisson_boundary=poisson_boundary,
            dirichlet_value=dirichlet_value,
            max_iterations=max_iterations,
            tolerance=tolerance,
            omega=omega,
        )

    elif cleaning_geometry == "curved":
        if metric is None:
            raise ValueError(
                "cleaning_geometry='curved' requires a metric object."
            )

        geom = metric.evaluate(0.0, grid)

        cleaned_state, report = clean_electric_constraint_curved(
            state=raw_state,
            t=0.0,
            grid=grid,
            geom=geom,
            sources=VacuumSources(),
            numerics=numerics,
            physical=physical,
            include_axion_coupling=True,
            poisson_method=poisson_solver,
            poisson_boundary=poisson_boundary,
            dirichlet_value=dirichlet_value,
            max_iterations=max_iterations,
            tolerance=tolerance,
            omega=omega,
        )

    else:
        raise ValueError(
            "cleaning_geometry must be either 'flat' or 'curved'."
        )

    return cleaned_state, report

def gaussian_axion_uniform_Bxy_constraint_solved_2d(
    grid: Grid,
    axion_amplitude: float = 1.0,
    axion_center: tuple[float, float] = (0.0, 0.0),
    axion_width: tuple[float, float] = (8.0, 8.0),
    axion_momentum_amplitude: float = 0.3,
    g_agamma: float = 0.03,
    B0: tuple[float, float, float] = (1.0, 0.5, 0.0),
    E0: tuple[float, float, float] = (0.0, 0.0, 0.0),
    dt_for_cleaning: float = 0.01,
    poisson_solver: str = "periodic_fft",
    poisson_boundary: str = "periodic",
    dirichlet_value: float = 0.0,
    max_iterations: int = 50_000,
    tolerance: float = 1.0e-8,
    omega: float | None = None,
    cleaning_geometry: str = "flat",
    metric=None,
) -> State:
    """
    YAML-friendly wrapper returning only the cleaned State.
    """
    state, report = gaussian_axion_uniform_Bxy_constraint_cleaned_2d(
        grid=grid,
        axion_amplitude=axion_amplitude,
        axion_center=axion_center,
        axion_width=axion_width,
        axion_momentum_amplitude=axion_momentum_amplitude,
        g_agamma=g_agamma,
        B0=B0,
        E0=E0,
        dt_for_cleaning=dt_for_cleaning,
        poisson_solver=poisson_solver,
        poisson_boundary=poisson_boundary,
        dirichlet_value=dirichlet_value,
        max_iterations=max_iterations,
        tolerance=tolerance,
        omega=omega,
        cleaning_geometry=cleaning_geometry,
        metric=metric,
    )

    print("\nInitial electric constraint cleaning report")
    print(f"method      = {report.method}")
    print(f"mean before = {report.mean_constraint_before:.6e}")
    print(f"L2 before   = {report.l2_constraint_before:.6e}")
    print(f"Linf before = {report.linf_constraint_before:.6e}")
    print(f"mean after  = {report.mean_constraint_after:.6e}")
    print(f"L2 after    = {report.l2_constraint_after:.6e}")
    print(f"Linf after  = {report.linf_constraint_after:.6e}")
    print(f"zero mode removed = {report.poisson_zero_mode_removed:.6e}")
    print(f"poisson iterations = {report.poisson_iterations}")
    print(f"poisson residual Linf = {report.poisson_residual_linf:.6e}")
    print(f"poisson converged = {report.poisson_converged}")
    print(f"method      = {report.method}")
    print(f"poisson iterations = {report.poisson_iterations}")
    print(f"poisson residual Linf = {report.poisson_residual_linf:.6e}")
    print(f"poisson converged = {report.poisson_converged}")

    return state