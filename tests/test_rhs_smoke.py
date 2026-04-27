"""
Smoke tests for RHS computation.

These tests do not verify full physical correctness. They verify that the RHS
can be evaluated without shape errors or missing method errors.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.core.state import State
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.geometry.schwarzschild_like import SmoothCompactObjectMetric2D
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.physics.sources import VacuumSources


def _make_state_2d(grid):
    X, Y = grid.coordinates_2d()

    a = np.exp(-0.5 * (X**2 + Y**2) / 10.0**2)
    Pi = 0.1 * a

    E = grid.zeros_vector()
    B = grid.zeros_vector()

    E[1] = 0.1 * a
    B[2] = 1.0

    return State(a=a, Pi=Pi, E=E, B=B)


def test_rhs_axion_flat_2d_smoke(grid_2d_compact, numerics, physical):
    grid = grid_2d_compact
    state = _make_state_2d(grid)

    rhs = RHSComputer(
        grid=grid,
        metric=FlatMetric(),
        potential=MassivePotential(m=physical.m_axion),
        numerics=numerics,
        physical=physical,
        sources=VacuumSources(),
        evolve_axion=True,
        evolve_maxwell=False,
        include_axion_em_coupling=True,
    )

    out = rhs(state, 0.0)

    assert out.a.shape == state.a.shape
    assert out.Pi.shape == state.Pi.shape
    assert np.all(np.isfinite(out.a))
    assert np.all(np.isfinite(out.Pi))


def test_rhs_axion_curved_2d_smoke(grid_2d_compact, numerics, physical):
    grid = grid_2d_compact
    state = _make_state_2d(grid)

    metric = SmoothCompactObjectMetric2D(
        conformal_amplitude=4.0,
        compactness=0.25,
        radius=12.0,
        center=(0.0, 0.0),
        plane="xy",
        plane_offset=2.0,
        lapse_floor=0.2,
    )

    rhs = RHSComputer(
        grid=grid,
        metric=metric,
        potential=MassivePotential(m=physical.m_axion),
        numerics=numerics,
        physical=physical,
        sources=VacuumSources(),
        evolve_axion=True,
        evolve_maxwell=True,
        include_axion_em_coupling=True,
    )

    out = rhs(state, 0.0)

    assert out.a.shape == state.a.shape
    assert out.Pi.shape == state.Pi.shape
    assert out.E.shape == state.E.shape
    assert out.B.shape == state.B.shape

    assert np.all(np.isfinite(out.a))
    assert np.all(np.isfinite(out.Pi))
    assert np.all(np.isfinite(out.E))
    assert np.all(np.isfinite(out.B))