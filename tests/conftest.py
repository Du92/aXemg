"""
Shared pytest fixtures for axion_em_gr tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from axion_em_gr.core.grid import Grid
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.geometry.schwarzschild_like import SmoothCompactObjectMetric2D
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters


@pytest.fixture
def grid_1d_periodic():
    return Grid(
        ndim=1,
        shape=(256,),
        bounds=((0.0, 2.0 * np.pi),),
        nghost=3,
    )


@pytest.fixture
def grid_2d_periodic():
    return Grid(
        ndim=2,
        shape=(96, 96),
        bounds=((0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi)),
        nghost=3,
    )


@pytest.fixture
def grid_2d_compact():
    return Grid(
        ndim=2,
        shape=(96, 96),
        bounds=((-40.0, 40.0), (-40.0, 40.0)),
        nghost=3,
    )


@pytest.fixture
def numerics():
    return NumericalParameters(
        dt=0.001,
        t_final=0.01,
        output_every=1,
        derivative_order=2,
    )


@pytest.fixture
def physical():
    return PhysicalParameters(
        m_axion=0.2,
        g_agamma=0.03,
    )


@pytest.fixture
def flat_metric():
    return FlatMetric()


@pytest.fixture
def compact_metric_2d():
    return SmoothCompactObjectMetric2D(
        conformal_amplitude=3.0,
        compactness=0.2,
        radius=10.0,
        center=(0.0, 0.0),
        plane="xy",
        plane_offset=2.0,
        lapse_floor=0.2,
    )