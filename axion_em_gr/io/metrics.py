"""
Simulation metrics.

These routines collect scalar diagnostics at the end of a run.
They are useful for parameter sweeps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.core.state import State
from axion_em_gr.core.tensors import contract_cov_contra, lower_vector
from axion_em_gr.geometry.base_metric import GeometryFields
from axion_em_gr.physics.constraints import (
    constraint_norms,
    electric_constraint,
    magnetic_constraint,
)
from axion_em_gr.physics.diagnostics import (
    axion_energy_flat,
    edotb_density,
    electromagnetic_energy_flat,
)
from axion_em_gr.physics.potentials import AxionPotential
from axion_em_gr.physics.sources import SourceModel
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.physics.diagnostics import axion_energy_geometry


@dataclass
class FinalMetrics:
    """
    Scalar summary of one completed simulation.
    """

    run_id: str
    status: str

    t_final: float

    max_abs_a: float
    l2_a: float
    max_abs_Pi: float

    max_abs_E: float
    max_abs_B: float
    max_abs_EdotB: float
    l2_EdotB: float

    axion_energy: float
    em_energy: float
    total_energy: float

    l2_div_B: float
    linf_div_B: float
    l2_div_E: float
    linf_div_E: float


def _volume_element(grid: Grid) -> float:
    if grid.ndim == 1:
        return grid.dx[0]
    if grid.ndim == 2:
        return grid.dx[0] * grid.dx[1]

    raise NotImplementedError("Metrics currently support 1D and 2D.")


def _l2_scalar(field: np.ndarray, grid: Grid) -> float:
    interior = grid.interior_slices
    return float(np.sqrt(np.sum(field[interior] ** 2) * _volume_element(grid)))


def _max_abs_scalar(field: np.ndarray, grid: Grid) -> float:
    interior = grid.interior_slices
    return float(np.max(np.abs(field[interior])))


def _max_abs_vector(vector: np.ndarray, grid: Grid) -> float:
    interior = grid.interior_slices
    data = vector[(slice(None), *interior)]
    return float(np.max(np.sqrt(np.sum(data**2, axis=0))))


def compute_final_metrics(
    run_id: str,
    status: str,
    state: State,
    grid: Grid,
    geom: GeometryFields,
    potential: AxionPotential,
    sources: SourceModel,
    numerics: NumericalParameters,
    physical: PhysicalParameters,
    include_axion_coupling: bool,
) -> FinalMetrics:
    """
    Compute final scalar metrics for one simulation.
    """
    max_abs_a = _max_abs_scalar(state.a, grid)
    l2_a = _l2_scalar(state.a, grid)
    max_abs_Pi = _max_abs_scalar(state.Pi, grid)

    if state.E is not None and state.B is not None:
        max_abs_E = _max_abs_vector(state.E, grid)
        max_abs_B = _max_abs_vector(state.B, grid)

        EdotB = edotb_density(state, geom)
        max_abs_EdotB = _max_abs_scalar(EdotB, grid)
        l2_EdotB = _l2_scalar(EdotB, grid)

        axion_energy = axion_energy_geometry(
            state=state,
            grid=grid,
            geom=geom,
            potential=potential,
        )

        em_energy = electromagnetic_energy_flat(
            state=state,
            grid=grid,
            geom=geom,
        )

        div_B = magnetic_constraint(
            state=state,
            grid=grid,
            geom=geom,
            numerics=numerics,
        )

        div_E = electric_constraint(
            state=state,
            t=numerics.t_final,
            grid=grid,
            geom=geom,
            sources=sources,
            numerics=numerics,
            physical=physical,
            include_axion_coupling=include_axion_coupling,
        )

        l2_div_B, linf_div_B = constraint_norms(div_B, grid)
        l2_div_E, linf_div_E = constraint_norms(div_E, grid)

    else:
        max_abs_E = 0.0
        max_abs_B = 0.0
        max_abs_EdotB = 0.0
        l2_EdotB = 0.0

        axion_energy = axion_energy_flat(
            state=state,
            grid=grid,
            potential=potential,
        )

        em_energy = 0.0

        l2_div_B = 0.0
        linf_div_B = 0.0
        l2_div_E = 0.0
        linf_div_E = 0.0

    total_energy = axion_energy + em_energy

    return FinalMetrics(
        run_id=run_id,
        status=status,
        t_final=numerics.t_final,
        max_abs_a=max_abs_a,
        l2_a=l2_a,
        max_abs_Pi=max_abs_Pi,
        max_abs_E=max_abs_E,
        max_abs_B=max_abs_B,
        max_abs_EdotB=max_abs_EdotB,
        l2_EdotB=l2_EdotB,
        axion_energy=axion_energy,
        em_energy=em_energy,
        total_energy=total_energy,
        l2_div_B=l2_div_B,
        linf_div_B=linf_div_B,
        l2_div_E=l2_div_E,
        linf_div_E=linf_div_E,
    )


def metrics_to_dict(metrics: FinalMetrics) -> dict[str, Any]:
    """
    Convert metrics dataclass to dictionary.
    """
    return asdict(metrics)