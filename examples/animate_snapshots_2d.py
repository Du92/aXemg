"""
Generic 2D animation example.

This reruns a simple 2D flat simulation and generates animations for:
    - a
    - Pi
    - |E|
    - |B|
    - E.B

Run with:
    python examples/animate_snapshots_2d.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.boundary import OutflowBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.initial_data.combined_setups_2d import gaussian_axion_em_ring_2d
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.animations import animate_default_2d_set
from axion_em_gr.visualization.plots_2d import ensure_output_dir


def main() -> None:
    output_dir = ensure_output_dir(Path("outputs") / "animations_2d_example")

    grid = Grid(
        ndim=2,
        shape=(160, 160),
        bounds=((-40.0, 40.0), (-40.0, 40.0)),
        nghost=3,
    )

    physical = PhysicalParameters(
        m_axion=0.15,
        g_agamma=0.03,
    )

    numerics = NumericalParameters(
        dt=0.004,
        t_final=4.0,
        output_every=25,
        derivative_order=2,
    )

    metric = FlatMetric()

    state0 = gaussian_axion_em_ring_2d(
        grid=grid,
        axion_amplitude=1.0,
        axion_center=(0.0, 0.0),
        axion_width=(8.0, 8.0),
        axion_momentum_amplitude=0.15,
        em_amplitude=0.18,
        em_center=(0.0, 0.0),
        em_width=(10.0, 10.0),
        background_Bz=1.0,
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

    solver = EvolutionSolver(
        grid=grid,
        rhs_computer=rhs,
        integrator=RK4(),
        boundary=OutflowBoundary(),
        numerics=numerics,
        save_snapshots=True,
        snapshot_every=25,
    )

    final_state, history = solver.evolve(state0)

    anim_dir = ensure_output_dir(output_dir / "animations")

    for path in animate_default_2d_set(
        history=history,
        grid=grid,
        output_dir=anim_dir,
        metric=metric,
        fps=12,
        overlay_geometry=None,
    ):
        print(path)


if __name__ == "__main__":
    main()