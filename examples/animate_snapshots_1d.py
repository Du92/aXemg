"""
Generic 1D animation example.

This reruns a simple 1D simulation and generates animations for:
    - a
    - Pi
    - |E|
    - |B|
    - E.B

Run with:
    python examples/animate_snapshots_1d.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.boundary import PeriodicBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.initial_data.combined_setups import gaussian_axion_em_1d
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.animations import animate_default_1d_set
from axion_em_gr.visualization.plots_2d import ensure_output_dir


def main() -> None:
    output_dir = ensure_output_dir(Path("outputs") / "animations_1d_example")

    grid = Grid(
        ndim=1,
        shape=(512,),
        bounds=((-50.0, 50.0),),
        nghost=3,
    )

    physical = PhysicalParameters(
        m_axion=0.25,
        g_agamma=0.04,
    )

    numerics = NumericalParameters(
        dt=0.01,
        t_final=8.0,
        output_every=20,
        derivative_order=2,
    )

    metric = FlatMetric()

    state0 = gaussian_axion_em_1d(
        grid=grid,
        axion_amplitude=1.0,
        axion_center=0.0,
        axion_width=8.0,
        axion_momentum_amplitude=0.15,
        ex_amplitude=0.2,
        ey_amplitude=0.0,
        bz_background=1.0,
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
        boundary=PeriodicBoundary(),
        numerics=numerics,
        save_snapshots=True,
        snapshot_every=20,
    )

    final_state, history = solver.evolve(state0)

    anim_dir = ensure_output_dir(output_dir / "animations")

    for path in animate_default_1d_set(
        history=history,
        grid=grid,
        output_dir=anim_dir,
        metric=metric,
        fps=18,
    ):
        print(path)


if __name__ == "__main__":
    main()