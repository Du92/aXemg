"""
Phase 7 example:

2D flat Cartesian axion-electromagnetic evolution.

Fields depend on (x, y), but E^i and B^i still have three components.

Run with:

    python examples/run_flat_axion_em_2d.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.boundary import PeriodicBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.initial_data.combined_setups_2d import (
    gaussian_axion_em_ring_2d,
)
from axion_em_gr.physics.constraints import (
    electric_constraint,
    magnetic_constraint,
)
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.plots_1d import plot_coupling_history, plot_history
from axion_em_gr.visualization.plots_2d import (
    ensure_output_dir,
    plot_EdotB_2d,
    plot_axion_state_2d,
    plot_constraint_heatmaps_2d,
    plot_em_summary_2d,
)


def main() -> None:
    output_dir = ensure_output_dir(Path("outputs") / "phase7_flat_axion_em_2d")

    grid = Grid(
        ndim=2,
        shape=(256, 256),
        bounds=((-50.0, 50.0), (-50.0, 50.0)),
        nghost=3,
    )

    physical = PhysicalParameters(
        m_axion=0.2,
        g_agamma=0.03,
    )

    numerics = NumericalParameters(
        dt=0.01,
        t_final=10.0,
        output_every=100,
        derivative_order=2,
    )

    state0 = gaussian_axion_em_ring_2d(
        grid=grid,
        axion_amplitude=1.0,
        axion_center=(0.0, 0.0),
        axion_width=(8.0, 8.0),
        axion_momentum_amplitude=0.3,
        em_amplitude=0.2,
        em_center=(0.0, 0.0),
        em_width=(12.0, 12.0),
        background_Bz=1.0,
    )

    metric = FlatMetric()
    sources = VacuumSources()
    potential = MassivePotential(m=physical.m_axion)

    rhs = RHSComputer(
        grid=grid,
        metric=metric,
        potential=potential,
        numerics=numerics,
        physical=physical,
        sources=sources,
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
    )

    geom0 = metric.evaluate(0.0, grid)

    for path in plot_axion_state_2d(
        grid=grid,
        state=state0,
        output_dir=output_dir,
        prefix="initial_axion",
    ):
        print(path)

    for path in plot_em_summary_2d(
        grid=grid,
        state=state0,
        output_dir=output_dir,
        prefix="initial_em",
    ):
        print(path)

    print(
        plot_EdotB_2d(
            grid=grid,
            state=state0,
            geom=geom0,
            output_dir=output_dir,
            filename="initial_EdotB.png",
        )
    )

    final_state, history = solver.evolve(state0)

    geom_final = metric.evaluate(numerics.t_final, grid)

    for path in plot_axion_state_2d(
        grid=grid,
        state=final_state,
        output_dir=output_dir,
        prefix="final_axion",
    ):
        print(path)

    for path in plot_em_summary_2d(
        grid=grid,
        state=final_state,
        output_dir=output_dir,
        prefix="final_em",
    ):
        print(path)

    print(
        plot_EdotB_2d(
            grid=grid,
            state=final_state,
            geom=geom_final,
            output_dir=output_dir,
            filename="final_EdotB.png",
        )
    )

    print(
        plot_history(
            history,
            output_dir=output_dir,
            filename="history.png",
        )
    )

    print(
        plot_coupling_history(
            history,
            output_dir=output_dir,
            filename="coupling_history.png",
        )
    )

    div_B = magnetic_constraint(
        state=final_state,
        grid=grid,
        geom=geom_final,
        numerics=numerics,
    )

    div_E = electric_constraint(
        state=final_state,
        t=numerics.t_final,
        grid=grid,
        geom=geom_final,
        sources=sources,
        numerics=numerics,
        physical=physical,
        include_axion_coupling=True,
    )

    for path in plot_constraint_heatmaps_2d(
        grid=grid,
        div_B=div_B,
        div_E=div_E,
        output_dir=output_dir,
        prefix="final_constraints",
    ):
        print(path)


if __name__ == "__main__":
    main()