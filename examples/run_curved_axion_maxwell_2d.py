"""
Phase 20A example:

Coupled axion-Maxwell evolution on a 2D compact-object metric.

This is the first example where:
    - the axion sector uses D_i D^i a,
    - Maxwell uses curved curl operators,
    - Gauss constraints use covariant divergence,
    - E_i B^i uses the spatial metric.

Run with:

    python examples/run_curved_axion_maxwell_2d.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.boundary import MixedBoundary, OutflowBoundary, SommerfeldBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.schwarzschild_like import SmoothCompactObjectMetric2D
from axion_em_gr.initial_data.combined_setups_2d import gaussian_axion_em_ring_2d
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.curved_diagnostics_2d import (
    make_curved_2d_diagnostic_report,
)
from axion_em_gr.visualization.geometry_plots import plot_geometry_2d
from axion_em_gr.visualization.plots_1d import plot_coupling_history, plot_history
from axion_em_gr.visualization.plots_2d import (
    ensure_output_dir,
    plot_EdotB_2d,
    plot_axion_state_2d,
    plot_em_summary_2d,
)


def main() -> None:
    output_dir = ensure_output_dir(Path("outputs") / "phase20A_curved_axion_maxwell_2d")

    grid = Grid(
        ndim=2,
        shape=(192, 192),
        bounds=((-60.0, 60.0), (-60.0, 60.0)),
        nghost=3,
    )

    physical = PhysicalParameters(
        m_axion=0.15,
        g_agamma=0.03,
    )

    numerics = NumericalParameters(
        dt=0.002,
        t_final=5.0,
        output_every=100,
        derivative_order=2,
    )

    metric = SmoothCompactObjectMetric2D(
        conformal_amplitude=4.0,
        compactness=0.25,
        radius=12.0,
        center=(0.0, 0.0),
        plane="xy",
        plane_offset=2.0,
        lapse_floor=0.2,
    )

    state0 = gaussian_axion_em_ring_2d(
        grid=grid,
        axion_amplitude=1.0,
        axion_center=(0.0, 0.0),
        axion_width=(10.0, 10.0),
        axion_momentum_amplitude=0.2,
        em_amplitude=0.15,
        em_center=(0.0, 0.0),
        em_width=(14.0, 14.0),
        background_Bz=1.0,
    )

    boundary = MixedBoundary(
        default=OutflowBoundary(),
        field_boundaries={
            "a": SommerfeldBoundary(asymptotic_value=0.0, center=(0.0, 0.0)),
            "Pi": SommerfeldBoundary(asymptotic_value=0.0, center=(0.0, 0.0)),
            "E": OutflowBoundary(),
            "B": OutflowBoundary(),
        },
    )

    potential = MassivePotential(m=physical.m_axion)
    sources = VacuumSources()

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
        boundary=boundary,
        numerics=numerics,
        save_snapshots=True,
        snapshot_every=100,
    )

    geom0 = metric.evaluate(0.0, grid)

    for path in plot_geometry_2d(
        grid=grid,
        geom=geom0,
        output_dir=output_dir,
        prefix="geometry",
    ):
        print(path)

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

    report_dir = ensure_output_dir(output_dir / "curved_diagnostics")

    for path in make_curved_2d_diagnostic_report(
        grid=grid,
        state=final_state,
        geom=geom_final,
        potential=potential,
        sources=sources,
        numerics=numerics,
        physical=physical,
        time=numerics.t_final,
        output_dir=report_dir,
        prefix="final",
    ):
        print(path)

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


if __name__ == "__main__":
    main()