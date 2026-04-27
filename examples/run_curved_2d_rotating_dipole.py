"""
Phase 19 example:

Axion cloud with prescribed rotating dipole on a 2D compact-object metric,
using curved 2D axion operators.

Run with:

    python examples/run_curved_2d_rotating_dipole.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.boundary import MixedBoundary, OutflowBoundary, SommerfeldBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.schwarzschild_like import SmoothCompactObjectMetric2D
from axion_em_gr.initial_data.ns_scenarios_2d import (
    axion_cloud_around_compact_object_2d,
)
from axion_em_gr.physics.background_em import RotatingDipoleBackground2D
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.background_em_plots import (
    plot_background_B_magnitude_2d,
    plot_background_EdotB_2d,
)
from axion_em_gr.visualization.curved_diagnostics_2d import (
    make_curved_2d_diagnostic_report,
)
from axion_em_gr.visualization.geometry_plots import plot_geometry_2d
from axion_em_gr.visualization.plots_1d import plot_coupling_history, plot_history
from axion_em_gr.visualization.plots_2d import (
    ensure_output_dir,
    plot_EdotB_2d,
    plot_axion_state_2d,
)


def main() -> None:
    output_dir = ensure_output_dir(
        Path("outputs") / "phase19_curved_2d_rotating_dipole"
    )

    grid = Grid(
        ndim=2,
        shape=(192, 192),
        bounds=((-60.0, 60.0), (-60.0, 60.0)),
        nghost=3,
    )

    physical = PhysicalParameters(
        m_axion=0.15,
        g_agamma=0.05,
    )

    numerics = NumericalParameters(
        dt=0.005,
        t_final=15.0,
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

    background = RotatingDipoleBackground2D(
        mu0=1.0,
        omega=0.3,
        inclination=0.7,
        phase0=0.0,
        center=(0.0, 0.0),
        plane="xy",
        plane_offset=2.0,
        B_scale=80.0,
        softening_radius=3.0,
        star_radius=6.0,
        include_induced_E=True,
        electric_scale=1.0,
        light_cylinder_limit=True,
        max_velocity=0.8,
        parallel_E_fraction=0.02,
        parallel_E_profile_radius=25.0,
    )

    state0 = axion_cloud_around_compact_object_2d(
        grid=grid,
        axion_amplitude=1.0,
        cloud_radius=18.0,
        cloud_width=8.0,
        center=(0.0, 0.0),
        axion_background=0.0,
        angular_modulation=0.15,
        azimuthal_mode=2,
        Pi_amplitude=0.0,
    )

    sources = VacuumSources()
    potential = MassivePotential(m=physical.m_axion)

    boundary = MixedBoundary(
        default=OutflowBoundary(),
        field_boundaries={
            "a": SommerfeldBoundary(asymptotic_value=0.0, center=(0.0, 0.0)),
            "Pi": SommerfeldBoundary(asymptotic_value=0.0, center=(0.0, 0.0)),
            "E": OutflowBoundary(),
            "B": OutflowBoundary(),
        },
    )

    rhs = RHSComputer(
        grid=grid,
        metric=metric,
        potential=potential,
        numerics=numerics,
        physical=physical,
        sources=sources,
        evolve_axion=True,
        evolve_maxwell=False,
        include_axion_em_coupling=True,
        background_em=background,
        background_em_mode="replace",
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
    state0_with_bg = rhs.state_with_background(state0, 0.0)

    for path in plot_geometry_2d(
        grid=grid,
        geom=geom0,
        output_dir=output_dir,
        prefix="initial_geometry",
    ):
        print(path)

    print(
        plot_background_B_magnitude_2d(
            grid=grid,
            background=background,
            t=0.0,
            output_dir=output_dir,
            filename="background_B_magnitude_t0.png",
        )
    )

    print(
        plot_background_EdotB_2d(
            grid=grid,
            background=background,
            t=0.0,
            output_dir=output_dir,
            filename="background_flat_EdotB_t0.png",
        )
    )

    for path in plot_axion_state_2d(
        grid=grid,
        state=state0,
        output_dir=output_dir,
        prefix="initial_axion",
    ):
        print(path)

    print(
        plot_EdotB_2d(
            grid=grid,
            state=state0_with_bg,
            geom=geom0,
            output_dir=output_dir,
            filename="initial_curved_EdotB.png",
        )
    )

    final_state, history = solver.evolve(state0)

    geom_final = metric.evaluate(numerics.t_final, grid)
    final_state_with_bg = rhs.state_with_background(final_state, numerics.t_final)

    for path in plot_axion_state_2d(
        grid=grid,
        state=final_state,
        output_dir=output_dir,
        prefix="final_axion",
    ):
        print(path)

    print(
        plot_EdotB_2d(
            grid=grid,
            state=final_state_with_bg,
            geom=geom_final,
            output_dir=output_dir,
            filename="final_curved_EdotB.png",
        )
    )

    report_dir = ensure_output_dir(output_dir / "curved_diagnostics")

    for path in make_curved_2d_diagnostic_report(
        grid=grid,
        state=final_state_with_bg,
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