"""
Phase 19 example:

2D axion evolution with covariant spatial operators on a smooth compact-object
metric.

This is the first 2D case where the axion RHS uses

    D_i D^i a

instead of the flat Laplacian.

Run with:

    python examples/run_curved_2d_axion_compact_metric.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.boundary import MixedBoundary, OutflowBoundary, SommerfeldBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.geometry.schwarzschild_like import SmoothCompactObjectMetric2D
from axion_em_gr.initial_data.ns_scenarios_2d import (
    axion_cloud_around_compact_object_2d,
)
from axion_em_gr.physics.comparison import difference_norms
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.curved_diagnostics_2d import (
    make_curved_2d_diagnostic_report,
)
from axion_em_gr.visualization.geometry_plots import plot_geometry_2d
from axion_em_gr.visualization.plots_1d import plot_history
from axion_em_gr.visualization.plots_2d import (
    ensure_output_dir,
    plot_axion_state_2d,
)


def run_case(label, metric, grid, state0, physical, numerics, output_dir):
    sources = VacuumSources()
    potential = MassivePotential(m=physical.m_axion)

    boundary = MixedBoundary(
        default=OutflowBoundary(),
        field_boundaries={
            "a": SommerfeldBoundary(asymptotic_value=0.0, center=(0.0, 0.0)),
            "Pi": SommerfeldBoundary(asymptotic_value=0.0, center=(0.0, 0.0)),
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
        include_axion_em_coupling=False,
    )

    solver = EvolutionSolver(
        grid=grid,
        rhs_computer=rhs,
        integrator=RK4(),
        boundary=boundary,
        numerics=numerics,
    )

    case_dir = ensure_output_dir(output_dir / label)

    geom0 = metric.evaluate(0.0, grid)

    for path in plot_geometry_2d(
        grid=grid,
        geom=geom0,
        output_dir=case_dir,
        prefix="geometry_initial",
    ):
        print(path)

    for path in plot_axion_state_2d(
        grid=grid,
        state=state0,
        output_dir=case_dir,
        prefix="initial_axion",
    ):
        print(path)

    final_state, history = solver.evolve(state0)

    geom_final = metric.evaluate(numerics.t_final, grid)

    for path in plot_axion_state_2d(
        grid=grid,
        state=final_state,
        output_dir=case_dir,
        prefix="final_axion",
    ):
        print(path)

    print(
        plot_history(
            history,
            output_dir=case_dir,
            filename="history.png",
        )
    )

    report_dir = ensure_output_dir(case_dir / "curved_diagnostics")

    # This report expects E/B if present, but it also works with E/B zero arrays.
    state_for_report = final_state

    for path in make_curved_2d_diagnostic_report(
        grid=grid,
        state=state_for_report,
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

    return final_state, history


def main() -> None:
    output_dir = ensure_output_dir(Path("outputs") / "phase19_curved_2d_axion")

    grid = Grid(
        ndim=2,
        shape=(192, 192),
        bounds=((-60.0, 60.0), (-60.0, 60.0)),
        nghost=3,
    )

    physical = PhysicalParameters(
        m_axion=0.15,
        g_agamma=0.0,
    )

    numerics = NumericalParameters(
        dt=0.005,
        t_final=15.0,
        output_every=100,
        derivative_order=2,
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
        Pi_amplitude=0.1,
    )

    flat_metric = FlatMetric()

    compact_metric = SmoothCompactObjectMetric2D(
        conformal_amplitude=4.0,
        compactness=0.25,
        radius=12.0,
        center=(0.0, 0.0),
        plane="xy",
        plane_offset=2.0,
        lapse_floor=0.2,
    )

    print("\nRunning flat reference")
    final_flat, hist_flat = run_case(
        label="flat",
        metric=flat_metric,
        grid=grid,
        state0=state0,
        physical=physical,
        numerics=numerics,
        output_dir=output_dir,
    )

    print("\nRunning compact metric")
    final_curved, hist_curved = run_case(
        label="compact_metric",
        metric=compact_metric,
        grid=grid,
        state0=state0,
        physical=physical,
        numerics=numerics,
        output_dir=output_dir,
    )

    geom_flat = flat_metric.evaluate(numerics.t_final, grid)
    geom_curved = compact_metric.evaluate(numerics.t_final, grid)

    norms = difference_norms(
        state_reference=final_flat,
        state_test=final_curved,
        geom_reference=geom_flat,
        geom_test=geom_curved,
        grid=grid,
        time=numerics.t_final,
    )

    print("\nCompact-metric minus flat difference summary")
    print(f"t = {norms.time:.6f}")
    print(f"max Δa        = {norms.max_delta_a:.8e}")
    print(f"L2  Δa        = {norms.l2_delta_a:.8e}")
    print(f"rel L2 Δa     = {norms.rel_l2_delta_a:.8e}")


if __name__ == "__main__":
    main()