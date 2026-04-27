"""
Phase 12C example:

2D constraint-cleaned axion-electromagnetic initial data.

The electric field is projected with:

    E^i_new = E^i - partial^i phi

where:

    Laplacian(phi) = C_E.

Run with:

    python examples/run_constraint_cleaned_2d.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.boundary import PeriodicBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.initial_data.combined_setups_2d import (
    gaussian_axion_uniform_Bxy_constraint_cleaned_2d,
)
from axion_em_gr.physics.constraints import (
    constraint_norms,
    electric_constraint,
    magnetic_constraint,
)
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.diagnostics_2d import make_full_2d_diagnostic_report
from axion_em_gr.visualization.plots_1d import (
    plot_coupling_history,
    plot_history,
)
from axion_em_gr.visualization.plots_2d import (
    ensure_output_dir,
    plot_EdotB_2d,
    plot_axion_state_2d,
    plot_constraint_heatmaps_2d,
    plot_em_summary_2d,
)


def main() -> None:
    output_dir = ensure_output_dir(Path("outputs") / "phase12C_constraint_cleaned_2d")

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

    state0, cleaning_report = gaussian_axion_uniform_Bxy_constraint_cleaned_2d(
        grid=grid,
        axion_amplitude=1.0,
        axion_center=(0.0, 0.0),
        axion_width=(8.0, 8.0),
        axion_momentum_amplitude=0.3,
        g_agamma=physical.g_agamma,
        B0=(1.0, 0.5, 0.0),
        E0=(0.0, 0.0, 0.0),
        dt_for_cleaning=numerics.dt,
        poisson_solver="jacobi",
        poisson_boundary="dirichlet",
        dirichlet_value=0.0,
        max_iterations=20_000,
        tolerance=1.0e-7,
        omega=2.0 / 3.0,
    )

    print("\nCleaning report from initial-data construction")
    print(f"mean before = {cleaning_report.mean_constraint_before:.8e}")
    print(f"L2 before   = {cleaning_report.l2_constraint_before:.8e}")
    print(f"Linf before = {cleaning_report.linf_constraint_before:.8e}")
    print(f"mean after  = {cleaning_report.mean_constraint_after:.8e}")
    print(f"L2 after    = {cleaning_report.l2_constraint_after:.8e}")
    print(f"Linf after  = {cleaning_report.linf_constraint_after:.8e}")
    print(f"zero mode removed = {cleaning_report.poisson_zero_mode_removed:.8e}")
    print(f"method = {cleaning_report.method}")
    print(f"poisson iterations = {cleaning_report.poisson_iterations}")
    print(f"poisson residual Linf = {cleaning_report.poisson_residual_linf:.8e}")
    print(f"poisson converged = {cleaning_report.poisson_converged}")

    metric = FlatMetric()
    sources = VacuumSources()
    potential = MassivePotential(m=physical.m_axion)

    geom0 = metric.evaluate(0.0, grid)

    C_E0 = electric_constraint(
        state=state0,
        t=0.0,
        grid=grid,
        geom=geom0,
        sources=sources,
        numerics=numerics,
        physical=physical,
        include_axion_coupling=True,
    )

    C_B0 = magnetic_constraint(
        state=state0,
        grid=grid,
        geom=geom0,
        numerics=numerics,
    )

    l2_E0, linf_E0 = constraint_norms(C_E0, grid)
    l2_B0, linf_B0 = constraint_norms(C_B0, grid)

    print("\nInitial constraints after cleaning")
    print(f"L2 CE   = {l2_E0:.8e}")
    print(f"Linf CE = {linf_E0:.8e}")
    print(f"L2 CB   = {l2_B0:.8e}")
    print(f"Linf CB = {linf_B0:.8e}")

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

    for path in plot_constraint_heatmaps_2d(
        grid=grid,
        div_B=C_B0,
        div_E=C_E0,
        output_dir=output_dir,
        prefix="initial_constraints",
    ):
        print(path)

    final_state, history = solver.evolve(state0)

    geom_final = metric.evaluate(numerics.t_final, grid)

    C_Ef = electric_constraint(
        state=final_state,
        t=numerics.t_final,
        grid=grid,
        geom=geom_final,
        sources=sources,
        numerics=numerics,
        physical=physical,
        include_axion_coupling=True,
    )

    C_Bf = magnetic_constraint(
        state=final_state,
        grid=grid,
        geom=geom_final,
        numerics=numerics,
    )

    l2_Ef, linf_Ef = constraint_norms(C_Ef, grid)
    l2_Bf, linf_Bf = constraint_norms(C_Bf, grid)

    print("\nFinal constraints")
    print(f"L2 CE   = {l2_Ef:.8e}")
    print(f"Linf CE = {linf_Ef:.8e}")
    print(f"L2 CB   = {l2_Bf:.8e}")
    print(f"Linf CB = {linf_Bf:.8e}")

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

    for path in plot_constraint_heatmaps_2d(
        grid=grid,
        div_B=C_Bf,
        div_E=C_Ef,
        output_dir=output_dir,
        prefix="final_constraints",
    ):
        print(path)

    report_dir = ensure_output_dir(output_dir / "final_diagnostics")

    for path in make_full_2d_diagnostic_report(
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
        center=(0.0, 0.0),
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