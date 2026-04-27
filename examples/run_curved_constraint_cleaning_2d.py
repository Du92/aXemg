"""
Phase 20B example:

Curved electric constraint cleaning in 2D.

This solves

    D_iD^i phi = C_E

and corrects

    E^i -> E^i - D^i phi.

Run with:

    python examples/run_curved_constraint_cleaning_2d.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.geometry.schwarzschild_like import SmoothCompactObjectMetric2D
from axion_em_gr.initial_data.combined_setups_2d import (
    gaussian_axion_uniform_Bxy_constraint_cleaned_2d,
)
from axion_em_gr.physics.constraints import (
    constraint_norms,
    electric_constraint,
    magnetic_constraint,
)
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.visualization.curved_diagnostics_2d import (
    make_curved_2d_diagnostic_report,
)
from axion_em_gr.visualization.geometry_plots import plot_geometry_2d
from axion_em_gr.visualization.plots_2d import (
    ensure_output_dir,
    plot_constraint_heatmaps_2d,
    plot_em_summary_2d,
)


def main() -> None:
    output_dir = ensure_output_dir(Path("outputs") / "phase20B_curved_cleaning_2d")

    grid = Grid(
        ndim=2,
        shape=(128, 128),
        bounds=((-60.0, 60.0), (-60.0, 60.0)),
        nghost=3,
    )

    physical = PhysicalParameters(
        m_axion=0.15,
        g_agamma=0.03,
    )

    numerics = NumericalParameters(
        dt=0.005,
        t_final=0.005,
        output_every=1,
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

    geom = metric.evaluate(0.0, grid)

    for path in plot_geometry_2d(
        grid=grid,
        geom=geom,
        output_dir=output_dir,
        prefix="geometry",
    ):
        print(path)

    state, report = gaussian_axion_uniform_Bxy_constraint_cleaned_2d(
        grid=grid,
        axion_amplitude=1.0,
        axion_center=(0.0, 0.0),
        axion_width=(10.0, 10.0),
        axion_momentum_amplitude=0.2,
        g_agamma=physical.g_agamma,
        B0=(1.0, 0.5, 0.0),
        E0=(0.0, 0.0, 0.0),
        dt_for_cleaning=numerics.dt,
        poisson_solver="jacobi",
        poisson_boundary="dirichlet",
        dirichlet_value=0.0,
        max_iterations=30_000,
        tolerance=1.0e-7,
        omega=2.0 / 3.0,
        cleaning_geometry="curved",
        metric=metric,
    )

    print("\nCurved cleaning report")
    print(f"method               = {report.method}")
    print(f"mean before          = {report.mean_constraint_before:.8e}")
    print(f"L2 before            = {report.l2_constraint_before:.8e}")
    print(f"Linf before          = {report.linf_constraint_before:.8e}")
    print(f"mean after           = {report.mean_constraint_after:.8e}")
    print(f"L2 after             = {report.l2_constraint_after:.8e}")
    print(f"Linf after           = {report.linf_constraint_after:.8e}")
    print(f"removed mean         = {report.poisson_zero_mode_removed:.8e}")
    print(f"poisson iterations   = {report.poisson_iterations}")
    print(f"poisson residual     = {report.poisson_residual_linf:.8e}")
    print(f"poisson converged    = {report.poisson_converged}")

    C_E = electric_constraint(
        state=state,
        t=0.0,
        grid=grid,
        geom=geom,
        sources=VacuumSources(),
        numerics=numerics,
        physical=physical,
        include_axion_coupling=True,
    )

    C_B = magnetic_constraint(
        state=state,
        grid=grid,
        geom=geom,
        numerics=numerics,
    )

    l2_E, linf_E = constraint_norms(C_E, grid)
    l2_B, linf_B = constraint_norms(C_B, grid)

    print("\nConstraint check after cleaning")
    print(f"L2 CE   = {l2_E:.8e}")
    print(f"Linf CE = {linf_E:.8e}")
    print(f"L2 CB   = {l2_B:.8e}")
    print(f"Linf CB = {linf_B:.8e}")

    for path in plot_em_summary_2d(
        grid=grid,
        state=state,
        output_dir=output_dir,
        prefix="cleaned_em",
    ):
        print(path)

    for path in plot_constraint_heatmaps_2d(
        grid=grid,
        div_B=C_B,
        div_E=C_E,
        output_dir=output_dir,
        prefix="cleaned_constraints",
    ):
        print(path)

    from axion_em_gr.physics.potentials import MassivePotential

    for path in make_curved_2d_diagnostic_report(
        grid=grid,
        state=state,
        geom=geom,
        potential=MassivePotential(m=physical.m_axion),
        sources=VacuumSources(),
        numerics=numerics,
        physical=physical,
        time=0.0,
        output_dir=output_dir / "curved_diagnostics",
        prefix="cleaned",
    ):
        print(path)


if __name__ == "__main__":
    main()