"""
Phase 18 example:

Axion evolution on a 1D isotropic Schwarzschild background.

This uses the existing 1D 3+1 curved RHS:

    partial_t a = N Pi + beta^x partial_x a

    partial_t Pi = beta^x partial_x Pi
                 + N [ D_i D^i a + K Pi
                       - D_i ln(N) D^i a
                       - dV/da ]

Run with:

    python examples/run_schwarzschild_axion_1d.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.boundary import OutflowBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.geometry.schwarzschild_like import SchwarzschildIsotropicMetric1D
from axion_em_gr.initial_data.axion_profiles import gaussian_axion_packet
from axion_em_gr.physics.comparison import difference_norms
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.geometry_plots import plot_geometry_1d
from axion_em_gr.visualization.plots_1d import (
    ensure_output_dir,
    plot_axion_state,
    plot_history,
    plot_scalar_difference_1d,
)


def run_case(label, metric, grid, physical, numerics, output_dir):
    state0 = gaussian_axion_packet(
        grid=grid,
        amplitude=1.0,
        center=35.0,
        width=5.0,
        momentum_amplitude=0.4,
    )

    potential = MassivePotential(m=physical.m_axion)

    rhs = RHSComputer(
        grid=grid,
        metric=metric,
        potential=potential,
        numerics=numerics,
        physical=physical,
        sources=VacuumSources(),
        evolve_axion=True,
        evolve_maxwell=False,
        include_axion_em_coupling=False,
    )

    solver = EvolutionSolver(
        grid=grid,
        rhs_computer=rhs,
        integrator=RK4(),
        boundary=OutflowBoundary(),
        numerics=numerics,
    )

    case_dir = ensure_output_dir(output_dir / label)

    geom0 = metric.evaluate(0.0, grid)

    print(
        plot_geometry_1d(
            grid=grid,
            geom=geom0,
            output_dir=case_dir,
            filename="geometry_initial.png",
            title=f"Geometry: {label}",
        )
    )

    print(
        plot_axion_state(
            grid,
            state0,
            title=f"Initial axion state: {label}",
            output_dir=case_dir,
            filename="initial_axion.png",
        )
    )

    final_state, history = solver.evolve(state0)

    print(
        plot_axion_state(
            grid,
            final_state,
            title=f"Final axion state: {label}",
            output_dir=case_dir,
            filename="final_axion.png",
        )
    )

    print(
        plot_history(
            history,
            output_dir=case_dir,
            filename="history.png",
        )
    )

    return state0, final_state, history


def main() -> None:
    output_dir = ensure_output_dir(Path("outputs") / "phase18_schwarzschild_axion_1d")

    # Isotropic radial coordinate r. Avoid the horizon r=M/2 by starting at r=4.
    grid = Grid(
        ndim=1,
        shape=(1024,),
        bounds=((4.0, 120.0),),
        nghost=3,
    )

    physical = PhysicalParameters(
        m_axion=0.15,
        g_agamma=0.0,
    )

    numerics = NumericalParameters(
        dt=0.005,
        t_final=45.0,
        output_every=200,
        derivative_order=2,
    )

    flat_metric = FlatMetric()

    schwarzschild_metric = SchwarzschildIsotropicMetric1D(
        mass=1.5,
        center=0.0,
        use_absolute_radius=False,
        radial_floor=1.0e-6,
        lapse_floor=1.0e-4,
        horizon_buffer=1.0e-3,
    )

    print("\nRunning flat reference")
    _, final_flat, hist_flat = run_case(
        label="flat",
        metric=flat_metric,
        grid=grid,
        physical=physical,
        numerics=numerics,
        output_dir=output_dir,
    )

    print("\nRunning Schwarzschild isotropic")
    _, final_schw, hist_schw = run_case(
        label="schwarzschild",
        metric=schwarzschild_metric,
        grid=grid,
        physical=physical,
        numerics=numerics,
        output_dir=output_dir,
    )

    comparison_dir = ensure_output_dir(output_dir / "comparison")

    print(
        plot_scalar_difference_1d(
            grid=grid,
            field_reference=final_flat.a,
            field_test=final_schw.a,
            label=r"$\Delta a$",
            output_dir=comparison_dir,
            filename="delta_a_final.png",
            title=r"Final $\Delta a = a_{\rm Schw}-a_{\rm flat}$",
        )
    )

    print(
        plot_scalar_difference_1d(
            grid=grid,
            field_reference=final_flat.Pi,
            field_test=final_schw.Pi,
            label=r"$\Delta \Pi$",
            output_dir=comparison_dir,
            filename="delta_Pi_final.png",
            title=r"Final $\Delta \Pi = \Pi_{\rm Schw}-\Pi_{\rm flat}$",
        )
    )

    geom_flat = flat_metric.evaluate(numerics.t_final, grid)
    geom_schw = schwarzschild_metric.evaluate(numerics.t_final, grid)

    norms = difference_norms(
        state_reference=final_flat,
        state_test=final_schw,
        geom_reference=geom_flat,
        geom_test=geom_schw,
        grid=grid,
        time=numerics.t_final,
    )

    print("\nSchwarzschild-minus-flat difference summary")
    print(f"t = {norms.time:.6f}")
    print(f"max Δa    = {norms.max_delta_a:.8e}")
    print(f"L2  Δa    = {norms.l2_delta_a:.8e}")
    print(f"rel L2 Δa = {norms.rel_l2_delta_a:.8e}")


if __name__ == "__main__":
    main()