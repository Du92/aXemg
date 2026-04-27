"""
Phase 16 example:

Gravitational wave crossing a magnetized axion halo with a localized axion
gradient.

This script compares:
    - flat background
    - prescribed TT gravitational-wave background

Run with:

    python examples/run_gw_axion_halo_1d.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.boundary import MixedBoundary, OutflowBoundary, SommerfeldBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.geometry.gw_tt import GWTTMetric1D
from axion_em_gr.initial_data.physical_scenarios_1d import (
    axion_halo_gradient_magnetized_1d,
)
from axion_em_gr.physics.comparison import difference_norms, edotb_profile
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.plots_1d import (
    ensure_output_dir,
    plot_EdotB_difference_1d,
    plot_EdotB_profile,
    plot_axion_state,
    plot_coupling_history,
    plot_em_state,
    plot_gw_metric_1d,
    plot_history,
    plot_scalar_difference_1d,
    plot_vector_difference_components_1d,
)


def build_boundary():
    """
    Use Sommerfeld-like boundaries for axion variables and outflow for EM.
    """
    sommerfeld = SommerfeldBoundary(
        asymptotic_value=0.0,
        wave_speed=1.0,
        center=(60.0, 0.0),
    )

    return MixedBoundary(
        default=OutflowBoundary(),
        field_boundaries={
            "a": sommerfeld,
            "Pi": sommerfeld,
            "E": OutflowBoundary(),
            "B": OutflowBoundary(),
        },
    )


def build_initial_state(grid, physical):
    return axion_halo_gradient_magnetized_1d(
        grid=grid,
        axion_amplitude=1.0,
        axion_center=45.0,
        axion_width=10.0,
        axion_background=0.0,
        gradient_amplitude=0.35,
        gradient_width=12.0,
        Pi_amplitude=0.15,
        Pi_width=10.0,
        g_agamma=physical.g_agamma,
        background_Bx=1.0,
        background_By=0.0,
        background_Bz=0.0,
        em_pulse_amplitude=0.15,
        em_pulse_center=40.0,
        em_pulse_width=8.0,
        em_pulse_polarization="y",
        em_pulse_propagation="right",
        constraint_solved_Ex=True,
    )


def run_case(label, metric, state0, grid, physical, numerics, output_dir):
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
        boundary=build_boundary(),
        numerics=numerics,
        save_snapshots=False,
    )

    case_dir = ensure_output_dir(output_dir / label)

    geom0 = metric.evaluate(0.0, grid)

    if label == "gw":
        print(
            plot_gw_metric_1d(
                grid=grid,
                geom=geom0,
                output_dir=case_dir,
                filename="gw_metric_initial.png",
                title="Initial TT gravitational-wave metric",
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

    print(
        plot_em_state(
            grid,
            state0,
            title=f"Initial EM state: {label}",
            output_dir=case_dir,
            filename="initial_em.png",
        )
    )

    print(
        plot_EdotB_profile(
            grid,
            state0,
            geom0,
            output_dir=case_dir,
            filename="initial_EdotB.png",
            title=rf"Initial $E_iB^i$: {label}",
        )
    )

    final_state, history = solver.evolve(state0)

    geom_final = metric.evaluate(numerics.t_final, grid)

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
        plot_em_state(
            grid,
            final_state,
            title=f"Final EM state: {label}",
            output_dir=case_dir,
            filename="final_em.png",
        )
    )

    print(
        plot_EdotB_profile(
            grid,
            final_state,
            geom_final,
            output_dir=case_dir,
            filename="final_EdotB.png",
            title=rf"Final $E_iB^i$: {label}",
        )
    )

    print(
        plot_history(
            history,
            output_dir=case_dir,
            filename="history.png",
        )
    )

    print(
        plot_coupling_history(
            history,
            output_dir=case_dir,
            filename="coupling_history.png",
        )
    )

    return final_state, history


def main() -> None:
    output_dir = ensure_output_dir(Path("outputs") / "phase16_gw_axion_halo_1d")

    grid = Grid(
        ndim=1,
        shape=(1024,),
        bounds=((0.0, 120.0),),
        nghost=3,
    )

    physical = PhysicalParameters(
        m_axion=0.15,
        g_agamma=0.03,
    )

    numerics = NumericalParameters(
        dt=0.005,
        t_final=45.0,
        output_every=200,
        derivative_order=2,
    )

    state0 = build_initial_state(grid, physical)

    flat_metric = FlatMetric()

    gw_metric = GWTTMetric1D(
        h_plus_amplitude=5.0e-2,
        h_cross_amplitude=3.0e-2,
        wavelength=50.0,
        omega=None,
        phase_plus=0.0,
        phase_cross=0.5,
        packet=True,
        packet_center=25.0,
        packet_width=25.0,
        direction=+1,
        compute_K_exact=True,
    )

    print("\nRunning flat reference case")
    final_flat, hist_flat = run_case(
        label="flat",
        metric=flat_metric,
        state0=state0,
        grid=grid,
        physical=physical,
        numerics=numerics,
        output_dir=output_dir,
    )

    print("\nRunning GW case")
    final_gw, hist_gw = run_case(
        label="gw",
        metric=gw_metric,
        state0=state0,
        grid=grid,
        physical=physical,
        numerics=numerics,
        output_dir=output_dir,
    )

    comparison_dir = ensure_output_dir(output_dir / "comparison")

    geom_flat_final = flat_metric.evaluate(numerics.t_final, grid)
    geom_gw_final = gw_metric.evaluate(numerics.t_final, grid)

    edotb_flat = edotb_profile(final_flat, geom_flat_final)
    edotb_gw = edotb_profile(final_gw, geom_gw_final)

    print(
        plot_scalar_difference_1d(
            grid=grid,
            field_reference=final_flat.a,
            field_test=final_gw.a,
            label=r"$\Delta a$",
            output_dir=comparison_dir,
            filename="delta_a_final.png",
            title=r"Final $\Delta a = a_{\rm GW}-a_{\rm flat}$",
        )
    )

    print(
        plot_scalar_difference_1d(
            grid=grid,
            field_reference=final_flat.Pi,
            field_test=final_gw.Pi,
            label=r"$\Delta \Pi$",
            output_dir=comparison_dir,
            filename="delta_Pi_final.png",
            title=r"Final $\Delta \Pi = \Pi_{\rm GW}-\Pi_{\rm flat}$",
        )
    )

    print(
        plot_vector_difference_components_1d(
            grid=grid,
            vector_reference=final_flat.E,
            vector_test=final_gw.E,
            vector_name="E",
            output_dir=comparison_dir,
            filename="delta_E_final.png",
            title=r"Final $\Delta E^i$",
        )
    )

    print(
        plot_vector_difference_components_1d(
            grid=grid,
            vector_reference=final_flat.B,
            vector_test=final_gw.B,
            vector_name="B",
            output_dir=comparison_dir,
            filename="delta_B_final.png",
            title=r"Final $\Delta B^i$",
        )
    )

    print(
        plot_EdotB_difference_1d(
            grid=grid,
            edotb_reference=edotb_flat,
            edotb_test=edotb_gw,
            output_dir=comparison_dir,
            filename="delta_EdotB_final.png",
            title=r"Final $\Delta(E_iB^i)$",
        )
    )

    norms = difference_norms(
        state_reference=final_flat,
        state_test=final_gw,
        geom_reference=geom_flat_final,
        geom_test=geom_gw_final,
        grid=grid,
        time=numerics.t_final,
    )

    print("\nFinal GW-minus-flat difference summary")
    print(f"t = {norms.time:.6f}")
    print(f"max Δa        = {norms.max_delta_a:.8e}")
    print(f"L2  Δa        = {norms.l2_delta_a:.8e}")
    print(f"rel L2 Δa     = {norms.rel_l2_delta_a:.8e}")
    print(f"max ΔE        = {norms.max_delta_E:.8e}")
    print(f"L2  ΔE        = {norms.l2_delta_E:.8e}")
    print(f"max ΔB        = {norms.max_delta_B:.8e}")
    print(f"L2  ΔB        = {norms.l2_delta_B:.8e}")
    print(f"max Δ(E.B)    = {norms.max_delta_EdotB:.8e}")
    print(f"L2  Δ(E.B)    = {norms.l2_delta_EdotB:.8e}")


if __name__ == "__main__":
    main()