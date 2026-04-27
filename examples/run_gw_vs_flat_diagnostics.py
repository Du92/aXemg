"""
Phase 6 example:

Advanced diagnostics comparing a flat background with a TT gravitational-wave
background.

This script computes:
- field-by-field final differences,
- time-dependent difference norms,
- Delta(E_i B^i),
- saved snapshots for later animations.

Run with:

    python examples/run_gw_vs_flat_diagnostics.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.boundary import PeriodicBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.geometry.gw_tt import GWTTMetric1D
from axion_em_gr.initial_data.combined_setups import (
    gaussian_axion_em_wave_background_Bx_1d,
)
from axion_em_gr.io.snapshot import save_history_snapshots
from axion_em_gr.physics.comparison import (
    difference_norms,
    edotb_profile,
)
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.plots_1d import (
    ensure_output_dir,
    plot_EdotB_difference_1d,
    plot_difference_norms,
    plot_relative_difference_norms,
    plot_scalar_difference_1d,
    plot_vector_difference_components_1d,
)


def run_case(
    metric,
    state0,
    grid,
    physical,
    numerics,
    save_snapshots: bool,
    snapshot_every: int,
):
    """
    Run one evolution case.
    """
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
        save_snapshots=save_snapshots,
        snapshot_every=snapshot_every,
    )

    final_state, history = solver.evolve(state0)

    return final_state, history


def main() -> None:
    output_dir = ensure_output_dir(Path("outputs") / "phase6_gw_vs_flat_diagnostics")
    snapshot_dir = ensure_output_dir(output_dir / "snapshots")

    grid = Grid(
        ndim=1,
        shape=(1024,),
        bounds=((0.0, 100.0),),
        nghost=3,
    )

    physical = PhysicalParameters(
        m_axion=0.2,
        g_agamma=0.03,
    )

    numerics = NumericalParameters(
        dt=0.005,
        t_final=30.0,
        output_every=200,
        derivative_order=2,
    )

    state0 = gaussian_axion_em_wave_background_Bx_1d(
        grid=grid,
        axion_amplitude=1.0,
        axion_center=35.0,
        axion_width=7.0,
        axion_momentum_amplitude=0.35,
        em_amplitude=0.15,
        em_center=40.0,
        em_width=8.0,
        background_Bx=1.0,
        propagation="right",
    )

    flat_metric = FlatMetric()

    gw_metric = GWTTMetric1D(
        h_plus_amplitude=5.0e-2,
        h_cross_amplitude=3.0e-2,
        wavelength=50.0,
        omega=None,
        phase_plus=0.0,
        phase_cross=0.5,
        packet=False,
        direction=+1,
        compute_K_exact=True,
    )

    print("Running flat reference case")
    final_flat, hist_flat = run_case(
        metric=flat_metric,
        state0=state0,
        grid=grid,
        physical=physical,
        numerics=numerics,
        save_snapshots=True,
        snapshot_every=200,
    )

    print("Running GW test case")
    final_gw, hist_gw = run_case(
        metric=gw_metric,
        state0=state0,
        grid=grid,
        physical=physical,
        numerics=numerics,
        save_snapshots=True,
        snapshot_every=200,
    )

    flat_snapshot_paths = save_history_snapshots(
        hist_flat,
        snapshot_dir / "flat",
        prefix="flat",
    )

    gw_snapshot_paths = save_history_snapshots(
        hist_gw,
        snapshot_dir / "gw",
        prefix="gw",
    )

    print(f"Saved {len(flat_snapshot_paths)} flat snapshots")
    print(f"Saved {len(gw_snapshot_paths)} GW snapshots")

    norms = []

    n_pairs = min(len(hist_flat.snapshots), len(hist_gw.snapshots))

    for i in range(n_pairs):
        t_flat = hist_flat.snapshot_times[i]
        t_gw = hist_gw.snapshot_times[i]

        if abs(t_flat - t_gw) > 1.0e-12:
            raise RuntimeError(
                f"Snapshot times do not match: flat={t_flat}, gw={t_gw}"
            )

        state_flat = hist_flat.snapshots[i]
        state_gw = hist_gw.snapshots[i]

        geom_flat = flat_metric.evaluate(t_flat, grid)
        geom_gw = gw_metric.evaluate(t_gw, grid)

        norms.append(
            difference_norms(
                state_reference=state_flat,
                state_test=state_gw,
                geom_reference=geom_flat,
                geom_test=geom_gw,
                grid=grid,
                time=t_flat,
            )
        )

    print(
        plot_difference_norms(
            norms,
            output_dir=output_dir,
            filename="difference_norms.png",
        )
    )

    print(
        plot_relative_difference_norms(
            norms,
            output_dir=output_dir,
            filename="relative_difference_norms.png",
        )
    )

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
            output_dir=output_dir,
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
            output_dir=output_dir,
            filename="delta_Pi_final.png",
            title=r"Final $\Delta \Pi = \Pi_{\rm GW}-\Pi_{\rm flat}$",
        )
    )

    if final_flat.E is not None and final_gw.E is not None:
        print(
            plot_vector_difference_components_1d(
                grid=grid,
                vector_reference=final_flat.E,
                vector_test=final_gw.E,
                vector_name="E",
                output_dir=output_dir,
                filename="delta_E_final.png",
                title=r"Final $\Delta E^i = E^i_{\rm GW}-E^i_{\rm flat}$",
            )
        )

    if final_flat.B is not None and final_gw.B is not None:
        print(
            plot_vector_difference_components_1d(
                grid=grid,
                vector_reference=final_flat.B,
                vector_test=final_gw.B,
                vector_name="B",
                output_dir=output_dir,
                filename="delta_B_final.png",
                title=r"Final $\Delta B^i = B^i_{\rm GW}-B^i_{\rm flat}$",
            )
        )

    print(
        plot_EdotB_difference_1d(
            grid=grid,
            edotb_reference=edotb_flat,
            edotb_test=edotb_gw,
            output_dir=output_dir,
            filename="delta_EdotB_final.png",
            title=r"Final $\Delta(E_iB^i)$",
        )
    )

    final_norm = norms[-1]

    print("\nFinal difference summary")
    print(f"t = {final_norm.time:.6f}")
    print(f"max Δa              = {final_norm.max_delta_a:.6e}")
    print(f"L2  Δa              = {final_norm.l2_delta_a:.6e}")
    print(f"rel L2 Δa           = {final_norm.rel_l2_delta_a:.6e}")
    print(f"max ΔE              = {final_norm.max_delta_E:.6e}")
    print(f"L2  ΔE              = {final_norm.l2_delta_E:.6e}")
    print(f"rel L2 ΔE           = {final_norm.rel_l2_delta_E:.6e}")
    print(f"max ΔB              = {final_norm.max_delta_B:.6e}")
    print(f"L2  ΔB              = {final_norm.l2_delta_B:.6e}")
    print(f"rel L2 ΔB           = {final_norm.rel_l2_delta_B:.6e}")
    print(f"max Δ(E.B)          = {final_norm.max_delta_EdotB:.6e}")
    print(f"L2  Δ(E.B)          = {final_norm.l2_delta_EdotB:.6e}")
    print(f"rel L2 Δ(E.B)       = {final_norm.rel_l2_delta_EdotB:.6e}")


if __name__ == "__main__":
    main()