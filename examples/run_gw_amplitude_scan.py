"""
Phase 6 amplitude scan:

Run several GW amplitudes and measure the final difference with respect to
the flat case.

Run with:

    python examples/run_gw_amplitude_scan.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from axion_em_gr.core.boundary import PeriodicBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.geometry.gw_tt import GWTTMetric1D
from axion_em_gr.initial_data.combined_setups import (
    gaussian_axion_em_wave_background_Bx_1d,
)
from axion_em_gr.physics.comparison import difference_norms
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.plots_1d import ensure_output_dir


def run_case(metric, state0, grid, physical, numerics):
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

    final_state, history = solver.evolve(state0)

    return final_state, history


def main() -> None:
    output_dir = ensure_output_dir(Path("outputs") / "phase6_gw_amplitude_scan")

    grid = Grid(
        ndim=1,
        shape=(512,),
        bounds=((0.0, 100.0),),
        nghost=3,
    )

    physical = PhysicalParameters(
        m_axion=0.2,
        g_agamma=0.03,
    )

    numerics = NumericalParameters(
        dt=0.005,
        t_final=20.0,
        output_every=400,
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

    print("Running flat reference")
    flat_metric = FlatMetric()
    final_flat, _ = run_case(
        metric=flat_metric,
        state0=state0,
        grid=grid,
        physical=physical,
        numerics=numerics,
    )

    geom_flat = flat_metric.evaluate(numerics.t_final, grid)

    amplitudes = [0.0, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 5.0e-2, 1.0e-1]

    final_delta_a = []
    final_delta_E = []
    final_delta_B = []
    final_delta_EdotB = []

    for amp in amplitudes:
        print(f"Running GW amplitude h_plus={amp:.3e}")

        gw_metric = GWTTMetric1D(
            h_plus_amplitude=amp,
            h_cross_amplitude=0.5 * amp,
            wavelength=50.0,
            omega=None,
            phase_plus=0.0,
            phase_cross=0.5,
            packet=False,
            direction=+1,
            compute_K_exact=True,
        )

        final_gw, _ = run_case(
            metric=gw_metric,
            state0=state0,
            grid=grid,
            physical=physical,
            numerics=numerics,
        )

        geom_gw = gw_metric.evaluate(numerics.t_final, grid)

        norms = difference_norms(
            state_reference=final_flat,
            state_test=final_gw,
            geom_reference=geom_flat,
            geom_test=geom_gw,
            grid=grid,
            time=numerics.t_final,
        )

        final_delta_a.append(norms.l2_delta_a)
        final_delta_E.append(norms.l2_delta_E)
        final_delta_B.append(norms.l2_delta_B)
        final_delta_EdotB.append(norms.l2_delta_EdotB)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(amplitudes, final_delta_a, marker="o", label=r"$||\Delta a||_2$")
    ax.plot(amplitudes, final_delta_E, marker="o", label=r"$||\Delta E||_2$")
    ax.plot(amplitudes, final_delta_B, marker="o", label=r"$||\Delta B||_2$")
    ax.plot(amplitudes, final_delta_EdotB, marker="o", label=r"$||\Delta(E_iB^i)||_2$")

    ax.set_xlabel(r"$h_+$ amplitude")
    ax.set_ylabel("final absolute difference norm")
    ax.legend()
    fig.tight_layout()

    path = output_dir / "gw_amplitude_scan.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(path)


if __name__ == "__main__":
    main()