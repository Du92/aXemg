"""
Compare flat background vs TT gravitational-wave background.

Run with:

    python examples/run_gw_vs_flat.py
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
    output_dir = ensure_output_dir(Path("outputs") / "phase5_gw_vs_flat")

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
        h_plus_amplitude=2.0e-2,
        h_cross_amplitude=1.0e-2,
        wavelength=50.0,
        omega=None,
        phase_plus=0.0,
        phase_cross=0.5,
        packet=False,
        direction=+1,
        compute_K_exact=True,
    )

    print("Running flat case")
    _, hist_flat = run_case(
        metric=flat_metric,
        state0=state0,
        grid=grid,
        physical=physical,
        numerics=numerics,
    )

    print("Running GW case")
    _, hist_gw = run_case(
        metric=gw_metric,
        state0=state0,
        grid=grid,
        physical=physical,
        numerics=numerics,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist_flat.times, hist_flat.max_EdotB, label=r"flat: $\max |E_iB^i|$")
    ax.plot(hist_gw.times, hist_gw.max_EdotB, label=r"GW: $\max |E_iB^i|$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\max |E_iB^i|$")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "compare_EdotB.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(path)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist_flat.times, hist_flat.max_E, label=r"flat: $\max |E|$")
    ax.plot(hist_gw.times, hist_gw.max_E, label=r"GW: $\max |E|$")
    ax.plot(hist_flat.times, hist_flat.max_B, label=r"flat: $\max |B|$", linestyle="--")
    ax.plot(hist_gw.times, hist_gw.max_B, label=r"GW: $\max |B|$", linestyle="--")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("field amplitude")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "compare_EM_amplitudes.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(path)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist_flat.times, hist_flat.max_a, label=r"flat: $\max |a|$")
    ax.plot(hist_gw.times, hist_gw.max_a, label=r"GW: $\max |a|$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\max |a|$")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "compare_axion_amplitude.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()