"""
Phase 14 example:

Compare 1D boundary conditions for an outgoing axion packet.

Run with:

    python examples/run_boundary_comparison_1d.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from axion_em_gr.core.boundary import (
    OutflowBoundary,
    PeriodicBoundary,
    SommerfeldBoundary,
)
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.initial_data.axion_profiles import gaussian_axion_packet
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.plots_1d import ensure_output_dir, plot_axion_state


def run_case(boundary, label: str):
    grid = Grid(
        ndim=1,
        shape=(1024,),
        bounds=((0.0, 100.0),),
        nghost=3,
    )

    physical = PhysicalParameters(
        m_axion=0.1,
        g_agamma=0.0,
    )

    numerics = NumericalParameters(
        dt=0.01,
        t_final=60.0,
        output_every=200,
        derivative_order=2,
    )

    # Give Pi a nonzero profile to generate outgoing motion.
    state0 = gaussian_axion_packet(
        grid=grid,
        amplitude=1.0,
        center=30.0,
        width=5.0,
        momentum_amplitude=0.7,
    )

    metric = FlatMetric()
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
        boundary=boundary,
        numerics=numerics,
    )

    final_state, history = solver.evolve(state0)

    return grid, state0, final_state, history


def main() -> None:
    output_dir = ensure_output_dir(Path("outputs") / "phase14_boundary_comparison_1d")

    cases = {
        "periodic": PeriodicBoundary(),
        "outflow": OutflowBoundary(),
        "sommerfeld": SommerfeldBoundary(asymptotic_value=0.0),
    }

    results = {}

    for label, boundary in cases.items():
        print(f"\nRunning boundary case: {label}")
        grid, state0, final_state, history = run_case(boundary, label)
        results[label] = (grid, state0, final_state, history)

        print(
            plot_axion_state(
                grid,
                final_state,
                title=f"Final axion field: {label}",
                output_dir=output_dir,
                filename=f"final_axion_{label}.png",
            )
        )

    # Combined final comparison.
    fig, ax = plt.subplots(figsize=(9, 4))

    for label, (grid, state0, final_state, history) in results.items():
        x = grid.coordinates_1d()[grid.interior_slices]
        a = final_state.a[grid.interior_slices]
        ax.plot(x, a, label=label)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$a$")
    ax.set_title("Final axion field for different boundary conditions")
    ax.legend()
    fig.tight_layout()

    path = output_dir / "final_axion_boundary_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(path)

    # History comparison.
    fig, ax = plt.subplots(figsize=(9, 4))

    for label, (grid, state0, final_state, history) in results.items():
        ax.plot(history.times, history.l2_a, label=label)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$||a||_2$")
    ax.set_title("Axion L2 norm for different boundary conditions")
    ax.legend()
    fig.tight_layout()

    path = output_dir / "l2_axion_boundary_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(path)


if __name__ == "__main__":
    main()