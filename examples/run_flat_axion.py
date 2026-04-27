"""
Phase 1 example:

Flat-space 1D evolution of a massive axion field.

Run with:

    python examples/run_flat_axion.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.boundary import PeriodicBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.initial_data.axion_profiles import gaussian_axion_packet
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.plots_1d import plot_axion_state, plot_history


def main() -> None:
    output_dir = Path("outputs") / "phase1_flat_axion"
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid(
        ndim=1,
        shape=(1024,),
        bounds=((0.0, 100.0),),
        nghost=3,
    )

    physical = PhysicalParameters(
        m_axion=0.2,
        g_agamma=0.0,
    )

    numerics = NumericalParameters(
        dt=0.02,
        t_final=50.0,
        output_every=100,
        derivative_order=2,
    )

    state0 = gaussian_axion_packet(
        grid=grid,
        amplitude=1.0,
        center=50.0,
        width=5.0,
        momentum_amplitude=0.0,
    )

    metric = FlatMetric()
    potential = MassivePotential(m=physical.m_axion)

    rhs = RHSComputer(
        grid=grid,
        metric=metric,
        potential=potential,
        numerics=numerics,
        physical=physical,
        evolve_axion=True,
        evolve_maxwell=False,
    )

    solver = EvolutionSolver(
        grid=grid,
        rhs_computer=rhs,
        integrator=RK4(),
        boundary=PeriodicBoundary(),
        numerics=numerics,
    )

    initial_plot = plot_axion_state(
        grid,
        state0,
        title="Initial axion field",
        output_dir=output_dir,
        filename="initial_axion_state.png",
    )
    print(f"Saved initial axion plot to: {initial_plot}")

    final_state, history = solver.evolve(state0)

    final_plot = plot_axion_state(
        grid,
        final_state,
        title="Final axion field",
        output_dir=output_dir,
        filename="final_axion_state.png",
    )
    print(f"Saved final axion plot to: {final_plot}")

    history_plot = plot_history(
        history,
        output_dir=output_dir,
        filename="history.png",
    )
    print(f"Saved history plot to: {history_plot}")


if __name__ == "__main__":
    main()