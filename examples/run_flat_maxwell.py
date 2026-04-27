"""
Phase 2 example:

Flat-space 1D vacuum Maxwell evolution.

Equations:

    ∂_t B^i = - epsilon^{ijk} ∂_j E_k
    ∂_t E^i = - epsilon^{ijk} ∂_j B_k

Run with:

    python examples/run_flat_maxwell.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.boundary import PeriodicBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.initial_data.electromagnetic_profiles import gaussian_em_pulse_1d
from axion_em_gr.physics.constraints import (
    electric_constraint_3p1_1d,
    magnetic_constraint_3p1_1d,
)
from axion_em_gr.physics.potentials import ZeroPotential
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.plots_1d import (
    plot_constraints_1d,
    plot_em_state,
    plot_history,
)
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters

physical = PhysicalParameters(
    m_axion=0.0,
    g_agamma=0.0,
)

def main() -> None:
    output_dir = Path("outputs") / "phase2_flat_maxwell"
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid(
        ndim=1,
        shape=(1024,),
        bounds=((0.0, 100.0),),
        nghost=3,
    )

    numerics = NumericalParameters(
        dt=0.02,
        t_final=50.0,
        output_every=100,
        derivative_order=2,
    )

    state0 = gaussian_em_pulse_1d(
        grid=grid,
        amplitude=1.0,
        center=50.0,
        width=5.0,
        polarization="y",
        propagation="right",
    )

    metric = FlatMetric()
    sources = VacuumSources()

    rhs = RHSComputer(
        grid=grid,
        metric=metric,
        potential=ZeroPotential(),
        numerics=numerics,
        physical=physical,
        sources=sources,
        evolve_axion=False,
        evolve_maxwell=True,
    )

    solver = EvolutionSolver(
        grid=grid,
        rhs_computer=rhs,
        integrator=RK4(),
        boundary=PeriodicBoundary(),
        numerics=numerics,
    )

    initial_plot = plot_em_state(
        grid,
        state0,
        title="Initial electromagnetic pulse",
        output_dir=output_dir,
        filename="initial_em_state.png",
    )
    print(f"Saved initial EM plot to: {initial_plot}")

    final_state, history = solver.evolve(state0)

    final_plot = plot_em_state(
        grid,
        final_state,
        title="Final electromagnetic pulse",
        output_dir=output_dir,
        filename="final_em_state.png",
    )
    print(f"Saved final EM plot to: {final_plot}")

    history_plot = plot_history(
        history,
        output_dir=output_dir,
        filename="history.png",
    )
    print(f"Saved history plot to: {history_plot}")

    geom = metric.evaluate(numerics.t_final, grid)

    div_B = magnetic_constraint_3p1_1d(
        state=final_state,
        grid=grid,
        geom=geom,
        numerics=numerics,
    )

    div_E = electric_constraint_3p1_1d(
        state=final_state,
        t=numerics.t_final,
        grid=grid,
        geom=geom,
        sources=sources,
        numerics=numerics,
    )

    constraint_plot = plot_constraints_1d(
        grid=grid,
        div_B=div_B,
        div_E=div_E,
        output_dir=output_dir,
        filename="constraints_final.png",
    )
    print(f"Saved constraint plot to: {constraint_plot}")


if __name__ == "__main__":
    main()
