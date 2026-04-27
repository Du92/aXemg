"""
Phase 8 example:

2D advanced diagnostics for flat axion-electromagnetic evolution.

This script generates:
- axion energy density,
- EM energy density,
- E_i B^i,
- constraints,
- Poynting vector,
- radial Poynting flux,
- central 1D slices.

Run with:

    python examples/run_flat_axion_em_2d_diagnostics.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.boundary import PeriodicBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.initial_data.combined_setups_2d import (
    gaussian_axion_em_ring_2d,
)
from axion_em_gr.physics.diagnostics import (
    axion_energy_flat,
    electromagnetic_energy_flat,
)
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.diagnostics_2d import (
    make_full_2d_diagnostic_report,
)
from axion_em_gr.visualization.plots_1d import (
    plot_coupling_history,
    plot_history,
)
from axion_em_gr.visualization.plots_2d import (
    ensure_output_dir,
    plot_EdotB_2d,
    plot_axion_state_2d,
    plot_em_summary_2d,
)


def main() -> None:
    output_dir = ensure_output_dir(
        Path("outputs") / "phase8_flat_axion_em_2d_diagnostics"
    )

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

    state0 = gaussian_axion_em_ring_2d(
        grid=grid,
        axion_amplitude=1.0,
        axion_center=(0.0, 0.0),
        axion_width=(8.0, 8.0),
        axion_momentum_amplitude=0.3,
        em_amplitude=0.2,
        em_center=(0.0, 0.0),
        em_width=(12.0, 12.0),
        background_Bz=1.0,
    )

    metric = FlatMetric()
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

    geom0 = metric.evaluate(0.0, grid)

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

    initial_report_dir = ensure_output_dir(output_dir / "initial_diagnostics")

    for path in make_full_2d_diagnostic_report(
        grid=grid,
        state=state0,
        geom=geom0,
        potential=potential,
        sources=sources,
        numerics=numerics,
        physical=physical,
        time=0.0,
        output_dir=initial_report_dir,
        prefix="initial",
        center=(0.0, 0.0),
    ):
        print(path)

    final_state, history = solver.evolve(state0)

    geom_final = metric.evaluate(numerics.t_final, grid)

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

    final_report_dir = ensure_output_dir(output_dir / "final_diagnostics")

    for path in make_full_2d_diagnostic_report(
        grid=grid,
        state=final_state,
        geom=geom_final,
        potential=potential,
        sources=sources,
        numerics=numerics,
        physical=physical,
        time=numerics.t_final,
        output_dir=final_report_dir,
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

    E_a = axion_energy_flat(
        state=final_state,
        grid=grid,
        potential=potential,
    )

    E_em = electromagnetic_energy_flat(
        state=final_state,
        grid=grid,
        geom=geom_final,
    )

    print("\nFinal integrated diagnostic energies")
    print(f"E_axion = {E_a:.8e}")
    print(f"E_EM    = {E_em:.8e}")
    print(f"E_total = {E_a + E_em:.8e}")


if __name__ == "__main__":
    main()