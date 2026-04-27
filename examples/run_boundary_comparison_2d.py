"""
Phase 14 example:

Compare 2D boundary conditions for a localized axion-EM setup.

Run with:

    python examples/run_boundary_comparison_2d.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.boundary import (
    MixedBoundary,
    OutflowBoundary,
    PeriodicBoundary,
    SommerfeldBoundary,
)
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.initial_data.combined_setups_2d import gaussian_axion_em_ring_2d
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.diagnostics_2d import make_full_2d_diagnostic_report
from axion_em_gr.visualization.plots_1d import plot_history
from axion_em_gr.visualization.plots_2d import ensure_output_dir, plot_axion_state_2d


def run_case(boundary, label: str):
    grid = Grid(
        ndim=2,
        shape=(128, 128),
        bounds=((-50.0, 50.0), (-50.0, 50.0)),
        nghost=3,
    )

    physical = PhysicalParameters(
        m_axion=0.2,
        g_agamma=0.03,
    )

    numerics = NumericalParameters(
        dt=0.01,
        t_final=8.0,
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
        boundary=boundary,
        numerics=numerics,
    )

    final_state, history = solver.evolve(state0)

    return grid, state0, final_state, history, metric, potential, sources, numerics, physical


def main() -> None:
    output_dir = ensure_output_dir(Path("outputs") / "phase14_boundary_comparison_2d")

    sommerfeld = SommerfeldBoundary(
        asymptotic_value=0.0,
        center=(0.0, 0.0),
    )

    mixed = MixedBoundary(
        default=OutflowBoundary(),
        field_boundaries={
            "a": sommerfeld,
            "Pi": sommerfeld,
            "E": OutflowBoundary(),
            "B": OutflowBoundary(),
        },
    )

    cases = {
        "periodic": PeriodicBoundary(),
        "outflow": OutflowBoundary(),
        "mixed_sommerfeld": mixed,
    }

    for label, boundary in cases.items():
        print(f"\nRunning 2D boundary case: {label}")

        (
            grid,
            state0,
            final_state,
            history,
            metric,
            potential,
            sources,
            numerics,
            physical,
        ) = run_case(boundary, label)

        case_dir = ensure_output_dir(output_dir / label)

        for path in plot_axion_state_2d(
            grid=grid,
            state=final_state,
            output_dir=case_dir,
            prefix="final_axion",
        ):
            print(path)

        print(
            plot_history(
                history,
                output_dir=case_dir,
                filename="history.png",
            )
        )

        geom_final = metric.evaluate(numerics.t_final, grid)

        for path in make_full_2d_diagnostic_report(
            grid=grid,
            state=final_state,
            geom=geom_final,
            potential=potential,
            sources=sources,
            numerics=numerics,
            physical=physical,
            time=numerics.t_final,
            output_dir=case_dir / "diagnostics",
            prefix="final",
            center=(0.0, 0.0),
        ):
            print(path)


if __name__ == "__main__":
    main()