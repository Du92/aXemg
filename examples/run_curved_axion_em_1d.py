"""
Phase 4 example:

Axion-electromagnetic evolution on a nontrivial 1D 3+1 background.

This tests:
- lapse N,
- shift beta^x,
- spatial metric gamma_ij,
- sqrt(gamma),
- K,
- axion-em coupling.

Run with:

    python examples/run_curved_axion_em_1d.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.boundary import PeriodicBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.diagonal_1d import DiagonalMetric1D
from axion_em_gr.initial_data.combined_setups import (
    gaussian_axion_uniform_magnetic_field_1d,
)
from axion_em_gr.physics.constraints import (
    electric_constraint_3p1_1d,
    magnetic_constraint_3p1_1d,
)
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.plots_1d import (
    plot_EdotB_profile,
    plot_axion_state,
    plot_constraints_1d,
    plot_coupling_history,
    plot_em_state,
    plot_geometry_1d,
    plot_history,
)


def main() -> None:
    output_dir = Path("outputs") / "phase4_curved_axion_em_1d"
    output_dir.mkdir(parents=True, exist_ok=True)

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
        t_final=40.0,
        output_every=200,
        derivative_order=2,
    )

    state0 = gaussian_axion_uniform_magnetic_field_1d(
        grid=grid,
        axion_amplitude=1.0,
        axion_center=35.0,
        axion_width=6.0,
        axion_momentum_amplitude=0.4,
        B0=(1.0, 0.0, 0.0),
        E0=(0.0, 0.0, 0.0),
    )

    metric = DiagonalMetric1D(
        lapse_amplitude=0.10,
        metric_amplitude=0.15,
        shift_amplitude=0.03,
        center=50.0,
        width=12.0,
        K_value=0.0,
    )

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

    print(
        plot_geometry_1d(
            grid,
            geom0,
            output_dir=output_dir,
            filename="geometry.png",
            title="Toy 1D 3+1 geometry",
        )
    )

    print(
        plot_axion_state(
            grid,
            state0,
            title="Initial axion field",
            output_dir=output_dir,
            filename="initial_axion_state.png",
        )
    )

    print(
        plot_em_state(
            grid,
            state0,
            title="Initial electromagnetic field",
            output_dir=output_dir,
            filename="initial_em_state.png",
        )
    )

    print(
        plot_EdotB_profile(
            grid,
            state0,
            geom0,
            output_dir=output_dir,
            filename="initial_EdotB.png",
            title=r"Initial $E_iB^i$",
        )
    )

    final_state, history = solver.evolve(state0)

    geom_final = metric.evaluate(numerics.t_final, grid)

    print(
        plot_axion_state(
            grid,
            final_state,
            title="Final axion field",
            output_dir=output_dir,
            filename="final_axion_state.png",
        )
    )

    print(
        plot_em_state(
            grid,
            final_state,
            title="Final electromagnetic field",
            output_dir=output_dir,
            filename="final_em_state.png",
        )
    )

    print(
        plot_EdotB_profile(
            grid,
            final_state,
            geom_final,
            output_dir=output_dir,
            filename="final_EdotB.png",
            title=r"Final $E_iB^i$",
        )
    )

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

    div_B = magnetic_constraint_3p1_1d(
        state=final_state,
        grid=grid,
        geom=geom_final,
        numerics=numerics,
    )

    div_E = electric_constraint_3p1_1d(
        state=final_state,
        t=numerics.t_final,
        grid=grid,
        geom=geom_final,
        sources=sources,
        numerics=numerics,
        physical=physical,
        include_axion_coupling=True,
    )

    print(
        plot_constraints_1d(
            grid=grid,
            div_B=div_B,
            div_E=div_E,
            output_dir=output_dir,
            filename="constraints_final.png",
        )
    )


if __name__ == "__main__":
    main()
