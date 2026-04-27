"""
Phase 12C example:

1D constraint-solved axion-electromagnetic initial data.

This uses:

    E^x = - g B^x a + const

so that:

    partial_x E^x + g B^x partial_x a ≈ 0.

Run with:

    python examples/run_constraint_solved_1d.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.boundary import PeriodicBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.initial_data.combined_setups import (
    gaussian_axion_uniform_Bx_constraint_solved_1d,
)
from axion_em_gr.physics.constraints import (
    constraint_norms,
    electric_constraint,
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
    plot_history,
)


def main() -> None:
    output_dir = Path("outputs") / "phase12C_constraint_solved_1d"
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
        dt=0.01,
        t_final=30.0,
        output_every=100,
        derivative_order=2,
    )

    state0 = gaussian_axion_uniform_Bx_constraint_solved_1d(
        grid=grid,
        axion_amplitude=1.0,
        axion_center=50.0,
        axion_width=6.0,
        axion_momentum_amplitude=0.4,
        g_agamma=physical.g_agamma,
        Bx=1.0,
        Ex_constant=0.0,
    )

    metric = FlatMetric()
    sources = VacuumSources()
    potential = MassivePotential(m=physical.m_axion)

    geom0 = metric.evaluate(0.0, grid)

    C0 = electric_constraint(
        state=state0,
        t=0.0,
        grid=grid,
        geom=geom0,
        sources=sources,
        numerics=numerics,
        physical=physical,
        include_axion_coupling=True,
    )

    l2_C0, linf_C0 = constraint_norms(C0, grid)

    print("\nInitial Gauss constraint")
    print(f"L2   = {l2_C0:.8e}")
    print(f"Linf = {linf_C0:.8e}")

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

    print(
        plot_axion_state(
            grid,
            state0,
            title="Initial constraint-solved axion field",
            output_dir=output_dir,
            filename="initial_axion.png",
        )
    )

    print(
        plot_em_state(
            grid,
            state0,
            title="Initial constraint-solved EM field",
            output_dir=output_dir,
            filename="initial_em.png",
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

    C_final = electric_constraint(
        state=final_state,
        t=numerics.t_final,
        grid=grid,
        geom=geom_final,
        sources=sources,
        numerics=numerics,
        physical=physical,
        include_axion_coupling=True,
    )

    l2_Cf, linf_Cf = constraint_norms(C_final, grid)

    print("\nFinal Gauss constraint")
    print(f"L2   = {l2_Cf:.8e}")
    print(f"Linf = {linf_Cf:.8e}")

    print(
        plot_axion_state(
            grid,
            final_state,
            title="Final axion field",
            output_dir=output_dir,
            filename="final_axion.png",
        )
    )

    print(
        plot_em_state(
            grid,
            final_state,
            title="Final EM field",
            output_dir=output_dir,
            filename="final_em.png",
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

    div_B = C0 * 0.0

    print(
        plot_constraints_1d(
            grid=grid,
            div_B=div_B,
            div_E=C_final,
            output_dir=output_dir,
            filename="final_constraints.png",
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


if __name__ == "__main__":
    main()