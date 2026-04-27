"""
Phase 5 example:

Axion-electromagnetic evolution on a prescribed TT gravitational-wave
background.

The GW propagates along the numerical x direction. The transverse plane is
(y, z), so the spatial metric is

    gamma_ij =
        [[1,       0,        0],
         [0, 1 + h_+,  h_x],
         [0,   h_x, 1 - h_+]]

Run with:

    python examples/run_gw_background.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.boundary import PeriodicBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.gw_tt import GWTTMetric1D
from axion_em_gr.initial_data.combined_setups import (
    gaussian_axion_em_wave_background_Bx_1d,
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
    plot_gw_metric_1d,
    plot_history,
)


def main() -> None:
    output_dir = Path("outputs") / "phase5_gw_background"
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

    metric = GWTTMetric1D(
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
        plot_gw_metric_1d(
            grid,
            geom0,
            output_dir=output_dir,
            filename="gw_metric_initial.png",
            title="Initial TT gravitational-wave background",
        )
    )

    print(
        plot_axion_state(
            grid,
            state0,
            title="Initial axion field on GW background",
            output_dir=output_dir,
            filename="initial_axion_state.png",
        )
    )

    print(
        plot_em_state(
            grid,
            state0,
            title="Initial electromagnetic field on GW background",
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
            title=r"Initial $E_iB^i$ on GW background",
        )
    )

    final_state, history = solver.evolve(state0)

    geom_final = metric.evaluate(numerics.t_final, grid)

    print(
        plot_gw_metric_1d(
            grid,
            geom_final,
            output_dir=output_dir,
            filename="gw_metric_final.png",
            title="Final TT gravitational-wave background",
        )
    )

    print(
        plot_axion_state(
            grid,
            final_state,
            title="Final axion field on GW background",
            output_dir=output_dir,
            filename="final_axion_state.png",
        )
    )

    print(
        plot_em_state(
            grid,
            final_state,
            title="Final electromagnetic field on GW background",
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
            title=r"Final $E_iB^i$ on GW background",
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