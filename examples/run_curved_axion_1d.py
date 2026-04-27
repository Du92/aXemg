"""
Phase 4 example:

Axion evolution on a nontrivial 1D 3+1 background.

The geometry is a toy diagonal metric:

    N = N(x)
    beta^x = beta^x(x)
    gamma_ij = diag(gxx(x), gyy(x), gzz(x))

This example tests the geometric axion terms:

    ∂_t a = N Pi + beta^x ∂_x a

    ∂_t Pi =
        beta^x ∂_x Pi
        + N [
            D_i D^i a
            + K Pi
            - D_i(ln N) D^i a
            - dV/da
        ].

Run with:

    python examples/run_curved_axion_1d.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.boundary import PeriodicBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.diagonal_1d import DiagonalMetric1D
from axion_em_gr.initial_data.axion_profiles import gaussian_axion_packet
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.plots_1d import (
    plot_axion_state,
    plot_geometry_1d,
    plot_history,
)


def main() -> None:
    output_dir = Path("outputs") / "phase4_curved_axion_1d"
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
        dt=0.005,
        t_final=40.0,
        output_every=200,
        derivative_order=2,
    )

    state0 = gaussian_axion_packet(
        grid=grid,
        amplitude=1.0,
        center=35.0,
        width=5.0,
        momentum_amplitude=0.0,
    )

    metric = DiagonalMetric1D(
        lapse_amplitude=0.15,
        metric_amplitude=0.20,
        shift_amplitude=0.05,
        center=50.0,
        width=12.0,
        K_value=0.0,
    )

    potential = MassivePotential(m=physical.m_axion)

    rhs = RHSComputer(
        grid=grid,
        metric=metric,
        potential=potential,
        numerics=numerics,
        physical=physical,
        evolve_axion=True,
        evolve_maxwell=False,
        include_axion_em_coupling=False,
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
            title="Initial axion field on curved 1D background",
            output_dir=output_dir,
            filename="initial_axion_state.png",
        )
    )

    final_state, history = solver.evolve(state0)

    print(
        plot_axion_state(
            grid,
            final_state,
            title="Final axion field on curved 1D background",
            output_dir=output_dir,
            filename="final_axion_state.png",
        )
    )

    print(
        plot_history(
            history,
            output_dir=output_dir,
            filename="history.png",
        )
    )


if __name__ == "__main__":
    main()