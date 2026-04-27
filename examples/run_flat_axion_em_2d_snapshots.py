"""
Phase 8 auxiliary example:

Run a 2D simulation and save snapshots for later animations.

Run with:

    python examples/run_flat_axion_em_2d_snapshots.py
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
from axion_em_gr.io.snapshot import save_history_snapshots
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.visualization.plots_2d import ensure_output_dir


def main() -> None:
    output_dir = ensure_output_dir(Path("outputs") / "phase8_2d_snapshots")
    snapshot_dir = ensure_output_dir(output_dir / "snapshots")

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
        save_snapshots=True,
        snapshot_every=100,
    )

    final_state, history = solver.evolve(state0)

    paths = save_history_snapshots(
        history=history,
        output_dir=snapshot_dir,
        prefix="snapshot_2d",
    )

    print(f"Saved {len(paths)} snapshots in {snapshot_dir}")


if __name__ == "__main__":
    main()