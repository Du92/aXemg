"""
Phase 9 example:

Create 2D animations from saved snapshots.

First run:

    python examples/run_flat_axion_em_2d_snapshots.py

Then run:

    python examples/run_2d_animations.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import PhysicalParameters
from axion_em_gr.io.snapshot import list_snapshot_files
from axion_em_gr.physics.potentials import MassivePotential
from axion_em_gr.visualization.animations_2d import (
    animate_EdotB,
    animate_axion_energy_density,
    animate_axion_field,
    animate_axion_momentum,
    animate_em_component,
    animate_em_energy_density,
    animate_poynting_quiver,
    animate_radial_flux,
    ensure_output_dir,
)


def main() -> None:
    snapshot_dir = Path("outputs") / "phase8_2d_snapshots" / "snapshots"
    output_dir = ensure_output_dir(Path("outputs") / "phase9_animations_2d")

    snapshot_files = list_snapshot_files(
        snapshot_dir=snapshot_dir,
        pattern="*.npz",
    )

    print(f"Found {len(snapshot_files)} snapshots")

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

    potential = MassivePotential(m=physical.m_axion)

    print(
        animate_axion_field(
            grid=grid,
            snapshot_files=snapshot_files,
            output_dir=output_dir,
            filename="axion_a.mp4",
            fps=15,
        )
    )

    print(
        animate_axion_momentum(
            grid=grid,
            snapshot_files=snapshot_files,
            output_dir=output_dir,
            filename="axion_Pi.mp4",
            fps=15,
        )
    )

    print(
        animate_EdotB(
            grid=grid,
            snapshot_files=snapshot_files,
            output_dir=output_dir,
            filename="EdotB.mp4",
            fps=15,
        )
    )

    print(
        animate_axion_energy_density(
            grid=grid,
            snapshot_files=snapshot_files,
            potential=potential,
            output_dir=output_dir,
            filename="rho_axion.mp4",
            fps=15,
        )
    )

    print(
        animate_em_energy_density(
            grid=grid,
            snapshot_files=snapshot_files,
            output_dir=output_dir,
            filename="rho_em.mp4",
            fps=15,
        )
    )

    print(
        animate_radial_flux(
            grid=grid,
            snapshot_files=snapshot_files,
            output_dir=output_dir,
            filename="radial_flux.mp4",
            fps=15,
            center=(0.0, 0.0),
        )
    )

    print(
        animate_em_component(
            grid=grid,
            snapshot_files=snapshot_files,
            field="E",
            component=2,
            output_dir=output_dir,
            filename="Ez.mp4",
            fps=15,
        )
    )

    print(
        animate_em_component(
            grid=grid,
            snapshot_files=snapshot_files,
            field="B",
            component=2,
            output_dir=output_dir,
            filename="Bz.mp4",
            fps=15,
        )
    )

    print(
        animate_poynting_quiver(
            grid=grid,
            snapshot_files=snapshot_files,
            output_path=output_dir / "poynting_quiver.mp4",
            fps=15,
            dpi=150,
            stride=10,
        )
    )


if __name__ == "__main__":
    main()