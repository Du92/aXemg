"""
Minimal Phase 9 animation script.

Creates only the most important animations:

    a(t,x,y)
    E_i B^i(t,x,y)
    rho_EM(t,x,y)

Run with:

    python examples/run_2d_animations_minimal.py
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.grid import Grid
from axion_em_gr.io.snapshot import list_snapshot_files
from axion_em_gr.visualization.animations_2d import (
    animate_EdotB,
    animate_axion_field,
    animate_em_energy_density,
    ensure_output_dir,
)


def main() -> None:
    snapshot_dir = Path("outputs") / "phase8_2d_snapshots" / "snapshots"
    output_dir = ensure_output_dir(Path("outputs") / "phase9_animations_2d_minimal")

    snapshot_files = list_snapshot_files(
        snapshot_dir=snapshot_dir,
        pattern="*.npz",
    )

    grid = Grid(
        ndim=2,
        shape=(256, 256),
        bounds=((-50.0, 50.0), (-50.0, 50.0)),
        nghost=3,
    )

    print(f"Found {len(snapshot_files)} snapshots")

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
        animate_EdotB(
            grid=grid,
            snapshot_files=snapshot_files,
            output_dir=output_dir,
            filename="EdotB.mp4",
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


if __name__ == "__main__":
    main()