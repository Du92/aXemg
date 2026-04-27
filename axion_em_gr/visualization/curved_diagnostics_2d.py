"""
Curved 2D diagnostic plots.
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.state import State
from axion_em_gr.geometry.base_metric import GeometryFields
from axion_em_gr.physics.constraints import electric_constraint, magnetic_constraint
from axion_em_gr.physics.diagnostics import (
    axion_energy_density_geometry,
    edotb_density,
    electromagnetic_energy_density_flat,
)
from axion_em_gr.physics.potentials import AxionPotential
from axion_em_gr.physics.sources import SourceModel
from axion_em_gr.visualization.plots_2d import (
    ensure_output_dir,
    plot_EdotB_2d,
    plot_constraint_heatmaps_2d,
    plot_scalar_heatmap_2d,
)


def make_curved_2d_diagnostic_report(
    grid: Grid,
    state: State,
    geom: GeometryFields,
    potential: AxionPotential,
    sources: SourceModel,
    numerics: NumericalParameters,
    physical: PhysicalParameters,
    time: float,
    output_dir: str | Path,
    prefix: str = "curved",
) -> list[Path]:
    """
    Generate diagnostic maps using geometry-aware scalar quantities.
    """
    if grid.ndim != 2:
        raise ValueError("make_curved_2d_diagnostic_report requires 2D.")

    output_path = ensure_output_dir(output_dir)
    paths: list[Path] = []

    rho_a_geom = axion_energy_density_geometry(
        state=state,
        grid=grid,
        geom=geom,
        potential=potential,
    )

    rho_em = electromagnetic_energy_density_flat(
        state=state,
        geom=geom,
    )

    edotb = edotb_density(
        state=state,
        geom=geom,
    )

    div_B = magnetic_constraint(
        state=state,
        grid=grid,
        geom=geom,
        numerics=numerics,
    )

    div_E = electric_constraint(
        state=state,
        t=time,
        grid=grid,
        geom=geom,
        sources=sources,
        numerics=numerics,
        physical=physical,
        include_axion_coupling=True,
    )

    paths.append(
        plot_scalar_heatmap_2d(
            grid=grid,
            scalar=rho_a_geom,
            title=rf"Geometry-aware axion energy density at $t={time:.3f}$",
            label=r"$\rho_a^{(\gamma)}$",
            output_dir=output_path,
            filename=f"{prefix}_rho_axion_geometry.png",
        )
    )

    paths.append(
        plot_scalar_heatmap_2d(
            grid=grid,
            scalar=rho_em,
            title=rf"EM energy density at $t={time:.3f}$",
            label=r"$\rho_{\rm EM}$",
            output_dir=output_path,
            filename=f"{prefix}_rho_em_geometry.png",
        )
    )

    paths.append(
        plot_scalar_heatmap_2d(
            grid=grid,
            scalar=geom.sqrt_gamma,
            title=r"Volume factor $\sqrt{\gamma}$",
            label=r"$\sqrt{\gamma}$",
            output_dir=output_path,
            filename=f"{prefix}_sqrt_gamma.png",
        )
    )

    paths.append(
        plot_EdotB_2d(
            grid=grid,
            state=state,
            geom=geom,
            output_dir=output_path,
            filename=f"{prefix}_EdotB_geometry.png",
        )
    )

    paths.extend(
        plot_constraint_heatmaps_2d(
            grid=grid,
            div_B=div_B,
            div_E=div_E,
            output_dir=output_path,
            prefix=f"{prefix}_constraints",
        )
    )

    paths.append(
        plot_scalar_heatmap_2d(
            grid=grid,
            scalar=edotb * geom.sqrt_gamma,
            title=rf"$\sqrt{{\gamma}} E_iB^i$ at $t={time:.3f}$",
            label=r"$\sqrt{\gamma}E_iB^i$",
            output_dir=output_path,
            filename=f"{prefix}_sqrtgamma_EdotB.png",
        )
    )

    return paths