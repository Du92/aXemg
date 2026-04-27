"""
High-level 2D diagnostic plotting routines.
"""

from __future__ import annotations

from pathlib import Path

from axion_em_gr.core.grid import Grid
from axion_em_gr.core.state import State
from axion_em_gr.geometry.base_metric import GeometryFields
from axion_em_gr.physics.constraints import electric_constraint, magnetic_constraint
from axion_em_gr.physics.diagnostics import (
    axion_energy_density_flat,
    edotb_density,
    electromagnetic_energy_density_flat,
    poynting_vector_flat,
    radial_flux_density_2d,
)
from axion_em_gr.physics.potentials import AxionPotential
from axion_em_gr.physics.slices import extract_x_slice, extract_y_slice
from axion_em_gr.physics.sources import SourceModel
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.visualization.plots_2d import (
    ensure_output_dir,
    plot_EdotB_2d,
    plot_constraint_heatmaps_2d,
    plot_energy_densities_2d,
    plot_multiple_slices_1d,
    plot_poynting_2d,
    plot_radial_flux_2d,
    plot_scalar_heatmap_2d,
)


def make_full_2d_diagnostic_report(
    grid: Grid,
    state: State,
    geom: GeometryFields,
    potential: AxionPotential,
    sources: SourceModel,
    numerics: NumericalParameters,
    physical: PhysicalParameters,
    time: float,
    output_dir: str | Path,
    prefix: str = "diagnostics",
    center: tuple[float, float] = (0.0, 0.0),
) -> list[Path]:
    """
    Generate a complete set of 2D diagnostic plots for one state.
    """
    if grid.ndim != 2:
        raise ValueError("make_full_2d_diagnostic_report requires a 2D grid.")

    output_path = ensure_output_dir(output_dir)

    paths: list[Path] = []

    rho_a = axion_energy_density_flat(
        state=state,
        grid=grid,
        potential=potential,
    )

    rho_em = electromagnetic_energy_density_flat(
        state=state,
        geom=geom,
    )

    paths.extend(
        plot_energy_densities_2d(
            grid=grid,
            rho_axion=rho_a,
            rho_em=rho_em,
            output_dir=output_path,
            prefix=f"{prefix}_energy",
        )
    )

    paths.append(
        plot_EdotB_2d(
            grid=grid,
            state=state,
            geom=geom,
            output_dir=output_path,
            filename=f"{prefix}_EdotB.png",
        )
    )

    EdotB = edotb_density(
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

    paths.extend(
        plot_constraint_heatmaps_2d(
            grid=grid,
            div_B=div_B,
            div_E=div_E,
            output_dir=output_path,
            prefix=f"{prefix}_constraints",
        )
    )

    if state.E is not None and state.B is not None:
        S = poynting_vector_flat(state)

        paths.append(
            plot_poynting_2d(
                grid=grid,
                S=S,
                output_dir=output_path,
                filename=f"{prefix}_poynting.png",
                stride=8,
                title=rf"Poynting vector at $t={time:.3f}$",
            )
        )

        S_r = radial_flux_density_2d(
            state=state,
            grid=grid,
            center=center,
        )

        paths.append(
            plot_radial_flux_2d(
                grid=grid,
                radial_flux=S_r,
                output_dir=output_path,
                filename=f"{prefix}_radial_flux.png",
            )
        )

    # Slices through y=0 and x=0.
    x_slice_a = extract_x_slice(grid, state.a, y_value=center[1])
    y_slice_a = extract_y_slice(grid, state.a, x_value=center[0])

    paths.append(
        plot_multiple_slices_1d(
            slices=[x_slice_a, y_slice_a],
            labels=[
                rf"$a(x,y={x_slice_a.fixed_value:.2f})$",
                rf"$a(x={y_slice_a.fixed_value:.2f},y)$",
            ],
            output_dir=output_path,
            filename=f"{prefix}_axion_center_slices.png",
            title=rf"Central axion slices at $t={time:.3f}$",
            ylabel=r"$a$",
        )
    )

    x_slice_EdotB = extract_x_slice(grid, EdotB, y_value=center[1])
    y_slice_EdotB = extract_y_slice(grid, EdotB, x_value=center[0])

    paths.append(
        plot_multiple_slices_1d(
            slices=[x_slice_EdotB, y_slice_EdotB],
            labels=[
                rf"$E_iB^i(x,y={x_slice_EdotB.fixed_value:.2f})$",
                rf"$E_iB^i(x={y_slice_EdotB.fixed_value:.2f},y)$",
            ],
            output_dir=output_path,
            filename=f"{prefix}_EdotB_center_slices.png",
            title=rf"Central $E_iB^i$ slices at $t={time:.3f}$",
            ylabel=r"$E_iB^i$",
        )
    )

    paths.append(
        plot_scalar_heatmap_2d(
            grid=grid,
            scalar=rho_a + rho_em,
            title=rf"Total diagnostic energy density at $t={time:.3f}$",
            label=r"$\rho_a+\rho_{\rm EM}$",
            output_dir=output_path,
            filename=f"{prefix}_rho_total.png",
        )
    )

    return paths