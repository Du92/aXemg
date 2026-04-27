"""
2D plotting utilities.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.core.state import State
from axion_em_gr.core.tensors import contract_cov_contra, lower_vector


def ensure_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _extent_2d(grid: Grid) -> tuple[float, float, float, float]:
    """
    Matplotlib extent for imshow.

    Returns:
        (xmin, xmax, ymin, ymax)
    """
    (xmin, xmax), (ymin, ymax) = grid.bounds
    return xmin, xmax, ymin, ymax


def plot_scalar_heatmap_2d(
    grid: Grid,
    scalar: np.ndarray,
    title: str,
    label: str,
    output_dir: str | Path = "outputs",
    filename: str = "scalar_2d.png",
) -> Path:
    """
    Save a 2D scalar heatmap over the interior domain.
    """
    if grid.ndim != 2:
        raise ValueError("plot_scalar_heatmap_2d requires a 2D grid.")

    output_path = ensure_output_dir(output_dir)

    interior = grid.interior_slices
    data = scalar[interior]

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        data.T,
        origin="lower",
        extent=_extent_2d(grid),
        aspect="auto",
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)

    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def plot_axion_state_2d(
    grid: Grid,
    state: State,
    output_dir: str | Path = "outputs",
    prefix: str = "axion",
) -> list[Path]:
    """
    Save heatmaps of a and Pi.
    """
    paths = []

    paths.append(
        plot_scalar_heatmap_2d(
            grid=grid,
            scalar=state.a,
            title=r"Axion field $a(x,y)$",
            label=r"$a$",
            output_dir=output_dir,
            filename=f"{prefix}_a.png",
        )
    )

    paths.append(
        plot_scalar_heatmap_2d(
            grid=grid,
            scalar=state.Pi,
            title=r"Axion momentum $\Pi(x,y)$",
            label=r"$\Pi$",
            output_dir=output_dir,
            filename=f"{prefix}_Pi.png",
        )
    )

    return paths


def plot_em_component_2d(
    grid: Grid,
    state: State,
    field: str,
    component: int,
    output_dir: str | Path = "outputs",
    filename: str = "em_component.png",
) -> Path:
    """
    Save heatmap of one EM component.

    field:
        "E" or "B"

    component:
        0, 1, or 2.
    """
    if state.E is None or state.B is None:
        raise ValueError("State must contain E and B.")

    component_labels = ["x", "y", "z"]

    if field == "E":
        scalar = state.E[component]
        label = rf"$E^{{{component_labels[component]}}}$"
    elif field == "B":
        scalar = state.B[component]
        label = rf"$B^{{{component_labels[component]}}}$"
    else:
        raise ValueError("field must be 'E' or 'B'.")

    return plot_scalar_heatmap_2d(
        grid=grid,
        scalar=scalar,
        title=label,
        label=label,
        output_dir=output_dir,
        filename=filename,
    )


def plot_em_summary_2d(
    grid: Grid,
    state: State,
    output_dir: str | Path = "outputs",
    prefix: str = "em",
) -> list[Path]:
    """
    Save heatmaps for all EM components.
    """
    paths = []
    component_labels = ["x", "y", "z"]

    for i in range(3):
        paths.append(
            plot_em_component_2d(
                grid=grid,
                state=state,
                field="E",
                component=i,
                output_dir=output_dir,
                filename=f"{prefix}_E{component_labels[i]}.png",
            )
        )

    for i in range(3):
        paths.append(
            plot_em_component_2d(
                grid=grid,
                state=state,
                field="B",
                component=i,
                output_dir=output_dir,
                filename=f"{prefix}_B{component_labels[i]}.png",
            )
        )

    return paths


def plot_EdotB_2d(
    grid: Grid,
    state: State,
    geom,
    output_dir: str | Path = "outputs",
    filename: str = "EdotB_2d.png",
) -> Path:
    """
    Save heatmap of E_i B^i.
    """
    if state.E is None or state.B is None:
        raise ValueError("State must contain E and B.")

    E_down = lower_vector(state.E, geom.gamma_down)
    EdotB = contract_cov_contra(E_down, state.B)

    return plot_scalar_heatmap_2d(
        grid=grid,
        scalar=EdotB,
        title=r"$E_iB^i$",
        label=r"$E_iB^i$",
        output_dir=output_dir,
        filename=filename,
    )


def plot_constraint_heatmaps_2d(
    grid: Grid,
    div_B: np.ndarray,
    div_E: np.ndarray,
    output_dir: str | Path = "outputs",
    prefix: str = "constraints",
) -> list[Path]:
    """
    Save heatmaps of divB and divE constraints.
    """
    paths = []

    paths.append(
        plot_scalar_heatmap_2d(
            grid=grid,
            scalar=div_B,
            title=r"$\partial_iB^i$",
            label=r"$\partial_iB^i$",
            output_dir=output_dir,
            filename=f"{prefix}_divB.png",
        )
    )

    paths.append(
        plot_scalar_heatmap_2d(
            grid=grid,
            scalar=div_E,
            title=r"$\partial_iE^i - \rho + gB^i\partial_i a$",
            label=r"$C_E$",
            output_dir=output_dir,
            filename=f"{prefix}_divE.png",
        )
    )

    return paths

def plot_energy_densities_2d(
    grid: Grid,
    rho_axion,
    rho_em,
    output_dir: str | Path = "outputs",
    prefix: str = "energy",
) -> list[Path]:
    """
    Save axion and electromagnetic energy-density heatmaps.
    """
    paths = []

    paths.append(
        plot_scalar_heatmap_2d(
            grid=grid,
            scalar=rho_axion,
            title=r"Axion energy density $\rho_a$",
            label=r"$\rho_a$",
            output_dir=output_dir,
            filename=f"{prefix}_rho_axion.png",
        )
    )

    paths.append(
        plot_scalar_heatmap_2d(
            grid=grid,
            scalar=rho_em,
            title=r"Electromagnetic energy density $\rho_{\rm EM}$",
            label=r"$\rho_{\rm EM}$",
            output_dir=output_dir,
            filename=f"{prefix}_rho_em.png",
        )
    )

    return paths


def plot_poynting_2d(
    grid: Grid,
    S,
    output_dir: str | Path = "outputs",
    filename: str = "poynting_2d.png",
    stride: int = 8,
    title: str = "Poynting vector in the x-y plane",
) -> Path:
    """
    Plot the in-plane Poynting vector (S^x, S^y) as a quiver plot,
    with |S_xy| as background.

    If the in-plane vector field is numerically zero, the function saves
    only the background heatmap and skips quiver arrows to avoid Matplotlib
    autoscaling warnings.
    """
    if grid.ndim != 2:
        raise ValueError("plot_poynting_2d requires a 2D grid.")

    output_path = ensure_output_dir(output_dir)

    X, Y = grid.coordinates_2d()
    interior = grid.interior_slices

    X_int = X[interior]
    Y_int = Y[interior]

    Sx = S[0][interior]
    Sy = S[1][interior]

    Smag = np.sqrt(Sx**2 + Sy**2)
    max_smag = float(np.max(Smag))

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        Smag.T,
        origin="lower",
        extent=_extent_2d(grid),
        aspect="auto",
    )

    if max_smag > 1.0e-14:
        ax.quiver(
            X_int[::stride, ::stride],
            Y_int[::stride, ::stride],
            Sx[::stride, ::stride],
            Sy[::stride, ::stride],
            angles="xy",
            scale_units="xy",
            scale=max_smag,
        )
    else:
        ax.text(
            0.5,
            1.02,
            r"in-plane $S^x,S^y$ numerically zero",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\sqrt{(S^x)^2+(S^y)^2}$")

    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def plot_radial_flux_2d(
    grid: Grid,
    radial_flux,
    output_dir: str | Path = "outputs",
    filename: str = "radial_flux_2d.png",
) -> Path:
    """
    Save heatmap of approximate radial Poynting flux S_r.
    """
    return plot_scalar_heatmap_2d(
        grid=grid,
        scalar=radial_flux,
        title=r"Radial Poynting flux $S_r$",
        label=r"$S_r$",
        output_dir=output_dir,
        filename=filename,
    )


def plot_slice_1d(
    slice_data,
    output_dir: str | Path = "outputs",
    filename: str = "slice_1d.png",
    title: str = "1D slice",
    ylabel: str = "field",
) -> Path:
    """
    Plot a 1D slice extracted from a 2D field.
    """
    output_path = ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(slice_data.coordinate, slice_data.values)

    if slice_data.fixed_axis == "y":
        ax.set_xlabel(r"$x$")
        label = rf"$y={slice_data.fixed_value:.3f}$"
    else:
        ax.set_xlabel(r"$y$")
        label = rf"$x={slice_data.fixed_value:.3f}$"

    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} ({label})")

    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def plot_multiple_slices_1d(
    slices,
    labels,
    output_dir: str | Path = "outputs",
    filename: str = "multiple_slices_1d.png",
    title: str = "1D slices",
    ylabel: str = "field",
) -> Path:
    """
    Plot several 1D slices on the same figure.
    """
    output_path = ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(8, 4))

    for slice_data, label in zip(slices, labels):
        ax.plot(slice_data.coordinate, slice_data.values, label=label)

    first = slices[0]

    if first.fixed_axis == "y":
        ax.set_xlabel(r"$x$")
    else:
        ax.set_xlabel(r"$y$")

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path