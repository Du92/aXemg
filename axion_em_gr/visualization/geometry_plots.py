"""
Geometry plotting utilities.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.geometry.base_metric import GeometryFields
from axion_em_gr.visualization.plots_2d import ensure_output_dir


def plot_geometry_1d(
    grid: Grid,
    geom: GeometryFields,
    output_dir: str | Path,
    filename: str = "geometry_1d.png",
    title: str = "1D geometry",
) -> Path:
    """
    Plot lapse, sqrt(gamma), gamma_xx and gamma^xx in 1D.
    """
    if grid.ndim != 1:
        raise ValueError("plot_geometry_1d requires a 1D grid.")

    output_path = ensure_output_dir(output_dir)

    x = grid.coordinates_1d()[grid.interior_slices]

    lapse = geom.lapse[grid.interior_slices]
    sqrt_gamma = geom.sqrt_gamma[grid.interior_slices]
    gamma_xx = geom.gamma_down[0, 0][grid.interior_slices]
    gamma_up_xx = geom.gamma_up[0, 0][grid.interior_slices]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(x, lapse, label=r"$\alpha$")
    ax.plot(x, sqrt_gamma, label=r"$\sqrt{\gamma}$")
    ax.plot(x, gamma_xx, label=r"$\gamma_{xx}$")
    ax.plot(x, gamma_up_xx, label=r"$\gamma^{xx}$")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel("geometry field")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def _extent_2d(grid: Grid) -> tuple[float, float, float, float]:
    (xmin, xmax), (ymin, ymax) = grid.bounds
    return xmin, xmax, ymin, ymax


def plot_geometry_scalar_2d(
    grid: Grid,
    scalar,
    output_dir: str | Path,
    filename: str,
    title: str,
    label: str,
) -> Path:
    """
    Generic 2D geometry scalar heatmap.
    """
    if grid.ndim != 2:
        raise ValueError("plot_geometry_scalar_2d requires a 2D grid.")

    output_path = ensure_output_dir(output_dir)

    data = scalar[grid.interior_slices]

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


def plot_geometry_2d(
    grid: Grid,
    geom: GeometryFields,
    output_dir: str | Path,
    prefix: str = "geometry",
) -> list[Path]:
    """
    Save standard 2D geometry maps.
    """
    paths = []

    paths.append(
        plot_geometry_scalar_2d(
            grid=grid,
            scalar=geom.lapse,
            output_dir=output_dir,
            filename=f"{prefix}_lapse.png",
            title=r"Lapse $\alpha$",
            label=r"$\alpha$",
        )
    )

    paths.append(
        plot_geometry_scalar_2d(
            grid=grid,
            scalar=geom.sqrt_gamma,
            output_dir=output_dir,
            filename=f"{prefix}_sqrt_gamma.png",
            title=r"$\sqrt{\gamma}$",
            label=r"$\sqrt{\gamma}$",
        )
    )

    paths.append(
        plot_geometry_scalar_2d(
            grid=grid,
            scalar=geom.gamma_down[0, 0],
            output_dir=output_dir,
            filename=f"{prefix}_gamma_xx.png",
            title=r"$\gamma_{xx}$",
            label=r"$\gamma_{xx}$",
        )
    )

    paths.append(
        plot_geometry_scalar_2d(
            grid=grid,
            scalar=geom.gamma_up[0, 0],
            output_dir=output_dir,
            filename=f"{prefix}_gamma_up_xx.png",
            title=r"$\gamma^{xx}$",
            label=r"$\gamma^{xx}$",
        )
    )

    return paths