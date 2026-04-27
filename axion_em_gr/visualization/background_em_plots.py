"""
Plot utilities for prescribed EM backgrounds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.physics.background_em import BackgroundEMField
from axion_em_gr.visualization.plots_2d import ensure_output_dir


def _extent_2d(grid: Grid) -> tuple[float, float, float, float]:
    (xmin, xmax), (ymin, ymax) = grid.bounds
    return xmin, xmax, ymin, ymax


def plot_background_B_magnitude_2d(
    grid: Grid,
    background: BackgroundEMField,
    t: float,
    output_dir: str | Path,
    filename: str = "background_B_magnitude.png",
) -> Path:
    """
    Plot |B| for a prescribed background.
    """
    output_path = ensure_output_dir(output_dir)

    _, B = background.evaluate(t, grid)

    interior = grid.interior_slices
    B_int = B[(slice(None), *interior)]

    Bmag = np.sqrt(np.sum(B_int**2, axis=0))

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        Bmag.T,
        origin="lower",
        extent=_extent_2d(grid),
        aspect="auto",
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(rf"$|\mathbf{{B}}_{{\rm bg}}|$, $t={t:.3f}$")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$|\mathbf{B}_{\rm bg}|$")

    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def plot_background_EdotB_2d(
    grid: Grid,
    background: BackgroundEMField,
    t: float,
    output_dir: str | Path,
    filename: str = "background_EdotB.png",
) -> Path:
    """
    Plot E.B for a prescribed background in flat space.
    """
    output_path = ensure_output_dir(output_dir)

    E, B = background.evaluate(t, grid)

    interior = grid.interior_slices

    EdotB = E[0] * B[0] + E[1] * B[1] + E[2] * B[2]

    data = EdotB[interior]

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        data.T,
        origin="lower",
        extent=_extent_2d(grid),
        aspect="auto",
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(rf"$\mathbf{{E}}_{{\rm bg}}\cdot\mathbf{{B}}_{{\rm bg}}$, $t={t:.3f}$")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$E_{\rm bg}\cdot B_{\rm bg}$")

    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def plot_background_B_quiver_2d(
    grid: Grid,
    background: BackgroundEMField,
    t: float,
    output_dir: str | Path,
    filename: str = "background_B_quiver.png",
    stride: int = 8,
) -> Path:
    """
    Plot in-plane B vector as quiver with |B| background.
    """
    output_path = ensure_output_dir(output_dir)

    X, Y = grid.coordinates_2d()
    _, B = background.evaluate(t, grid)

    interior = grid.interior_slices

    X_int = X[interior]
    Y_int = Y[interior]

    Bx = B[0][interior]
    By = B[1][interior]

    Bmag = np.sqrt(B[0][interior] ** 2 + B[1][interior] ** 2 + B[2][interior] ** 2)

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        Bmag.T,
        origin="lower",
        extent=_extent_2d(grid),
        aspect="auto",
    )

    max_b = float(np.max(np.sqrt(Bx**2 + By**2)))

    if max_b > 1.0e-14:
        ax.quiver(
            X_int[::stride, ::stride],
            Y_int[::stride, ::stride],
            Bx[::stride, ::stride],
            By[::stride, ::stride],
            angles="xy",
            scale_units="xy",
            scale=max_b,
        )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(rf"In-plane $\mathbf{{B}}_{{\rm bg}}$, $t={t:.3f}$")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$|\mathbf{B}_{\rm bg}|$")

    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path