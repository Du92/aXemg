"""
2D animation utilities.

These functions build animations from saved .npz snapshots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.io.snapshot import load_state_npz
from axion_em_gr.physics.diagnostics import (
    axion_energy_density_flat,
    edotb_density,
    electromagnetic_energy_density_flat,
    poynting_vector_flat,
    radial_flux_density_2d,
)
from axion_em_gr.physics.potentials import AxionPotential


def ensure_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _extent_2d(grid: Grid) -> tuple[float, float, float, float]:
    """
    Matplotlib imshow extent.
    """
    (xmin, xmax), (ymin, ymax) = grid.bounds
    return xmin, xmax, ymin, ymax


def _interior_data(grid: Grid, array: np.ndarray) -> np.ndarray:
    """
    Return interior data for scalar arrays.
    """
    return array[grid.interior_slices]


def _compute_global_vmin_vmax(
    snapshot_files: list[Path],
    field_getter: Callable,
    symmetric: bool = False,
) -> tuple[float, float]:
    """
    Compute global color scale over all snapshots.
    """
    values_min = []
    values_max = []

    for path in snapshot_files:
        state, time = load_state_npz(path)
        data = field_getter(state, time)

        values_min.append(float(np.nanmin(data)))
        values_max.append(float(np.nanmax(data)))

    vmin = min(values_min)
    vmax = max(values_max)

    if symmetric:
        amp = max(abs(vmin), abs(vmax))
        return -amp, amp

    return vmin, vmax


def animate_scalar_from_snapshots(
    grid: Grid,
    snapshot_files: list[Path],
    field_getter: Callable,
    output_path: str | Path,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    fps: int = 15,
    dpi: int = 150,
    symmetric: bool = False,
    fixed_color_scale: bool = True,
) -> Path:
    """
    Generic scalar-field animation from snapshots.

    Parameters
    ----------
    grid:
        Numerical grid.
    snapshot_files:
        List of snapshot files.
    field_getter:
        Function with signature field_getter(state, time) -> scalar array.
    output_path:
        Output .mp4 or .gif path.
    title:
        Figure title prefix.
    colorbar_label:
        Label for colorbar.
    cmap:
        Matplotlib colormap.
    fps:
        Frames per second.
    symmetric:
        If True, use symmetric color scale around zero.
    fixed_color_scale:
        If True, compute one global color scale for all frames.
    """
    if grid.ndim != 2:
        raise ValueError("animate_scalar_from_snapshots requires a 2D grid.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_state, first_time = load_state_npz(snapshot_files[0])
    first_data = _interior_data(grid, field_getter(first_state, first_time))

    if fixed_color_scale:
        vmin, vmax = _compute_global_vmin_vmax(
            snapshot_files=snapshot_files,
            field_getter=lambda state, time: _interior_data(
                grid,
                field_getter(state, time),
            ),
            symmetric=symmetric,
        )
    else:
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        first_data.T,
        origin="lower",
        extent=_extent_2d(grid),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(f"{title}, t={first_time:.3f}")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    fig.tight_layout()

    def update(frame_index: int):
        state, time = load_state_npz(snapshot_files[frame_index])
        data = _interior_data(grid, field_getter(state, time))

        im.set_data(data.T)

        if not fixed_color_scale:
            if symmetric:
                amp = float(np.max(np.abs(data)))
                im.set_clim(-amp, amp)
            else:
                im.set_clim(float(np.min(data)), float(np.max(data)))

        ax.set_title(f"{title}, t={time:.3f}")

        return (im,)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(snapshot_files),
        interval=1000 / fps,
        blit=False,
    )

    if output_path.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi)
    else:
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi)

    plt.close(fig)

    return output_path


def animate_axion_field(
    grid: Grid,
    snapshot_files: list[Path],
    output_dir: str | Path,
    filename: str = "axion_a.mp4",
    fps: int = 15,
) -> Path:
    """
    Animate the axion field a(t,x,y).
    """
    output_dir = ensure_output_dir(output_dir)

    return animate_scalar_from_snapshots(
        grid=grid,
        snapshot_files=snapshot_files,
        field_getter=lambda state, time: state.a,
        output_path=output_dir / filename,
        title=r"Axion field $a(t,x,y)$",
        colorbar_label=r"$a$",
        cmap="viridis",
        fps=fps,
        symmetric=True,
        fixed_color_scale=True,
    )


def animate_axion_momentum(
    grid: Grid,
    snapshot_files: list[Path],
    output_dir: str | Path,
    filename: str = "axion_Pi.mp4",
    fps: int = 15,
) -> Path:
    """
    Animate the axion momentum Pi(t,x,y).
    """
    output_dir = ensure_output_dir(output_dir)

    return animate_scalar_from_snapshots(
        grid=grid,
        snapshot_files=snapshot_files,
        field_getter=lambda state, time: state.Pi,
        output_path=output_dir / filename,
        title=r"Axion momentum $\Pi(t,x,y)$",
        colorbar_label=r"$\Pi$",
        cmap="viridis",
        fps=fps,
        symmetric=True,
        fixed_color_scale=True,
    )


def animate_EdotB(
    grid: Grid,
    snapshot_files: list[Path],
    output_dir: str | Path,
    filename: str = "EdotB.mp4",
    fps: int = 15,
) -> Path:
    """
    Animate E_i B^i.

    For Phase 9 we assume flat metric.
    """
    output_dir = ensure_output_dir(output_dir)
    metric = FlatMetric()

    def getter(state, time):
        geom = metric.evaluate(time, grid)
        return edotb_density(state, geom)

    return animate_scalar_from_snapshots(
        grid=grid,
        snapshot_files=snapshot_files,
        field_getter=getter,
        output_path=output_dir / filename,
        title=r"$E_iB^i(t,x,y)$",
        colorbar_label=r"$E_iB^i$",
        cmap="coolwarm",
        fps=fps,
        symmetric=True,
        fixed_color_scale=True,
    )


def animate_axion_energy_density(
    grid: Grid,
    snapshot_files: list[Path],
    potential: AxionPotential,
    output_dir: str | Path,
    filename: str = "rho_axion.mp4",
    fps: int = 15,
) -> Path:
    """
    Animate axion energy density.
    """
    output_dir = ensure_output_dir(output_dir)

    def getter(state, time):
        return axion_energy_density_flat(
            state=state,
            grid=grid,
            potential=potential,
        )

    return animate_scalar_from_snapshots(
        grid=grid,
        snapshot_files=snapshot_files,
        field_getter=getter,
        output_path=output_dir / filename,
        title=r"Axion energy density $\rho_a(t,x,y)$",
        colorbar_label=r"$\rho_a$",
        cmap="magma",
        fps=fps,
        symmetric=False,
        fixed_color_scale=True,
    )


def animate_em_energy_density(
    grid: Grid,
    snapshot_files: list[Path],
    output_dir: str | Path,
    filename: str = "rho_em.mp4",
    fps: int = 15,
) -> Path:
    """
    Animate electromagnetic energy density.
    """
    output_dir = ensure_output_dir(output_dir)
    metric = FlatMetric()

    def getter(state, time):
        geom = metric.evaluate(time, grid)
        return electromagnetic_energy_density_flat(
            state=state,
            geom=geom,
        )

    return animate_scalar_from_snapshots(
        grid=grid,
        snapshot_files=snapshot_files,
        field_getter=getter,
        output_path=output_dir / filename,
        title=r"Electromagnetic energy density $\rho_{\rm EM}(t,x,y)$",
        colorbar_label=r"$\rho_{\rm EM}$",
        cmap="plasma",
        fps=fps,
        symmetric=False,
        fixed_color_scale=True,
    )


def animate_radial_flux(
    grid: Grid,
    snapshot_files: list[Path],
    output_dir: str | Path,
    filename: str = "radial_flux.mp4",
    fps: int = 15,
    center: tuple[float, float] = (0.0, 0.0),
) -> Path:
    """
    Animate approximate radial Poynting flux S_r.
    """
    output_dir = ensure_output_dir(output_dir)

    def getter(state, time):
        return radial_flux_density_2d(
            state=state,
            grid=grid,
            center=center,
        )

    return animate_scalar_from_snapshots(
        grid=grid,
        snapshot_files=snapshot_files,
        field_getter=getter,
        output_path=output_dir / filename,
        title=r"Radial Poynting flux $S_r(t,x,y)$",
        colorbar_label=r"$S_r$",
        cmap="coolwarm",
        fps=fps,
        symmetric=True,
        fixed_color_scale=True,
    )


def animate_em_component(
    grid: Grid,
    snapshot_files: list[Path],
    field: str,
    component: int,
    output_dir: str | Path,
    filename: str,
    fps: int = 15,
) -> Path:
    """
    Animate one electromagnetic component.

    field:
        "E" or "B"

    component:
        0, 1, or 2.
    """
    output_dir = ensure_output_dir(output_dir)

    component_labels = ["x", "y", "z"]

    if field not in ("E", "B"):
        raise ValueError("field must be 'E' or 'B'.")

    if component not in (0, 1, 2):
        raise ValueError("component must be 0, 1, or 2.")

    def getter(state, time):
        if field == "E":
            if state.E is None:
                raise ValueError("Snapshot has no E field.")
            return state.E[component]

        if state.B is None:
            raise ValueError("Snapshot has no B field.")
        return state.B[component]

    label = rf"${field}^{{{component_labels[component]}}}$"

    return animate_scalar_from_snapshots(
        grid=grid,
        snapshot_files=snapshot_files,
        field_getter=getter,
        output_path=output_dir / filename,
        title=rf"{label}(t,x,y)",
        colorbar_label=label,
        cmap="viridis",
        fps=fps,
        symmetric=True,
        fixed_color_scale=True,
    )


def animate_poynting_quiver(
    grid: Grid,
    snapshot_files: list[Path],
    output_path: str | Path,
    fps: int = 15,
    dpi: int = 150,
    stride: int = 8,
) -> Path:
    """
    Animate the in-plane Poynting vector (S^x, S^y) with |S_xy| background.

    This is more expensive than scalar animations.
    """
    if grid.ndim != 2:
        raise ValueError("animate_poynting_quiver requires a 2D grid.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    X, Y = grid.coordinates_2d()
    interior = grid.interior_slices

    X_int = X[interior]
    Y_int = Y[interior]

    first_state, first_time = load_state_npz(snapshot_files[0])
    S0 = poynting_vector_flat(first_state)

    Sx0 = S0[0][interior]
    Sy0 = S0[1][interior]
    Smag0 = np.sqrt(Sx0**2 + Sy0**2)

    global_max = 0.0

    for path in snapshot_files:
        state, _ = load_state_npz(path)
        S = poynting_vector_flat(state)
        Sx = S[0][interior]
        Sy = S[1][interior]
        Smag = np.sqrt(Sx**2 + Sy**2)
        global_max = max(global_max, float(np.max(Smag)))

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        Smag0.T,
        origin="lower",
        extent=_extent_2d(grid),
        aspect="auto",
        vmin=0.0,
        vmax=max(global_max, 1.0e-14),
        cmap="plasma",
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(rf"Poynting vector, $t={first_time:.3f}$")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\sqrt{(S^x)^2+(S^y)^2}$")

    if global_max > 1.0e-14:
        quiv = ax.quiver(
            X_int[::stride, ::stride],
            Y_int[::stride, ::stride],
            Sx0[::stride, ::stride],
            Sy0[::stride, ::stride],
            angles="xy",
            scale_units="xy",
            scale=global_max,
        )
    else:
        quiv = None
        ax.text(
            0.5,
            1.02,
            r"in-plane $S^x,S^y$ numerically zero",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    def update(frame_index: int):
        state, time = load_state_npz(snapshot_files[frame_index])
        S = poynting_vector_flat(state)

        Sx = S[0][interior]
        Sy = S[1][interior]
        Smag = np.sqrt(Sx**2 + Sy**2)

        im.set_data(Smag.T)

        if quiv is not None:
            quiv.set_UVC(
                Sx[::stride, ::stride],
                Sy[::stride, ::stride],
            )

        ax.set_title(rf"Poynting vector, $t={time:.3f}$")

        if quiv is not None:
            return im, quiv

        return (im,)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(snapshot_files),
        interval=1000 / fps,
        blit=False,
    )

    if output_path.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi)
    else:
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi)

    plt.close(fig)

    return output_path