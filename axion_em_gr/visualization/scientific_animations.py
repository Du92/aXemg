"""
Scientific multi-panel animations.

This module builds on the simple snapshot animation infrastructure but creates
publication/presentation-friendly multi-panel animations.

It intentionally avoids fragile dynamic contour handling. Geometry overlays
are drawn as static contours once, assuming fixed background metrics.

Supported quantities:
    - a
    - Pi
    - E
    - B
    - EdotB

Here:
    E     means |E|
    B     means |B|
    EdotB means E_i B^i.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def ensure_output_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_snapshots_and_times(history):
    """
    Extract snapshots and snapshot_times from an EvolutionHistory-like object.
    """
    if not hasattr(history, "snapshots") or len(history.snapshots) == 0:
        raise ValueError(
            "No snapshots found. Run with save_snapshots=True and snapshot_every > 0."
        )

    snapshots = list(history.snapshots)

    if hasattr(history, "snapshot_times") and len(history.snapshot_times) > 0:
        times = list(history.snapshot_times)
    else:
        times = list(range(len(snapshots)))

    n = min(len(snapshots), len(times))

    return snapshots[:n], times[:n]


def evaluate_geom(metric: Any | None, t: float, grid):
    if metric is None:
        return None

    return metric.evaluate(t, grid)


def vector_norm(vector: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(vector**2, axis=0))


def edotb(state, geom=None) -> np.ndarray:
    """
    Compute E_i B^i.

    If geom is None:
        use flat Euclidean contraction.
    If geom is provided:
        use gamma_ij E^j B^i.
    """
    if state.E is None or state.B is None:
        raise ValueError("EdotB requires E and B.")

    if geom is None:
        return np.sum(state.E * state.B, axis=0)

    result = np.zeros_like(state.a)

    for i in range(3):
        for j in range(3):
            result += geom.gamma_down[i, j] * state.E[j] * state.B[i]

    return result


def extract_quantity(state, quantity: str, geom=None) -> np.ndarray:
    """
    Extract one scalar quantity from a State.
    """
    if quantity == "a":
        return state.a

    if quantity == "Pi":
        return state.Pi

    if quantity == "E":
        if state.E is None:
            raise ValueError("State has no E field.")
        return vector_norm(state.E)

    if quantity == "B":
        if state.B is None:
            raise ValueError("State has no B field.")
        return vector_norm(state.B)

    if quantity == "EdotB":
        return edotb(state, geom=geom)

    raise ValueError(f"Unknown quantity: {quantity!r}")


def collect_quantity_data(
    history,
    grid,
    quantity: str,
    metric=None,
) -> tuple[np.ndarray, list[float]]:
    """
    Return data array with shape:

        (nframes, *grid.shape)

    over the physical interior.
    """
    snapshots, times = get_snapshots_and_times(history)
    interior = grid.interior_slices

    data = []

    for state, t in zip(snapshots, times):
        geom = evaluate_geom(metric, t, grid)
        q = extract_quantity(state, quantity, geom=geom)
        data.append(q[interior].copy())

    return np.asarray(data), times


def robust_clim(data: np.ndarray, symmetric: bool = False) -> tuple[float, float]:
    """
    Compute robust color limits using percentiles.

    This avoids a single spike dominating the colorbar.
    """
    finite = data[np.isfinite(data)]

    if finite.size == 0:
        return -1.0, 1.0

    if symmetric:
        vmax = float(np.percentile(np.abs(finite), 99.0))
        if vmax == 0.0:
            vmax = 1.0
        return -vmax, vmax

    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.0))

    if np.isclose(vmin, vmax):
        margin = 1.0 if vmax == 0.0 else 0.05 * abs(vmax)
        return vmin - margin, vmax + margin

    return vmin, vmax


def save_animation(anim: FuncAnimation, output_path: str | Path, fps: int) -> Path:
    """
    Save MP4 if possible, otherwise GIF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")

    try:
        anim.save(output_path, writer="ffmpeg", fps=fps, dpi=160)
        return output_path
    except Exception as exc:
        gif_path = output_path.with_suffix(".gif")
        print(f"Could not save mp4 because: {exc}")
        print(f"Saving gif instead: {gif_path}")
        anim.save(gif_path, writer="pillow", fps=fps)
        return gif_path


def extent_from_grid_2d(grid) -> list[float]:
    X, Y = grid.coordinates_2d()
    interior = grid.interior_slices

    X_int = X[interior]
    Y_int = Y[interior]

    return [
        float(np.min(X_int)),
        float(np.max(X_int)),
        float(np.min(Y_int)),
        float(np.max(Y_int)),
    ]


def interior_coordinates_2d(grid):
    X, Y = grid.coordinates_2d()
    interior = grid.interior_slices
    return X[interior], Y[interior]


def quantity_label(quantity: str) -> str:
    labels = {
        "a": r"$a$",
        "Pi": r"$\Pi$",
        "E": r"$|\mathbf{E}|$",
        "B": r"$|\mathbf{B}|$",
        "EdotB": r"$E_iB^i$",
    }

    return labels.get(quantity, quantity)


def default_cmap(quantity: str) -> str:
    if quantity in ("a", "Pi", "EdotB"):
        return "RdBu_r"

    return "viridis"


def quantity_is_symmetric(quantity: str) -> bool:
    return quantity in ("a", "Pi", "EdotB")


# ---------------------------------------------------------------------
# Static overlays
# ---------------------------------------------------------------------

def draw_static_geometry_overlay(
    ax,
    grid,
    metric=None,
    overlay: str | None = None,
    t: float = 0.0,
    levels: int = 8,
) -> None:
    """
    Draw static geometry contours.

    This avoids dynamic contour removal, which caused compatibility problems
    with some Matplotlib versions.
    """
    if metric is None or overlay is None:
        return

    geom = metric.evaluate(t, grid)

    X, Y = interior_coordinates_2d(grid)

    if overlay == "lapse":
        data = geom.lapse[grid.interior_slices]
    elif overlay == "sqrt_gamma":
        data = geom.sqrt_gamma[grid.interior_slices]
    else:
        return

    try:
        ax.contour(
            X,
            Y,
            data,
            levels=levels,
            colors="white",
            linewidths=0.5,
            alpha=0.65,
        )
    except Exception as exc:
        print(f"Geometry overlay skipped: {exc}")


def draw_radius_circle(
    ax,
    radius: float | None = None,
    center: tuple[float, float] = (0.0, 0.0),
    color: str = "white",
) -> None:
    """
    Draw a static circle representing a compact object surface or reference radius.
    """
    if radius is None or radius <= 0.0:
        return

    circle = plt.Circle(
        center,
        radius,
        fill=False,
        color=color,
        linewidth=1.0,
        linestyle="--",
        alpha=0.9,
    )

    ax.add_patch(circle)


# ---------------------------------------------------------------------
# 2D multi-panel animations
# ---------------------------------------------------------------------

def animate_multipanel_2d(
    history,
    grid,
    quantities: list[str],
    output_path: str | Path,
    metric=None,
    overlay: str | None = None,
    radius: float | None = None,
    radius_center: tuple[float, float] = (0.0, 0.0),
    fps: int = 12,
    interval_ms: int = 90,
    title: str | None = None,
    robust_limits: bool = True,
) -> Path:
    """
    Create a 2D multi-panel animation.

    Examples
    --------
    quantities = ["a", "Pi", "EdotB", "B"]

    quantities = ["a", "EdotB", "E", "B"]
    """
    if grid.ndim != 2:
        raise ValueError("animate_multipanel_2d requires a 2D grid.")

    if len(quantities) == 0:
        raise ValueError("At least one quantity is required.")

    snapshots, times = get_snapshots_and_times(history)

    data_by_q = {}
    for quantity in quantities:
        data, _ = collect_quantity_data(
            history=history,
            grid=grid,
            quantity=quantity,
            metric=metric,
        )
        data_by_q[quantity] = data

    nframes = len(times)

    if len(quantities) == 1:
        nrows, ncols = 1, 1
    elif len(quantities) == 2:
        nrows, ncols = 1, 2
    elif len(quantities) <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 2, 3

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.6 * nrows),
        squeeze=False,
    )

    flat_axes = axes.ravel()
    extent = extent_from_grid_2d(grid)

    images = []
    time_texts = []

    for idx, quantity in enumerate(quantities):
        ax = flat_axes[idx]
        data = data_by_q[quantity]

        if robust_limits:
            vmin, vmax = robust_clim(
                data,
                symmetric=quantity_is_symmetric(quantity),
            )
        else:
            vmin = float(np.min(data))
            vmax = float(np.max(data))

        im = ax.imshow(
            data[0].T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=default_cmap(quantity),
            vmin=vmin,
            vmax=vmax,
        )

        draw_static_geometry_overlay(
            ax=ax,
            grid=grid,
            metric=metric,
            overlay=overlay,
            t=times[0],
        )

        draw_radius_circle(
            ax=ax,
            radius=radius,
            center=radius_center,
        )

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(quantity_label(quantity))

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(quantity_label(quantity))

        txt = ax.text(
            0.02,
            0.96,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="white",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
        )

        images.append((quantity, im))
        time_texts.append(txt)

    for j in range(len(quantities), len(flat_axes)):
        flat_axes[j].axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()

    def update(frame: int):
        artists = []

        for quantity, im in images:
            im.set_data(data_by_q[quantity][frame].T)
            artists.append(im)

        for txt in time_texts:
            txt.set_text(f"t = {times[frame]:.6f}")
            artists.append(txt)

        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=nframes,
        interval=interval_ms,
        blit=False,
    )

    saved = save_animation(anim, output_path, fps=fps)
    plt.close(fig)

    return saved


def animate_axion_em_summary_2d(
    history,
    grid,
    output_path: str | Path,
    metric=None,
    overlay: str | None = None,
    radius: float | None = None,
    radius_center: tuple[float, float] = (0.0, 0.0),
    fps: int = 12,
    title: str | None = None,
) -> Path:
    """
    Default 4-panel animation for coupled axion-EM simulations:

        a, EdotB, |E|, |B|
    """
    return animate_multipanel_2d(
        history=history,
        grid=grid,
        quantities=["a", "EdotB", "E", "B"],
        output_path=output_path,
        metric=metric,
        overlay=overlay,
        radius=radius,
        radius_center=radius_center,
        fps=fps,
        title=title or r"Axion--electromagnetic evolution",
    )


def animate_axion_summary_2d(
    history,
    grid,
    output_path: str | Path,
    metric=None,
    overlay: str | None = None,
    radius: float | None = None,
    radius_center: tuple[float, float] = (0.0, 0.0),
    fps: int = 12,
    title: str | None = None,
) -> Path:
    """
    Default 4-panel animation focused on the axion sector:

        a, Pi, EdotB, |B|
    """
    return animate_multipanel_2d(
        history=history,
        grid=grid,
        quantities=["a", "Pi", "EdotB", "B"],
        output_path=output_path,
        metric=metric,
        overlay=overlay,
        radius=radius,
        radius_center=radius_center,
        fps=fps,
        title=title or r"Axion-sector response",
    )


# ---------------------------------------------------------------------
# 1D spacetime/waterfall plots from snapshots
# ---------------------------------------------------------------------

def make_spacetime_map_1d(
    history,
    grid,
    quantity: str,
    output_path: str | Path,
    metric=None,
    title: str | None = None,
    cmap: str | None = None,
) -> Path:
    """
    Create a static spacetime map quantity(t,x) from 1D snapshots.

    This is often better than a movie for GW/halo 1D cases.
    """
    if grid.ndim != 1:
        raise ValueError("make_spacetime_map_1d requires a 1D grid.")

    data, times = collect_quantity_data(
        history=history,
        grid=grid,
        quantity=quantity,
        metric=metric,
    )

    x = grid.coordinates_1d()[grid.interior_slices]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    vmin, vmax = robust_clim(
        data,
        symmetric=quantity_is_symmetric(quantity),
    )

    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[
            float(np.min(x)),
            float(np.max(x)),
            float(np.min(times)),
            float(np.max(times)),
        ],
        cmap=cmap or default_cmap(quantity),
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    ax.set_title(title or rf"Spacetime map of {quantity_label(quantity)}")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(quantity_label(quantity))

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    return output_path


def make_default_spacetime_maps_1d(
    history,
    grid,
    output_dir: str | Path,
    metric=None,
) -> list[Path]:
    """
    Generate default 1D spacetime maps.
    """
    output_dir = ensure_output_dir(output_dir)
    paths = []

    for quantity in ["a", "Pi", "E", "B", "EdotB"]:
        try:
            path = make_spacetime_map_1d(
                history=history,
                grid=grid,
                quantity=quantity,
                output_path=Path(output_dir) / f"spacetime_{quantity}.png",
                metric=metric,
            )
            paths.append(path)
        except Exception as exc:
            print(f"Skipping spacetime map {quantity}: {exc}")

    return paths