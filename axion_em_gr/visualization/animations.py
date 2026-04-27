"""
Simple and robust animation utilities.

This module only depends on objects that already exist in the project:
    - Grid
    - State
    - metric.evaluate(t, grid)
    - history.snapshots
    - history.snapshot_times

Animated quantities:
    - a
    - Pi
    - |E|
    - |B|
    - E.B

The contraction E.B is computed as:
    E_i B^i = gamma_ij E^j B^i
if a metric is provided, otherwise with the flat Euclidean contraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def ensure_output_dir(path: str | Path) -> Path:
    """
    Create output directory if needed.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_snapshots(history) -> list:
    """
    Return snapshots from a history object.
    """
    if not hasattr(history, "snapshots"):
        raise AttributeError("History object has no attribute 'snapshots'.")

    snapshots = list(history.snapshots)

    if len(snapshots) == 0:
        raise ValueError(
            "history.snapshots is empty. Run the solver with save_snapshots=True "
            "and snapshot_every > 0."
        )

    return snapshots


def _get_snapshot_times(history, n: int) -> list[float]:
    """
    Return snapshot times if available. Otherwise use frame index.
    """
    if hasattr(history, "snapshot_times") and len(history.snapshot_times) > 0:
        times = list(history.snapshot_times)
    else:
        times = list(range(n))

    if len(times) != n:
        m = min(len(times), n)
        return times[:m]

    return times


def _vector_norm(vector: np.ndarray) -> np.ndarray:
    """
    Compute sqrt(Vx^2 + Vy^2 + Vz^2).
    """
    return np.sqrt(np.sum(vector**2, axis=0))


def _edotb(state, geom=None) -> np.ndarray:
    """
    Compute E_i B^i.

    If geom is None, use flat contraction:
        E.B = E^x B^x + E^y B^y + E^z B^z.

    If geom is provided, use:
        E_i B^i = gamma_ij E^j B^i.
    """
    if state.E is None or state.B is None:
        raise ValueError("E.B requires state.E and state.B.")

    if geom is None:
        return np.sum(state.E * state.B, axis=0)

    result = np.zeros_like(state.a)

    for i in range(3):
        for j in range(3):
            result += geom.gamma_down[i, j] * state.E[j] * state.B[i]

    return result


def _extract_quantity(state, quantity: str, geom=None) -> np.ndarray:
    """
    Extract scalar quantity from state.
    """
    if quantity == "a":
        return state.a

    if quantity == "Pi":
        return state.Pi

    if quantity == "E":
        if state.E is None:
            raise ValueError("State has no E field.")
        return _vector_norm(state.E)

    if quantity == "B":
        if state.B is None:
            raise ValueError("State has no B field.")
        return _vector_norm(state.B)

    if quantity == "EdotB":
        return _edotb(state, geom=geom)

    raise ValueError(f"Unknown quantity: {quantity!r}")


def _evaluate_geom(metric: Any | None, t: float, grid):
    """
    Evaluate metric if available.
    """
    if metric is None:
        return None

    return metric.evaluate(t, grid)


def _save_animation(anim: FuncAnimation, output_path: Path, fps: int) -> Path:
    """
    Save animation. Try mp4 first. If ffmpeg fails, save gif.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".gif":
        anim.save(output_path, writer="pillow", fps=fps)
        return output_path

    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")

    try:
        anim.save(output_path, writer="ffmpeg", fps=fps, dpi=150)
        return output_path
    except Exception as exc:
        gif_path = output_path.with_suffix(".gif")
        print(f"Could not save mp4 because: {exc}")
        print(f"Saving gif instead: {gif_path}")
        anim.save(gif_path, writer="pillow", fps=fps)
        return gif_path


def animate_quantity_1d(
    history,
    grid,
    quantity: str,
    output_path: str | Path,
    metric=None,
    fps: int = 15,
    interval_ms: int = 80,
    fixed_ylim: bool = True,
) -> Path:
    """
    Animate one 1D quantity.

    quantity:
        "a", "Pi", "E", "B", "EdotB"
    """
    snapshots = _get_snapshots(history)
    times = _get_snapshot_times(history, len(snapshots))

    n = min(len(snapshots), len(times))
    snapshots = snapshots[:n]
    times = times[:n]

    x = grid.coordinates_1d()
    interior = grid.interior_slices

    data = []

    for state, t in zip(snapshots, times):
        geom = _evaluate_geom(metric, t, grid)
        q = _extract_quantity(state, quantity, geom=geom)
        data.append(q[interior].copy())

    data = np.asarray(data)
    x_int = x[interior]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    line, = ax.plot(x_int, data[0])

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(quantity)
    ax.set_title(f"{quantity} evolution")

    if fixed_ylim:
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            ymin = float(np.min(finite))
            ymax = float(np.max(finite))
            if np.isclose(ymin, ymax):
                margin = 1.0 if ymax == 0.0 else 0.05 * abs(ymax)
            else:
                margin = 0.05 * (ymax - ymin)
            ax.set_ylim(ymin - margin, ymax + margin)

    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    def update(frame: int):
        line.set_ydata(data[frame])
        time_text.set_text(f"t = {times[frame]:.6f}")
        return line, time_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(data),
        interval=interval_ms,
        blit=False,
    )

    output_path = Path(output_path)
    saved = _save_animation(anim, output_path, fps=fps)
    plt.close(fig)

    return saved


def animate_quantity_2d(
    history,
    grid,
    quantity: str,
    output_path: str | Path,
    metric=None,
    fps: int = 12,
    interval_ms: int = 90,
    fixed_clim: bool = True,
    overlay: str | None = None,
) -> Path:
    """
    Animate one 2D quantity.

    quantity:
        "a", "Pi", "E", "B", "EdotB"

    overlay:
        None, "lapse", or "sqrt_gamma"
    """
    snapshots = _get_snapshots(history)
    times = _get_snapshot_times(history, len(snapshots))

    n = min(len(snapshots), len(times))
    snapshots = snapshots[:n]
    times = times[:n]

    X, Y = grid.coordinates_2d()
    interior = grid.interior_slices

    data = []
    overlay_data = []

    for state, t in zip(snapshots, times):
        geom = _evaluate_geom(metric, t, grid)
        q = _extract_quantity(state, quantity, geom=geom)
        data.append(q[interior].copy())

        if overlay is None or geom is None:
            overlay_data.append(None)
        elif overlay == "lapse":
            overlay_data.append(geom.lapse[interior].copy())
        elif overlay == "sqrt_gamma":
            overlay_data.append(geom.sqrt_gamma[interior].copy())
        else:
            raise ValueError("overlay must be None, 'lapse', or 'sqrt_gamma'.")

    data = np.asarray(data)

    X_int = X[interior]
    Y_int = Y[interior]

    extent = [
        float(np.min(X_int)),
        float(np.max(X_int)),
        float(np.min(Y_int)),
        float(np.max(Y_int)),
    ]

    fig, ax = plt.subplots(figsize=(7, 6))

    if fixed_clim:
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            if np.isclose(vmin, vmax):
                margin = 1.0 if vmax == 0.0 else 0.05 * abs(vmax)
                vmin -= margin
                vmax += margin
        else:
            vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = None, None

    im = ax.imshow(
        data[0].T,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(quantity)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(f"{quantity} evolution")

    time_text = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
    )

    contour_artists = []

    def draw_contours(frame: int):
        nonlocal contour_artists

        for artist in contour_artists:
            try:
                artist.remove()
            except Exception:
                pass

        contour_artists = []

        if overlay_data[frame] is not None:
            cs = ax.contour(
                X_int,
                Y_int,
                overlay_data[frame],
                levels=8,
                colors="white",
                linewidths=0.6,
                alpha=0.8,
            )
            contour_artists = list(cs.collections)

    draw_contours(0)

    def update(frame: int):
        im.set_data(data[frame].T)

        if not fixed_clim:
            finite = data[frame][np.isfinite(data[frame])]
            if finite.size > 0:
                vmin_f = float(np.min(finite))
                vmax_f = float(np.max(finite))
                if np.isclose(vmin_f, vmax_f):
                    margin = 1.0 if vmax_f == 0.0 else 0.05 * abs(vmax_f)
                    vmin_f -= margin
                    vmax_f += margin
                im.set_clim(vmin_f, vmax_f)

        draw_contours(frame)
        time_text.set_text(f"t = {times[frame]:.6f}")

        return [im, time_text, *contour_artists]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(data),
        interval=interval_ms,
        blit=False,
    )

    output_path = Path(output_path)
    saved = _save_animation(anim, output_path, fps=fps)
    plt.close(fig)

    return saved


def animate_default_1d_set(
    history,
    grid,
    output_dir: str | Path,
    metric=None,
    fps: int = 15,
) -> list[Path]:
    """
    Generate default 1D animations.
    """
    output_dir = ensure_output_dir(output_dir)

    quantities = ["a", "Pi", "E", "B", "EdotB"]
    paths = []

    for quantity in quantities:
        try:
            path = animate_quantity_1d(
                history=history,
                grid=grid,
                quantity=quantity,
                output_path=Path(output_dir) / f"{quantity}.mp4",
                metric=metric,
                fps=fps,
            )
            paths.append(path)
        except Exception as exc:
            print(f"Skipping {quantity}: {exc}")

    return paths


def animate_default_2d_set(
    history,
    grid,
    output_dir: str | Path,
    metric=None,
    fps: int = 12,
    overlay: str | None = "lapse",
) -> list[Path]:
    """
    Generate default 2D animations.
    """
    output_dir = ensure_output_dir(output_dir)

    quantities = ["a", "Pi", "E", "B", "EdotB"]
    paths = []

    for quantity in quantities:
        try:
            path = animate_quantity_2d(
                history=history,
                grid=grid,
                quantity=quantity,
                output_path=Path(output_dir) / f"{quantity}.mp4",
                metric=metric,
                fps=fps,
                overlay=overlay if metric is not None else None,
            )
            paths.append(path)
        except Exception as exc:
            print(f"Skipping {quantity}: {exc}")

    return paths