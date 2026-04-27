"""
Simple 1D plotting utilities.

All plots are saved to disk instead of shown interactively.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.core.state import State


def ensure_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_axion_state(
    grid: Grid,
    state: State,
    title: str = "Axion field",
    output_dir: str | Path = "outputs",
    filename: str = "axion_state.png",
) -> Path:
    """
    Save a plot of a and Pi on the interior domain.
    """
    output_path = ensure_output_dir(output_dir)

    x = grid.coordinates_1d()[grid.interior_slices]
    interior = grid.interior_slices

    a = state.a[interior]
    Pi = state.Pi[interior]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, a, label=r"$a$")
    ax.plot(x, Pi, label=r"$\Pi$", linestyle="--")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("field value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def plot_em_state(
    grid: Grid,
    state: State,
    title: str = "Electromagnetic field",
    output_dir: str | Path = "outputs",
    filename: str = "em_state.png",
) -> Path:
    """
    Save a plot of E^i and B^i on the interior domain.
    """
    if state.E is None or state.B is None:
        raise ValueError("State must contain E and B fields to plot EM state.")

    output_path = ensure_output_dir(output_dir)

    x = grid.coordinates_1d()[grid.interior_slices]
    interior = grid.interior_slices

    fig, ax = plt.subplots(figsize=(10, 5))

    component_labels = ["x", "y", "z"]

    for i in range(3):
        ax.plot(
            x,
            state.E[i][interior],
            label=rf"$E^{{{component_labels[i]}}}$",
        )

    for i in range(3):
        ax.plot(
            x,
            state.B[i][interior],
            linestyle="--",
            label=rf"$B^{{{component_labels[i]}}}$",
        )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel("field value")
    ax.set_title(title)
    ax.legend(ncol=3)
    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def plot_history(
    history,
    output_dir: str | Path = "outputs",
    filename: str = "history.png",
) -> Path:
    """
    Save basic evolution diagnostics.

    This function is robust against diagnostic arrays with slightly different
    lengths. It truncates each series to the shortest compatible length.
    """
    output_path = ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(8, 4))

    def safe_plot(y_values, label):
        if not hasattr(history, "times"):
            return

        if y_values is None:
            return

        if len(y_values) == 0 or len(history.times) == 0:
            return

        n = min(len(history.times), len(y_values))

        ax.plot(history.times[:n], y_values[:n], label=label)

    if hasattr(history, "max_a"):
        safe_plot(history.max_a, r"$\max |a|$")

    if hasattr(history, "l2_a"):
        safe_plot(history.l2_a, r"$||a||_2$")

    if hasattr(history, "max_E"):
        safe_plot(history.max_E, r"$\max |E|$")

    if hasattr(history, "max_B"):
        safe_plot(history.max_B, r"$\max |B|$")

    if hasattr(history, "l2_em"):
        safe_plot(history.l2_em, r"$||EM||_2$")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel("diagnostic")
    ax.legend()
    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def plot_constraints_1d(
    grid: Grid,
    div_B: np.ndarray,
    div_E: np.ndarray,
    output_dir: str | Path = "outputs",
    filename: str = "constraints.png",
) -> Path:
    """
    Save the 1D Maxwell constraints.
    """
    output_path = ensure_output_dir(output_dir)

    x = grid.coordinates_1d()[grid.interior_slices]
    interior = grid.interior_slices

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, div_B[interior], label=r"$D_iB^i$")
    ax.plot(x, div_E[interior], label=r"$D_iE^i$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("constraint value")
    ax.legend()
    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path

def plot_EdotB_profile(
    grid: Grid,
    state: State,
    geom,
    output_dir: str | Path = "outputs",
    filename: str = "edotb_profile.png",
    title: str = r"$E_iB^i$ profile",
) -> Path:
    """
    Save a plot of E_i B^i on the interior domain.
    """
    from axion_em_gr.core.tensors import contract_cov_contra, lower_vector

    if state.E is None or state.B is None:
        raise ValueError("State must contain E and B fields.")

    output_path = ensure_output_dir(output_dir)

    x = grid.coordinates_1d()[grid.interior_slices]
    interior = grid.interior_slices

    E_down = lower_vector(state.E, geom.gamma_down)
    EdotB = contract_cov_contra(E_down, state.B)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, EdotB[interior], label=r"$E_iB^i$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$E_iB^i$")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def plot_coupling_history(
    history,
    output_dir: str | Path = "outputs",
    filename: str = "coupling_history.png",
) -> Path:
    """
    Save coupling-related diagnostics.
    """
    output_path = ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(8, 4))

    if hasattr(history, "max_EdotB") and len(history.max_EdotB) > 0:
        ax.plot(history.times, history.max_EdotB, label=r"$\max |E_iB^i|$")

    if hasattr(history, "l2_EdotB") and len(history.l2_EdotB) > 0:
        ax.plot(history.times, history.l2_EdotB, label=r"$||E_iB^i||_2$")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel("coupling diagnostic")
    ax.legend()
    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path

def plot_geometry_1d(
    grid: Grid,
    geom,
    output_dir: str | Path = "outputs",
    filename: str = "geometry.png",
    title: str = "3+1 geometry fields",
) -> Path:
    """
    Save a plot of lapse, shift beta^x, sqrt(gamma), and K.
    """
    output_path = ensure_output_dir(output_dir)

    x = grid.coordinates_1d()[grid.interior_slices]
    interior = grid.interior_slices

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(x, geom.lapse[interior], label=r"$N$")
    ax.plot(x, geom.shift[0][interior], label=r"$\beta^x$")
    ax.plot(x, geom.sqrt_gamma[interior], label=r"$\sqrt{\gamma}$")
    ax.plot(x, geom.K[interior], label=r"$K$")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path

def plot_gw_metric_1d(
    grid: Grid,
    geom,
    output_dir: str | Path = "outputs",
    filename: str = "gw_metric.png",
    title: str = "Gravitational-wave TT metric",
) -> Path:
    """
    Save a plot of the relevant TT metric components.

    For the 1D GW propagating along x:

        h_+     = (gamma_yy - gamma_zz)/2
        h_cross = gamma_yz
    """
    output_path = ensure_output_dir(output_dir)

    x = grid.coordinates_1d()[grid.interior_slices]
    interior = grid.interior_slices

    gamma_yy = geom.gamma_down[1, 1][interior]
    gamma_zz = geom.gamma_down[2, 2][interior]
    gamma_yz = geom.gamma_down[1, 2][interior]

    h_plus = 0.5 * (gamma_yy - gamma_zz)
    h_cross = gamma_yz

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(x, h_plus, label=r"$h_+$")
    ax.plot(x, h_cross, label=r"$h_\times$")
    ax.plot(x, geom.K[interior], label=r"$K$", linestyle="--")
    ax.plot(x, geom.sqrt_gamma[interior] - 1.0, label=r"$\sqrt{\gamma}-1$", linestyle=":")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel("metric perturbation")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path

def plot_scalar_difference_1d(
    grid: Grid,
    field_reference,
    field_test,
    label: str,
    output_dir: str | Path = "outputs",
    filename: str = "scalar_difference.png",
    title: str = "Scalar difference",
) -> Path:
    """
    Save field_test - field_reference for a scalar field.
    """
    output_path = ensure_output_dir(output_dir)

    x = grid.coordinates_1d()[grid.interior_slices]
    interior = grid.interior_slices

    delta = field_test - field_reference

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, delta[interior], label=label)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def plot_vector_difference_components_1d(
    grid: Grid,
    vector_reference,
    vector_test,
    vector_name: str,
    output_dir: str | Path = "outputs",
    filename: str = "vector_difference.png",
    title: str = "Vector difference",
) -> Path:
    """
    Save component-wise vector difference.
    """
    output_path = ensure_output_dir(output_dir)

    x = grid.coordinates_1d()[grid.interior_slices]
    interior = grid.interior_slices

    delta = vector_test - vector_reference

    component_labels = ["x", "y", "z"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(3):
        ax.plot(
            x,
            delta[i][interior],
            label=rf"$\Delta {vector_name}^{{{component_labels[i]}}}$",
        )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(rf"$\Delta {vector_name}^i$")
    ax.set_title(title)
    ax.legend(ncol=3)
    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def plot_difference_norms(
    norms,
    output_dir: str | Path = "outputs",
    filename: str = "difference_norms.png",
) -> Path:
    """
    Plot time series of difference norms.
    """
    output_path = ensure_output_dir(output_dir)

    times = [n.time for n in norms]

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(times, [n.l2_delta_a for n in norms], label=r"$||\Delta a||_2$")
    ax.plot(times, [n.l2_delta_E for n in norms], label=r"$||\Delta E||_2$")
    ax.plot(times, [n.l2_delta_B for n in norms], label=r"$||\Delta B||_2$")
    ax.plot(times, [n.l2_delta_EdotB for n in norms], label=r"$||\Delta(E_iB^i)||_2$")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel("absolute difference norm")
    ax.legend()
    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def plot_relative_difference_norms(
    norms,
    output_dir: str | Path = "outputs",
    filename: str = "relative_difference_norms.png",
) -> Path:
    """
    Plot time series of relative difference norms.
    """
    output_path = ensure_output_dir(output_dir)

    times = [n.time for n in norms]

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(times, [n.rel_l2_delta_a for n in norms], label=r"$||\Delta a||_2/||a||_2$")
    ax.plot(times, [n.rel_l2_delta_E for n in norms], label=r"$||\Delta E||_2/||E||_2$")
    ax.plot(times, [n.rel_l2_delta_B for n in norms], label=r"$||\Delta B||_2/||B||_2$")
    ax.plot(
        times,
        [n.rel_l2_delta_EdotB for n in norms],
        label=r"$||\Delta(E_iB^i)||_2/||E_iB^i||_2$",
    )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel("relative difference")
    ax.legend()
    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path


def plot_EdotB_difference_1d(
    grid: Grid,
    edotb_reference,
    edotb_test,
    output_dir: str | Path = "outputs",
    filename: str = "delta_EdotB.png",
    title: str = r"$\Delta(E_iB^i)$",
) -> Path:
    """
    Save difference in E_iB^i.
    """
    output_path = ensure_output_dir(output_dir)

    x = grid.coordinates_1d()[grid.interior_slices]
    interior = grid.interior_slices

    delta = edotb_test - edotb_reference

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, delta[interior], label=r"$\Delta(E_iB^i)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\Delta(E_iB^i)$")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    save_path = output_path / filename
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return save_path