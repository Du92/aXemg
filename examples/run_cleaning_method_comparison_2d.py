"""
Phase 15C example:

Compare electric constraint-cleaning methods in 2D.

Run with:

    python examples/run_cleaning_method_comparison_2d.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from axion_em_gr.core.grid import Grid
from axion_em_gr.initial_data.combined_setups_2d import (
    gaussian_axion_uniform_Bxy_constraint_cleaned_2d,
)
from axion_em_gr.visualization.plots_2d import ensure_output_dir, plot_em_summary_2d


def main() -> None:
    output_dir = ensure_output_dir(Path("outputs") / "phase15C_cleaning_comparison_2d")

    grid = Grid(
        ndim=2,
        shape=(128, 128),
        bounds=((-50.0, 50.0), (-50.0, 50.0)),
        nghost=3,
    )

    cases = [
        {
            "label": "periodic_fft",
            "poisson_solver": "periodic_fft",
            "poisson_boundary": "periodic",
            "max_iterations": 1,
            "tolerance": 1.0e-10,
            "omega": None,
        },
        {
            "label": "jacobi_dirichlet",
            "poisson_solver": "jacobi",
            "poisson_boundary": "dirichlet",
            "max_iterations": 20_000,
            "tolerance": 1.0e-7,
            "omega": 2.0 / 3.0,
        },
        {
            "label": "jacobi_neumann",
            "poisson_solver": "jacobi",
            "poisson_boundary": "neumann",
            "max_iterations": 20_000,
            "tolerance": 1.0e-7,
            "omega": 2.0 / 3.0,
        },
        {
            "label": "sor_dirichlet",
            "poisson_solver": "sor",
            "poisson_boundary": "dirichlet",
            "max_iterations": 5_000,
            "tolerance": 1.0e-7,
            "omega": 1.7,
        },
    ]

    labels = []
    l2_before = []
    l2_after = []
    linf_after = []
    iterations = []
    residuals = []

    for case in cases:
        label = case["label"]
        print("\n" + "=" * 80)
        print(f"Cleaning case: {label}")
        print("=" * 80)

        state, report = gaussian_axion_uniform_Bxy_constraint_cleaned_2d(
            grid=grid,
            axion_amplitude=1.0,
            axion_center=(0.0, 0.0),
            axion_width=(8.0, 8.0),
            axion_momentum_amplitude=0.3,
            g_agamma=0.03,
            B0=(1.0, 0.5, 0.0),
            E0=(0.0, 0.0, 0.0),
            dt_for_cleaning=0.01,
            poisson_solver=case["poisson_solver"],
            poisson_boundary=case["poisson_boundary"],
            max_iterations=case["max_iterations"],
            tolerance=case["tolerance"],
            omega=case["omega"],
        )

        print(f"method = {report.method}")
        print(f"L2 before = {report.l2_constraint_before:.8e}")
        print(f"L2 after  = {report.l2_constraint_after:.8e}")
        print(f"Linf after = {report.linf_constraint_after:.8e}")
        print(f"iterations = {report.poisson_iterations}")
        print(f"residual Linf = {report.poisson_residual_linf:.8e}")
        print(f"converged = {report.poisson_converged}")

        labels.append(label)
        l2_before.append(report.l2_constraint_before)
        l2_after.append(report.l2_constraint_after)
        linf_after.append(report.linf_constraint_after)
        iterations.append(report.poisson_iterations)
        residuals.append(report.poisson_residual_linf)

        case_dir = ensure_output_dir(output_dir / label)

        for path in plot_em_summary_2d(
            grid=grid,
            state=state,
            output_dir=case_dir,
            prefix="cleaned_em",
        ):
            print(path)

    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(x, l2_before, marker="o", label=r"$||C_E||_2$ before")
    ax.semilogy(x, l2_after, marker="o", label=r"$||C_E||_2$ after")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("constraint norm")
    ax.legend()
    fig.tight_layout()

    path = output_dir / "cleaning_l2_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(path)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(x, linf_after, marker="o", label=r"$||C_E||_\infty$ after")
    ax.semilogy(x, residuals, marker="o", label="Poisson residual Linf")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("value")
    ax.legend()
    fig.tight_layout()

    path = output_dir / "cleaning_residual_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(path)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, iterations)
    ax.set_ylabel("iterations")
    ax.set_title("Poisson iterations")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    path = output_dir / "cleaning_iterations.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()