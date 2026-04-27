"""
Plotting utilities for parameter sweeps.
"""

from __future__ import annotations

from pathlib import Path

import csv
import math

import matplotlib.pyplot as plt


def read_sweep_summary_csv(path: str | Path) -> list[dict]:
    """
    Read a sweep summary CSV into a list of dictionaries.

    Numeric conversion is attempted field by field.
    """
    csv_path = Path(path)

    rows = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            converted = {}

            for key, value in row.items():
                if value is None:
                    converted[key] = value
                    continue

                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value

            rows.append(converted)

    return rows


def _only_ok_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("status") == "ok"]


def plot_metric_vs_parameter(
    rows: list[dict],
    parameter_key: str,
    metric_key: str,
    output_path: str | Path,
    logx: bool = False,
    logy: bool = False,
) -> Path:
    """
    Plot one metric against one parameter.

    parameter_key should usually be like:
        "param.physics.g_agamma"

    metric_key examples:
        "max_abs_EdotB"
        "l2_EdotB"
        "axion_energy"
        "em_energy"
        "l2_div_E"
    """
    ok_rows = _only_ok_rows(rows)

    if not ok_rows:
        raise ValueError("No successful rows to plot.")

    x = [row[parameter_key] for row in ok_rows]
    y = [row[metric_key] for row in ok_rows]

    pairs = sorted(zip(x, y), key=lambda pair: pair[0])

    x_sorted = [p[0] for p in pairs]
    y_sorted = [p[1] for p in pairs]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(x_sorted, y_sorted, marker="o")

    ax.set_xlabel(parameter_key)
    ax.set_ylabel(metric_key)

    if logx:
        ax.set_xscale("log")

    if logy:
        ax.set_yscale("log")

    fig.tight_layout()

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_two_parameter_heatmap(
    rows: list[dict],
    parameter_x: str,
    parameter_y: str,
    metric_key: str,
    output_path: str | Path,
) -> Path:
    """
    Plot a heatmap for a two-parameter grid sweep.

    Assumes the sweep generated a Cartesian grid in parameter_x and parameter_y.
    """
    ok_rows = _only_ok_rows(rows)

    if not ok_rows:
        raise ValueError("No successful rows to plot.")

    xs = sorted(set(row[parameter_x] for row in ok_rows))
    ys = sorted(set(row[parameter_y] for row in ok_rows))

    x_index = {value: i for i, value in enumerate(xs)}
    y_index = {value: j for j, value in enumerate(ys)}

    data = [[math.nan for _ in xs] for _ in ys]

    for row in ok_rows:
        i = x_index[row[parameter_x]]
        j = y_index[row[parameter_y]]
        data[j][i] = row[metric_key]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[min(xs), max(xs), min(ys), max(ys)],
    )

    ax.set_xlabel(parameter_x)
    ax.set_ylabel(parameter_y)
    ax.set_title(metric_key)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_key)

    fig.tight_layout()

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def make_default_sweep_plots(
    summary_csv: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """
    Create simple default plots if recognizable parameters are present.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_sweep_summary_csv(summary_csv)

    if not rows:
        return []

    keys = set(rows[0].keys())
    paths: list[Path] = []

    candidate_metrics = [
        "max_abs_EdotB",
        "l2_EdotB",
        "axion_energy",
        "em_energy",
        "total_energy",
        "l2_div_E",
    ]

    candidate_params = [
        "param.physics.g_agamma",
        "param.physics.m_axion",
        "param.initial_data.background_Bz",
        "param.initial_data.Bx",
        "param.geometry.h_plus_amplitude",
        "param.geometry.mass",
        "param.geometry.compactness",
        "param.geometry.conformal_amplitude",
        "param.background_em.omega",
    ]

    present_params = [p for p in candidate_params if p in keys]
    present_metrics = [m for m in candidate_metrics if m in keys]

    if len(present_params) == 1:
        p = present_params[0]

        for m in present_metrics:
            paths.append(
                plot_metric_vs_parameter(
                    rows=rows,
                    parameter_key=p,
                    metric_key=m,
                    output_path=output_dir / f"{m}_vs_{p.replace('.', '_')}.png",
                    logx=False,
                    logy=False,
                )
            )

    if len(present_params) >= 2:
        p0 = present_params[0]
        p1 = present_params[1]

        for m in present_metrics:
            paths.append(
                plot_two_parameter_heatmap(
                    rows=rows,
                    parameter_x=p0,
                    parameter_y=p1,
                    metric_key=m,
                    output_path=output_dir / (
                        f"heatmap_{m}_vs_"
                        f"{p0.replace('.', '_')}_"
                        f"{p1.replace('.', '_')}.png"
                    ),
                )
            )

    return paths