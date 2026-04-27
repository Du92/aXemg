"""
Plot a metric from a sweep summary CSV.

Examples
--------
Plot max_abs_EdotB versus g_agamma:

    python examples/plot_sweep_summary.py \
        --csv outputs/sweep_constraint_solved_1d_g/sweep_summary.csv \
        --x param.physics.g_agamma \
        --y max_abs_EdotB

Plot axion energy versus compactness:

    python examples/plot_sweep_summary.py \
        --csv outputs/sweep_compactness_2d/sweep_summary.csv \
        --x param.geometry.compactness \
        --y axion_energy

Use log scale:

    python examples/plot_sweep_summary.py \
        --csv outputs/sweep_gw_axion_halo_h/sweep_summary.csv \
        --x param.geometry.h_plus_amplitude \
        --y l2_EdotB \
        --logy
"""

from __future__ import annotations

import argparse
from pathlib import Path

from axion_em_gr.visualization.sweep_plots import (
    plot_metric_vs_parameter,
    read_sweep_summary_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one metric against one swept parameter."
    )

    parser.add_argument(
        "--csv",
        required=True,
        type=str,
        help="Path to sweep_summary.csv.",
    )

    parser.add_argument(
        "--x",
        required=True,
        type=str,
        help=(
            "Parameter column for the horizontal axis, e.g. "
            "param.physics.g_agamma or param.geometry.compactness."
        ),
    )

    parser.add_argument(
        "--y",
        required=True,
        type=str,
        help=(
            "Metric column for the vertical axis, e.g. "
            "max_abs_EdotB, l2_EdotB, axion_energy, em_energy."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="Optional output PNG path. If omitted, a path is created automatically.",
    )

    parser.add_argument(
        "--logx",
        action="store_true",
        help="Use logarithmic scale on the x axis.",
    )

    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use logarithmic scale on the y axis.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    rows = read_sweep_summary_csv(csv_path)

    if len(rows) == 0:
        raise ValueError(f"No rows found in CSV file: {csv_path}")

    available_columns = set(rows[0].keys())

    if args.x not in available_columns:
        available = "\n".join(sorted(available_columns))
        raise KeyError(
            f"Column {args.x!r} not found in {csv_path}.\n\n"
            f"Available columns:\n{available}"
        )

    if args.y not in available_columns:
        available = "\n".join(sorted(available_columns))
        raise KeyError(
            f"Column {args.y!r} not found in {csv_path}.\n\n"
            f"Available columns:\n{available}"
        )

    if args.output is None:
        output_path = (
            csv_path.parent
            / "plots"
            / f"{args.y}_vs_{args.x.replace('.', '_')}.png"
        )
    else:
        output_path = Path(args.output)

    path = plot_metric_vs_parameter(
        rows=rows,
        parameter_key=args.x,
        metric_key=args.y,
        output_path=output_path,
        logx=args.logx,
        logy=args.logy,
    )

    print(path)


if __name__ == "__main__":
    main()