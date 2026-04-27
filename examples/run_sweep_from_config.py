"""
Run a parameter sweep from a YAML configuration file.

Example:

    python examples/run_sweep_from_config.py --config config/sweep_flat_axion_em_2d.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from axion_em_gr.io.config_loader import load_yaml_config
from axion_em_gr.io.sweep import run_sweep
from axion_em_gr.visualization.sweep_plots import make_default_sweep_plots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a parameter sweep from YAML config."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to sweep YAML configuration file.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable automatic sweep summary plots.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    config = load_yaml_config(config_path)

    if "sweep" not in config:
        raise ValueError("Config file does not contain a 'sweep' block.")

    rows, summary_path = run_sweep(config)

    print("\nSweep finished.")
    print(f"Number of cases: {len(rows)}")
    print(f"Summary CSV: {summary_path}")

    if not args.no_plots:
        plot_dir = summary_path.parent / "plots"
        paths = make_default_sweep_plots(
            summary_csv=summary_path,
            output_dir=plot_dir,
        )

        print("\nSweep plots:")
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()