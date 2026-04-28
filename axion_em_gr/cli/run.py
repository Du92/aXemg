"""
Command-line entry point for running a simulation from YAML.

Equivalent to:

    python examples/run_from_config.py --config path/to/config.yaml

Installed command:

    axemg-run --config config/canonical/curved_axion_maxwell_2d.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from axion_em_gr.initial_data.factory import build_initial_state
from axion_em_gr.io.config_loader import build_simulation_objects, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run axion_em_gr simulation from YAML config."
    )

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to YAML configuration file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)

    config = load_yaml_config(config_path)
    objects = build_simulation_objects(config)

    grid = objects["grid"]
    metric = objects["metric"]
    solver = objects["solver"]

    state0 = build_initial_state(config, grid, metric=metric)

    output_dir = config.get("output", {}).get("directory", "outputs/from_config")

    print(f"\nRunning from config: {config_path}")
    print(f"Output directory:    {output_dir}\n")

    final_state, history = solver.evolve(state0)

    print("\nRun completed.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()