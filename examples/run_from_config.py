"""
Run a simulation from a YAML configuration file.

Examples:

    python examples/run_from_config.py --config config/flat_axion_1d.yaml

    python examples/run_from_config.py --config config/flat_axion_em_2d.yaml

    python examples/run_from_config.py --config config/gw_background_1d.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from axion_em_gr.initial_data.factory import build_initial_state
from axion_em_gr.io.config_loader import (
    build_simulation_objects,
    load_yaml_config,
)
from axion_em_gr.io.output_manager import save_basic_outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run axion-electrodynamics simulation from YAML config."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    config = load_yaml_config(config_path)

    objects = build_simulation_objects(config)

    grid = objects["grid"]
    physical = objects["physical"]
    numerics = objects["numerics"]
    metric = objects["metric"]
    potential = objects["potential"]
    sources = objects["sources"]
    solver = objects["solver"]

    state0 = build_initial_state(config, grid, metric=metric)

    print("\nSimulation loaded from config")
    print(f"config      = {config_path}")
    print(f"ndim        = {grid.ndim}")
    print(f"shape       = {grid.shape}")
    print(f"bounds      = {grid.bounds}")
    print(f"dt          = {numerics.dt}")
    print(f"t_final     = {numerics.t_final}")
    print(f"m_axion     = {physical.m_axion}")
    print(f"g_agamma    = {physical.g_agamma}")
    print(f"geometry    = {config.get('geometry', {}).get('type', 'flat')}")
    print(f"initial     = {config.get('initial_data', {}).get('type')}")
    print("\nStarting evolution...\n")

    final_state, history = solver.evolve(state0)

    print("\nEvolution finished.")
    print("Saving outputs...\n")

    paths = save_basic_outputs(
        config=config,
        grid=grid,
        state0=state0,
        final_state=final_state,
        history=history,
        metric=metric,
        potential=potential,
        sources=sources,
        numerics=numerics,
        physical=physical,
    )

    print("\nSaved files:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
