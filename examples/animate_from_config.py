"""
Run a YAML-configured simulation and generate simple animations.

Examples
--------
    python examples/animate_from_config.py \
        --config config/curved_axion_maxwell_2d.yaml

    python examples/animate_from_config.py \
        --config config/gw_axion_halo_1d.yaml

    python examples/animate_from_config.py \
        --config config/curved_constraint_cleaned_2d.yaml \
        --overlay sqrt_gamma
"""

from __future__ import annotations

import argparse
from pathlib import Path

from axion_em_gr.initial_data.factory import build_initial_state
from axion_em_gr.io.config_loader import build_simulation_objects, load_yaml_config
from axion_em_gr.visualization.animations import (
    animate_default_1d_set,
    animate_default_2d_set,
    ensure_output_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a YAML simulation and generate animations."
    )

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to YAML configuration file.",
    )

    parser.add_argument(
        "--fps",
        default=12,
        type=int,
        help="Frames per second.",
    )

    parser.add_argument(
        "--overlay",
        default="lapse",
        choices=["none", "lapse", "sqrt_gamma"],
        help="2D overlay contours.",
    )

    parser.add_argument(
        "--snapshot-every",
        default=None,
        type=int,
        help="Override snapshot interval.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_yaml_config(args.config)
    objects = build_simulation_objects(config)

    grid = objects["grid"]
    metric = objects["metric"]
    solver = objects["solver"]

    state0 = build_initial_state(config, grid, metric=metric)

    # Force snapshots on for animations, regardless of YAML.
    solver.save_snapshots = True

    if args.snapshot_every is not None:
        solver.snapshot_every = args.snapshot_every
    elif getattr(solver, "snapshot_every", None) is None:
        solver.snapshot_every = max(1, objects["numerics"].output_every)

    final_state, history = solver.evolve(state0)

    output_dir_cfg = config.get("output", {}).get("directory", "outputs/from_config")
    output_dir = ensure_output_dir(Path(output_dir_cfg) / "animations")

    overlay = None if args.overlay == "none" else args.overlay

    if grid.ndim == 1:
        paths = animate_default_1d_set(
            history=history,
            grid=grid,
            output_dir=output_dir,
            metric=metric,
            fps=args.fps,
        )

    elif grid.ndim == 2:
        paths = animate_default_2d_set(
            history=history,
            grid=grid,
            output_dir=output_dir,
            metric=metric,
            fps=args.fps,
            overlay=overlay,
        )

    else:
        raise NotImplementedError("Animations currently support only 1D and 2D.")

    print("\nAnimations written to:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()